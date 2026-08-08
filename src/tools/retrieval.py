"""The retrieval tool: the RAG pipeline as a capability an agent can invoke.

This module is the seam between the deterministic pipeline in :mod:`src.rag` and
the non-deterministic agent that will drive it. Nothing here retrieves anything
itself -- it wraps :class:`~src.rag.retriever.HybridRetriever` and re-presents it
in the one format a language model can act on: a name, a description, a typed
argument, and text coming back.

**Why the tool returns fragments and not an answer.** The tempting alternative is
to call :class:`~src.rag.generator.Generator` inside the tool and hand the agent
a finished paragraph. Published comparisons of agentic and non-agentic RAG test
the other shape -- a single tool returning documents, with the agent deciding
whether to reformulate or answer -- and it is the shape that preserves the
agent's leverage. An agent given raw evidence can notice that two fragments
disagree, that the corpus answers half the question, or that a second query is
needed. An agent given a finished paragraph can only forward it, and every
reasoning step the agent might have contributed has already been spent inside
the tool. The generator keeps its job as the non-agentic baseline that this
system has to beat on measurements, not as a step in the agent's loop.

**Why reranking stays.** The same comparisons find that an agent's iterative
re-retrieval rarely improves on what an explicit cross-encoder rerank already
produced, while costing multiples in tokens and latency. So the agentic layer is
added for what it is genuinely good at -- interpreting intent, reformulating,
deciding when to stop -- and the precision stage stays exactly where it was.

**Why the tool returns text and an artifact.** The model needs prose it can read
and cite; the application needs the original :class:`Document` objects to render
sources and to audit an answer afterwards. LangChain's ``content_and_artifact``
response format carries both: the string goes into the conversation and costs
tokens, the artifact rides alongside on the ``ToolMessage`` and costs none.
Flattening documents into the prompt and losing the objects would mean rebuilding
provenance later by string-matching, which is exactly the fragility the metadata
was carried through the pipeline to avoid.
"""

from __future__ import annotations

import logging

from langchain_core.documents import Document
from langchain_core.tools import BaseTool, tool

logger = logging.getLogger("pipeline.tools.retrieval")

# Per-fragment character budget. Chunks are already token-bounded by
# MAX_CHUNK_TOKENS, so this is a backstop against an outlier -- a table or a
# recital that chunked badly -- rather than routine truncation. Tool results are
# the largest uncontrolled contributor to an agent's context, and an agent that
# runs several searches pays this cost once per fragment per search.
MAX_FRAGMENT_CHARS = 1_500

# Total budget across all fragments in one call. Bounds the worst case: k
# fragments that each hit the per-fragment cap.
MAX_RESULT_CHARS = 8_000

# Returned verbatim when the corpus has nothing. Phrased as an instruction
# rather than an error because the recipient is a model choosing its next
# action: a bare "no results" invites a retry of the same failing query, while
# naming the two useful moves -- rephrase, or concede -- makes the useful path
# the obvious one.
NO_RESULTS = (
    "Sin resultados para esta consulta. El corpus no contiene fragmentos "
    "relevantes, o la consulta usa términos que no aparecen en él. Reformula "
    "con la terminología del propio corpus (por ejemplo, el vocabulario del "
    "articulado en lugar de una paráfrasis coloquial) o indica al usuario que "
    "esta información no está disponible. No respondas con conocimiento propio."
)

DEFAULT_TOOL_NAME = "search_regulations"

# The description is the tool's real interface. The model never reads this
# module; it reads these lines and decides from them whether this tool applies.
# Three things earn their place here and are not padding: what the corpus
# actually contains (so the model can tell an answerable question from an
# unanswerable one before spending a call), how to phrase a query for a hybrid
# dense+BM25 index (so exact identifiers survive into the sparse arm), and what
# comes back (so the model plans on citing markers rather than inventing them).
DEFAULT_TOOL_DESCRIPTION = """\
Busca fragmentos literales en el corpus normativo indexado (Reglamento Europeo \
de Inteligencia Artificial y Reglamento General de Protección de Datos, texto \
consolidado: articulado, considerandos y anexos).

Úsala siempre que la pregunta dependa de lo que dice la norma: obligaciones, \
plazos, definiciones, clasificación de riesgo, régimen sancionador o el ámbito \
de aplicación. Es la única fuente admisible para esas afirmaciones.

Redacta la consulta con la terminología de la norma y conserva los \
identificadores exactos ("artículo 6", "Anexo III", "GPAI") en lugar de \
parafrasearlos: el índice combina búsqueda semántica y léxica, y esos términos \
son lo que ancla la segunda.

Una consulta por asunto. Si la pregunta cubre dos asuntos (por ejemplo, una \
obligación y su sanción), haz dos búsquedas.

Devuelve fragmentos numerados [1], [2]... con su fuente, página y sección. Cita \
esos marcadores en la respuesta. Si no devuelve nada, la norma indexada no lo \
cubre.\
"""


def _format_fragment(marker: int, document: Document) -> str:
    """Renders one chunk as a cited block the model can quote from.

    The provenance goes in a header above the text rather than in a separate
    list at the end. A model attributing a claim reads the fragment and its
    label together; splitting them forces it to hold a mapping in working
    memory, and mis-attribution is the failure that follows.
    """
    metadata = document.metadata
    header_parts = [f"[{marker}]", f"Fuente: {metadata.get('source', 'desconocido')}"]

    page = metadata.get("page")
    if page is not None:
        header_parts.append(f"Página: {page}")

    headings = metadata.get("headings") or []
    if headings:
        header_parts.append(f"Sección: {' > '.join(headings)}")

    text = document.page_content.strip()
    if len(text) > MAX_FRAGMENT_CHARS:
        # Marked rather than silently cut: a model that can see the fragment was
        # truncated can say so, instead of reporting a partial list as complete.
        text = text[:MAX_FRAGMENT_CHARS].rstrip() + " […fragmento truncado]"

    return " | ".join(header_parts) + "\n" + text


def _format_results(documents: list[Document]) -> str:
    """Joins the fragments into one block, stopping at the character budget."""
    blocks: list[str] = []
    used = 0

    for marker, document in enumerate(documents, start=1):
        block = _format_fragment(marker, document)
        # Stop on the first fragment that would overflow rather than skipping it
        # and continuing. Results arrive best-first, so everything after an
        # overflow is weaker anyway, and contiguous markers keep the numbering
        # honest -- a gap in [1][2][4] is a citation the model cannot resolve.
        if used + len(block) > MAX_RESULT_CHARS:
            logger.debug(
                f"Result budget reached at fragment {marker}; "
                f"returning {marker - 1} of {len(documents)}."
            )
            break
        blocks.append(block)
        used += len(block)

    return "\n\n---\n\n".join(blocks)


def build_retrieval_tool(
    retriever=None,
    *,
    name: str = DEFAULT_TOOL_NAME,
    description: str = DEFAULT_TOOL_DESCRIPTION,
) -> BaseTool:
    """Builds the corpus search tool bound to a retriever.

    Args:
        retriever: Anything exposing LangChain's ``invoke(query) ->
            list[Document]``. Defaults to the configured
            :class:`~src.rag.retriever.HybridRetriever`. Injected rather than
            imported at module scope so that tests can pass a fake, and so that
            importing this module neither opens a database connection nor loads
            the cross-encoder weights.
        name: The tool name the model sees and calls.
        description: The tool description the model reasons over. Override to
            retarget the tool at a different corpus -- the mechanics below are
            corpus-independent, only these lines are not.

    Returns:
        A ``BaseTool`` whose result is ``(text, documents)``: numbered prose for
        the model, and the original documents for the application.
    """
    if retriever is None:
        from src.config.config import get_retriever

        retriever = get_retriever()

    @tool(name, description=description, response_format="content_and_artifact")
    def search_corpus(query: str) -> tuple[str, list[Document]]:
        """Signature only; `description` above is what the model actually reads."""
        documents = retriever.invoke(query)

        if not documents:
            logger.info(f"No fragments for tool query: {query!r}")
            # The artifact stays a list so callers can treat the return type
            # uniformly instead of guarding against None on the empty path.
            return NO_RESULTS, []

        logger.info(f"Retrieved {len(documents)} fragments for query: {query!r}")
        return _format_results(documents), documents

    return search_corpus
