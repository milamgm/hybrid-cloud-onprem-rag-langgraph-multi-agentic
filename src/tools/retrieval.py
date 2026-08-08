"""Expose corpus retrieval as an agent tool."""

from __future__ import annotations

import logging

from langchain_core.documents import Document
from langchain_core.tools import BaseTool, tool

logger = logging.getLogger("pipeline.tools.retrieval")

MAX_FRAGMENT_CHARS = 1_500
MAX_RESULT_CHARS = 8_000
NO_RESULTS = (
    "No results for this query. The corpus does not contain relevant passages, "
    "or the query uses terms absent from it. Rephrase using the corpus "
    "terminology or tell the user that this information is unavailable. Do not "
    "answer from your own knowledge."
)
DEFAULT_TOOL_NAME = "search_regulations"
DEFAULT_TOOL_DESCRIPTION = """\
Searches the indexed regulatory corpus: the consolidated EU AI Act and GDPR,
including articles, recitals, and annexes.

Use it for questions about legal obligations, deadlines, definitions, risk
classification, penalties, or scope. Use the regulation's terminology and keep
exact identifiers such as "Article 6", "Annex III", and "GPAI". Search one
topic at a time. Returns numbered passages with source, page, and section; cite
their markers in the response.\
"""


def _format_fragment(marker: int, document: Document) -> str:
    """Render a chunk with provenance."""
    metadata = document.metadata
    header_parts = [f"[{marker}]", f"Source: {metadata.get('source', 'unknown')}"]

    page = metadata.get("page")
    if page is not None:
        header_parts.append(f"Page: {page}")

    headings = metadata.get("headings") or []
    if headings:
        header_parts.append(f"Section: {' > '.join(headings)}")

    text = document.page_content.strip()
    if len(text) > MAX_FRAGMENT_CHARS:
        text = text[:MAX_FRAGMENT_CHARS].rstrip() + " […truncated]"

    return " | ".join(header_parts) + "\n" + text


def _format_results(documents: list[Document]) -> str:
    """Join fragments within the result character budget."""
    blocks: list[str] = []
    used = 0

    for marker, document in enumerate(documents, start=1):
        block = _format_fragment(marker, document)
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
    """Build a corpus search tool."""
    if retriever is None:
        from src.config.config import get_retriever

        retriever = get_retriever()

    @tool(name, description=description, response_format="content_and_artifact")
    def search_corpus(query: str) -> tuple[str, list[Document]]:
        """Search the configured corpus."""
        documents = retriever.invoke(query)

        if not documents:
            logger.info(f"No fragments for tool query: {query!r}")
            return NO_RESULTS, []

        logger.info(f"Retrieved {len(documents)} fragments for query: {query!r}")
        return _format_results(documents), documents

    return search_corpus
