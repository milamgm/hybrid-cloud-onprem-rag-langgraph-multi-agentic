"""Generation: the G in RAG.

Takes the question and the chunks the retriever found, assembles them into a
grounded prompt, calls the LLM, and returns an answer with citations. This is
the only place in the pipeline where new text is produced.

**Grounding is the whole job.** A language model asked a question it cannot
answer from the supplied context will answer it anyway, from training data, in
the same confident register as a correct answer. The system prompt below is the
control for that: answer only from the context, and say so when the context is
insufficient. It is a prompt, not a library, and it is the single highest-impact
line of code in this module.

**Citations are what make the answer auditable.** Every piece of metadata the
pipeline has carried since the loader -- `source`, `page`, `headings` -- exists
for this moment. Without citations a user cannot verify a claim, and a wrong
answer is indistinguishable from a right one.

**One protocol, two deployments.** The module talks to any OpenAI-compatible
Chat Completions endpoint. Azure AI Foundry, vLLM, NVIDIA NIM and LM Studio all
speak it, so moving from a laptop-sized model to a self-hosted production model
is a change of base URL and model name, not a change of code. That portability
is why Chat Completions is used here rather than the newer Responses API, which
the on-premise serving stacks do not implement.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger("pipeline.generator")

# The refusal string is a constant so callers can detect the "no answer" case
# without parsing prose, and so it stays identical across every path that can
# produce it (empty retrieval, and the model declining).
NO_ANSWER = "No encuentro esa información en los documentos disponibles."

DEFAULT_SYSTEM_PROMPT = f"""\
Eres un asistente documental. Respondes exclusivamente a partir del CONTEXTO \
que se te proporciona.

Reglas:
1. Usa únicamente la información del CONTEXTO. No añadas conocimiento propio, \
aunque estés seguro de que es correcto.
2. Si el CONTEXTO no contiene la respuesta, responde exactamente: \
"{NO_ANSWER}" y no añadas nada más.
3. Cita cada afirmación con el marcador del fragmento del que procede, en \
formato [1], [2]. Una afirmación sin cita no es aceptable.
4. Si el CONTEXTO se contradice, exponlo en lugar de elegir una versión.
5. Responde en el idioma de la pregunta, de forma concisa y directa.\
"""


@dataclass
class Citation:
    """A retrieved chunk, numbered so the answer can point at it."""

    marker: int
    source: str
    page: int | None = None
    headings: list[str] = field(default_factory=list)

    def label(self) -> str:
        """Human-readable reference, e.g. '[1] handbook.pdf, p. 3 - Deployment'."""
        parts = [self.source]
        if self.page is not None:
            parts.append(f"p. {self.page}")
        if self.headings:
            parts.append(self.headings[-1])
        return f"[{self.marker}] " + ", ".join(parts)


@dataclass
class GeneratedAnswer:
    """An answer plus everything needed to audit it."""

    text: str
    citations: list[Citation] = field(default_factory=list)
    context_documents: list[Document] = field(default_factory=list)

    @property
    def is_refusal(self) -> bool:
        """True when the model declined for lack of grounding."""
        return NO_ANSWER in self.text

    def formatted_sources(self) -> str:
        """The citation list, for display under the answer."""
        return "\n".join(citation.label() for citation in self.citations)


class Generator:
    """Turns retrieved chunks into a grounded, cited answer.

    Attributes:
        llm: Any LangChain chat model. Injected rather than constructed here so
            that cloud and on-premise deployments differ only in configuration,
            and so tests can substitute a fake without a network.
        system_prompt: The grounding contract. Override with care -- weakening
            rule 2 is what turns a retrieval system into a confident fabricator.
    """

    def __init__(
        self,
        llm: Any,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ):
        self._llm = llm
        self._system_prompt = system_prompt

    def _build_citations(self, documents: list[Document]) -> list[Citation]:
        """Numbers the documents 1..n, matching the markers used in the prompt."""
        citations = []
        for position, doc in enumerate(documents, start=1):
            citations.append(
                Citation(
                    marker=position,
                    source=doc.metadata.get("source", "desconocido"),
                    page=doc.metadata.get("page"),
                    headings=list(doc.metadata.get("headings") or []),
                )
            )
        return citations

    def _format_context(
        self, documents: list[Document], citations: list[Citation]
    ) -> str:
        """Renders the chunks as a numbered block the model can cite from.

        Each fragment is labelled with its own marker and provenance. Stating
        the source inside the block, not only in a separate list, is what lets
        the model attribute a claim to the right document when several
        fragments disagree.
        """
        blocks = []
        for doc, citation in zip(documents, citations, strict=True):
            header = f"[{citation.marker}] Fuente: {citation.source}"
            if citation.page is not None:
                header += f" | Página: {citation.page}"
            if citation.headings:
                header += f" | Sección: {' > '.join(citation.headings)}"
            blocks.append(f"{header}\n{doc.page_content}")

        return "\n\n---\n\n".join(blocks)

    def generate(self, question: str, documents: list[Document]) -> GeneratedAnswer:
        """Answers `question` using only `documents`.

        Args:
            question: The user's question, verbatim.
            documents: Chunks from the retriever, best first. Order is
                preserved into the prompt because models attend less reliably
                to the middle of a long context -- the "lost in the middle"
                effect -- so the strongest evidence belongs at the edges.

        Returns:
            The answer, its citations, and the context it was given.
        """
        if not documents:
            # Calling the model with no context invites it to answer from
            # training data, which is precisely the failure this module exists
            # to prevent. Refuse without spending a request.
            logger.info("No documents retrieved; refusing without calling the LLM.")
            return GeneratedAnswer(text=NO_ANSWER)

        citations = self._build_citations(documents)
        context = self._format_context(documents, citations)

        messages = [
            SystemMessage(content=self._system_prompt),
            HumanMessage(content=f"CONTEXTO:\n{context}\n\nPREGUNTA: {question}"),
        ]

        response = self._llm.invoke(messages)
        text = response.content if hasattr(response, "content") else str(response)

        answer = GeneratedAnswer(
            text=text.strip(),
            citations=citations,
            context_documents=documents,
        )

        if answer.is_refusal:
            logger.info(f"Model declined for lack of grounding: {question!r}")

        return answer
