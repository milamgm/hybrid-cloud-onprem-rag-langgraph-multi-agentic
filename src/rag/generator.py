"""Generate grounded, cited RAG answers."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger("pipeline.generator")

NO_ANSWER = "I could not find that information in the available documents."
DEFAULT_SYSTEM_PROMPT = f"""\
You are a document assistant. Answer only from the supplied CONTEXT.

Rules:
1. Use only information in the CONTEXT; do not add outside knowledge.
2. If the CONTEXT does not contain the answer, reply exactly: "{NO_ANSWER}"
3. Cite every claim with its passage marker, for example [1] or [2].
4. State contradictions in the CONTEXT instead of choosing a version.
5. Answer concisely in the question's language.\
"""


@dataclass
class Citation:
    """A retrieved chunk reference."""

    marker: int
    source: str
    page: int | None = None
    headings: list[str] = field(default_factory=list)

    def label(self) -> str:
        """Return a human-readable citation."""
        parts = [self.source]
        if self.page is not None:
            parts.append(f"p. {self.page}")
        if self.headings:
            parts.append(self.headings[-1])
        return f"[{self.marker}] " + ", ".join(parts)


@dataclass
class GeneratedAnswer:
    """A generated answer and its supporting context."""

    text: str
    citations: list[Citation] = field(default_factory=list)
    context_documents: list[Document] = field(default_factory=list)

    @property
    def is_refusal(self) -> bool:
        """Whether the answer lacks grounding."""
        return NO_ANSWER in self.text

    def formatted_sources(self) -> str:
        """Format citations for display."""
        return "\n".join(citation.label() for citation in self.citations)


class Generator:
    """Turn retrieved chunks into a grounded, cited answer."""

    def __init__(self, llm: Any, system_prompt: str = DEFAULT_SYSTEM_PROMPT):
        self._llm = llm
        self._system_prompt = system_prompt

    def _build_citations(self, documents: list[Document]) -> list[Citation]:
        """Number documents for prompt citations."""
        return [
            Citation(
                marker=position,
                source=doc.metadata.get("source", "unknown"),
                page=doc.metadata.get("page"),
                headings=list(doc.metadata.get("headings") or []),
            )
            for position, doc in enumerate(documents, start=1)
        ]

    def _format_context(
        self, documents: list[Document], citations: list[Citation]
    ) -> str:
        """Render chunks as a numbered, citable context block."""
        blocks = []
        for doc, citation in zip(documents, citations, strict=True):
            header = f"[{citation.marker}] Source: {citation.source}"
            if citation.page is not None:
                header += f" | Page: {citation.page}"
            if citation.headings:
                header += f" | Section: {' > '.join(citation.headings)}"
            blocks.append(f"{header}\n{doc.page_content}")

        return "\n\n---\n\n".join(blocks)

    def generate(self, question: str, documents: list[Document]) -> GeneratedAnswer:
        """Answer a question using only retrieved documents."""
        if not documents:
            logger.info("No documents retrieved; refusing without calling the LLM.")
            return GeneratedAnswer(text=NO_ANSWER)

        citations = self._build_citations(documents)
        context = self._format_context(documents, citations)
        messages = [
            SystemMessage(content=self._system_prompt),
            HumanMessage(content=f"CONTEXT:\n{context}\n\nQUESTION: {question}"),
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
