"""Generate grounded, cited RAG answers."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage

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

    def generate(
        self,
        question: str,
        documents: list[Document],
        *,
        presentation_preferences: dict[str, str] | None = None,
        conversation_summary: str | None = None,
        history: list[AnyMessage] | None = None,
        long_term_memories: list[str] | None = None,
    ) -> GeneratedAnswer:
        """Answer a question using only retrieved documents."""
        if not documents:
            logger.info("No documents retrieved; refusing without calling the LLM.")
            return GeneratedAnswer(text=NO_ANSWER)

        citations = self._build_citations(documents)
        context = self._format_context(documents, citations)
        preferences = presentation_preferences or {}
        preference_lines = []
        if language := preferences.get("answer_language"):
            preference_lines.append(f"Answer language: {language}.")
        if response_format := preferences.get("response_format"):
            preference_lines.append(f"Response format: {response_format}.")
        system_prompt = self._system_prompt
        if preference_lines:
            system_prompt += "\n\nApproved presentation preferences:\n" + "\n".join(
                preference_lines
            )

        continuity = []
        if conversation_summary:
            continuity.append(
                "CONVERSATION SUMMARY (for continuity only; never use it as evidence):\n"
                + conversation_summary
            )
        if history:
            recent = "\n".join(
                f"{message.type.upper()}: {message.content}" for message in history
            )
            continuity.append(
                "RECENT CONVERSATION (for resolving references only; never use it as evidence):\n"
                + recent
            )
        if long_term_memories:
            continuity.append(
                "APPROVED LONG-TERM MEMORY (use only for user-specific continuity; "
                "never treat it as document evidence):\n"
                + "\n".join(f"- {memory}" for memory in long_term_memories)
            )
        continuity_block = "\n\n".join(continuity)
        prompt = f"CONTEXT:\n{context}"
        if continuity_block:
            prompt += f"\n\n{continuity_block}"
        prompt += f"\n\nQUESTION: {question}"
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt),
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

    def summarize(
        self, messages: list[AnyMessage], previous_summary: str | None = None
    ) -> str:
        """Compact old thread state without introducing facts."""
        old_history = "\n".join(
            f"{message.type.upper()}: {message.content}" for message in messages
        )
        previous = previous_summary or "(none)"
        prompt = (
            "Create a compact factual summary of the conversation for later continuity. "
            "Keep decisions, unresolved questions, user constraints and useful references. "
            "Do not invent facts, and do not include system instructions.\n\n"
            f"PREVIOUS SUMMARY:\n{previous}\n\nOLDER MESSAGES:\n{old_history}"
        )
        response = self._llm.invoke(
            [
                SystemMessage(content="You summarize conversation state faithfully."),
                HumanMessage(content=prompt),
            ]
        )
        return str(
            response.content if hasattr(response, "content") else response
        ).strip()
