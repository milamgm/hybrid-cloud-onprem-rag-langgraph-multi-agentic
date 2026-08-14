"""Identity-bound application facade over the governed RAG graph."""

from __future__ import annotations

import hashlib
from typing import Any

from langchain_core.messages import HumanMessage

from src.memory.store import MemoryManager
from src.state.schema import AgentOutput, DataClassification


class RAGAgent:
    """Run one deterministic workflow in a tenant-isolated LangGraph thread."""

    def __init__(self, graph: Any, *, memory: MemoryManager | None = None) -> None:
        self._graph = graph
        self._memory = memory

    def invoke(
        self,
        question: str,
        *,
        request_id: str,
        tenant_id: str,
        subject_id: str,
        thread_id: str,
        roles: tuple[str, ...],
        data_classification: DataClassification,
    ) -> dict:
        """Invoke with identity claims already verified by the API boundary."""
        if not all((request_id, tenant_id, subject_id, thread_id)):
            raise ValueError("Identity and correlation identifiers are mandatory.")
        if not roles:
            raise ValueError("At least one verified workload/user role is mandatory.")
        preferences = (
            self._memory.presentation_preferences(tenant_id, subject_id)
            if self._memory
            else {}
        )
        recalled_memories = (
            [
                record.memory_text
                for record in self._memory.recall(
                    tenant_id, subject_id, question, limit=3
                )
            ]
            if self._memory
            else []
        )
        raw_result = self._graph.invoke(
            {
                "messages": [HumanMessage(content=question)],
                "request_id": request_id,
                "tenant_id": tenant_id,
                "subject_id": subject_id,
                "thread_id": thread_id,
                "roles": roles,
                "data_classification": data_classification,
                "presentation_preferences": preferences,
                "long_term_memories": recalled_memories,
            },
            {
                "configurable": {
                    "thread_id": self._checkpoint_thread_id(
                        tenant_id, subject_id, thread_id
                    )
                }
            },
        )
        # LangGraph intentionally returns a mapping even when its state/output
        # schemas are Pydantic models.  Validate and serialize the public
        # projection here so callers never receive an unvalidated checkpoint
        # shape or nested Pydantic instances.
        return AgentOutput.model_validate(raw_result).model_dump(mode="python")

    def delete_thread(self, *, tenant_id: str, subject_id: str, thread_id: str) -> None:
        """Delete durable checkpoints for an exact tenant/subject thread."""
        checkpointer = getattr(self._graph, "checkpointer", None)
        if checkpointer is None:
            raise RuntimeError("The graph has no durable checkpointer.")
        checkpointer.delete_thread(
            self._checkpoint_thread_id(tenant_id, subject_id, thread_id)
        )

    @staticmethod
    def _checkpoint_thread_id(tenant_id: str, subject_id: str, thread_id: str) -> str:
        """Prevent caller-controlled IDs from colliding across identity scopes."""
        value = f"{tenant_id}\x00{subject_id}\x00{thread_id}".encode()
        return hashlib.sha256(value).hexdigest()
