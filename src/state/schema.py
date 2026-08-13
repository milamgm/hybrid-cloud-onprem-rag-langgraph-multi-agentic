"""Minimal checkpoint schema for a tenant-isolated institutional workflow."""

from __future__ import annotations

import os
from operator import add
from typing import Annotated, Any, Literal, NotRequired

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from typing_extensions import TypedDict

DataClassification = Literal["public", "internal", "confidential", "restricted"]
WorkflowStatus = Literal[
    "received",
    "retrieved",
    "generated",
    "validated",
    "completed",
    "blocked",
]
MAX_CHECKPOINT_MESSAGES = int(os.getenv("SHORT_TERM_MAX_MESSAGES", "24"))
SUMMARY_TRIGGER_MESSAGES = int(os.getenv("SUMMARY_TRIGGER_MESSAGES", "12"))
SUMMARY_KEEP_MESSAGES = int(os.getenv("SUMMARY_KEEP_MESSAGES", "8"))
MAX_CONTEXT_TOKENS = int(os.getenv("SHORT_TERM_MAX_TOKENS", "3000"))


def add_bounded_messages(
    left: list[AnyMessage], right: list[AnyMessage]
) -> list[AnyMessage]:
    """Apply LangGraph semantics while preserving system instructions."""
    messages = list(add_messages(left, right))
    system = [message for message in messages if message.type == "system"]
    non_system = [message for message in messages if message.type != "system"]
    return (system + non_system[-MAX_CHECKPOINT_MESSAGES:])[-MAX_CHECKPOINT_MESSAGES:]


def estimate_message_tokens(message: AnyMessage) -> int:
    """Cheap provider-independent token estimate for context budgeting."""
    return max(1, len(str(message.content)) // 4)


def context_messages(messages: list[AnyMessage]) -> list[AnyMessage]:
    """Keep system instructions and the newest messages within the token budget."""
    system = [message for message in messages if message.type == "system"]
    non_system = [message for message in messages if message.type != "system"]
    selected: list[AnyMessage] = []
    budget = sum(estimate_message_tokens(message) for message in system)
    for message in reversed(non_system):
        cost = estimate_message_tokens(message)
        if selected and budget + cost > MAX_CONTEXT_TOKENS:
            break
        selected.append(message)
        budget += cost
    return system + list(reversed(selected))


class CitationReference(TypedDict):
    """A provenance reference without retrieved document content."""

    marker: int
    source: str
    page: int | None


class AgentInput(TypedDict):
    """Identity-bound input supplied by the trusted API boundary."""

    messages: list[AnyMessage]
    request_id: str
    tenant_id: str
    subject_id: str
    thread_id: str
    roles: tuple[str, ...]
    data_classification: DataClassification
    presentation_preferences: NotRequired[dict[str, str]]
    conversation_summary: NotRequired[str]
    long_term_memories: NotRequired[list[str]]


class AgentOutput(TypedDict):
    """Only the validated response leaves the orchestration boundary."""

    response_text: str
    citations: list[CitationReference]
    status: Literal["completed", "blocked"]
    request_id: str


class AgentState(TypedDict):
    """Durable execution state; no secrets or retrieved document bodies.

    Messages are encrypted by the production checkpointer and bounded to limit
    privacy exposure and checkpoint growth. RAG bodies live only in a TTL-bound
    context cache referenced by ``retrieval_handle``.
    """

    messages: Annotated[list[AnyMessage], add_bounded_messages]
    request_id: str
    tenant_id: str
    subject_id: str
    thread_id: str
    roles: tuple[str, ...]
    data_classification: DataClassification
    presentation_preferences: NotRequired[dict[str, str]]
    conversation_summary: NotRequired[str]
    long_term_memories: NotRequired[list[str]]
    retrieval_handle: NotRequired[str]
    citations: NotRequired[list[CitationReference]]
    response_text: NotRequired[str]
    status: NotRequired[WorkflowStatus]
    security_events: Annotated[list[dict[str, Any]], add]
    governance_event_ids: Annotated[list[str], add]
