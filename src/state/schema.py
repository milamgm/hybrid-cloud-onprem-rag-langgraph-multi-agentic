"""Minimal checkpoint schema for a tenant-isolated institutional workflow."""

from __future__ import annotations

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
MAX_CHECKPOINT_MESSAGES = 24


def add_bounded_messages(
    left: list[AnyMessage], right: list[AnyMessage]
) -> list[AnyMessage]:
    """Apply LangGraph message semantics and cap checkpoint growth."""
    return list(add_messages(left, right))[-MAX_CHECKPOINT_MESSAGES:]


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
    retrieval_handle: NotRequired[str]
    citations: NotRequired[list[CitationReference]]
    response_text: NotRequired[str]
    status: NotRequired[WorkflowStatus]
    security_events: Annotated[list[dict[str, Any]], add]
    governance_event_ids: Annotated[list[str], add]
