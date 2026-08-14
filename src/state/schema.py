"""Validated, tenant-scoped state for the institutional RAG workflow.

The graph state is a checkpoint boundary, not a general-purpose application
cache.  The models below deliberately contain identifiers, bounded message
history, references and control evidence only.  Retrieved document bodies stay
in the TTL-bound context cache and are represented here by ``retrieval_handle``.

LangGraph applies reducers to individual state channels.  Reducers therefore
validate their own updates as well as combining them.  This matters because
LangGraph documents Pydantic validation as an input-boundary guarantee; node
updates are still processed by the channel reducers.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from typing import Annotated, Any, Literal

from langchain_core.messages import AnyMessage, BaseMessage
from langgraph.graph import add_messages
from pydantic import BaseModel, ConfigDict, Field, field_validator

DataClassification = Literal["public", "internal", "confidential", "restricted"]
WorkflowStatus = Literal[
    "received",
    "retrieved",
    "generated",
    "validated",
    "completed",
    "blocked",
]


def _bounded_env_int(name: str, default: int, *, minimum: int, maximum: int) -> int:
    """Read a bounded operational limit without accepting unsafe values."""
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer.") from exc
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}.")
    return value


MAX_CHECKPOINT_MESSAGES = _bounded_env_int(
    "SHORT_TERM_MAX_MESSAGES", 24, minimum=4, maximum=256
)
SUMMARY_TRIGGER_MESSAGES = _bounded_env_int(
    "SUMMARY_TRIGGER_MESSAGES", 12, minimum=2, maximum=MAX_CHECKPOINT_MESSAGES
)
SUMMARY_KEEP_MESSAGES = _bounded_env_int(
    "SUMMARY_KEEP_MESSAGES", 8, minimum=1, maximum=MAX_CHECKPOINT_MESSAGES
)
MAX_CONTEXT_TOKENS = _bounded_env_int(
    "SHORT_TERM_MAX_TOKENS", 3000, minimum=128, maximum=100_000
)
MAX_SECURITY_EVENTS = _bounded_env_int(
    "SHORT_TERM_MAX_SECURITY_EVENTS", 64, minimum=1, maximum=1024
)
MAX_GOVERNANCE_EVENT_IDS = _bounded_env_int(
    "SHORT_TERM_MAX_GOVERNANCE_EVENTS", 64, minimum=1, maximum=1024
)
MAX_CITATIONS = _bounded_env_int(
    "MAX_RESPONSE_CITATIONS", 128, minimum=1, maximum=1024
)
MAX_MESSAGE_CONTENT_CHARS = _bounded_env_int(
    "SHORT_TERM_MAX_MESSAGE_CHARS", 20_000, minimum=256, maximum=1_000_000
)


class _ValidatedModel(BaseModel):
    """Common fail-closed configuration for data crossing a state boundary."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        validate_assignment=True,
        revalidate_instances="always",
    )

    # Keep the existing node read API stable for nested records as well as the
    # top-level state.  Writes still go through Pydantic/reducer validation.
    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)


def _validate_identifier(value: str) -> str:
    """Reject ambiguous/control-bearing identity values without normalizing them."""
    if not value or value != value.strip():
        raise ValueError("Identifiers must be non-empty and have no surrounding whitespace.")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError("Identifiers must not contain control characters.")
    return value


class CitationReference(_ValidatedModel):
    """A bounded provenance reference; never a retrieved document body."""

    marker: int = Field(strict=True, ge=1, le=MAX_CITATIONS)
    source: str = Field(min_length=1, max_length=512)
    page: int | None = Field(default=None, strict=True, ge=0, le=10_000_000)

    @field_validator("source")
    @classmethod
    def source_must_be_safe(cls, value: str) -> str:
        return _validate_identifier(value)


class SecurityEvent(_ValidatedModel):
    """Structured control outcome retained in the transactional checkpoint."""

    control: str = Field(min_length=1, max_length=128)
    outcome: Literal["blocked", "allowed"]
    reason: str = Field(min_length=1, max_length=256)

    @field_validator("control", "reason")
    @classmethod
    def control_text_must_be_safe(cls, value: str) -> str:
        return _validate_identifier(value)


def _validated_messages(value: Iterable[AnyMessage] | None) -> list[AnyMessage]:
    """Validate reducer input without relying on a later model reconstruction."""
    messages = list(value or [])
    for message in messages:
        if not isinstance(message, BaseMessage):
            raise TypeError("State messages must be LangChain message instances.")
        if len(str(message.content)) > MAX_MESSAGE_CONTENT_CHARS:
            raise ValueError(
                f"A message exceeds the {MAX_MESSAGE_CONTENT_CHARS}-character limit."
            )
    return messages


def add_bounded_messages(
    left: list[AnyMessage] | None, right: list[AnyMessage] | None
) -> list[AnyMessage]:
    """Apply LangGraph message semantics and enforce a durable history bound.

    ``add_messages`` preserves message-id replacement semantics, which is safer
    for human review/correction than a blind ``operator.add`` reducer.  System
    instructions are retained ahead of the newest non-system turns.
    """
    messages = list(add_messages(_validated_messages(left), _validated_messages(right)))
    system = [message for message in messages if message.type == "system"]
    non_system = [message for message in messages if message.type != "system"]
    if len(system) >= MAX_CHECKPOINT_MESSAGES:
        return system[-MAX_CHECKPOINT_MESSAGES:]
    remaining = MAX_CHECKPOINT_MESSAGES - len(system)
    return system + non_system[-remaining:]


def estimate_message_tokens(message: AnyMessage) -> int:
    """Cheap provider-independent token estimate for context budgeting."""
    return max(1, len(str(message.content)) // 4)


def context_messages(messages: list[AnyMessage]) -> list[AnyMessage]:
    """Keep system instructions and the newest messages within the token budget."""
    messages = _validated_messages(messages)
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


def replace_optional_text(left: str | None, right: str | None) -> str | None:
    """Explicit replacement reducer for bounded text channels."""
    if right is not None:
        if not isinstance(right, str):
            raise TypeError("State text channels accept strings or None.")
        if len(right) > MAX_MESSAGE_CONTENT_CHARS:
            raise ValueError("State text exceeds the configured size limit.")
        if any(ord(character) < 32 and character not in "\n\t" for character in right):
            raise ValueError("State text contains a control character.")
    return right


def replace_citations(
    left: list[CitationReference] | None,
    right: list[CitationReference] | None,
) -> list[CitationReference]:
    """Replace citations with a validated, bounded provenance set."""
    del left
    citations = [CitationReference.model_validate(item) for item in (right or [])]
    if len(citations) > MAX_CITATIONS:
        raise ValueError(f"A response cannot contain more than {MAX_CITATIONS} citations.")
    markers = [citation.marker for citation in citations]
    if len(markers) != len(set(markers)):
        raise ValueError("Citation markers must be unique within a response.")
    return citations


def append_security_events(
    left: list[SecurityEvent] | None, right: list[SecurityEvent] | None
) -> list[SecurityEvent]:
    """Append validated control outcomes and keep checkpoint growth bounded."""
    current = [SecurityEvent.model_validate(item) for item in (left or [])]
    updates = [SecurityEvent.model_validate(item) for item in (right or [])]
    return (current + updates)[-MAX_SECURITY_EVENTS:]


def _validate_event_id(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("Governance event IDs must be strings.")
    return _validate_identifier(value)


def append_governance_event_ids(
    left: list[str] | None, right: list[str] | None
) -> list[str]:
    """Append auditable evidence references without unbounded checkpoint growth."""
    current = [_validate_event_id(value) for value in (left or [])]
    updates = [_validate_event_id(value) for value in (right or [])]
    return (current + updates)[-MAX_GOVERNANCE_EVENT_IDS:]


_STATUS_ORDER: dict[WorkflowStatus, int] = {
    "received": 1,
    "retrieved": 2,
    "generated": 3,
    "validated": 4,
    "completed": 5,
    "blocked": 5,
}


def reduce_status(
    left: WorkflowStatus | None, right: WorkflowStatus | None
) -> WorkflowStatus | None:
    """Permit only known lifecycle transitions; terminal runs may be re-entered.

    A new invocation on the same durable thread starts at ``received`` after a
    previous ``completed``/``blocked`` run.  All other backwards transitions
    are rejected so a malformed or manually edited update cannot move the graph
    to an ambiguous lifecycle state.
    """
    if right is None:
        return left
    if right not in _STATUS_ORDER:
        raise ValueError(f"Unknown workflow status: {right!r}.")
    if left is None or left == right:
        return right
    if left in {"completed", "blocked"} and right == "received":
        return right
    if right == "blocked":
        return right
    if _STATUS_ORDER[right] >= _STATUS_ORDER[left]:
        return right
    raise ValueError(f"Invalid workflow status transition: {left!r} -> {right!r}.")


class AgentInput(_ValidatedModel):
    """Strict, identity-bound input accepted from the trusted API boundary."""

    messages: list[AnyMessage] = Field(min_length=1)
    request_id: str = Field(min_length=1, max_length=256)
    tenant_id: str = Field(min_length=1, max_length=256)
    subject_id: str = Field(min_length=1, max_length=256)
    thread_id: str = Field(min_length=1, max_length=256)
    roles: tuple[str, ...] = Field(min_length=1, max_length=32)
    data_classification: DataClassification
    presentation_preferences: dict[str, str] = Field(default_factory=dict, max_length=32)
    conversation_summary: str | None = Field(default=None, max_length=MAX_MESSAGE_CONTENT_CHARS)
    long_term_memories: list[str] = Field(default_factory=list, max_length=32)

    @field_validator("request_id", "tenant_id", "subject_id", "thread_id")
    @classmethod
    def identifiers_must_be_safe(cls, value: str) -> str:
        return _validate_identifier(value)

    @field_validator("roles")
    @classmethod
    def roles_must_be_verified_values(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value):
            raise ValueError("Roles must not contain duplicates.")
        for role in value:
            _validate_identifier(role)
        return value

    @field_validator("messages")
    @classmethod
    def messages_must_contain_a_question(
        cls, value: list[AnyMessage]
    ) -> list[AnyMessage]:
        messages = _validated_messages(value)
        if not any(message.type == "human" and str(message.content).strip() for message in messages):
            raise ValueError("The workflow requires a non-empty human message.")
        return messages

    @field_validator("long_term_memories")
    @classmethod
    def memories_must_be_bounded(cls, value: list[str]) -> list[str]:
        if any(not item.strip() or len(item) > MAX_MESSAGE_CONTENT_CHARS for item in value):
            raise ValueError("Long-term memory entries must be non-empty and bounded.")
        return value

    @field_validator("conversation_summary")
    @classmethod
    def summary_must_be_safe(cls, value: str | None) -> str | None:
        return replace_optional_text(None, value)


class AgentOutput(_ValidatedModel):
    """Only this validated projection leaves the orchestration boundary."""

    response_text: str = Field(min_length=1, max_length=MAX_MESSAGE_CONTENT_CHARS)
    citations: list[CitationReference] = Field(max_length=MAX_CITATIONS)
    status: Literal["completed", "blocked"]
    request_id: str = Field(min_length=1, max_length=256)

    @field_validator("request_id")
    @classmethod
    def output_request_id_must_be_safe(cls, value: str) -> str:
        return _validate_identifier(value)

    @field_validator("response_text")
    @classmethod
    def response_text_must_be_safe(cls, value: str) -> str:
        return replace_optional_text(None, value) or value


class AgentState(_ValidatedModel):
    """Durable internal state; document bodies and secrets are intentionally absent.

    Identity fields have empty defaults solely so small isolated graph tests can
    exercise the message reducer.  Production entrypoints use ``AgentInput``,
    which requires every identity and authorization field and rejects extras.
    """

    messages: Annotated[list[AnyMessage], add_bounded_messages] = Field(
        default_factory=list
    )
    request_id: str = Field(default="", max_length=256)
    tenant_id: str = Field(default="", max_length=256)
    subject_id: str = Field(default="", max_length=256)
    thread_id: str = Field(default="", max_length=256)
    roles: tuple[str, ...] = Field(default_factory=tuple, max_length=32)
    data_classification: DataClassification = "public"
    presentation_preferences: dict[str, str] = Field(default_factory=dict, max_length=32)
    conversation_summary: Annotated[str | None, replace_optional_text] = None
    long_term_memories: list[str] = Field(default_factory=list, max_length=32)
    retrieval_handle: Annotated[str | None, replace_optional_text] = None
    citations: Annotated[list[CitationReference], replace_citations] = Field(
        default_factory=list
    )
    response_text: Annotated[str | None, replace_optional_text] = None
    status: Annotated[WorkflowStatus | None, reduce_status] = None
    security_events: Annotated[list[SecurityEvent], append_security_events] = Field(
        default_factory=list
    )
    governance_event_ids: Annotated[list[str], append_governance_event_ids] = Field(
        default_factory=list
    )

    @field_validator("request_id", "tenant_id", "subject_id", "thread_id")
    @classmethod
    def state_identifiers_must_be_safe(cls, value: str) -> str:
        if value:
            return _validate_identifier(value)
        return value

__all__ = [
    "AgentInput",
    "AgentOutput",
    "AgentState",
    "CitationReference",
    "DataClassification",
    "MAX_CHECKPOINT_MESSAGES",
    "MAX_CONTEXT_TOKENS",
    "SecurityEvent",
    "SUMMARY_KEEP_MESSAGES",
    "SUMMARY_TRIGGER_MESSAGES",
    "WorkflowStatus",
    "add_bounded_messages",
    "append_governance_event_ids",
    "append_security_events",
    "context_messages",
    "estimate_message_tokens",
    "reduce_status",
    "replace_citations",
    "replace_optional_text",
]
