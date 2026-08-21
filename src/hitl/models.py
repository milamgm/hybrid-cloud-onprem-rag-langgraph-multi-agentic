"""Strict, JSON-safe HITL request and response models."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class _HITLModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        validate_assignment=True,
        protected_namespaces=(),
    )


class HumanApprovalPrompt(_HITLModel):
    """The only payload exposed by the interrupt to a reviewer."""

    approval_request_id: str = Field(min_length=1, max_length=128)
    investigation_id: str = Field(min_length=1, max_length=128)
    requested_action: Literal["hold", "release", "review"]
    risk_score: float = Field(ge=0, le=1)
    finding_codes: tuple[str, ...] = Field(min_length=1, max_length=64)
    report_id: str = Field(min_length=1, max_length=128)
    evidence_ids: tuple[str, ...] = Field(min_length=1, max_length=64)
    policy_citation_ids: tuple[str, ...] = Field(min_length=1, max_length=8)
    idempotency_key: str = Field(min_length=1, max_length=256)
    message: str = Field(min_length=1, max_length=1_000)


class ApprovalDecision(_HITLModel):
    """Verified human input returned through ``Command(resume=...)``."""

    approval_request_id: str = Field(min_length=1, max_length=128)
    decision: Literal["approved", "rejected"]
    approver_ref: str = Field(min_length=1, max_length=128)
    reason: str = Field(min_length=1, max_length=2_000)
    decided_at: datetime


class HITLStatus(_HITLModel):
    thread_id: str = Field(min_length=1, max_length=256)
    investigation_id: str = Field(min_length=1, max_length=128)
    status: Literal[
        "not_found",
        "running",
        "awaiting_human",
        "execution_pending",
        "completed",
        "rejected",
    ]
    approval_request_id: str | None = None
    interrupts: tuple[dict[str, object], ...] = ()
    checkpoint_id: str | None = None


class HITLResumeResult(_HITLModel):
    thread_id: str = Field(min_length=1, max_length=256)
    status: Literal[
        "awaiting_human", "approved", "rejected", "execution_requested", "completed"
    ]
    approval_request_id: str
    state: dict[str, object] = Field(default_factory=dict)
    event_ids: tuple[str, ...] = ()
