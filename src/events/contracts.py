"""Versioned, validated contracts for the forensic event flow.

The payloads intentionally contain references and derived risk signals rather
than document bodies or unrestricted user input. This keeps the event log
auditable without turning it into an accidental PII store.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Generic, Literal, TypeVar
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

RISK_ALERT_TYPE = "risk.transaction.alert.v1"
FACTS_REQUESTED_TYPE = "forensic.investigation.facts_requested.v1"
FACTS_READY_TYPE = "forensic.investigation.facts_ready.v1"
APPROVAL_REQUESTED_TYPE = "forensic.investigation.approval_requested.v1"
HUMAN_APPROVAL_GRANTED_TYPE = "forensic.investigation.approval_granted.v1"
EXECUTION_ORDER_REQUESTED_TYPE = "forensic.execution.order_requested.v1"


class _EventModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        validate_assignment=True,
        protected_namespaces=(),
    )


class TransactionSample(_EventModel):
    """Minimal model input used by the mock XGBoost publisher."""

    transaction_id: str = Field(min_length=1, max_length=128)
    tenant_id: str = Field(min_length=1, max_length=128)
    subject_id: str = Field(min_length=1, max_length=128)
    amount: float = Field(ge=0, le=100_000_000)
    velocity_24h: int = Field(ge=0, le=1_000_000)
    geo_distance_km: float = Field(ge=0, le=50_000)
    channel: Literal["card", "transfer", "cash", "online", "unknown"] = "unknown"


class TransactionRiskAlert(_EventModel):
    """Immutable risk signal emitted by an analytical model."""

    alert_id: str = Field(min_length=1, max_length=128)
    transaction_id: str = Field(min_length=1, max_length=128)
    tenant_id: str = Field(min_length=1, max_length=128)
    subject_id: str = Field(min_length=1, max_length=128)
    risk_score: float = Field(ge=0, le=1)
    severity: Literal["low", "medium", "high", "critical"]
    model_name: Literal["xgboost"] = "xgboost"
    model_version: str = Field(min_length=1, max_length=64)
    signal_codes: tuple[str, ...] = Field(min_length=1, max_length=32)
    occurred_at: datetime
    data_classification: Literal["internal", "confidential", "restricted"] = "restricted"


class TransactionFacts(_EventModel):
    """Reader output passed to the reasoning stage through an event."""

    fact_id: str = Field(min_length=1, max_length=128)
    transaction_id: str = Field(min_length=1, max_length=128)
    tenant_id: str = Field(min_length=1, max_length=128)
    source_system: str = Field(min_length=1, max_length=128)
    observed_at: datetime
    attributes: dict[str, str | int | float | bool] = Field(default_factory=dict, max_length=64)


class ForensicFinding(_EventModel):
    finding_code: str = Field(min_length=1, max_length=128)
    rationale: str = Field(min_length=1, max_length=2_000)
    confidence: float = Field(ge=0, le=1)
    disposition: Literal["clear", "monitor", "escalate"]


class FactsRequested(_EventModel):
    investigation_id: str = Field(min_length=1, max_length=128)
    request_id: str = Field(min_length=1, max_length=128)
    alert: TransactionRiskAlert


class FactsReady(_EventModel):
    investigation_id: str = Field(min_length=1, max_length=128)
    request_id: str = Field(min_length=1, max_length=128)
    alert: TransactionRiskAlert
    facts: TransactionFacts


class ApprovalRequested(_EventModel):
    investigation_id: str = Field(min_length=1, max_length=128)
    request_id: str = Field(min_length=1, max_length=128)
    approval_request_id: str = Field(min_length=1, max_length=128)
    alert: TransactionRiskAlert
    findings: tuple[ForensicFinding, ...] = Field(min_length=1, max_length=64)
    requested_action: Literal["hold", "review", "clear"]


class HumanApprovalGranted(_EventModel):
    """Contract reserved for the future human-approval service."""

    investigation_id: str = Field(min_length=1, max_length=128)
    approval_request_id: str = Field(min_length=1, max_length=128)
    approver_ref: str = Field(min_length=1, max_length=128)
    decision: Literal["approved", "rejected"]
    decided_at: datetime
    reason: str = Field(min_length=1, max_length=2_000)


class ExecutionOrderRequested(_EventModel):
    """Contract only; no producer exists before human approval is implemented."""

    investigation_id: str = Field(min_length=1, max_length=128)
    approval_request_id: str = Field(min_length=1, max_length=128)
    order_type: Literal["hold", "release", "review"]
    idempotency_key: str = Field(min_length=1, max_length=256)


PayloadT = TypeVar("PayloadT")


class EventEnvelope(_EventModel, Generic[PayloadT]):
    """CloudEvents-shaped envelope with explicit schema and trace metadata."""

    specversion: Literal["1.0"] = "1.0"
    id: str = Field(min_length=1, max_length=128)
    source: str = Field(min_length=1, max_length=256)
    type: str = Field(min_length=1, max_length=256)
    subject: str = Field(min_length=1, max_length=256)
    time: datetime
    datacontenttype: Literal["application/json"] = "application/json"
    schema_version: Literal["1.0"] = "1.0"
    traceparent: str | None = Field(default=None, max_length=512)
    data: PayloadT


def make_event(
    *,
    event_type: str,
    source: str,
    subject: str,
    data: PayloadT,
    traceparent: str | None = None,
    event_id: str | None = None,
) -> EventEnvelope[PayloadT]:
    """Create an event, allowing deterministic ids for retry-safe outbox writes."""

    return EventEnvelope(
        id=event_id or str(uuid4()),
        source=source,
        type=event_type,
        subject=subject,
        time=datetime.now(timezone.utc),
        traceparent=traceparent,
        data=data,
    )
