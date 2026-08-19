"""Versioned, minimised contracts for the forensic case workflow.

Events carry model signals, references and bounded evidence summaries.  They do
not carry raw account records, RAG chunks or free-form SQL, so the broker is
not a secondary core-banking database.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Generic, Literal, TypeVar
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

RISK_ALERT_TYPE = "risk.transaction.alert.v1"
EVIDENCE_COLLECTION_REQUESTED_TYPE = "forensic.case.evidence_requested.v1"
EVIDENCE_READY_TYPE = "forensic.case.evidence_ready.v1"
CASE_REVIEW_REQUESTED_TYPE = "forensic.case.review_requested.v1"
HUMAN_APPROVAL_GRANTED_TYPE = "forensic.case.approval_granted.v1"
EXECUTION_ORDER_REQUESTED_TYPE = "forensic.execution.order_requested.v1"

# Compatibility names for deployments that configured the original topic names.
FACTS_REQUESTED_TYPE = EVIDENCE_COLLECTION_REQUESTED_TYPE
FACTS_READY_TYPE = EVIDENCE_READY_TYPE
APPROVAL_REQUESTED_TYPE = CASE_REVIEW_REQUESTED_TYPE


class _EventModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, validate_assignment=True, protected_namespaces=())


class TransactionSample(_EventModel):
    """Minimal, immutable model input; ``customer_id`` is a reference only."""

    transaction_id: str = Field(min_length=1, max_length=128)
    tenant_id: str = Field(min_length=1, max_length=128)
    customer_id: str = Field(min_length=1, max_length=128)
    amount: float = Field(ge=0, le=100_000_000)
    velocity_24h: int = Field(ge=0, le=1_000_000)
    geo_distance_km: float = Field(ge=0, le=50_000)
    channel: Literal["card", "transfer", "cash", "online", "unknown"] = "unknown"


class TransactionRiskAlert(_EventModel):
    """Model signal that opens a case; it is never an execution instruction."""

    alert_id: str = Field(min_length=1, max_length=128)
    transaction_id: str = Field(min_length=1, max_length=128)
    tenant_id: str = Field(min_length=1, max_length=128)
    customer_id: str = Field(min_length=1, max_length=128)
    risk_score: float = Field(ge=0, le=1)
    severity: Literal["low", "medium", "high", "critical"]
    model_name: Literal["xgboost"] = "xgboost"
    model_version: str = Field(min_length=1, max_length=64)
    signal_codes: tuple[str, ...] = Field(min_length=1, max_length=32)
    occurred_at: datetime
    data_classification: Literal["internal", "confidential", "restricted"] = "restricted"


class CoreBankingEvidence(_EventModel):
    """Bounded output of approved, parameterised core/AML read operations."""

    evidence_id: str = Field(min_length=1, max_length=128)
    tenant_id: str = Field(min_length=1, max_length=128)
    customer_id: str = Field(min_length=1, max_length=128)
    source_system: str = Field(min_length=1, max_length=128)
    observed_at: datetime
    attributes: dict[str, str | int | float | bool] = Field(default_factory=dict, max_length=64)


class ReadOnlyEvidence(_EventModel):
    """Bounded result from an allowlisted read tool other than core banking."""

    evidence_id: str = Field(min_length=1, max_length=128)
    tenant_id: str = Field(min_length=1, max_length=128)
    customer_id: str = Field(min_length=1, max_length=128)
    source_system: str = Field(min_length=1, max_length=128)
    observed_at: datetime
    attributes: dict[str, str | int | float | bool] = Field(default_factory=dict, max_length=64)


class PolicyCitation(_EventModel):
    """Provenance-only result from the curated policy/RAG corpus."""

    citation_id: str = Field(min_length=1, max_length=128)
    source: str = Field(min_length=1, max_length=512)
    page: int | None = Field(default=None, ge=1)
    section: str | None = Field(default=None, max_length=256)
    excerpt: str = Field(min_length=1, max_length=1_500)


class CaseEvidenceBundle(_EventModel):
    """Validated, data-minimised evidence supplied to the reasoning node."""

    bundle_id: str = Field(min_length=1, max_length=128)
    core_banking: CoreBankingEvidence
    policy_citations: tuple[PolicyCitation, ...] = Field(min_length=1, max_length=8)
    collected_at: datetime


class ForensicFinding(_EventModel):
    finding_code: str = Field(min_length=1, max_length=128)
    rationale: str = Field(min_length=1, max_length=2_000)
    confidence: float = Field(ge=0, le=1)
    disposition: Literal["clear", "monitor", "escalate"]


AgentName = Literal[
    "transaction_analyst",
    "customer_risk_analyst",
    "network_analyst",
    "policy_compliance_analyst",
    "case_lead",
]


class AgentAssessment(_EventModel):
    """Structured, attributable output from one bounded forensic agent."""

    agent_name: AgentName
    summary: str = Field(min_length=1, max_length=4_000)
    findings: tuple[ForensicFinding, ...] = Field(min_length=1, max_length=16)
    evidence_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=64)
    policy_citation_ids: tuple[str, ...] = Field(default_factory=tuple, max_length=16)
    information_gaps: tuple[str, ...] = Field(default_factory=tuple, max_length=16)


class ForensicAnalysis(_EventModel):
    """The case-lead decision package assembled from specialist assessments."""

    summary: str = Field(min_length=1, max_length=4_000)
    findings: tuple[ForensicFinding, ...] = Field(min_length=1, max_length=64)
    recommended_action: Literal["hold", "review", "clear"]
    agent_assessments: tuple[AgentAssessment, ...] = Field(min_length=1, max_length=8)


class InvestigationReport(_EventModel):
    """Human-reviewable case artefact; claims must link to evidence references."""

    report_id: str = Field(min_length=1, max_length=128)
    investigation_id: str = Field(min_length=1, max_length=128)
    summary: str = Field(min_length=1, max_length=4_000)
    findings: tuple[ForensicFinding, ...] = Field(min_length=1, max_length=64)
    evidence_ids: tuple[str, ...] = Field(min_length=1, max_length=64)
    policy_citation_ids: tuple[str, ...] = Field(min_length=1, max_length=8)
    recommended_action: Literal["hold", "review", "clear"]
    created_at: datetime
    agent_assessments: tuple[AgentAssessment, ...] = Field(default_factory=tuple)


class EvidenceCollectionRequested(_EventModel):
    investigation_id: str = Field(min_length=1, max_length=128)
    request_id: str = Field(min_length=1, max_length=128)
    alert: TransactionRiskAlert


class EvidenceReady(_EventModel):
    investigation_id: str = Field(min_length=1, max_length=128)
    request_id: str = Field(min_length=1, max_length=128)
    alert: TransactionRiskAlert
    evidence: CaseEvidenceBundle


class ApprovalRequested(_EventModel):
    """Notification for the analyst work queue, emitted only after a report exists."""

    investigation_id: str = Field(min_length=1, max_length=128)
    request_id: str = Field(min_length=1, max_length=128)
    approval_request_id: str = Field(min_length=1, max_length=128)
    report: InvestigationReport


class HumanApprovalGranted(_EventModel):
    investigation_id: str = Field(min_length=1, max_length=128)
    approval_request_id: str = Field(min_length=1, max_length=128)
    approver_ref: str = Field(min_length=1, max_length=128)
    decision: Literal["approved", "rejected"]
    decided_at: datetime
    reason: str = Field(min_length=1, max_length=2_000)


class ExecutionOrderRequested(_EventModel):
    investigation_id: str = Field(min_length=1, max_length=128)
    approval_request_id: str = Field(min_length=1, max_length=128)
    order_type: Literal["hold", "release", "review"]
    idempotency_key: str = Field(min_length=1, max_length=256)


# Deprecated aliases keep imports stable while the event schema changes in place.
FactsRequested = EvidenceCollectionRequested
FactsReady = EvidenceReady
TransactionFacts = CoreBankingEvidence

PayloadT = TypeVar("PayloadT")


class EventEnvelope(_EventModel, Generic[PayloadT]):
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


def make_event(*, event_type: str, source: str, subject: str, data: PayloadT, traceparent: str | None = None, event_id: str | None = None) -> EventEnvelope[PayloadT]:
    return EventEnvelope(id=event_id or str(uuid4()), source=source, type=event_type, subject=subject, time=datetime.now(timezone.utc), traceparent=traceparent, data=data)
