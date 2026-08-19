"""Durable, event-separated banking case investigation graph."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Literal

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from pydantic import BaseModel, ConfigDict, Field

from src.agents.forensic import ForensicReasoner
from src.events.contracts import (
    APPROVAL_REQUESTED_TYPE,
    EVIDENCE_COLLECTION_REQUESTED_TYPE,
    EXECUTION_ORDER_REQUESTED_TYPE,
    HUMAN_APPROVAL_GRANTED_TYPE,
    AgentAssessment,
    ApprovalRequested,
    CaseEvidenceBundle,
    EvidenceCollectionRequested,
    ExecutionOrderRequested,
    ForensicAnalysis,
    ForensicFinding,
    HumanApprovalGranted,
    InvestigationReport,
    TransactionRiskAlert,
    make_event,
)
from src.events.transport import EventPublisher
from src.hitl.models import ApprovalDecision, HumanApprovalPrompt


def append_event_ids(current: list[str], update: list[str]) -> list[str]:
    return list(dict.fromkeys([*current, *update]))[-128:]


def append_findings(
    current: list[ForensicFinding], update: list[ForensicFinding]
) -> list[ForensicFinding]:
    return [*current, *update][-64:]


class ForensicInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    alert: TransactionRiskAlert
    investigation_id: str = Field(min_length=1, max_length=128)
    request_id: str = Field(min_length=1, max_length=128)
    stage: Literal["alert_received", "evidence_ready", "approved"]
    evidence: CaseEvidenceBundle | None = None


class ForensicState(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    alert: TransactionRiskAlert | None = None
    evidence: CaseEvidenceBundle | None = None
    investigation_id: str = ""
    request_id: str = ""
    stage: Literal[
        "alert_received",
        "evidence_requested",
        "evidence_ready",
        "review_requested",
        "approved",
        "rejected",
        "execution_requested",
    ] = "alert_received"
    findings: Annotated[list[ForensicFinding], append_findings] = Field(
        default_factory=list
    )
    report: InvestigationReport | None = None
    event_ids: Annotated[list[str], append_event_ids] = Field(default_factory=list)
    approval_request_id: str | None = None
    human_decision: ApprovalDecision | None = None


class ForensicOutput(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)
    investigation_id: str
    stage: str
    report: InvestigationReport | None = None
    findings: list[ForensicFinding] = Field(default_factory=list)
    event_ids: list[str] = Field(default_factory=list)
    approval_request_id: str | None = None


class MockForensicReasoner:
    """Deterministic test double; production injects ``ForensicInvestigationTeam``."""

    async def analyze(
        self, alert: TransactionRiskAlert, evidence: CaseEvidenceBundle
    ) -> ForensicAnalysis:
        history_count = int(
            evidence.core_banking.attributes.get("historical_alert_count", 0)
        )
        if alert.risk_score >= 0.65 or history_count > 1:
            findings = (
                ForensicFinding(
                    finding_code="elevated_transaction_risk",
                    rationale="Model signal and validated customer/transaction evidence require analyst disposition.",
                    confidence=max(alert.risk_score, 0.8),
                    disposition="escalate",
                ),
            )
            action = "hold"
        else:
            findings = (
                ForensicFinding(
                    finding_code="insufficient_adverse_signal",
                    rationale="Validated evidence does not meet the escalation threshold.",
                    confidence=1 - alert.risk_score,
                    disposition="monitor",
                ),
            )
            action = "review"
        return ForensicAnalysis(
            summary="Deterministic development assessment; not an LLM agent output.",
            findings=findings,
            recommended_action=action,
            agent_assessments=(
                AgentAssessment(
                    agent_name="case_lead",
                    summary="Mock assessment used only in development and tests.",
                    findings=findings,
                    evidence_ids=(evidence.core_banking.evidence_id,),
                    policy_citation_ids=tuple(
                        item.citation_id for item in evidence.policy_citations
                    ),
                ),
            ),
        )


class ForensicGraphDependencies:
    def __init__(
        self,
        *,
        evidence_publisher: EventPublisher,
        review_publisher: EventPublisher,
        approval_granted_publisher: EventPublisher | None = None,
        execution_publisher: EventPublisher | None = None,
        reasoner: ForensicReasoner,
        source: str = "service://forensic-orchestrator",
    ) -> None:
        self.evidence_publisher = evidence_publisher
        self.review_publisher = review_publisher
        self.approval_granted_publisher = approval_granted_publisher
        self.execution_publisher = execution_publisher
        self.reasoner = reasoner
        self.source = source


async def _route_stage(
    state: ForensicState,
) -> Literal[
    "request_evidence", "prepare_case_for_review", "request_execution_order", "__end__"
]:
    if state.stage == "alert_received":
        return "request_evidence"
    if state.stage == "evidence_ready":
        return "prepare_case_for_review"
    if state.stage == "approved":
        return "request_execution_order"
    if state.stage in {"rejected", "execution_requested"}:
        return "__end__"
    raise ValueError(f"unsupported forensic graph stage: {state.stage}")


async def _route_after_human_approval(
    state: ForensicState,
) -> Literal["request_execution_order", "__end__"]:
    return "request_execution_order" if state.stage == "approved" else "__end__"


def _alert_from_state(state: ForensicState) -> TransactionRiskAlert | None:
    return (
        TransactionRiskAlert.model_validate(
            state.alert.model_dump(mode="python")
            if isinstance(state.alert, BaseModel)
            else state.alert,
            strict=False,
        )
        if state.alert is not None
        else None
    )


def _evidence_from_state(state: ForensicState) -> CaseEvidenceBundle | None:
    return (
        CaseEvidenceBundle.model_validate(
            state.evidence.model_dump(mode="python")
            if isinstance(state.evidence, BaseModel)
            else state.evidence,
            strict=False,
        )
        if state.evidence is not None
        else None
    )


def _report_from_state(state: ForensicState) -> InvestigationReport | None:
    return (
        InvestigationReport.model_validate(
            state.report.model_dump(mode="python")
            if isinstance(state.report, BaseModel)
            else state.report,
            strict=False,
        )
        if state.report is not None
        else None
    )


def _decision_from_state(state: ForensicState) -> ApprovalDecision | None:
    return (
        ApprovalDecision.model_validate(
            state.human_decision.model_dump(mode="python")
            if isinstance(state.human_decision, BaseModel)
            else state.human_decision,
            strict=False,
        )
        if state.human_decision is not None
        else None
    )


def build_forensic_graph(dependencies: ForensicGraphDependencies, *, checkpointer=None):
    """Build a graph that pauses only after a human-reviewable case report exists."""
    if checkpointer is None:
        raise ValueError("forensic HITL graph requires a checkpointer")

    async def request_evidence(state: ForensicState):
        alert = _alert_from_state(state)
        if alert is None:
            raise ValueError("forensic graph requires an alert")
        request = EvidenceCollectionRequested(
            investigation_id=state.investigation_id,
            request_id=state.request_id,
            alert=alert,
        )
        event = make_event(
            event_type=EVIDENCE_COLLECTION_REQUESTED_TYPE,
            source=dependencies.source,
            subject=state.investigation_id,
            data=request,
            event_id=f"{state.investigation_id}:evidence_requested:v1",
        )
        await dependencies.evidence_publisher.publish(event, key=alert.tenant_id)
        return {"stage": "evidence_requested", "event_ids": [event.id]}

    async def prepare_case_for_review(state: ForensicState):
        alert = _alert_from_state(state)
        evidence = _evidence_from_state(state)
        if alert is None or evidence is None:
            raise ValueError("case preparation requires alert and evidence")
        analysis = await dependencies.reasoner.analyze(alert, evidence)
        report = InvestigationReport(
            report_id=f"report-{state.investigation_id}:v1",
            investigation_id=state.investigation_id,
            summary=analysis.summary,
            findings=analysis.findings,
            evidence_ids=(evidence.core_banking.evidence_id,),
            policy_citation_ids=tuple(
                item.citation_id for item in evidence.policy_citations
            ),
            recommended_action=analysis.recommended_action,
            created_at=datetime.now(UTC),
            agent_assessments=analysis.agent_assessments,
        )
        approval_request_id = f"approval-{state.investigation_id}:v1"
        request = ApprovalRequested(
            investigation_id=state.investigation_id,
            request_id=state.request_id,
            approval_request_id=approval_request_id,
            report=report,
        )
        event = make_event(
            event_type=APPROVAL_REQUESTED_TYPE,
            source=dependencies.source,
            subject=state.investigation_id,
            data=request,
            event_id=f"{state.investigation_id}:review_requested:v1",
        )
        await dependencies.review_publisher.publish(event, key=alert.tenant_id)
        return {
            "stage": "review_requested",
            "findings": list(analysis.findings),
            "report": report,
            "event_ids": [event.id],
            "approval_request_id": approval_request_id,
        }

    async def human_approval(state: ForensicState):
        """No side effect precedes the interrupt: re-entry is therefore safe."""
        alert = _alert_from_state(state)
        report = _report_from_state(state)
        approval_id = state.approval_request_id
        if alert is None or report is None or not approval_id:
            raise ValueError("human approval requires a complete case report")
        prompt = HumanApprovalPrompt(
            approval_request_id=approval_id,
            investigation_id=state.investigation_id,
            requested_action=report.recommended_action,
            risk_score=alert.risk_score,
            finding_codes=tuple(f.finding_code for f in report.findings),
            report_id=report.report_id,
            evidence_ids=report.evidence_ids,
            policy_citation_ids=report.policy_citation_ids,
            idempotency_key=f"{state.investigation_id}:execution:v1",
            message="Review the cited case report and approve or reject the proposed control action.",
        )
        decision = ApprovalDecision.model_validate(
            interrupt(prompt.model_dump(mode="json"))
        )
        if decision.approval_request_id != approval_id:
            raise ValueError("resumed approval does not match the pending request")
        event_ids: list[str] = []
        if dependencies.approval_granted_publisher is not None:
            granted = HumanApprovalGranted(
                investigation_id=state.investigation_id,
                approval_request_id=approval_id,
                approver_ref=decision.approver_ref,
                decision=decision.decision,
                decided_at=decision.decided_at,
                reason=decision.reason,
            )
            event = make_event(
                event_type=HUMAN_APPROVAL_GRANTED_TYPE,
                source=dependencies.source,
                subject=state.investigation_id,
                data=granted,
                event_id=f"{state.investigation_id}:approval_granted:v1",
            )
            await dependencies.approval_granted_publisher.publish(
                event, key=alert.tenant_id
            )
            event_ids.append(event.id)
        return {
            "stage": "approved" if decision.decision == "approved" else "rejected",
            "human_decision": decision,
            "event_ids": event_ids,
        }

    async def request_execution_order(state: ForensicState):
        if dependencies.execution_publisher is None:
            raise RuntimeError("execution publisher is required after human approval")
        alert = _alert_from_state(state)
        report = _report_from_state(state)
        decision = _decision_from_state(state)
        approval_id = state.approval_request_id
        if (
            alert is None
            or report is None
            or decision is None
            or approval_id is None
            or decision.decision != "approved"
        ):
            raise ValueError("approved case report is required for an execution order")
        order = ExecutionOrderRequested(
            investigation_id=state.investigation_id,
            approval_request_id=approval_id,
            order_type=report.recommended_action,
            idempotency_key=f"{state.investigation_id}:execution:v1",
        )
        event = make_event(
            event_type=EXECUTION_ORDER_REQUESTED_TYPE,
            source=dependencies.source,
            subject=state.investigation_id,
            data=order,
            event_id=f"{state.investigation_id}:execution_order_requested:v1",
        )
        await dependencies.execution_publisher.publish(event, key=alert.tenant_id)
        return {"stage": "execution_requested", "event_ids": [event.id]}

    builder = StateGraph(
        ForensicState,
        input_schema=ForensicInput,
        output_schema=ForensicOutput,
    )
    builder.add_node("request_evidence", request_evidence)
    builder.add_node("prepare_case_for_review", prepare_case_for_review)
    builder.add_node("human_approval", human_approval)
    builder.add_node("request_execution_order", request_execution_order)
    builder.add_conditional_edges(START, _route_stage)
    builder.add_edge("request_evidence", END)
    builder.add_edge("prepare_case_for_review", "human_approval")
    builder.add_conditional_edges(
        "human_approval",
        _route_after_human_approval,
    )
    builder.add_edge("request_execution_order", END)
    return builder.compile(checkpointer=checkpointer)
