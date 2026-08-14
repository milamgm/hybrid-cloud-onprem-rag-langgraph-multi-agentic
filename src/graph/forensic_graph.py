"""Event-separated LangGraph stages for a transaction forensic investigation."""

from __future__ import annotations

from typing import Annotated, Literal, Protocol

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field

from src.events.contracts import (
    APPROVAL_REQUESTED_TYPE,
    FACTS_REQUESTED_TYPE,
    ApprovalRequested,
    FactsRequested,
    ForensicFinding,
    TransactionFacts,
    TransactionRiskAlert,
    make_event,
)
from src.events.transport import EventPublisher


def append_event_ids(current: list[str], update: list[str]) -> list[str]:
    """Reducer that appends unique event ids while bounding state growth."""

    merged = list(dict.fromkeys([*current, *update]))
    return merged[-128:]


def append_findings(current: list[ForensicFinding], update: list[ForensicFinding]) -> list[ForensicFinding]:
    """Reducer that appends findings and enforces a bounded investigation state."""

    return [*current, *update][-64:]


class ForensicInput(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    alert: TransactionRiskAlert
    investigation_id: str = Field(min_length=1, max_length=128)
    request_id: str = Field(min_length=1, max_length=128)
    stage: Literal["alert_received", "facts_ready"]
    facts: TransactionFacts | None = None


class ForensicState(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    alert: TransactionRiskAlert | None = None
    facts: TransactionFacts | None = None
    investigation_id: str = ""
    request_id: str = ""
    stage: Literal["alert_received", "facts_requested", "facts_ready", "approval_requested"] = "alert_received"
    findings: Annotated[list[ForensicFinding], append_findings] = Field(default_factory=list)
    event_ids: Annotated[list[str], append_event_ids] = Field(default_factory=list)
    approval_request_id: str | None = None


class ForensicOutput(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    investigation_id: str
    stage: str
    findings: list[ForensicFinding] = Field(default_factory=list)
    event_ids: list[str] = Field(default_factory=list)
    approval_request_id: str | None = None


class ForensicReasoner(Protocol):
    async def analyze(self, alert: TransactionRiskAlert, facts: TransactionFacts) -> list[ForensicFinding]:
        """Run the agent reasoning stage over validated facts."""


class MockForensicReasoner:
    async def analyze(self, alert: TransactionRiskAlert, facts: TransactionFacts) -> list[ForensicFinding]:
        history_count = int(facts.attributes.get("historical_alert_count", 0))
        if alert.risk_score >= 0.65 or history_count > 1:
            return [
                ForensicFinding(
                    finding_code="elevated_transaction_risk",
                    rationale="The model score or prior alert history requires human review.",
                    confidence=max(alert.risk_score, 0.8),
                    disposition="escalate",
                )
            ]
        return [
            ForensicFinding(
                finding_code="insufficient_adverse_signal",
                rationale="No escalation threshold was reached by the validated facts.",
                confidence=1 - alert.risk_score,
                disposition="monitor",
            )
        ]


class ForensicGraphDependencies:
    def __init__(
        self,
        *,
        facts_publisher: EventPublisher,
        approval_publisher: EventPublisher,
        reasoner: ForensicReasoner,
        source: str = "service://forensic-orchestrator",
    ) -> None:
        self.facts_publisher = facts_publisher
        self.approval_publisher = approval_publisher
        self.reasoner = reasoner
        self.source = source


def _route_stage(state: ForensicState) -> Literal["request_facts", "reason_and_request_approval"]:
    return "request_facts" if state.get("stage") == "alert_received" else "reason_and_request_approval"


def build_forensic_graph(dependencies: ForensicGraphDependencies):
    """Build a graph whose cross-service boundaries are durable domain events."""

    async def request_facts(state: ForensicState):
        alert = state.get("alert")
        if alert is None:
            raise ValueError("forensic graph requires an alert")
        request = FactsRequested(
            investigation_id=state["investigation_id"],
            request_id=state["request_id"],
            alert=alert,
        )
        event = make_event(
            event_type=FACTS_REQUESTED_TYPE,
            source=dependencies.source,
            subject=state["investigation_id"],
            data=request,
            event_id=f"{state['investigation_id']}:facts_requested:v1",
        )
        await dependencies.facts_publisher.publish(event, key=alert.tenant_id)
        return {"stage": "facts_requested", "event_ids": [event.id]}

    async def reason_and_request_approval(state: ForensicState):
        alert = state.get("alert")
        facts = state.get("facts")
        if alert is None or facts is None:
            raise ValueError("reasoning stage requires alert and facts")
        findings = await dependencies.reasoner.analyze(alert, facts)
        approval_request_id = f"approval-{state['investigation_id']}:v1"
        request = ApprovalRequested(
            investigation_id=state["investigation_id"],
            request_id=state["request_id"],
            approval_request_id=approval_request_id,
            alert=alert,
            findings=tuple(findings),
            requested_action="hold" if any(f.disposition == "escalate" for f in findings) else "review",
        )
        event = make_event(
            event_type=APPROVAL_REQUESTED_TYPE,
            source=dependencies.source,
            subject=state["investigation_id"],
            data=request,
            event_id=f"{state['investigation_id']}:approval_requested:v1",
        )
        await dependencies.approval_publisher.publish(event, key=alert.tenant_id)
        return {
            "stage": "approval_requested",
            "findings": findings,
            "event_ids": [event.id],
            "approval_request_id": approval_request_id,
        }

    builder = StateGraph(ForensicState, input_schema=ForensicInput, output_schema=ForensicOutput)
    builder.add_node("request_facts", request_facts)
    builder.add_node("reason_and_request_approval", reason_and_request_approval)
    builder.add_conditional_edges(START, _route_stage)
    builder.add_edge("request_facts", END)
    builder.add_edge("reason_and_request_approval", END)
    return builder.compile()
