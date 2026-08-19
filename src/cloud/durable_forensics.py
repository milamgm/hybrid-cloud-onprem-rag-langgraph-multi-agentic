"""Azure Durable Functions adapter for the multi-agent forensic workflow.

The orchestrator schedules idempotent activities only.  Agent/LLM calls,
database reads and broker I/O stay inside activities because Durable Functions
replays orchestrators and therefore requires deterministic orchestrator code.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

from src.agents.forensic import ForensicReasoner
from src.events.contracts import (
    APPROVAL_REQUESTED_TYPE,
    EXECUTION_ORDER_REQUESTED_TYPE,
    ApprovalRequested,
    CaseEvidenceBundle,
    EvidenceCollectionRequested,
    ExecutionOrderRequested,
    InvestigationReport,
    TransactionRiskAlert,
    make_event,
)
from src.events.transport import EventPublisher
from src.forensics.evidence import CaseEvidenceCollector

COLLECT_EVIDENCE_ACTIVITY = "collect_forensic_evidence"
RUN_AGENTS_ACTIVITY = "run_forensic_agents"
PUBLISH_REVIEW_ACTIVITY = "publish_forensic_review"
PUBLISH_EXECUTION_ACTIVITY = "publish_forensic_execution"
HUMAN_APPROVAL_EVENT = "human_approval"


class DurableOrchestrationContext(Protocol):
    """Small protocol that keeps the deterministic orchestration unit-testable."""

    def get_input(self) -> dict[str, Any]: ...

    def call_activity(self, name: str, input_: dict[str, Any]) -> Any: ...

    def wait_for_external_event(self, name: str) -> Any: ...


def fraud_case_orchestrator(context: DurableOrchestrationContext):
    """Coordinate evidence, agents and human approval through Durable Functions.

    Do not add LLM calls, clocks, random IDs, database work, HTTP requests or
    log side effects here.  Durable Functions replays this generator; those
    operations belong in the named activities below.
    """

    case = context.get_input()
    evidence = yield context.call_activity(COLLECT_EVIDENCE_ACTIVITY, case)
    report = yield context.call_activity(
        RUN_AGENTS_ACTIVITY, {**case, "evidence": evidence}
    )
    yield context.call_activity(PUBLISH_REVIEW_ACTIVITY, {**case, "report": report})
    decision = yield context.wait_for_external_event(HUMAN_APPROVAL_EVENT)
    if decision.get("decision") != "approved":
        return {
            "status": "rejected",
            "investigation_id": case["investigation_id"],
            "report_id": report["report_id"],
        }
    yield context.call_activity(
        PUBLISH_EXECUTION_ACTIVITY,
        {**case, "report": report, "approval": decision},
    )
    return {
        "status": "execution_requested",
        "investigation_id": case["investigation_id"],
        "report_id": report["report_id"],
    }


@dataclass(slots=True)
class DurableForensicActivities:
    """Concrete work performed by Azure Function activity handlers."""

    collector: CaseEvidenceCollector
    reasoner: ForensicReasoner
    review_publisher: EventPublisher
    execution_publisher: EventPublisher
    source: str = "service://durable-forensic-orchestrator"

    async def collect_evidence(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = EvidenceCollectionRequested(
            investigation_id=str(payload["investigation_id"]),
            request_id=str(payload["request_id"]),
            alert=TransactionRiskAlert.model_validate(payload["alert"]),
        )
        evidence = await self.collector.collect(request)
        return evidence.model_dump(mode="json")

    async def run_agents(self, payload: dict[str, Any]) -> dict[str, Any]:
        alert = TransactionRiskAlert.model_validate(payload["alert"])
        evidence = CaseEvidenceBundle.model_validate(payload["evidence"])
        analysis = await self.reasoner.analyze(alert, evidence)
        report = InvestigationReport(
            report_id=f"report-{payload['investigation_id']}:v1",
            investigation_id=str(payload["investigation_id"]),
            summary=analysis.summary,
            findings=analysis.findings,
            evidence_ids=(evidence.core_banking.evidence_id,),
            policy_citation_ids=tuple(
                citation.citation_id for citation in evidence.policy_citations
            ),
            recommended_action=analysis.recommended_action,
            created_at=datetime.now(UTC),
            agent_assessments=analysis.agent_assessments,
        )
        return report.model_dump(mode="json")

    async def publish_review(self, payload: dict[str, Any]) -> None:
        alert = TransactionRiskAlert.model_validate(payload["alert"])
        report = InvestigationReport.model_validate(payload["report"])
        approval_request_id = f"approval-{payload['investigation_id']}:v1"
        event = make_event(
            event_type=APPROVAL_REQUESTED_TYPE,
            source=self.source,
            subject=str(payload["investigation_id"]),
            event_id=f"{payload['investigation_id']}:review_requested:v1",
            data=ApprovalRequested(
                investigation_id=str(payload["investigation_id"]),
                request_id=str(payload["request_id"]),
                approval_request_id=approval_request_id,
                report=report,
            ),
        )
        await self.review_publisher.publish(event, key=alert.tenant_id)

    async def publish_execution(self, payload: dict[str, Any]) -> None:
        alert = TransactionRiskAlert.model_validate(payload["alert"])
        report = InvestigationReport.model_validate(payload["report"])
        approval = payload["approval"]
        approval_request_id = str(approval["approval_request_id"])
        event = make_event(
            event_type=EXECUTION_ORDER_REQUESTED_TYPE,
            source=self.source,
            subject=str(payload["investigation_id"]),
            event_id=f"{payload['investigation_id']}:execution_order_requested:v1",
            data=ExecutionOrderRequested(
                investigation_id=str(payload["investigation_id"]),
                approval_request_id=approval_request_id,
                order_type=report.recommended_action,
                idempotency_key=f"{payload['investigation_id']}:execution:v1",
            ),
        )
        await self.execution_publisher.publish(event, key=alert.tenant_id)
