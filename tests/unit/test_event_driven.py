"""Contract tests for alert -> evidence -> report -> HITL -> command."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from langgraph.checkpoint.memory import InMemorySaver

from src.events.contracts import (
    APPROVAL_REQUESTED_TYPE,
    EVIDENCE_COLLECTION_REQUESTED_TYPE,
    EVIDENCE_READY_TYPE,
    EXECUTION_ORDER_REQUESTED_TYPE,
    HUMAN_APPROVAL_GRANTED_TYPE,
    RISK_ALERT_TYPE,
    EventEnvelope,
    TransactionSample,
)
from src.events.consumers import EvidenceReadyConsumer, EvidenceRequestedConsumer, RiskAlertConsumer
from src.events.risk_model import MockXGBoostModel
from src.events.settings import EventTopologySettings
from src.events.transport import InMemoryEventBus, InMemoryEventLedger
from src.forensics.evidence import CaseEvidenceCollector, MockCoreBankingReadGateway, MockPolicyRAGGateway
from src.graph.forensic_graph import ForensicGraphDependencies, MockForensicReasoner, build_forensic_graph
from src.hitl.service import HumanApprovalService


def _dependencies(bus):
    return ForensicGraphDependencies(evidence_publisher=bus, review_publisher=bus, approval_granted_publisher=bus, execution_publisher=bus, reasoner=MockForensicReasoner())


def _collector() -> CaseEvidenceCollector:
    return CaseEvidenceCollector(core_banking=MockCoreBankingReadGateway(), policy_rag=MockPolicyRAGGateway())


def _sample(transaction_id: str) -> TransactionSample:
    return TransactionSample(transaction_id=transaction_id, tenant_id="tenant-a", customer_id="customer-001", amount=80_000, velocity_24h=25, geo_distance_km=3_000, channel="transfer")


def test_alert_is_enriched_and_reported_before_human_interrupt() -> None:
    async def scenario() -> None:
        bus, ledger = InMemoryEventBus(), InMemoryEventLedger()
        graph = build_forensic_graph(_dependencies(bus), checkpointer=InMemorySaver())
        alerts = RiskAlertConsumer(graph=graph, ledger=ledger)
        evidence = EvidenceRequestedConsumer(collector=_collector(), publisher=bus, ledger=ledger)
        ready = EvidenceReadyConsumer(graph=graph, ledger=ledger)
        async with (bus.subscribe(RISK_ALERT_TYPE, "intake") as alert_sub, bus.subscribe(EVIDENCE_COLLECTION_REQUESTED_TYPE, "evidence") as request_sub, bus.subscribe(EVIDENCE_READY_TYPE, "reasoning") as ready_sub, bus.subscribe(APPROVAL_REQUESTED_TYPE, "analyst-work-queue") as review_sub):
            await MockXGBoostModel(bus).score_and_publish(_sample("tx-001"))
            await alerts.process(await alert_sub.__anext__())
            await evidence.process(await request_sub.__anext__())
            paused = await ready.process(await ready_sub.__anext__())
            review = await review_sub.__anext__()
        assert "__interrupt__" in paused
        report = review.event.data.report
        assert report.evidence_ids == ("core-investigation-alert-tx-001",)
        assert report.policy_citation_ids
        assert report.recommended_action == "hold"
        assert [event.type for event in bus.published] == [RISK_ALERT_TYPE, EVIDENCE_COLLECTION_REQUESTED_TYPE, EVIDENCE_READY_TYPE, APPROVAL_REQUESTED_TYPE]
    asyncio.run(scenario())


def test_human_reviewer_can_resume_a_customer_case_without_becoming_state_scope() -> None:
    async def scenario() -> None:
        bus, ledger = InMemoryEventBus(), InMemoryEventLedger()
        graph = build_forensic_graph(_dependencies(bus), checkpointer=InMemorySaver())
        alerts = RiskAlertConsumer(graph=graph, ledger=ledger)
        evidence = EvidenceRequestedConsumer(collector=_collector(), publisher=bus, ledger=ledger)
        ready = EvidenceReadyConsumer(graph=graph, ledger=ledger)
        service = HumanApprovalService(graph)
        async with (bus.subscribe(RISK_ALERT_TYPE, "intake") as alert_sub, bus.subscribe(EVIDENCE_COLLECTION_REQUESTED_TYPE, "evidence") as request_sub, bus.subscribe(EVIDENCE_READY_TYPE, "reasoning") as ready_sub, bus.subscribe(APPROVAL_REQUESTED_TYPE, "analyst") as review_sub, bus.subscribe(HUMAN_APPROVAL_GRANTED_TYPE, "audit") as granted_sub, bus.subscribe(EXECUTION_ORDER_REQUESTED_TYPE, "executor") as execution_sub):
            await MockXGBoostModel(bus).score_and_publish(_sample("tx-hitl"))
            await alerts.process(await alert_sub.__anext__()); await evidence.process(await request_sub.__anext__()); await ready.process(await ready_sub.__anext__())
            review = await review_sub.__anext__()
            case_id = "investigation-alert-tx-hitl"
            status = await service.status(thread_id=case_id, tenant_id="tenant-a")
            assert status.status == "awaiting_human"
            assert status.interrupts[0]["report_id"] == review.event.data.report.report_id
            resumed = await service.resume(thread_id=case_id, tenant_id="tenant-a", approver_ref="analyst-007", approver_roles=("risk.investigation.approve",), approval_request_id=status.approval_request_id, decision="approved", reason="Evidence and cited procedure reviewed.")
            assert resumed.status == "execution_requested"
            assert (await granted_sub.__anext__()).event.data.approver_ref == "analyst-007"
            assert (await execution_sub.__anext__()).event.data.order_type == "hold"
    asyncio.run(scenario())


def test_duplicate_alert_is_acknowledged_without_starting_a_second_case() -> None:
    async def scenario() -> None:
        bus, ledger = InMemoryEventBus(), InMemoryEventLedger()
        graph = build_forensic_graph(_dependencies(bus), checkpointer=InMemorySaver())
        consumer = RiskAlertConsumer(graph=graph, ledger=ledger)
        async with bus.subscribe(RISK_ALERT_TYPE, "intake") as subscription:
            event = await MockXGBoostModel(bus).score_and_publish(_sample("tx-duplicate"))
            delivery = await subscription.__anext__()
            first, second = await consumer.process(delivery), await consumer.process(delivery)
        assert first["stage"] == "evidence_requested" and second == {"status": "duplicate"}
        assert [item.type for item in bus.published] == [event.type, EVIDENCE_COLLECTION_REQUESTED_TYPE]
    asyncio.run(scenario())


def test_topology_defaults_are_mode_specific(monkeypatch) -> None:
    monkeypatch.setenv("INFRASTRUCTURE_MODE", "cloud"); assert EventTopologySettings.from_env().stream_backend == "azure_event_hubs"
    monkeypatch.setenv("INFRASTRUCTURE_MODE", "onprem"); assert EventTopologySettings.from_env().transactional_backend == "rabbitmq"


def test_invalid_event_is_dead_lettered() -> None:
    async def scenario() -> None:
        bus, ledger = InMemoryEventBus(), InMemoryEventLedger()
        class NeverCalledGraph:
            async def ainvoke(self, input, *, config): raise AssertionError("invalid payload must not reach graph")
        consumer = RiskAlertConsumer(graph=NeverCalledGraph(), ledger=ledger)
        async with bus.subscribe(RISK_ALERT_TYPE, "intake") as subscription:
            await bus.publish(EventEnvelope[dict](id="invalid-alert", source="test", type=RISK_ALERT_TYPE, subject="tx-invalid", time=datetime.now(timezone.utc), data={}))
            result = await consumer.process(await subscription.__anext__())
        assert result == {"status": "dead_lettered"} and len(bus.dead_letters) == 1
    asyncio.run(scenario())
