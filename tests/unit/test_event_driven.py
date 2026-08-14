"""Contract and end-to-end tests for the event-driven forensic path."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from src.events.contracts import (
    APPROVAL_REQUESTED_TYPE,
    EventEnvelope,
    FACTS_READY_TYPE,
    FACTS_REQUESTED_TYPE,
    RISK_ALERT_TYPE,
    TransactionSample,
)
from src.events.consumers import FactsReadyConsumer, FactsRequestedConsumer, MockTransactionFactsReader, RiskAlertConsumer
from src.events.risk_model import MockXGBoostModel
from src.events.settings import EventTopologySettings
from src.events.transport import InMemoryEventBus, InMemoryEventLedger
from src.graph.forensic_graph import ForensicGraphDependencies, MockForensicReasoner, build_forensic_graph


def test_event_pipeline_injects_alert_and_stops_at_human_approval() -> None:
    async def scenario() -> None:
        bus = InMemoryEventBus()
        ledger = InMemoryEventLedger()
        dependencies = ForensicGraphDependencies(
            facts_publisher=bus,
            approval_publisher=bus,
            reasoner=MockForensicReasoner(),
        )
        graph = build_forensic_graph(dependencies)
        alert_consumer = RiskAlertConsumer(graph=graph, ledger=ledger)
        facts_consumer = FactsRequestedConsumer(
            reader=MockTransactionFactsReader(),
            publisher=bus,
            ledger=ledger,
        )
        ready_consumer = FactsReadyConsumer(graph=graph, ledger=ledger)

        async with (
            bus.subscribe("risk.transaction.alert.v1", "model") as alerts,
            bus.subscribe(FACTS_REQUESTED_TYPE, "reader") as facts_requested,
            bus.subscribe(FACTS_READY_TYPE, "reasoner") as facts_ready,
            bus.subscribe(APPROVAL_REQUESTED_TYPE, "approval") as approvals,
        ):
            model = MockXGBoostModel(bus)
            await model.score_and_publish(
                TransactionSample(
                    transaction_id="tx-001",
                    tenant_id="tenant-a",
                    subject_id="subject-001",
                    amount=80_000,
                    velocity_24h=25,
                    geo_distance_km=3_000,
                    channel="transfer",
                )
            )

            alert_delivery = await alerts.__anext__()
            await alert_consumer.process(alert_delivery)
            assert alert_delivery.acknowledged is True

            facts_delivery = await facts_requested.__anext__()
            await facts_consumer.process(facts_delivery)
            assert facts_delivery.acknowledged is True

            ready_delivery = await facts_ready.__anext__()
            await ready_consumer.process(ready_delivery)
            assert ready_delivery.acknowledged is True

            approval_delivery = await approvals.__anext__()
            assert approval_delivery.event.type == APPROVAL_REQUESTED_TYPE
            assert approval_delivery.event.data.investigation_id == "investigation-alert-tx-001"
            assert approval_delivery.event.data.requested_action == "hold"

        event_types = [event.type for event in bus.published]
        assert event_types == [
            "risk.transaction.alert.v1",
            FACTS_REQUESTED_TYPE,
            FACTS_READY_TYPE,
            APPROVAL_REQUESTED_TYPE,
        ]
        assert not any("order_requested" in event_type for event_type in event_types)

    asyncio.run(scenario())


def test_duplicate_alert_is_acknowledged_without_restarting_graph() -> None:
    async def scenario() -> None:
        bus = InMemoryEventBus()
        ledger = InMemoryEventLedger()
        graph = build_forensic_graph(
            ForensicGraphDependencies(
                facts_publisher=bus,
                approval_publisher=bus,
                reasoner=MockForensicReasoner(),
            )
        )
        consumer = RiskAlertConsumer(graph=graph, ledger=ledger)
        async with bus.subscribe("risk.transaction.alert.v1", "model") as alerts:
            event = await MockXGBoostModel(bus).score_and_publish(
                TransactionSample(
                    transaction_id="tx-duplicate",
                    tenant_id="tenant-a",
                    subject_id="subject-001",
                    amount=1,
                    velocity_24h=0,
                    geo_distance_km=0,
                )
            )
            delivery = await alerts.__anext__()
            first = await consumer.process(delivery)
            second = await consumer.process(delivery)

        assert first["stage"] == "facts_requested"
        assert second == {"status": "duplicate"}
        assert [item.type for item in bus.published] == [event.type, FACTS_REQUESTED_TYPE]

    asyncio.run(scenario())


def test_topology_defaults_are_mode_specific(monkeypatch) -> None:
    monkeypatch.setenv("INFRASTRUCTURE_MODE", "cloud")
    cloud = EventTopologySettings.from_env()
    assert cloud.stream_backend == "azure_event_hubs"
    assert cloud.transactional_backend == "azure_service_bus"
    assert cloud.reactive_backend == "azure_event_grid"

    monkeypatch.setenv("INFRASTRUCTURE_MODE", "onprem")
    onprem = EventTopologySettings.from_env()
    assert onprem.stream_backend == "kafka"
    assert onprem.transactional_backend == "rabbitmq"
    assert onprem.reactive_backend == "nats"


def test_invalid_event_is_dead_lettered_and_not_acknowledged_as_success() -> None:
    async def scenario() -> None:
        bus = InMemoryEventBus()
        ledger = InMemoryEventLedger()

        class NeverCalledGraph:
            async def ainvoke(self, input, *, config):
                raise AssertionError("invalid payload must not reach the graph")

        consumer = RiskAlertConsumer(graph=NeverCalledGraph(), ledger=ledger)
        async with bus.subscribe(RISK_ALERT_TYPE, "model") as alerts:
            await bus.publish(
                EventEnvelope[dict](
                    id="invalid-alert",
                    source="test",
                    type=RISK_ALERT_TYPE,
                    subject="tx-invalid",
                    time=datetime.now(timezone.utc),
                    data={},
                )
            )
            delivery = await alerts.__anext__()
            result = await consumer.process(delivery)

        assert result == {"status": "dead_lettered"}
        assert len(bus.dead_letters) == 1
        assert delivery.acknowledged is True

    asyncio.run(scenario())
