"""Composition-root wiring for the provider-neutral forensic pipeline."""

from __future__ import annotations

from dataclasses import dataclass

from src.events.consumers import (
    FactsReadyConsumer,
    FactsRequestedConsumer,
    MockTransactionFactsReader,
    RiskAlertConsumer,
)
from src.events.risk_model import MockXGBoostModel
from src.events.settings import EventTopologySettings
from src.events.transport import EventLedger, EventPublisher, InMemoryEventBus, InMemoryEventLedger
from src.graph.forensic_graph import ForensicGraphDependencies, MockForensicReasoner, build_forensic_graph


@dataclass(frozen=True, slots=True)
class EventBindings:
    """One publisher per architectural layer, supplied by deployment code."""

    stream: EventPublisher
    transactional: EventPublisher
    reactive: EventPublisher

    @classmethod
    def in_memory(cls, bus: InMemoryEventBus) -> "EventBindings":
        return cls(stream=bus, transactional=bus, reactive=bus)


@dataclass(slots=True)
class ForensicRuntime:
    settings: EventTopologySettings
    risk_model: MockXGBoostModel
    graph: object
    alert_consumer: RiskAlertConsumer
    facts_requested_consumer: FactsRequestedConsumer
    facts_ready_consumer: FactsReadyConsumer


def build_forensic_runtime(
    settings: EventTopologySettings,
    bindings: EventBindings,
    *,
    ledger: EventLedger | None = None,
) -> ForensicRuntime:
    """Wire services without importing or constructing broker SDK clients.

    Production callers create the appropriate adapter from
    ``src.events.adapters`` and pass it in ``bindings``. The stateful ledger is
    also a port in this first slice; replace it with a durable unique-key store
    before using more than one process.
    """

    event_ledger = ledger or InMemoryEventLedger()
    graph = build_forensic_graph(
        ForensicGraphDependencies(
            facts_publisher=bindings.transactional,
            approval_publisher=bindings.reactive,
            reasoner=MockForensicReasoner(),
        )
    )
    return ForensicRuntime(
        settings=settings,
        risk_model=MockXGBoostModel(bindings.stream),
        graph=graph,
        alert_consumer=RiskAlertConsumer(graph=graph, ledger=event_ledger),
        facts_requested_consumer=FactsRequestedConsumer(
            reader=MockTransactionFactsReader(),
            publisher=bindings.transactional,
            ledger=event_ledger,
        ),
        facts_ready_consumer=FactsReadyConsumer(graph=graph, ledger=event_ledger),
    )
