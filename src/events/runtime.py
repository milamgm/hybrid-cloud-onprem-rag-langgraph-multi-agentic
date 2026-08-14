"""Composition-root wiring for the provider-neutral forensic pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from src.audit.events import AuditService
from src.events.consumers import (
    EvidenceReadyConsumer,
    EvidenceRequestedConsumer,
    RiskAlertConsumer,
)
from src.events.risk_model import MockXGBoostModel
from src.events.settings import EventTopologySettings
from src.events.transport import EventLedger, EventPublisher, InMemoryEventBus, InMemoryEventLedger
from src.graph.forensic_graph import ForensicGraphDependencies, MockForensicReasoner, build_forensic_graph
from src.hitl.service import HumanApprovalService
from src.forensics.evidence import CaseEvidenceCollector, MockCoreBankingReadGateway, MockPolicyRAGGateway


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
    evidence_requested_consumer: EvidenceRequestedConsumer
    evidence_ready_consumer: EvidenceReadyConsumer
    human_approval: HumanApprovalService


def build_forensic_runtime(
    settings: EventTopologySettings,
    bindings: EventBindings,
    *,
    ledger: EventLedger | None = None,
    checkpointer: Any = None,
    audit: AuditService | None = None,
) -> ForensicRuntime:
    """Wire services without importing or constructing broker SDK clients.

    Production callers create the appropriate adapter from
    ``src.events.adapters`` and pass it in ``bindings``. The stateful ledger is
    also a port in this first slice; replace it with a durable unique-key store
    before using more than one process.
    """

    if ledger is None and (
        settings.infrastructure_mode != "memory"
        and os.getenv("DEPLOYMENT_ENVIRONMENT", "development").strip().lower() == "production"
    ):
        raise ValueError("production runtime requires a durable EventLedger")
    if checkpointer is None:
        raise ValueError("forensic runtime requires a LangGraph checkpointer")
    event_ledger = ledger or InMemoryEventLedger()
    graph = build_forensic_graph(
        ForensicGraphDependencies(
            evidence_publisher=bindings.transactional,
            review_publisher=bindings.reactive,
            approval_granted_publisher=bindings.reactive,
            execution_publisher=bindings.transactional,
            reasoner=MockForensicReasoner(),
        ),
        checkpointer=checkpointer,
    )
    return ForensicRuntime(
        settings=settings,
        risk_model=MockXGBoostModel(bindings.stream),
        graph=graph,
        alert_consumer=RiskAlertConsumer(graph=graph, ledger=event_ledger),
        evidence_requested_consumer=EvidenceRequestedConsumer(
            collector=CaseEvidenceCollector(
                core_banking=MockCoreBankingReadGateway(),
                policy_rag=MockPolicyRAGGateway(),
            ),
            publisher=bindings.transactional,
            ledger=event_ledger,
        ),
        evidence_ready_consumer=EvidenceReadyConsumer(graph=graph, ledger=event_ledger),
        human_approval=HumanApprovalService(
            graph,
            audit=audit or AuditService.from_environment(),
        ),
    )
