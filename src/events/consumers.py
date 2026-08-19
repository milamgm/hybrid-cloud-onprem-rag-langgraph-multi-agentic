"""Consumers for alert intake, asynchronous case enrichment and reasoning."""

from __future__ import annotations

from typing import Any, Protocol

from langgraph.types import Command
from pydantic import ValidationError

from src.events.contracts import (
    EVIDENCE_COLLECTION_REQUESTED_TYPE,
    EVIDENCE_READY_TYPE,
    RISK_ALERT_TYPE,
    EvidenceCollectionRequested,
    EvidenceReady,
    TransactionRiskAlert,
    make_event,
)
from src.events.transport import EventDelivery, EventLedger, EventPublisher, EventSubscription
from src.forensics.evidence import CaseEvidenceCollector, MockCoreBankingReadGateway, MockPolicyRAGGateway
from src.hitl.tracing import case_graph_config


class ForensicGraph(Protocol):
    async def ainvoke(self, input: dict[str, Any], *, config: dict[str, Any]) -> dict[str, Any]: ...


class BaseEventConsumer:
    def __init__(self, *, ledger: EventLedger, max_attempts: int = 5) -> None:
        if max_attempts < 1: raise ValueError("max_attempts must be positive")
        self._ledger, self._max_attempts = ledger, max_attempts

    async def _claim(self, delivery: EventDelivery) -> bool: return await self._ledger.claim(delivery.event.id)

    async def _retry_or_dead_letter(self, delivery: EventDelivery, reason: str) -> None:
        if delivery.delivery_attempt >= self._max_attempts: await delivery.dead_letter(reason)
        else: await delivery.retry(reason)


class RiskAlertConsumer(BaseEventConsumer):
    """Validate a model alert and open a durable case; no analyst is paged yet."""
    def __init__(self, *, graph: ForensicGraph, ledger: EventLedger, max_attempts: int = 5) -> None:
        super().__init__(ledger=ledger, max_attempts=max_attempts); self._graph = graph

    async def process(self, delivery: EventDelivery) -> dict[str, Any]:
        try:
            if delivery.event.type != RISK_ALERT_TYPE: raise ValueError(f"unexpected event type: {delivery.event.type}")
            alert = TransactionRiskAlert.model_validate(delivery.event.data)
        except (ValidationError, ValueError) as exc:
            await delivery.dead_letter(f"invalid risk alert: {exc}"); return {"status": "dead_lettered"}
        if not await self._claim(delivery): await delivery.ack(); return {"status": "duplicate"}
        investigation_id, request_id = f"investigation-{alert.alert_id}", f"{delivery.event.id}:request:v1"
        try:
            result = await self._graph.ainvoke({"alert": alert, "investigation_id": investigation_id, "request_id": request_id, "stage": "alert_received"}, config=case_graph_config(tenant_id=alert.tenant_id, case_id=investigation_id, request_id=request_id))
        except Exception as exc:
            await self._ledger.release(delivery.event.id); await self._retry_or_dead_letter(delivery, f"case intake failed: {exc}"); return {"status": "retrying"}
        await self._ledger.complete(delivery.event.id); await delivery.ack(); return result

    async def run(self, subscription: EventSubscription) -> None:
        async for delivery in subscription: await self.process(delivery)


class EvidenceRequestedConsumer(BaseEventConsumer):
    """Use controlled read tools to enrich a case, outside the reasoning graph."""
    def __init__(self, *, collector: CaseEvidenceCollector, publisher: EventPublisher, ledger: EventLedger, max_attempts: int = 5, source: str = "service://forensic-evidence-worker") -> None:
        super().__init__(ledger=ledger, max_attempts=max_attempts); self._collector, self._publisher, self._source = collector, publisher, source

    async def process(self, delivery: EventDelivery) -> dict[str, Any]:
        try:
            if delivery.event.type != EVIDENCE_COLLECTION_REQUESTED_TYPE: raise ValueError(f"unexpected event type: {delivery.event.type}")
            request = EvidenceCollectionRequested.model_validate(delivery.event.data)
        except (ValidationError, ValueError) as exc:
            await delivery.dead_letter(f"invalid evidence request: {exc}"); return {"status": "dead_lettered"}
        if not await self._claim(delivery): await delivery.ack(); return {"status": "duplicate"}
        try:
            evidence = await self._collector.collect(request)
            output = make_event(event_type=EVIDENCE_READY_TYPE, source=self._source, subject=request.investigation_id, data=EvidenceReady(investigation_id=request.investigation_id, request_id=request.request_id, alert=request.alert, evidence=evidence), traceparent=delivery.event.traceparent, event_id=f"{request.investigation_id}:evidence_ready:v1")
            await self._publisher.publish(output, key=request.alert.tenant_id)
        except Exception as exc:
            await self._ledger.release(delivery.event.id); await self._retry_or_dead_letter(delivery, f"evidence collection failed: {exc}"); return {"status": "retrying"}
        await self._ledger.complete(delivery.event.id); await delivery.ack(); return {"status": "published", "event_id": output.id}

    async def run(self, subscription: EventSubscription) -> None:
        async for delivery in subscription: await self.process(delivery)


class EvidenceReadyConsumer(BaseEventConsumer):
    """Resume reasoning with validated evidence, producing a report then HITL."""
    def __init__(self, *, graph: ForensicGraph, ledger: EventLedger, max_attempts: int = 5) -> None:
        super().__init__(ledger=ledger, max_attempts=max_attempts); self._graph = graph

    async def process(self, delivery: EventDelivery) -> dict[str, Any]:
        try:
            if delivery.event.type != EVIDENCE_READY_TYPE: raise ValueError(f"unexpected event type: {delivery.event.type}")
            ready = EvidenceReady.model_validate(delivery.event.data)
        except (ValidationError, ValueError) as exc:
            await delivery.dead_letter(f"invalid evidence-ready event: {exc}"); return {"status": "dead_lettered"}
        if not await self._claim(delivery): await delivery.ack(); return {"status": "duplicate"}
        try:
            result = await self._graph.ainvoke(
                Command(
                    update={
                        "alert": ready.alert,
                        "evidence": ready.evidence,
                        "investigation_id": ready.investigation_id,
                        "request_id": ready.request_id,
                        "stage": "evidence_ready",
                    },
                    goto="prepare_case_for_review",
                ),
                config=case_graph_config(
                    tenant_id=ready.alert.tenant_id,
                    case_id=ready.investigation_id,
                    request_id=ready.request_id,
                ),
            )
        except Exception as exc:
            await self._ledger.release(delivery.event.id); await self._retry_or_dead_letter(delivery, f"case reasoning failed: {exc}"); return {"status": "retrying"}
        await self._ledger.complete(delivery.event.id); await delivery.ack(); return result

    async def run(self, subscription: EventSubscription) -> None:
        async for delivery in subscription: await self.process(delivery)


# Kept as public aliases while callers migrate their worker names.
FactsRequestedConsumer = EvidenceRequestedConsumer
FactsReadyConsumer = EvidenceReadyConsumer
MockTransactionFactsReader = MockCoreBankingReadGateway
