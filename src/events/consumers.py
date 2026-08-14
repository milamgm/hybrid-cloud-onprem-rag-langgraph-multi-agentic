"""Event consumers for the alert-to-investigation pipeline."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from typing import Any, Protocol

from pydantic import ValidationError

from src.events.contracts import (
    FACTS_READY_TYPE,
    FACTS_REQUESTED_TYPE,
    RISK_ALERT_TYPE,
    FactsReady,
    FactsRequested,
    TransactionFacts,
    TransactionRiskAlert,
    make_event,
)
from src.events.transport import EventDelivery, EventLedger, EventPublisher, EventSubscription


class ForensicGraph(Protocol):
    async def ainvoke(self, input: dict[str, Any], *, config: dict[str, Any]) -> dict[str, Any]:
        """Start one idempotent graph stage."""


class TransactionFactsReader(Protocol):
    async def read(self, request: FactsRequested) -> TransactionFacts:
        """Read transaction facts without knowing who requested them."""


class MockTransactionFactsReader:
    """Deterministic reader for local development and contract tests."""

    async def read(self, request: FactsRequested) -> TransactionFacts:
        alert = request.alert
        digest = sha256(alert.transaction_id.encode("utf-8")).hexdigest()
        return TransactionFacts(
            fact_id=f"facts-{request.investigation_id}",
            transaction_id=alert.transaction_id,
            tenant_id=alert.tenant_id,
            source_system="mock-ledger",
            observed_at=datetime.now(timezone.utc),
            attributes={
                "ledger_match": True,
                "counterparty_ref": f"cp-{digest[:12]}",
                "historical_alert_count": int(digest[:2], 16) % 4,
            },
        )


class BaseEventConsumer:
    def __init__(self, *, ledger: EventLedger, max_attempts: int = 5) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        self._ledger = ledger
        self._max_attempts = max_attempts

    async def _claim(self, delivery: EventDelivery) -> bool:
        return await self._ledger.claim(delivery.event.id)

    async def _retry_or_dead_letter(self, delivery: EventDelivery, reason: str) -> None:
        if delivery.delivery_attempt >= self._max_attempts:
            await delivery.dead_letter(reason)
        else:
            await delivery.retry(reason)


class RiskAlertConsumer(BaseEventConsumer):
    """Consumes model alerts and injects them into the graph input state."""

    def __init__(self, *, graph: ForensicGraph, ledger: EventLedger, max_attempts: int = 5) -> None:
        super().__init__(ledger=ledger, max_attempts=max_attempts)
        self._graph = graph

    async def process(self, delivery: EventDelivery) -> dict[str, Any]:
        try:
            event = delivery.event
            if event.type != RISK_ALERT_TYPE:
                raise ValueError(f"unexpected event type: {event.type}")
            alert = TransactionRiskAlert.model_validate(event.data)
        except (ValidationError, ValueError) as exc:
            await delivery.dead_letter(f"invalid risk alert: {exc}")
            return {"status": "dead_lettered"}

        if not await self._claim(delivery):
            await delivery.ack()
            return {"status": "duplicate"}

        investigation_id = f"investigation-{alert.alert_id}"
        request_id = f"{delivery.event.id}:request:v1"
        try:
            result = await self._graph.ainvoke(
                {
                    "alert": alert,
                    "investigation_id": investigation_id,
                    "request_id": request_id,
                    "stage": "alert_received",
                },
                config={"configurable": {"thread_id": investigation_id}},
            )
        except Exception as exc:
            await self._ledger.release(delivery.event.id)
            await self._retry_or_dead_letter(delivery, f"graph start failed: {exc}")
            return {"status": "retrying"}

        await self._ledger.complete(delivery.event.id)
        await delivery.ack()
        return result

    async def run(self, subscription: EventSubscription) -> None:
        async for delivery in subscription:
            await self.process(delivery)


class FactsRequestedConsumer(BaseEventConsumer):
    """Reads data in a separate consumer and emits only a validated result."""

    def __init__(
        self,
        *,
        reader: TransactionFactsReader,
        publisher: EventPublisher,
        ledger: EventLedger,
        max_attempts: int = 5,
        source: str = "service://transaction-facts-reader",
    ) -> None:
        super().__init__(ledger=ledger, max_attempts=max_attempts)
        self._reader = reader
        self._publisher = publisher
        self._source = source

    async def process(self, delivery: EventDelivery) -> dict[str, Any]:
        try:
            event = delivery.event
            if event.type != FACTS_REQUESTED_TYPE:
                raise ValueError(f"unexpected event type: {event.type}")
            request = FactsRequested.model_validate(event.data)
        except (ValidationError, ValueError) as exc:
            await delivery.dead_letter(f"invalid facts request: {exc}")
            return {"status": "dead_lettered"}

        if not await self._claim(delivery):
            await delivery.ack()
            return {"status": "duplicate"}

        try:
            facts = await self._reader.read(request)
            ready = FactsReady(
                investigation_id=request.investigation_id,
                request_id=request.request_id,
                alert=request.alert,
                facts=facts,
            )
            output = make_event(
                event_type=FACTS_READY_TYPE,
                source=self._source,
                subject=request.investigation_id,
                data=ready,
                traceparent=event.traceparent,
                event_id=f"{request.investigation_id}:facts_ready:v1",
            )
            await self._publisher.publish(output, key=request.alert.tenant_id)
        except Exception as exc:
            await self._ledger.release(delivery.event.id)
            await self._retry_or_dead_letter(delivery, f"facts read failed: {exc}")
            return {"status": "retrying"}

        await self._ledger.complete(delivery.event.id)
        await delivery.ack()
        return {"status": "published", "event_id": output.id}

    async def run(self, subscription: EventSubscription) -> None:
        async for delivery in subscription:
            await self.process(delivery)


class FactsReadyConsumer(BaseEventConsumer):
    """Injects data-ready events into the reasoning/approval graph stage."""

    def __init__(self, *, graph: ForensicGraph, ledger: EventLedger, max_attempts: int = 5) -> None:
        super().__init__(ledger=ledger, max_attempts=max_attempts)
        self._graph = graph

    async def process(self, delivery: EventDelivery) -> dict[str, Any]:
        try:
            event = delivery.event
            if event.type != FACTS_READY_TYPE:
                raise ValueError(f"unexpected event type: {event.type}")
            ready = FactsReady.model_validate(event.data)
        except (ValidationError, ValueError) as exc:
            await delivery.dead_letter(f"invalid facts-ready event: {exc}")
            return {"status": "dead_lettered"}

        if not await self._claim(delivery):
            await delivery.ack()
            return {"status": "duplicate"}

        try:
            result = await self._graph.ainvoke(
                {
                    "alert": ready.alert,
                    "facts": ready.facts,
                    "investigation_id": ready.investigation_id,
                    "request_id": ready.request_id,
                    "stage": "facts_ready",
                },
                config={"configurable": {"thread_id": ready.investigation_id}},
            )
        except Exception as exc:
            await self._ledger.release(delivery.event.id)
            await self._retry_or_dead_letter(delivery, f"reasoning failed: {exc}")
            return {"status": "retrying"}

        await self._ledger.complete(delivery.event.id)
        await delivery.ack()
        return result

    async def run(self, subscription: EventSubscription) -> None:
        async for delivery in subscription:
            await self.process(delivery)
