"""Broker-neutral event ports and a deterministic in-memory implementation."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Protocol

from src.events.contracts import EventEnvelope


class EventPublisher(Protocol):
    async def publish(self, event: EventEnvelope[Any], *, key: str | None = None) -> None:
        """Publish an event durably according to the concrete transport policy."""


class EventLedger(Protocol):
    async def claim(self, event_id: str) -> bool:
        """Atomically claim an event id using a durable uniqueness boundary."""

    async def complete(self, event_id: str) -> None:
        """Mark the side effect complete."""

    async def release(self, event_id: str) -> None:
        """Release a failed claim so a retry can process it."""


class EventDelivery(Protocol):
    event: EventEnvelope[Any]
    delivery_attempt: int

    async def ack(self) -> None:
        """Acknowledge only after the side effect has completed."""

    async def retry(self, reason: str) -> None:
        """Request redelivery without advancing the source checkpoint."""

    async def dead_letter(self, reason: str) -> None:
        """Move an invalid or exhausted message to a durable dead-letter path."""


class EventSubscription(Protocol):
    async def __aenter__(self) -> "EventSubscription":
        ...

    async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        ...

    def __aiter__(self) -> "EventSubscription":
        ...

    async def __anext__(self) -> EventDelivery:
        ...


@dataclass(slots=True)
class DeadLetter:
    event: EventEnvelope[Any]
    reason: str
    delivery_attempt: int


@dataclass(slots=True)
class _InMemoryDelivery:
    event: EventEnvelope[Any]
    _queue: asyncio.Queue["_InMemoryDelivery"]
    _dead_letters: list[DeadLetter]
    delivery_attempt: int = 1
    acknowledged: bool = False

    async def ack(self) -> None:
        self.acknowledged = True

    async def retry(self, reason: str) -> None:
        self.delivery_attempt += 1
        self._queue.put_nowait(self)

    async def dead_letter(self, reason: str) -> None:
        self._dead_letters.append(DeadLetter(self.event, reason, self.delivery_attempt))
        self.acknowledged = True


class _InMemorySubscription:
    def __init__(self, bus: "InMemoryEventBus", event_type: str, group: str) -> None:
        self._bus = bus
        self._event_type = event_type
        self._group = group
        self._queue: asyncio.Queue[_InMemoryDelivery] = asyncio.Queue()
        self._active = False

    async def __aenter__(self) -> "_InMemorySubscription":
        self._active = True
        self._bus._subscriptions.setdefault((self._event_type, self._group), []).append(self._queue)
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        queues = self._bus._subscriptions.get((self._event_type, self._group), [])
        if self._queue in queues:
            queues.remove(self._queue)
        self._active = False

    def __aiter__(self) -> "_InMemorySubscription":
        return self

    async def __anext__(self) -> _InMemoryDelivery:
        if not self._active:
            raise StopAsyncIteration
        return await self._queue.get()


class InMemoryEventBus:
    """Test/dev transport with explicit subscriptions and manual ack semantics."""

    def __init__(self) -> None:
        self._subscriptions: dict[tuple[str, str], list[asyncio.Queue[_InMemoryDelivery]]] = {}
        self.dead_letters: list[DeadLetter] = []
        self.published: list[EventEnvelope[Any]] = []

    def subscribe(self, event_type: str, group: str) -> EventSubscription:
        return _InMemorySubscription(self, event_type, group)

    async def publish(self, event: EventEnvelope[Any], *, key: str | None = None) -> None:
        del key  # Partition keys are applied by concrete broker bindings.
        self.published.append(event)
        for (event_type, _group), queues in self._subscriptions.items():
            if event_type != event.type:
                continue
            for queue in queues:
                queue.put_nowait(_InMemoryDelivery(event, queue, self.dead_letters))


class InMemoryEventLedger:
    """Small idempotency ledger mirroring a durable unique-event table."""

    def __init__(self) -> None:
        self._states: dict[str, str] = {}

    async def claim(self, event_id: str) -> bool:
        if self._states.get(event_id) in {"processing", "completed"}:
            return False
        self._states[event_id] = "processing"
        return True

    async def complete(self, event_id: str) -> None:
        self._states[event_id] = "completed"

    async def release(self, event_id: str) -> None:
        self._states.pop(event_id, None)
