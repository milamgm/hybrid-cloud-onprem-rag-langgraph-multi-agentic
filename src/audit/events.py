"""Asynchronous audit events, kept separate from transactional agent state."""

from __future__ import annotations

import hashlib
import json
import os
import queue
import threading
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4


@dataclass(frozen=True)
class AuditEvent:
    """An append-only event containing hashes and references, not raw content."""

    event_type: str
    tenant_id: str
    subject_id: str
    thread_id: str
    request_id: str
    payload: dict[str, Any]
    event_id: str = field(default_factory=lambda: str(uuid4()))
    occurred_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


class AuditSink(Protocol):
    def emit(self, event: AuditEvent) -> None: ...

    def close(self) -> None: ...


class InMemoryAuditSink:
    """Deterministic sink for tests and local inspection."""

    def __init__(self) -> None:
        self.events: list[AuditEvent] = []

    def emit(self, event: AuditEvent) -> None:
        self.events.append(event)

    def close(self) -> None:
        return None


class JsonlAuditSink:
    """Local append-only sink; production should use the Event Hubs path."""

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or Path(
            os.getenv("AUDIT_EVIDENCE_PATH", "var/evidence/audit.jsonl")
        )
        self._lock = threading.Lock()

    def emit(self, event: AuditEvent) -> None:
        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(asdict(event), sort_keys=True) + "\n")

    def close(self) -> None:
        return None


class AzureEventHubAuditSink:
    """Event Hubs producer; the archival consumer is intentionally separate."""

    def __init__(self, producer: Any) -> None:
        self._producer = producer

    @classmethod
    def from_environment(cls) -> AzureEventHubAuditSink:
        from azure.eventhub import EventHubProducerClient
        from azure.identity import DefaultAzureCredential

        namespace = os.getenv("AZURE_EVENT_HUB_NAMESPACE")
        event_hub = os.getenv("AZURE_EVENT_HUB_NAME", "agent-audit")
        if not namespace:
            raise ValueError(
                "AZURE_EVENT_HUB_NAMESPACE is required for Event Hubs audit."
            )
        producer = EventHubProducerClient(
            fully_qualified_namespace=namespace,
            eventhub_name=event_hub,
            credential=DefaultAzureCredential(),
        )
        return cls(producer)

    def emit(self, event: AuditEvent) -> None:
        batch = self._producer.create_batch()
        body = json.dumps(asdict(event), sort_keys=True).encode("utf-8")
        batch.add(body)
        self._producer.send_batch(batch)

    def close(self) -> None:
        self._producer.close()


class BufferedAuditSink:
    """Bounded background delivery so audit I/O does not own graph latency."""

    def __init__(self, sink: AuditSink, *, max_queue: int = 1000) -> None:
        self._sink = sink
        self._queue: queue.Queue[AuditEvent | None] = queue.Queue(maxsize=max_queue)
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()

    def emit(self, event: AuditEvent) -> None:
        self._queue.put_nowait(event)

    def close(self) -> None:
        self._queue.put(None)
        self._queue.join()
        self._worker.join(timeout=5)
        self._sink.close()

    def _run(self) -> None:
        while True:
            event = self._queue.get()
            try:
                if event is None:
                    return
                self._sink.emit(event)
            finally:
                self._queue.task_done()


class AuditService:
    """Create safe audit records and dispatch them through an independent sink."""

    def __init__(self, sink: AuditSink | None = None) -> None:
        self._sink = sink or InMemoryAuditSink()

    @classmethod
    def from_environment(cls) -> AuditService:
        backend = os.getenv("AUDIT_BACKEND", "jsonl").lower()
        if backend == "eventhub":
            return cls(BufferedAuditSink(AzureEventHubAuditSink.from_environment()))
        if backend == "jsonl":
            return cls(JsonlAuditSink())
        if backend == "memory":
            return cls(InMemoryAuditSink())
        raise ValueError("AUDIT_BACKEND must be 'eventhub', 'jsonl' or 'memory'.")

    def record(
        self,
        event_type: str,
        *,
        tenant_id: str,
        subject_id: str,
        thread_id: str,
        request_id: str,
        payload: dict[str, Any],
    ) -> AuditEvent:
        safe_payload = _safe_payload(payload)
        event = AuditEvent(
            event_type,
            tenant_id,
            subject_id,
            thread_id,
            request_id,
            safe_payload,
        )
        self._sink.emit(event)
        return event

    def close(self) -> None:
        self._sink.close()


def _safe_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Replace content with a stable digest while retaining audit evidence."""
    safe: dict[str, Any] = {}
    for key, value in payload.items():
        if key in {"prompt", "response", "summary", "content"}:
            safe[f"{key}_sha256"] = hashlib.sha256(str(value).encode()).hexdigest()
        else:
            safe[key] = value
    return safe
