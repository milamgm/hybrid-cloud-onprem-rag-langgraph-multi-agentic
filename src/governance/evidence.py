"""Tamper-evident governance evidence without retaining sensitive content."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from opentelemetry import metrics, trace

EventType = Literal[
    "version_registered",
    "policy_decision",
    "metric_recorded",
    "incident_reported",
    "risk_reviewed",
]
_SENSITIVE_KEYS = {"prompt", "output", "secret", "token", "password", "api_key"}


@dataclass(frozen=True)
class GovernanceEvent:
    """An immutable evidence record containing references, never raw content."""

    event_type: EventType
    subject: str
    payload: dict[str, Any]
    correlation_id: str = field(default_factory=lambda: str(uuid4()))
    event_id: str = field(default_factory=lambda: str(uuid4()))
    occurred_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    previous_hash: str | None = None
    event_hash: str | None = None

    def canonical_payload(self) -> str:
        """Return the stable representation used to calculate the evidence hash."""
        data = asdict(self)
        data.pop("event_hash")
        return json.dumps(
            data, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )

    def with_hash(self, previous_hash: str | None) -> GovernanceEvent:
        """Link this record to its predecessor and calculate its SHA-256 hash."""
        event = GovernanceEvent(
            event_type=self.event_type,
            subject=self.subject,
            payload=self.payload,
            correlation_id=self.correlation_id,
            event_id=self.event_id,
            occurred_at=self.occurred_at,
            previous_hash=previous_hash,
        )
        digest = hashlib.sha256(event.canonical_payload().encode("utf-8")).hexdigest()
        data = asdict(event)
        data["event_hash"] = digest
        return GovernanceEvent(**data)


class EvidenceLedger:
    """Append and verify a local hash-chained evidence journal.

    Production deployments should forward this evidence to an immutable SIEM or
    WORM store. The chain makes local-file changes detectable; it cannot stop a
    privileged attacker from replacing the entire file.
    """

    def __init__(self, path: Path | None = None) -> None:
        configured_path = os.getenv(
            "GOVERNANCE_EVIDENCE_PATH", "var/evidence/governance.jsonl"
        )
        self._path = path or Path(configured_path)
        self._lock = threading.Lock()

    def append(self, event: GovernanceEvent) -> GovernanceEvent:
        """Append a validated event, linked to the current journal head."""
        self._validate_payload(event.payload)
        with self._lock:
            previous_hash = self._last_hash()
            stored_event = event.with_hash(previous_hash)
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as evidence_file:
                evidence_file.write(
                    json.dumps(asdict(stored_event), sort_keys=True) + "\n"
                )
            return stored_event

    def verify(self) -> bool:
        """Return whether every event hash and chain link is intact."""
        if not self._path.exists():
            return True
        previous_hash = None
        for line in self._path.read_text(encoding="utf-8").splitlines():
            data = json.loads(line)
            event_hash = data.pop("event_hash", None)
            event = GovernanceEvent(**data)
            expected = event.with_hash(previous_hash).event_hash
            if event_hash != expected:
                return False
            previous_hash = event_hash
        return True

    def _last_hash(self) -> str | None:
        if not self._path.exists():
            return None
        lines = self._path.read_text(encoding="utf-8").splitlines()
        return json.loads(lines[-1])["event_hash"] if lines else None

    @staticmethod
    def _validate_payload(payload: dict[str, Any]) -> None:
        def walk(value: Any) -> None:
            if isinstance(value, dict):
                for key, child in value.items():
                    normalized_key = key.lower().replace("-", "_")
                    if any(
                        sensitive_key in normalized_key
                        for sensitive_key in _SENSITIVE_KEYS
                    ):
                        raise ValueError(f"Evidence payload must not contain {key!r}.")
                    walk(child)
            elif isinstance(value, list):
                for child in value:
                    walk(child)

        walk(payload)


class GovernanceService:
    """Record governed lifecycle events and emit vendor-neutral telemetry."""

    def __init__(self, ledger: EvidenceLedger | None = None) -> None:
        self._ledger = ledger or EvidenceLedger()
        self._tracer = trace.get_tracer("onyx.governance")
        self._meter = metrics.get_meter("onyx.governance")
        self._event_counter = self._meter.create_counter("onyx.governance.events")

    def register_version(
        self, subject: str, *, version: str, artifact_sha256: str, policy_version: str
    ) -> GovernanceEvent:
        return self._record(
            "version_registered",
            subject,
            {
                "version": version,
                "artifact_sha256": artifact_sha256,
                "policy_version": policy_version,
            },
        )

    def record_policy_decision(
        self,
        subject: str,
        *,
        policy_id: str,
        policy_version: str,
        decision: str,
        reason_code: str,
    ) -> GovernanceEvent:
        return self._record(
            "policy_decision",
            subject,
            {
                "policy_id": policy_id,
                "policy_version": policy_version,
                "decision": decision,
                "reason_code": reason_code,
            },
        )

    def record_metric(
        self, subject: str, *, name: str, value: float, unit: str
    ) -> GovernanceEvent:
        return self._record(
            "metric_recorded", subject, {"name": name, "value": value, "unit": unit}
        )

    def report_incident(
        self, subject: str, *, incident_id: str, severity: str, status: str
    ) -> GovernanceEvent:
        return self._record(
            "incident_reported",
            subject,
            {"incident_id": incident_id, "severity": severity, "status": status},
        )

    def record_risk_review(
        self,
        subject: str,
        *,
        review_id: str,
        risk_level: str,
        reviewer_id: str,
        outcome: str,
    ) -> GovernanceEvent:
        return self._record(
            "risk_reviewed",
            subject,
            {
                "review_id": review_id,
                "risk_level": risk_level,
                "reviewer_id": reviewer_id,
                "outcome": outcome,
            },
        )

    def _record(
        self, event_type: EventType, subject: str, payload: dict[str, Any]
    ) -> GovernanceEvent:
        event = self._ledger.append(GovernanceEvent(event_type, subject, payload))
        attributes = {
            "governance.event_type": event.event_type,
            "governance.subject": event.subject,
            "governance.event_hash": event.event_hash or "",
            "governance.correlation_id": event.correlation_id,
        }
        with self._tracer.start_as_current_span(
            "governance.evidence", attributes=attributes
        ):
            self._event_counter.add(
                1, attributes={"governance.event_type": event.event_type}
            )
        return event


def configure_governance_telemetry() -> None:
    """Configure the supported cloud or standard on-premise OTel exporter.

    Call once from the application bootstrap, before constructing services. It
    intentionally does not run during import because OpenTelemetry providers
    are process-global.
    """
    mode = os.getenv("INFRASTRUCTURE_MODE", "on_premise").lower()
    if mode == "cloud":
        connection_string = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")
        if not connection_string:
            raise ValueError(
                "Cloud governance telemetry requires APPLICATIONINSIGHTS_CONNECTION_STRING."
            )
        from azure.monitor.opentelemetry import configure_azure_monitor

        configure_azure_monitor(connection_string=connection_string)
        return

    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
    from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
        OTLPMetricExporter,
    )
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    resource = Resource.create(
        {"service.name": "onyx", "service.namespace": "governance"}
    )
    trace_provider = TracerProvider(resource=resource)
    trace_provider.add_span_processor(
        BatchSpanProcessor(
            OTLPSpanExporter(endpoint=f"{endpoint.rstrip('/')}/v1/traces")
        )
    )
    metric_reader = PeriodicExportingMetricReader(
        OTLPMetricExporter(endpoint=f"{endpoint.rstrip('/')}/v1/metrics")
    )
    trace.set_tracer_provider(trace_provider)
    metrics.set_meter_provider(
        MeterProvider(resource=resource, metric_readers=[metric_reader])
    )
