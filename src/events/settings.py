"""Environment-driven event topology selection.

The application depends on ports, not on a broker SDK. Deployments select a
cloud or on-premise binding at the composition root and inject its clients.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

InfrastructureMode = Literal["cloud", "onprem", "memory"]
StreamBackend = Literal["azure_event_hubs", "kafka", "memory"]
TransactionalBackend = Literal["azure_service_bus", "aws_sqs", "rabbitmq", "ibm_mq", "memory"]
ReactiveBackend = Literal["azure_event_grid", "aws_eventbridge", "nats", "knative", "memory"]


@dataclass(frozen=True, slots=True)
class EventTopologySettings:
    """Validated names and roles for the event topology."""

    infrastructure_mode: InfrastructureMode
    stream_backend: StreamBackend
    transactional_backend: TransactionalBackend
    reactive_backend: ReactiveBackend
    risk_alert_topic: str = "risk-alerts"
    facts_requested_topic: str = "forensic-evidence-requested"
    facts_ready_topic: str = "forensic-evidence-ready"
    approval_requested_topic: str = "forensic-approval-requested"
    approval_granted_topic: str = "forensic-approval-granted"
    consumer_group: str = "forensic-investigator"

    @classmethod
    def from_env(cls) -> "EventTopologySettings":
        mode = os.getenv("INFRASTRUCTURE_MODE", "onprem").strip().lower()
        if mode not in {"cloud", "onprem", "memory"}:
            raise ValueError("INFRASTRUCTURE_MODE must be cloud, onprem, or memory")

        defaults = {
            "cloud": ("azure_event_hubs", "azure_service_bus", "azure_event_grid"),
            "onprem": ("kafka", "rabbitmq", "nats"),
            "memory": ("memory", "memory", "memory"),
        }
        stream, transactional, reactive = defaults[mode]
        configured_stream = os.getenv("EVENT_STREAM_BACKEND", "").strip() or stream
        configured_transactional = os.getenv("EVENT_TRANSACTIONAL_BACKEND", "").strip() or transactional
        configured_reactive = os.getenv("EVENT_REACTIVE_BACKEND", "").strip() or reactive
        valid_streams = {"azure_event_hubs", "kafka", "memory"}
        valid_transactional = {"azure_service_bus", "aws_sqs", "rabbitmq", "ibm_mq", "memory"}
        valid_reactive = {"azure_event_grid", "aws_eventbridge", "nats", "knative", "memory"}
        if configured_stream not in valid_streams:
            raise ValueError(f"unsupported EVENT_STREAM_BACKEND: {configured_stream}")
        if configured_transactional not in valid_transactional:
            raise ValueError(f"unsupported EVENT_TRANSACTIONAL_BACKEND: {configured_transactional}")
        if configured_reactive not in valid_reactive:
            raise ValueError(f"unsupported EVENT_REACTIVE_BACKEND: {configured_reactive}")
        if os.getenv("DEPLOYMENT_ENVIRONMENT", "development").strip().lower() == "production" and (
            "memory" in {configured_stream, configured_transactional, configured_reactive}
        ):
            raise ValueError("memory event transports are forbidden in production")
        return cls(
            infrastructure_mode=mode,
            stream_backend=configured_stream,
            transactional_backend=configured_transactional,
            reactive_backend=configured_reactive,
            risk_alert_topic=os.getenv("EVENT_RISK_ALERT_TOPIC", "risk-alerts"),
            facts_requested_topic=os.getenv("EVENT_EVIDENCE_REQUESTED_TOPIC", os.getenv("EVENT_FACTS_REQUESTED_TOPIC", "forensic-evidence-requested")),
            facts_ready_topic=os.getenv("EVENT_EVIDENCE_READY_TOPIC", os.getenv("EVENT_FACTS_READY_TOPIC", "forensic-evidence-ready")),
            approval_requested_topic=os.getenv("EVENT_APPROVAL_REQUESTED_TOPIC", "forensic-approval-requested"),
            approval_granted_topic=os.getenv("EVENT_APPROVAL_GRANTED_TOPIC", "forensic-approval-granted"),
            consumer_group=os.getenv("EVENT_CONSUMER_GROUP", "forensic-investigator"),
        )

    def as_dict(self) -> dict[str, str]:
        return {
            "infrastructure_mode": self.infrastructure_mode,
            "stream_backend": self.stream_backend,
            "transactional_backend": self.transactional_backend,
            "reactive_backend": self.reactive_backend,
            "risk_alert_topic": self.risk_alert_topic,
            "facts_requested_topic": self.facts_requested_topic,
            "facts_ready_topic": self.facts_ready_topic,
            "approval_requested_topic": self.approval_requested_topic,
            "approval_granted_topic": self.approval_granted_topic,
            "consumer_group": self.consumer_group,
        }
