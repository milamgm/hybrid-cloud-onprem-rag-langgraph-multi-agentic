"""Asynchronous analytical publisher used as a safe XGBoost stand-in."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from src.events.contracts import RISK_ALERT_TYPE, TransactionRiskAlert, TransactionSample, make_event
from src.events.transport import EventPublisher


@dataclass(frozen=True, slots=True)
class MockXGBoostModel:
    """Deterministic mock preserving the production publisher boundary."""

    publisher: EventPublisher
    model_version: str = "mock-xgboost-0.1"
    source: str = "model://xgboost/mock"

    async def score_and_publish(self, sample: TransactionSample):
        score = min(
            1.0,
            0.05
            + min(sample.amount / 100_000, 0.45)
            + min(sample.velocity_24h / 100, 0.25)
            + min(sample.geo_distance_km / 10_000, 0.25),
        )
        severity = "critical" if score >= 0.85 else "high" if score >= 0.65 else "medium" if score >= 0.35 else "low"
        signals = ["amount"] if sample.amount >= 50_000 else []
        if sample.velocity_24h >= 20:
            signals.append("velocity")
        if sample.geo_distance_km >= 2_000:
            signals.append("geo_velocity")
        if not signals:
            signals.append("baseline")

        alert = TransactionRiskAlert(
            alert_id=f"alert-{sample.transaction_id}",
            transaction_id=sample.transaction_id,
            tenant_id=sample.tenant_id,
            customer_id=sample.customer_id,
            risk_score=score,
            severity=severity,
            model_version=self.model_version,
            signal_codes=tuple(signals),
            occurred_at=datetime.now(timezone.utc),
        )
        event = make_event(
            event_type=RISK_ALERT_TYPE,
            source=self.source,
            subject=sample.transaction_id,
            data=alert,
            event_id=f"risk-alert:{alert.alert_id}:v1",
        )
        await self.publisher.publish(event, key=sample.tenant_id)
        return event
