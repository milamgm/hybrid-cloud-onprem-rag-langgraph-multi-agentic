"""Azure Functions v2 entrypoint for cloud forensic investigations.

Deployment code must call ``configure_durable_forensic_activities`` during
startup with real, least-privilege gateways.  This module deliberately has no
fallback to mock data in order to fail closed in cloud production.
"""

from __future__ import annotations

import json
import os
from importlib import import_module
from typing import Any

import azure.durable_functions as df
import azure.functions as func

from src.cloud.durable_forensics import (
    HUMAN_APPROVAL_EVENT,
    DurableForensicActivities,
)
from src.cloud.durable_forensics import (
    fraud_case_orchestrator as run_fraud_case_orchestration,
)
from src.events.contracts import TransactionRiskAlert

app = df.DFApp(http_auth_level=func.AuthLevel.FUNCTION)
_activities: DurableForensicActivities | None = None


def configure_durable_forensic_activities(
    activities: DurableForensicActivities,
) -> None:
    """Inject real collectors, agents and publishers from the cloud composition root."""
    global _activities
    _activities = activities


def _configure_from_environment() -> None:
    """Load a deployment-owned composition factory without embedding credentials.

    Set ``FORENSIC_DURABLE_ACTIVITIES_FACTORY`` to
    ``package.module:factory``. The factory must return a fully configured
    ``DurableForensicActivities`` that uses managed identity/Key Vault and
    scoped read gateways. Leaving it unset is intentional for local imports;
    an activity then fails closed through ``_configured_activities``.
    """
    factory_path = os.getenv("FORENSIC_DURABLE_ACTIVITIES_FACTORY", "").strip()
    if not factory_path:
        return
    module_name, separator, attribute = factory_path.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError(
            "FORENSIC_DURABLE_ACTIVITIES_FACTORY must use package.module:factory"
        )
    factory = getattr(import_module(module_name), attribute)
    activities = factory()
    if not isinstance(activities, DurableForensicActivities):
        raise TypeError(
            "durable activities factory must return DurableForensicActivities"
        )
    configure_durable_forensic_activities(activities)


def _configured_activities() -> DurableForensicActivities:
    if _activities is None:
        raise RuntimeError(
            "Durable forensic activities are not configured; inject real scoped gateways."
        )
    return _activities


_configure_from_environment()


@app.event_hub_message_trigger(
    arg_name="event",
    event_hub_name="%FORENSIC_ALERT_EVENT_HUB%",
    connection="FORENSIC_EVENT_HUB_CONNECTION",
)
@app.durable_client_input(client_name="client")
async def start_fraud_case(
    event: func.EventHubEvent,
    client: df.DurableOrchestrationClient,
) -> None:
    """Translate a validated XGBoost alert into one durable case instance."""
    envelope = json.loads(event.get_body().decode("utf-8"))
    alert = TransactionRiskAlert.model_validate(envelope["data"])
    investigation_id = f"investigation-{alert.alert_id}"
    await client.start_new(
        "fraud_case_orchestrator",
        investigation_id,
        {
            "investigation_id": investigation_id,
            "request_id": f"{envelope['id']}:request:v1",
            "alert": alert.model_dump(mode="json"),
        },
    )


@app.orchestration_trigger(context_name="context")
def fraud_case_orchestrator(
    context: df.DurableOrchestrationContext,
):
    return run_fraud_case_orchestration(context)


@app.activity_trigger(input_name="payload")
async def collect_forensic_evidence(payload: dict[str, Any]) -> dict[str, Any]:
    return await _configured_activities().collect_evidence(payload)


@app.activity_trigger(input_name="payload")
async def run_forensic_agents(payload: dict[str, Any]) -> dict[str, Any]:
    return await _configured_activities().run_agents(payload)


@app.activity_trigger(input_name="payload")
async def publish_forensic_review(payload: dict[str, Any]) -> None:
    await _configured_activities().publish_review(payload)


@app.activity_trigger(input_name="payload")
async def publish_forensic_execution(payload: dict[str, Any]) -> None:
    await _configured_activities().publish_execution(payload)


@app.route(route="forensic/cases/{case_id}/approval", methods=["POST"])
@app.durable_client_input(client_name="client")
async def submit_human_approval(
    request: func.HttpRequest,
    client: df.DurableOrchestrationClient,
) -> func.HttpResponse:
    """Raise the approval event after APIM/Entra has authenticated the caller.

    The production APIM policy must validate the human's role and pass an
    approval payload with ``approval_request_id``, ``decision``,
    ``approver_ref`` and ``reason``.  This function never grants approval from
    an unverified request body by itself.
    """
    try:
        payload = request.get_json()
        if payload["decision"] not in {"approved", "rejected"}:
            raise ValueError("decision must be approved or rejected")
        for field in ("approval_request_id", "approver_ref", "reason"):
            if not isinstance(payload.get(field), str) or not payload[field].strip():
                raise ValueError(f"{field} is required")
    except (KeyError, TypeError, ValueError) as error:
        return func.HttpResponse(str(error), status_code=422)
    case_id = request.route_params["case_id"]
    await client.raise_event(case_id, HUMAN_APPROVAL_EVENT, payload)
    return func.HttpResponse(status_code=202)
