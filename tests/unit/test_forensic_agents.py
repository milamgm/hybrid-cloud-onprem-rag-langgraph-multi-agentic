"""Unit tests for specialist tool boundaries and Durable case coordination."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime

from src.agents.forensic import (
    ForensicToolGateways,
    _customer_risk_tools,
    _network_tools,
    _policy_tools,
    _transaction_tools,
)
from src.cloud.durable_forensics import (
    COLLECT_EVIDENCE_ACTIVITY,
    HUMAN_APPROVAL_EVENT,
    PUBLISH_EXECUTION_ACTIVITY,
    PUBLISH_REVIEW_ACTIVITY,
    RUN_AGENTS_ACTIVITY,
    fraud_case_orchestrator,
)
from src.events.contracts import EvidenceCollectionRequested, TransactionRiskAlert
from src.forensics.evidence import (
    MockCoreBankingReadGateway,
    MockCustomerRiskReadGateway,
    MockNetworkReadGateway,
    MockPolicyRAGGateway,
    MockScreeningReadGateway,
)


def _request() -> EvidenceCollectionRequested:
    return EvidenceCollectionRequested(
        investigation_id="investigation-alert-001",
        request_id="request-001",
        alert=TransactionRiskAlert(
            alert_id="alert-001",
            transaction_id="tx-001",
            tenant_id="tenant-a",
            customer_id="customer-001",
            risk_score=0.91,
            severity="critical",
            model_version="test-xgboost-1",
            signal_codes=("amount", "velocity"),
            occurred_at=datetime.now(UTC),
        ),
    )


def _gateways() -> ForensicToolGateways:
    return ForensicToolGateways(
        core_banking=MockCoreBankingReadGateway(),
        customer_risk=MockCustomerRiskReadGateway(),
        network=MockNetworkReadGateway(),
        screening=MockScreeningReadGateway(),
        policy_rag=MockPolicyRAGGateway(),
    )


def test_each_specialist_gets_only_its_allowlisted_tools() -> None:
    request, gateways = _request(), _gateways()

    assert [tool.name for tool in _transaction_tools(request, gateways)] == [
        "read_transaction_case_context"
    ]
    assert [tool.name for tool in _customer_risk_tools(request, gateways)] == [
        "read_customer_risk_profile",
        "read_subject_screening",
    ]
    assert [tool.name for tool in _network_tools(request, gateways)] == [
        "read_transaction_network"
    ]
    assert [tool.name for tool in _policy_tools(request, gateways)] == [
        "search_internal_policy"
    ]


def test_tool_results_remain_bound_to_the_current_case_scope() -> None:
    async def scenario() -> None:
        request, gateways = _request(), _gateways()
        transaction = _transaction_tools(request, gateways)[0]
        customer = _customer_risk_tools(request, gateways)[0]
        network = _network_tools(request, gateways)[0]

        transaction_result = json.loads(await transaction.ainvoke({}))
        customer_result = json.loads(await customer.ainvoke({}))
        network_result = json.loads(await network.ainvoke({}))

        for result in (transaction_result, customer_result, network_result):
            assert result["tenant_id"] == "tenant-a"
            assert result["customer_id"] == "customer-001"
            assert result["evidence_id"].endswith("investigation-alert-001")

    asyncio.run(scenario())


class _DurableContext:
    def __init__(self, input_value: dict) -> None:
        self._input = input_value

    def get_input(self) -> dict:
        return self._input

    def call_activity(self, name: str, input_: dict) -> tuple[str, dict]:
        return name, input_

    def wait_for_external_event(self, name: str) -> tuple[str, str]:
        return "event", name


def _case() -> dict:
    return {
        "investigation_id": "investigation-alert-001",
        "request_id": "request-001",
        "alert": {"alert_id": "alert-001"},
    }


def test_durable_orchestrator_runs_agents_outside_the_replayable_orchestrator() -> None:
    flow = fraud_case_orchestrator(_DurableContext(_case()))

    assert next(flow) == (COLLECT_EVIDENCE_ACTIVITY, _case())
    assert flow.send({"bundle_id": "evidence-001"}) == (
        RUN_AGENTS_ACTIVITY,
        {**_case(), "evidence": {"bundle_id": "evidence-001"}},
    )
    report = {"report_id": "report-001"}
    assert flow.send(report) == (
        PUBLISH_REVIEW_ACTIVITY,
        {**_case(), "report": report},
    )
    assert flow.send(None) == ("event", HUMAN_APPROVAL_EVENT)
    assert flow.send(
        {"decision": "approved", "approval_request_id": "approval-001"}
    ) == (
        PUBLISH_EXECUTION_ACTIVITY,
        {
            **_case(),
            "report": report,
            "approval": {"decision": "approved", "approval_request_id": "approval-001"},
        },
    )
    try:
        flow.send(None)
    except StopIteration as completed:
        assert completed.value == {
            "status": "execution_requested",
            "investigation_id": "investigation-alert-001",
            "report_id": "report-001",
        }
    else:  # pragma: no cover - protects the generator contract above
        raise AssertionError("orchestration did not finish")


def test_durable_orchestrator_stops_on_rejected_human_decision() -> None:
    flow = fraud_case_orchestrator(_DurableContext(_case()))
    next(flow)
    flow.send({"bundle_id": "evidence-001"})
    flow.send({"report_id": "report-001"})
    flow.send(None)
    try:
        flow.send({"decision": "rejected"})
    except StopIteration as completed:
        assert completed.value == {
            "status": "rejected",
            "investigation_id": "investigation-alert-001",
            "report_id": "report-001",
        }
    else:  # pragma: no cover - protects the generator contract above
        raise AssertionError("rejected case unexpectedly continued")
