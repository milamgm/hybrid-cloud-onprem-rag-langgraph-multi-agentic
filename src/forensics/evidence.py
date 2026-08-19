"""Read-only evidence tools used by the asynchronous enrichment worker.

The agent is deliberately not given a database connection or a raw-SQL tool.
Production composition supplies allowlisted query operations through this port;
the worker applies tenant/customer constraints before any data access.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from hashlib import sha256
from typing import Any, Protocol

from src.events.contracts import (
    CaseEvidenceBundle,
    CoreBankingEvidence,
    EvidenceCollectionRequested,
    PolicyCitation,
    ReadOnlyEvidence,
)


class CoreBankingReadGateway(Protocol):
    async def read_case_context(
        self, request: EvidenceCollectionRequested
    ) -> CoreBankingEvidence:
        """Run only the approved, parameterised customer/transaction reads."""


class PolicyRAGGateway(Protocol):
    async def retrieve_policy(
        self,
        request: EvidenceCollectionRequested,
        *,
        focus: str | None = None,
    ) -> tuple[PolicyCitation, ...]:
        """Retrieve citations from the curated policy corpus, never from the web."""


class CustomerRiskReadGateway(Protocol):
    async def read_customer_risk(
        self, request: EvidenceCollectionRequested
    ) -> ReadOnlyEvidence:
        """Read the bounded KYC/CDD and customer-risk profile for this case only."""


class NetworkReadGateway(Protocol):
    async def read_transaction_network(
        self, request: EvidenceCollectionRequested
    ) -> ReadOnlyEvidence:
        """Read a bounded counterparty/network summary for this case only."""


class ScreeningReadGateway(Protocol):
    async def screen_subjects(
        self, request: EvidenceCollectionRequested
    ) -> ReadOnlyEvidence:
        """Read a bounded, approved sanctions/adverse-media screening summary."""


class NamedQueryClient(Protocol):
    async def execute_named_query(
        self, operation: str, parameters: dict[str, str]
    ) -> dict[str, Any]:
        """Execute a repository-owned prepared statement or stored procedure."""


class ParameterizedCoreBankingGateway:
    """Production adapter boundary for core/AML/CRM repositories.

    ``operation`` is an allowlisted repository name, not user/model-provided
    SQL.  The implementing service account must be read-only and constrained
    to a tenant row-level-security policy as a defence in depth measure.
    """

    OPERATION = "forensics.case_context.v1"

    def __init__(
        self, client: NamedQueryClient, *, source_system: str = "core-banking"
    ) -> None:
        self._client = client
        self._source_system = source_system

    async def read_case_context(
        self, request: EvidenceCollectionRequested
    ) -> CoreBankingEvidence:
        alert = request.alert
        row = await self._client.execute_named_query(
            self.OPERATION,
            {
                "tenant_id": alert.tenant_id,
                "customer_id": alert.customer_id,
                "transaction_id": alert.transaction_id,
            },
        )
        attributes = row.get("attributes")
        if not isinstance(attributes, dict):
            raise ValueError("case-context query must return bounded attributes")
        return CoreBankingEvidence(
            evidence_id=f"core-{request.investigation_id}",
            tenant_id=alert.tenant_id,
            customer_id=alert.customer_id,
            source_system=self._source_system,
            observed_at=datetime.now(UTC),
            attributes=attributes,
        )


class _ParameterizedReadGateway:
    """Base for a repository-owned, tenant-scoped forensic read operation."""

    OPERATION = ""
    SOURCE_SYSTEM = ""

    def __init__(self, client: NamedQueryClient) -> None:
        self._client = client

    async def _read(self, request: EvidenceCollectionRequested) -> ReadOnlyEvidence:
        alert = request.alert
        row = await self._client.execute_named_query(
            self.OPERATION,
            {
                "tenant_id": alert.tenant_id,
                "customer_id": alert.customer_id,
                "transaction_id": alert.transaction_id,
            },
        )
        attributes = row.get("attributes")
        if not isinstance(attributes, dict):
            raise ValueError(f"{self.OPERATION} must return bounded attributes")
        if len(attributes) > 64:
            raise ValueError(f"{self.OPERATION} returned too many attributes")
        if not all(
            isinstance(value, str | int | float | bool) for value in attributes.values()
        ):
            raise ValueError(f"{self.OPERATION} returned an unsupported attribute type")
        return ReadOnlyEvidence(
            evidence_id=f"{self.SOURCE_SYSTEM}-{request.investigation_id}",
            tenant_id=alert.tenant_id,
            customer_id=alert.customer_id,
            source_system=self.SOURCE_SYSTEM,
            observed_at=datetime.now(UTC),
            attributes=attributes,
        )


class ParameterizedCustomerRiskGateway(_ParameterizedReadGateway):
    OPERATION = "forensics.customer_risk_profile.v1"
    SOURCE_SYSTEM = "customer-risk"

    async def read_customer_risk(
        self, request: EvidenceCollectionRequested
    ) -> ReadOnlyEvidence:
        return await self._read(request)


class ParameterizedNetworkGateway(_ParameterizedReadGateway):
    OPERATION = "forensics.transaction_network.v1"
    SOURCE_SYSTEM = "transaction-network"

    async def read_transaction_network(
        self, request: EvidenceCollectionRequested
    ) -> ReadOnlyEvidence:
        return await self._read(request)


class ParameterizedScreeningGateway(_ParameterizedReadGateway):
    OPERATION = "forensics.subject_screening.v1"
    SOURCE_SYSTEM = "screening"

    async def screen_subjects(
        self, request: EvidenceCollectionRequested
    ) -> ReadOnlyEvidence:
        return await self._read(request)


class CaseEvidenceCollector:
    """Collect independent read-only sources concurrently and validate a bundle."""

    def __init__(
        self, *, core_banking: CoreBankingReadGateway, policy_rag: PolicyRAGGateway
    ) -> None:
        self._core_banking = core_banking
        self._policy_rag = policy_rag

    async def collect(self, request: EvidenceCollectionRequested) -> CaseEvidenceBundle:
        core, policy_citations = await asyncio.gather(
            self._core_banking.read_case_context(request),
            self._policy_rag.retrieve_policy(request),
        )
        if (
            core.tenant_id != request.alert.tenant_id
            or core.customer_id != request.alert.customer_id
        ):
            raise PermissionError("core-banking evidence scope does not match the case")
        if not policy_citations:
            raise ValueError("policy RAG returned no governed citations")
        return CaseEvidenceBundle(
            bundle_id=f"evidence-{request.investigation_id}",
            core_banking=core,
            policy_citations=policy_citations,
            collected_at=datetime.now(UTC),
        )


class MockCoreBankingReadGateway:
    """Deterministic development substitute for an approved read repository."""

    async def read_case_context(
        self, request: EvidenceCollectionRequested
    ) -> CoreBankingEvidence:
        alert = request.alert
        digest = sha256(
            f"{alert.tenant_id}:{alert.customer_id}:{alert.transaction_id}".encode()
        ).hexdigest()
        return CoreBankingEvidence(
            evidence_id=f"core-{request.investigation_id}",
            tenant_id=alert.tenant_id,
            customer_id=alert.customer_id,
            source_system="mock-core-banking-readonly",
            observed_at=datetime.now(UTC),
            attributes={
                "transaction_found": True,
                "counterparty_ref": f"cp-{digest[:12]}",
                "historical_alert_count": int(digest[:2], 16) % 4,
                "kyc_review_current": True,
            },
        )


class MockPolicyRAGGateway:
    """Development substitute for the curated, access-controlled AML corpus."""

    async def retrieve_policy(
        self,
        request: EvidenceCollectionRequested,
        *,
        focus: str | None = None,
    ) -> tuple[PolicyCitation, ...]:
        del focus
        return (
            PolicyCitation(
                citation_id=f"policy-{request.investigation_id}-1",
                source="AML/Fraud investigation procedure",
                section="Alert investigation and disposition",
                excerpt="Review the customer risk profile, transaction activity and documented red flags before a human disposition.",
            ),
        )


class MockCustomerRiskReadGateway:
    async def read_customer_risk(
        self, request: EvidenceCollectionRequested
    ) -> ReadOnlyEvidence:
        return ReadOnlyEvidence(
            evidence_id=f"customer-risk-{request.investigation_id}",
            tenant_id=request.alert.tenant_id,
            customer_id=request.alert.customer_id,
            source_system="mock-customer-risk-readonly",
            observed_at=datetime.now(UTC),
            attributes={
                "kyc_review_current": True,
                "customer_risk_rating": "medium",
                "pep_match": False,
            },
        )


class MockNetworkReadGateway:
    async def read_transaction_network(
        self, request: EvidenceCollectionRequested
    ) -> ReadOnlyEvidence:
        return ReadOnlyEvidence(
            evidence_id=f"transaction-network-{request.investigation_id}",
            tenant_id=request.alert.tenant_id,
            customer_id=request.alert.customer_id,
            source_system="mock-transaction-network-readonly",
            observed_at=datetime.now(UTC),
            attributes={
                "related_counterparties": 3,
                "shared_device_cluster": False,
                "prior_related_alerts": 1,
            },
        )


class MockScreeningReadGateway:
    async def screen_subjects(
        self, request: EvidenceCollectionRequested
    ) -> ReadOnlyEvidence:
        return ReadOnlyEvidence(
            evidence_id=f"screening-{request.investigation_id}",
            tenant_id=request.alert.tenant_id,
            customer_id=request.alert.customer_id,
            source_system="mock-screening-readonly",
            observed_at=datetime.now(UTC),
            attributes={
                "sanctions_match": False,
                "adverse_media_match": False,
                "screening_status": "clear",
            },
        )
