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
)


class CoreBankingReadGateway(Protocol):
    async def read_case_context(self, request: EvidenceCollectionRequested) -> CoreBankingEvidence:
        """Run only the approved, parameterised customer/transaction reads."""


class PolicyRAGGateway(Protocol):
    async def retrieve_policy(self, request: EvidenceCollectionRequested) -> tuple[PolicyCitation, ...]:
        """Retrieve citations from the curated policy corpus, never from the web."""


class NamedQueryClient(Protocol):
    async def execute_named_query(self, operation: str, parameters: dict[str, str]) -> dict[str, Any]:
        """Execute a repository-owned prepared statement or stored procedure."""


class ParameterizedCoreBankingGateway:
    """Production adapter boundary for core/AML/CRM repositories.

    ``operation`` is an allowlisted repository name, not user/model-provided
    SQL.  The implementing service account must be read-only and constrained
    to a tenant row-level-security policy as a defence in depth measure.
    """

    OPERATION = "forensics.case_context.v1"

    def __init__(self, client: NamedQueryClient, *, source_system: str = "core-banking") -> None:
        self._client = client
        self._source_system = source_system

    async def read_case_context(self, request: EvidenceCollectionRequested) -> CoreBankingEvidence:
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


class CaseEvidenceCollector:
    """Collect independent read-only sources concurrently and validate a bundle."""

    def __init__(self, *, core_banking: CoreBankingReadGateway, policy_rag: PolicyRAGGateway) -> None:
        self._core_banking = core_banking
        self._policy_rag = policy_rag

    async def collect(self, request: EvidenceCollectionRequested) -> CaseEvidenceBundle:
        core, policy_citations = await asyncio.gather(
            self._core_banking.read_case_context(request),
            self._policy_rag.retrieve_policy(request),
        )
        if core.tenant_id != request.alert.tenant_id or core.customer_id != request.alert.customer_id:
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

    async def read_case_context(self, request: EvidenceCollectionRequested) -> CoreBankingEvidence:
        alert = request.alert
        digest = sha256(f"{alert.tenant_id}:{alert.customer_id}:{alert.transaction_id}".encode()).hexdigest()
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

    async def retrieve_policy(self, request: EvidenceCollectionRequested) -> tuple[PolicyCitation, ...]:
        return (
            PolicyCitation(
                citation_id=f"policy-{request.investigation_id}-1",
                source="AML/Fraud investigation procedure",
                section="Alert investigation and disposition",
                excerpt="Review the customer risk profile, transaction activity and documented red flags before a human disposition.",
            ),
        )
