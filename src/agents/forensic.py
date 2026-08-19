"""Tool-bounded specialist agents for financial-crime case investigation.

The specialists are intentionally not given database clients, generic HTTP
clients, web search, or write tools.  Each receives only a small set of
closure-bound read tools for the investigation already in progress.  This lets
the model decide how to use relevant evidence without letting it expand the
tenant, customer, or transaction scope.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Protocol

from langchain.agents import create_agent
from langchain_core.tools import BaseTool, tool

from src.events.contracts import (
    AgentAssessment,
    CaseEvidenceBundle,
    EvidenceCollectionRequested,
    ForensicAnalysis,
    TransactionRiskAlert,
)
from src.forensics.evidence import (
    CoreBankingReadGateway,
    CustomerRiskReadGateway,
    NetworkReadGateway,
    PolicyRAGGateway,
    ScreeningReadGateway,
)


class ForensicReasoner(Protocol):
    """Port implemented by the specialist team or a test double."""

    async def analyze(
        self, alert: TransactionRiskAlert, evidence: CaseEvidenceBundle
    ) -> ForensicAnalysis: ...


@dataclass(frozen=True, slots=True)
class ForensicToolGateways:
    """The only read capabilities delegated to forensic specialists."""

    core_banking: CoreBankingReadGateway
    customer_risk: CustomerRiskReadGateway
    network: NetworkReadGateway
    screening: ScreeningReadGateway
    policy_rag: PolicyRAGGateway


def _json(value: Any) -> str:
    return value.model_dump_json()


def _transaction_tools(
    request: EvidenceCollectionRequested,
    gateways: ForensicToolGateways,
) -> list[BaseTool]:
    @tool("read_transaction_case_context")
    async def read_transaction_case_context() -> str:
        """Read the approved core-banking evidence for this exact case."""
        return _json(await gateways.core_banking.read_case_context(request))

    return [read_transaction_case_context]


def _customer_risk_tools(
    request: EvidenceCollectionRequested,
    gateways: ForensicToolGateways,
) -> list[BaseTool]:
    @tool("read_customer_risk_profile")
    async def read_customer_risk_profile() -> str:
        """Read the bounded KYC/CDD and customer-risk profile for this exact case."""
        return _json(await gateways.customer_risk.read_customer_risk(request))

    @tool("read_subject_screening")
    async def read_subject_screening() -> str:
        """Read the approved sanctions and adverse-media screening result for this case."""
        return _json(await gateways.screening.screen_subjects(request))

    return [read_customer_risk_profile, read_subject_screening]


def _network_tools(
    request: EvidenceCollectionRequested,
    gateways: ForensicToolGateways,
) -> list[BaseTool]:
    @tool("read_transaction_network")
    async def read_transaction_network() -> str:
        """Read the bounded counterparty and linked-activity summary for this exact case."""
        return _json(await gateways.network.read_transaction_network(request))

    return [read_transaction_network]


def _policy_tools(
    request: EvidenceCollectionRequested,
    gateways: ForensicToolGateways,
) -> list[BaseTool]:
    @tool("search_internal_policy")
    async def search_internal_policy() -> str:
        """Retrieve relevant citations from the internal, curated policy corpus only."""
        focus = ", ".join(request.alert.signal_codes)
        citations = await gateways.policy_rag.retrieve_policy(request, focus=focus)
        return "[" + ",".join(_json(citation) for citation in citations) + "]"

    return [search_internal_policy]


class _SpecialistAgent:
    """One dynamic tool-calling loop with a fixed role and capability set."""

    def __init__(
        self,
        *,
        name: str,
        llm: Any,
        system_prompt: str,
        tool_builder: Any,
    ) -> None:
        self._name = name
        self._llm = llm
        self._system_prompt = system_prompt
        self._tool_builder = tool_builder

    async def assess(
        self,
        request: EvidenceCollectionRequested,
        evidence: CaseEvidenceBundle,
        gateways: ForensicToolGateways,
    ) -> AgentAssessment:
        agent = create_agent(
            self._llm,
            self._tool_builder(request, gateways),
            name=self._name,
            response_format=AgentAssessment,
            system_prompt=self._system_prompt,
        )
        result = await agent.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Investigate this alert using your tools before concluding. "
                            "Do not infer facts absent from tool output.\n\n"
                            f"ALERT:\n{request.alert.model_dump_json()}\n\n"
                            f"INITIAL_VALIDATED_EVIDENCE:\n{evidence.model_dump_json()}"
                        ),
                    }
                ]
            }
        )
        structured = result.get("structured_response")
        if structured is None:
            raise ValueError(f"{self._name} did not produce a structured assessment")
        assessment = AgentAssessment.model_validate(structured)
        # The agent is assigned an identity by the workflow, never by the model.
        return assessment.model_copy(update={"agent_name": self._name})


class _CaseLeadAgent:
    """Synthesises specialist work; it cannot call business-system tools."""

    def __init__(self, llm: Any) -> None:
        self._llm = llm

    async def analyze(
        self,
        *,
        request: EvidenceCollectionRequested,
        evidence: CaseEvidenceBundle,
        assessments: tuple[AgentAssessment, ...],
    ) -> ForensicAnalysis:
        @tool("read_specialist_case_dossier")
        def read_specialist_case_dossier() -> str:
            """Read the validated alert, evidence and specialist assessments for this case."""
            return (
                '{"alert":'
                + request.alert.model_dump_json()
                + ',"evidence":'
                + evidence.model_dump_json()
                + ',"assessments":['
                + ",".join(_json(item) for item in assessments)
                + "]}"
            )

        agent = create_agent(
            self._llm,
            [read_specialist_case_dossier],
            name="case_lead",
            response_format=ForensicAnalysis,
            system_prompt=(
                "You are the lead financial-crime investigator. Use the dossier tool "
                "before writing a structured recommendation. Reconcile disagreements, "
                "cite only supplied evidence IDs/citations, call out material gaps, and "
                "recommend hold, review, or clear. You may not initiate actions, file a "
                "report, or claim certainty beyond the evidence."
            ),
        )
        result = await agent.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "Produce the investigation decision package for this case.",
                    }
                ]
            }
        )
        structured = result.get("structured_response")
        if structured is None:
            raise ValueError("case_lead did not produce a structured analysis")
        return ForensicAnalysis.model_validate(structured)


class ForensicInvestigationTeam:
    """Four specialists plus a lead, run as bounded and attributable agents.

    The specialists execute in parallel because their tools read distinct,
    scoped evidence domains.  The lead receives their structured outputs, not
    their tool credentials.  No agent has a write tool; the graph retains the
    mandatory human approval gate for any operational action.
    """

    def __init__(self, *, llm: Any, gateways: ForensicToolGateways) -> None:
        self._gateways = gateways
        self._specialists = (
            _SpecialistAgent(
                name="transaction_analyst",
                llm=llm,
                tool_builder=_transaction_tools,
                system_prompt=(
                    "You are a transaction-analysis specialist. Inspect the approved "
                    "core-banking context, identify transaction-level anomalies and "
                    "return evidence-grounded findings. Never access or propose actions "
                    "outside this case."
                ),
            ),
            _SpecialistAgent(
                name="customer_risk_analyst",
                llm=llm,
                tool_builder=_customer_risk_tools,
                system_prompt=(
                    "You are a KYC/CDD and screening specialist. Use only your approved "
                    "customer-profile and screening tools. Distinguish a screening lead "
                    "from a verified match and identify missing due-diligence evidence."
                ),
            ),
            _SpecialistAgent(
                name="network_analyst",
                llm=llm,
                tool_builder=_network_tools,
                system_prompt=(
                    "You are a transaction-network specialist. Use the network summary "
                    "to identify linkage or typology indicators. Do not invent graph "
                    "relationships that are not returned by the tool."
                ),
            ),
            _SpecialistAgent(
                name="policy_compliance_analyst",
                llm=llm,
                tool_builder=_policy_tools,
                system_prompt=(
                    "You are a financial-crime policy specialist. Search the curated "
                    "internal policy corpus, map evidence to cited procedures, and explain "
                    "the required analyst review. Do not use public web sources."
                ),
            ),
        )
        self._lead = _CaseLeadAgent(llm)

    async def analyze(
        self, alert: TransactionRiskAlert, evidence: CaseEvidenceBundle
    ) -> ForensicAnalysis:
        request = EvidenceCollectionRequested(
            investigation_id=f"investigation-{alert.alert_id}",
            request_id=f"{alert.alert_id}:agent-analysis:v1",
            alert=alert,
        )
        assessments = tuple(
            await asyncio.gather(
                *(
                    specialist.assess(request, evidence, self._gateways)
                    for specialist in self._specialists
                )
            )
        )
        return await self._lead.analyze(
            request=request,
            evidence=evidence,
            assessments=assessments,
        )
