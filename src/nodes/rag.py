"""Deterministic RAG nodes with isolated ephemeral context and cleanup."""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from src.governance.evidence import GovernanceService
from src.memory.cache import ContextCache, ContextUnavailable
from src.rag.generator import NO_ANSWER, Generator
from src.security.injection import InjectionViolation, PromptInjectionMiddleware
from src.security.output_validation import OutputValidationMiddleware
from src.state.schema import AgentState, CitationReference

BLOCKED_RESPONSE = "I can't provide a response to that request."


@dataclass(frozen=True)
class RAGDependencies:
    """Infrastructure ports supplied by the composition root."""

    retriever: Any
    generator: Generator
    context_cache: ContextCache
    injection: PromptInjectionMiddleware
    output_validation: OutputValidationMiddleware
    governance: GovernanceService


def _question(state: AgentState) -> str:
    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage):
            return str(message.content)
    raise ValueError("The workflow requires a human message.")


def _documents(state: AgentState, cache: ContextCache):
    handle = state.get("retrieval_handle")
    if not handle:
        raise ContextUnavailable("The workflow has no RAG context handle.")
    return cache.get(
        handle=handle,
        tenant_id=state["tenant_id"],
        thread_id=state["thread_id"],
    )


def _blocked(control: str, reason: str) -> dict:
    return {
        "status": "blocked",
        "response_text": BLOCKED_RESPONSE,
        "citations": [],
        "messages": [AIMessage(content=BLOCKED_RESPONSE)],
        "security_events": [
            {"control": control, "outcome": "blocked", "reason": reason}
        ],
    }


def build_rag_nodes(dependencies: RAGDependencies) -> dict[str, Any]:
    """Return stateless node functions closed over explicit infrastructure."""

    def guard_input(state: AgentState) -> dict:
        try:
            dependencies.injection.enforce(_question(state))
        except InjectionViolation:
            return _blocked("prompt_injection", "direct_attack")
        return {"status": "received"}

    def retrieve(state: AgentState) -> dict:
        documents = dependencies.retriever.invoke(_question(state))
        try:
            handle = dependencies.context_cache.put(
                tenant_id=state["tenant_id"],
                thread_id=state["thread_id"],
                classification=state["data_classification"],
                documents=documents,
            )
        except (ContextUnavailable, ValueError):
            return _blocked("ephemeral_context", "classification_or_cache_failure")
        citations: list[CitationReference] = [
            {
                "marker": marker,
                "source": document.metadata.get("source", "unknown"),
                "page": document.metadata.get("page"),
            }
            for marker, document in enumerate(documents, 1)
        ]
        return {
            "retrieval_handle": handle,
            "citations": citations,
            "status": "retrieved",
        }

    def guard_retrieval(state: AgentState) -> dict:
        try:
            documents = _documents(state, dependencies.context_cache)
            dependencies.injection.enforce(
                _question(state), [document.page_content for document in documents]
            )
        except InjectionViolation:
            return _blocked("rag_prompt_injection", "indirect_attack")
        except ContextUnavailable:
            return _blocked("ephemeral_context", "expired_or_isolation_failure")
        return {}

    def generate(state: AgentState) -> dict:
        try:
            documents = _documents(state, dependencies.context_cache)
        except ContextUnavailable:
            return _blocked("ephemeral_context", "expired_before_generation")
        answer = dependencies.generator.generate(
            _question(state),
            documents,
            presentation_preferences=state.get("presentation_preferences", {}),
        )
        return {"response_text": answer.text, "status": "generated"}

    def validate_output(state: AgentState) -> dict:
        if state.get("status") == "blocked":
            return {}
        response = state["response_text"]
        citations = state.get("citations", [])
        result = dependencies.output_validation.validate(
            response,
            available_citations=[citation["marker"] for citation in citations],
            require_citations=bool(citations) and response != NO_ANSWER,
        )
        if not result.allowed:
            return _blocked("output_validation", "policy_violation")
        return {
            "response_text": result.text,
            "status": "validated",
            "messages": [AIMessage(content=result.text)],
        }

    def record_evidence(state: AgentState) -> dict:
        decision = "allow" if state["status"] == "validated" else "block"
        reason_code = "validated"
        if decision == "block":
            events = state.get("security_events", [])
            reason_code = events[-1]["control"] if events else "workflow_blocked"
        event = dependencies.governance.record_policy_decision(
            f"tenant:{state['tenant_id']}:rag.response",
            policy_id="response-validation",
            policy_version="2",
            decision=decision,
            reason_code=reason_code,
            correlation_id=state["request_id"],
        )
        return {
            "status": "completed" if decision == "allow" else "blocked",
            "governance_event_ids": [event.event_id],
        }

    def cleanup_context(state: AgentState) -> dict:
        if handle := state.get("retrieval_handle"):
            with suppress(ContextUnavailable):
                dependencies.context_cache.delete(
                    handle=handle,
                    tenant_id=state["tenant_id"],
                    thread_id=state["thread_id"],
                )
        return {}

    return {
        "guard_input": guard_input,
        "retrieve": retrieve,
        "guard_retrieval": guard_retrieval,
        "generate": generate,
        "validate_output": validate_output,
        "record_evidence": record_evidence,
        "cleanup_context": cleanup_context,
    }
