"""Deterministic RAG nodes with isolated ephemeral context and cleanup."""

from __future__ import annotations

import os
from contextlib import suppress
from dataclasses import dataclass
from inspect import signature
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage

from src.audit.events import AuditService
from src.governance.evidence import GovernanceService
from src.memory.cache import ContextCache, ContextUnavailable
from src.rag.generator import NO_ANSWER, Generator
from src.security.injection import InjectionViolation, PromptInjectionMiddleware
from src.security.output_validation import OutputValidationMiddleware
from src.state.schema import (
    SUMMARY_KEEP_MESSAGES,
    SUMMARY_TRIGGER_MESSAGES,
    AgentState,
    CitationReference,
    context_messages,
    estimate_message_tokens,
)

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
    audit: AuditService | None = None
    summarizer: Any = None


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

    if dependencies.audit is not None:
        audit = dependencies.audit
    elif os.getenv("DEPLOYMENT_ENVIRONMENT") == "production":
        audit = AuditService.from_environment()
    else:
        audit = AuditService()

    def manage_context(state: AgentState) -> dict:
        """Summarize old turns and trim the active buffer before every run."""
        messages = state["messages"]
        active = context_messages(messages)
        non_system = [message for message in messages if message.type != "system"]
        threshold_reached = len(non_system) > SUMMARY_TRIGGER_MESSAGES or sum(
            estimate_message_tokens(message) for message in messages
        ) > int(os.getenv("SUMMARY_TRIGGER_TOKENS", "1800"))
        if not threshold_reached:
            return {}

        keep_ids = {message.id for message in active[-SUMMARY_KEEP_MESSAGES:]}
        old_messages = [
            message
            for message in non_system
            if message.id not in keep_ids and message.id is not None
        ]
        updates: dict[str, Any] = {
            "messages": [RemoveMessage(id=message.id) for message in old_messages]
        }
        summarizer = dependencies.summarizer
        if summarizer is None and hasattr(dependencies.generator, "summarize"):
            summarizer = dependencies.generator
        if summarizer is not None and old_messages:
            updates["conversation_summary"] = summarizer.summarize(
                old_messages, state.get("conversation_summary")
            )
        return updates

    def guard_input(state: AgentState) -> dict:
        audit.record(
            "prompt.received",
            tenant_id=state["tenant_id"],
            subject_id=state["subject_id"],
            thread_id=state["thread_id"],
            request_id=state["request_id"],
            payload={
                "prompt": _question(state),
                "long_term_memory_count": len(state.get("long_term_memories", [])),
            },
        )
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
        audit.record(
            "retrieval.completed",
            tenant_id=state["tenant_id"],
            subject_id=state["subject_id"],
            thread_id=state["thread_id"],
            request_id=state["request_id"],
            payload={
                "document_count": len(documents),
                "sources": [citation["source"] for citation in citations],
            },
        )
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
        kwargs = {"presentation_preferences": state.get("presentation_preferences", {})}
        parameters = signature(dependencies.generator.generate).parameters
        if "conversation_summary" in parameters:
            kwargs["conversation_summary"] = state.get("conversation_summary")
        if "history" in parameters:
            kwargs["history"] = state["messages"][:-1]
        if "long_term_memories" in parameters:
            kwargs["long_term_memories"] = state.get("long_term_memories", [])
        answer = dependencies.generator.generate(_question(state), documents, **kwargs)
        audit.record(
            "response.generated",
            tenant_id=state["tenant_id"],
            subject_id=state["subject_id"],
            thread_id=state["thread_id"],
            request_id=state["request_id"],
            payload={"response": answer.text},
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
        audit.record(
            "policy.decision",
            tenant_id=state["tenant_id"],
            subject_id=state["subject_id"],
            thread_id=state["thread_id"],
            request_id=state["request_id"],
            payload={
                "decision": decision,
                "reason_code": reason_code,
                "governance_event_id": event.event_id,
            },
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
        "manage_context": manage_context,
        "retrieve": retrieve,
        "guard_retrieval": guard_retrieval,
        "generate": generate,
        "validate_output": validate_output,
        "record_evidence": record_evidence,
        "cleanup_context": cleanup_context,
    }
