"""Composition root for the interactive RAG application."""

from __future__ import annotations

import os
from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver

from src.agents.rag_agent import RAGAgent
from src.app.development_controls import (
    DevelopmentInjectionGuard,
    DevelopmentOutputValidation,
)
from src.audit.events import AuditService
from src.governance.evidence import GovernanceService
from src.graph.persistence import open_checkpointer
from src.graph.rag_graph import build_rag_graph
from src.memory.cache import InMemoryContextCache, RedisContextCache
from src.nodes.rag import RAGDependencies
from src.security.injection import PromptInjectionMiddleware
from src.security.output_validation import OutputValidationMiddleware


def _is_production() -> bool:
    return os.getenv("DEPLOYMENT_ENVIRONMENT", "development").lower() == "production"


def _security_profile() -> str:
    default = "managed" if _is_production() else "development"
    profile = os.getenv("RAG_SECURITY_PROFILE", default).strip().lower()
    if profile not in {"development", "managed"}:
        raise ValueError("RAG_SECURITY_PROFILE must be 'development' or 'managed'.")
    if _is_production() and profile != "managed":
        raise ValueError("Production requires RAG_SECURITY_PROFILE=managed.")
    return profile


def _context_cache() -> Any:
    backend = os.getenv(
        "RAG_CONTEXT_CACHE", "redis" if _is_production() else "memory"
    ).lower()
    if backend == "memory":
        if _is_production():
            raise ValueError("In-memory RAG context is forbidden in production.")
        return InMemoryContextCache()
    if backend == "redis":
        return RedisContextCache.from_environment()
    raise ValueError("RAG_CONTEXT_CACHE must be 'memory' or 'redis'.")


def _checkpointer(stack: ExitStack) -> Any:
    backend = os.getenv("LANGGRAPH_CHECKPOINTER")
    if not backend and not _is_production():
        return InMemorySaver()
    return stack.enter_context(open_checkpointer())


@dataclass
class RAGApplication:
    """Live application resources retained for the lifetime of the UI process."""

    agent: RAGAgent
    mode: str
    security_profile: str
    indexer: Any
    _stack: ExitStack = field(repr=False)
    _audit: AuditService = field(repr=False)

    @property
    def indexed_chunks(self) -> int:
        return self.indexer.count

    def close(self) -> None:
        self._audit.close()
        self._stack.close()


def build_rag_application() -> RAGApplication:
    """Build a runnable RAG agent from the selected infrastructure profile."""
    from src.config.config import get_generator, get_indexer, get_retriever

    mode = os.getenv("INFRASTRUCTURE_MODE", "on_premise").lower()
    if mode not in {"cloud", "on_premise"}:
        raise ValueError("INFRASTRUCTURE_MODE must be 'cloud' or 'on_premise'.")

    stack = ExitStack()
    audit = AuditService.from_environment()
    try:
        profile = _security_profile()
        if profile == "managed":
            injection: Any = PromptInjectionMiddleware()
            output_validation: Any = OutputValidationMiddleware()
        else:
            injection = DevelopmentInjectionGuard()
            output_validation = DevelopmentOutputValidation()

        indexer = get_indexer()
        dependencies = RAGDependencies(
            retriever=get_retriever(),
            generator=get_generator(),
            context_cache=_context_cache(),
            injection=injection,
            output_validation=output_validation,
            governance=GovernanceService(),
            audit=audit,
        )
        graph = build_rag_graph(dependencies, checkpointer=_checkpointer(stack))
        return RAGApplication(
            agent=RAGAgent(graph),
            mode=mode,
            security_profile=profile,
            indexer=indexer,
            _stack=stack,
            _audit=audit,
        )
    except Exception:
        audit.close()
        stack.close()
        raise
