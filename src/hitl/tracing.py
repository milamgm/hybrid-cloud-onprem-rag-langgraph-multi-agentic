"""Trace metadata shared by cloud and on-premise HITL invocations."""

from __future__ import annotations

import os
from typing import Any

from src.graph.thread_identity import case_checkpoint_thread_id, checkpoint_thread_id


def graph_config(
    *,
    tenant_id: str,
    subject_id: str,
    thread_id: str,
    request_id: str | None = None,
) -> dict[str, Any]:
    """Build a safe LangGraph config without placing business data in metadata."""

    internal_thread_id = checkpoint_thread_id(tenant_id, subject_id, thread_id)
    config: dict[str, Any] = {
        "configurable": {"thread_id": internal_thread_id},
        "metadata": {
            "tenant_id": tenant_id,
            "public_thread_id": thread_id,
            **({"request_id": request_id} if request_id else {}),
        },
        "tags": ["forensic-investigation", "human-in-the-loop"],
    }
    if os.getenv("LANGFUSE_ENABLED", "false").strip().lower() == "true":
        try:
            from langfuse.langchain import CallbackHandler
        except ImportError as exc:  # pragma: no cover - deployment dependency
            raise RuntimeError(
                "LANGFUSE_ENABLED requires the langfuse package"
            ) from exc
        config["callbacks"] = [CallbackHandler()]
    return config


def case_graph_config(
    *, tenant_id: str, case_id: str, request_id: str | None = None
) -> dict[str, Any]:
    """Config for the case graph; reviewer identity is authorization, not state scope."""
    config = {
        "configurable": {"thread_id": case_checkpoint_thread_id(tenant_id, case_id)},
        "metadata": {
            "tenant_id": tenant_id,
            "public_thread_id": case_id,
            **({"request_id": request_id} if request_id else {}),
        },
        "tags": ["forensic-investigation", "human-in-the-loop"],
    }
    if os.getenv("LANGFUSE_ENABLED", "false").strip().lower() == "true":
        try:
            from langfuse.langchain import CallbackHandler
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "LANGFUSE_ENABLED requires the langfuse package"
            ) from exc
        config["callbacks"] = [CallbackHandler()]
    return config
