"""Compose the governed RAG graph without giving the model tool autonomy."""

from __future__ import annotations

from typing import Any, Literal

from langgraph.graph import END, START, StateGraph

from src.nodes.rag import RAGDependencies, build_rag_nodes
from src.state.schema import AgentInput, AgentOutput, AgentState


def _after_guard(state: AgentState) -> Literal["retrieve", "record_evidence"]:
    return "record_evidence" if state.get("status") == "blocked" else "retrieve"


def _after_retrieval_guard(state: AgentState) -> Literal["generate", "record_evidence"]:
    return "record_evidence" if state.get("status") == "blocked" else "generate"


def _after_retrieve(state: AgentState) -> Literal["guard_retrieval", "record_evidence"]:
    return "record_evidence" if state.get("status") == "blocked" else "guard_retrieval"


def build_rag_graph(dependencies: RAGDependencies, *, checkpointer: Any = None):
    """Build a deterministic RAG graph with security gates around untrusted data."""
    nodes = build_rag_nodes(dependencies)
    builder = StateGraph(AgentState, input_schema=AgentInput, output_schema=AgentOutput)
    for name, node in nodes.items():
        builder.add_node(name, node)
    builder.add_edge(START, "guard_input")
    builder.add_conditional_edges("guard_input", _after_guard)
    builder.add_conditional_edges("retrieve", _after_retrieve)
    builder.add_conditional_edges("guard_retrieval", _after_retrieval_guard)
    builder.add_edge("generate", "validate_output")
    builder.add_edge("validate_output", "record_evidence")
    builder.add_edge("record_evidence", "cleanup_context")
    builder.add_edge("cleanup_context", END)
    return builder.compile(checkpointer=checkpointer)
