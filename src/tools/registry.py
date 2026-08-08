"""Register the tools assigned to an agent."""

from __future__ import annotations

from langchain_core.tools import BaseTool

from src.tools.retrieval import build_retrieval_tool
from src.tools.web_search import build_web_search_tool


def build_agent_tools(
    *,
    retriever=None,
    infrastructure_mode: str | None = None,
) -> list[BaseTool]:
    """Build the allowlisted tools for an agent workflow."""
    tools = [build_retrieval_tool(retriever=retriever)]
    web_search = build_web_search_tool(infrastructure_mode=infrastructure_mode)
    if web_search is not None:
        tools.append(web_search)
    return tools
