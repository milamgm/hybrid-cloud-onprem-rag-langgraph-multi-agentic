"""Tests for cloud-only web search."""

import pytest
from langchain_core.tools import BaseTool, tool

from src.tools.web_search import WebSearchPolicy, build_web_search_tool


@tool
def fake_web_search(query: str) -> str:
    """Return a fixed response."""
    return query


class ToolFactory:
    def __init__(self):
        self.kwargs = None

    def __call__(self, **kwargs) -> BaseTool:
        self.kwargs = kwargs
        return fake_web_search


def test_onprem_never_exposes_web_search():
    tool = build_web_search_tool(
        infrastructure_mode="on_premise",
        policy=WebSearchPolicy.APPROVED_WEB,
        allowed_domains=["example.com"],
        api_key="test-key",
    )

    assert tool is None


def test_internal_only_policy_disables_cloud_web_search():
    tool = build_web_search_tool(
        infrastructure_mode="cloud",
        policy=WebSearchPolicy.INTERNAL_ONLY,
    )

    assert tool is None


def test_approved_cloud_search_uses_tavily_safe_defaults():
    factory = ToolFactory()

    tool = build_web_search_tool(
        infrastructure_mode="cloud",
        policy=WebSearchPolicy.APPROVED_WEB,
        allowed_domains=["example.com", "docs.example.com"],
        api_key="test-key",
        tool_factory=factory,
    )

    assert tool is fake_web_search
    assert factory.kwargs["name"] == "search_web"
    assert factory.kwargs["include_domains"] == [
        "example.com",
        "docs.example.com",
    ]
    assert factory.kwargs["include_answer"] is False
    assert factory.kwargs["include_raw_content"] is False
    assert factory.kwargs["max_results"] == 5


@pytest.mark.parametrize(
    ("allowed_domains", "api_key", "message"),
    [
        ([], "test-key", "WEB_SEARCH_ALLOWED_DOMAINS"),
        (["example.com"], None, "TAVILY_API_KEY"),
    ],
)
def test_approved_cloud_search_requires_its_security_configuration(
    allowed_domains, api_key, message
):
    with pytest.raises(ValueError, match=message):
        build_web_search_tool(
            infrastructure_mode="cloud",
            policy=WebSearchPolicy.APPROVED_WEB,
            allowed_domains=allowed_domains,
            api_key=api_key,
        )
