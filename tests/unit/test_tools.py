"""Tests for the agent tool registry."""

from langchain_core.documents import Document

from src.tools.registry import build_agent_tools


class FakeRetriever:
    def invoke(self, query: str) -> list[Document]:
        return [Document(page_content=query, metadata={"source": "policy.pdf"})]


def test_registry_exposes_only_the_corpus_search_tool():
    tools = build_agent_tools(retriever=FakeRetriever())

    assert [tool.name for tool in tools] == ["search_corpus"]


def test_registry_tool_uses_the_injected_retriever():
    tool = build_agent_tools(retriever=FakeRetriever())[0]

    result = tool.invoke({"query": "business case"})

    assert "business case" in result
    assert "Source: policy.pdf" in result
