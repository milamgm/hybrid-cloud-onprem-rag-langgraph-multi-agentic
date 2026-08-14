"""Tests for the agent state schema."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from src.state.schema import (
    MAX_CHECKPOINT_MESSAGES,
    AgentInput,
    AgentState,
    append_security_events,
    reduce_status,
    replace_citations,
)


def test_state_appends_messages_from_nodes():
    def respond(_state: AgentState):
        return {"messages": [AIMessage(content="Hello")]}

    graph = (
        StateGraph(AgentState)
        .add_node("respond", respond)
        .add_edge(START, "respond")
        .add_edge("respond", END)
        .compile()
    )

    result = graph.invoke({"messages": [HumanMessage(content="Hi")]})

    assert [message.content for message in result["messages"]] == ["Hi", "Hello"]


def test_state_replaces_a_message_with_the_same_id():
    original = HumanMessage(content="Initial", id="message-1")
    replacement = HumanMessage(content="Updated", id="message-1")

    def correct(_state: AgentState):
        return {"messages": [replacement]}

    graph = (
        StateGraph(AgentState)
        .add_node("correct", correct)
        .add_edge(START, "correct")
        .add_edge("correct", END)
        .compile()
    )

    result = graph.invoke({"messages": [original]})

    assert [message.content for message in result["messages"]] == ["Updated"]


def test_state_bounds_durable_message_history():
    def respond(_state: AgentState):
        return {"messages": [AIMessage(content="latest")]}

    graph = (
        StateGraph(AgentState)
        .add_node("respond", respond)
        .add_edge(START, "respond")
        .add_edge("respond", END)
        .compile()
    )
    messages = [HumanMessage(content=f"message-{index}") for index in range(40)]

    result = graph.invoke({"messages": messages})

    assert len(result["messages"]) == MAX_CHECKPOINT_MESSAGES
    assert result["messages"][-1].content == "latest"


def test_input_schema_rejects_unknown_internal_channels():
    with pytest.raises(ValidationError):
        AgentInput.model_validate(
            {
                "messages": [HumanMessage(content="Hi")],
                "request_id": "request-1",
                "tenant_id": "bank-a",
                "subject_id": "alice",
                "thread_id": "case-1",
                "roles": ("analyst",),
                "data_classification": "internal",
                "security_events": [],
            }
        )


def test_reducers_validate_updates_and_enforce_lifecycle():
    assert reduce_status(None, "received") == "received"
    assert reduce_status("received", "retrieved") == "retrieved"
    assert reduce_status("completed", "received") == "received"
    with pytest.raises(ValueError, match="transition"):
        reduce_status("generated", "received")

    with pytest.raises(ValueError, match="unique"):
        replace_citations([], [{"marker": 1, "source": "handbook.pdf"}, {"marker": 1, "source": "other.pdf"}])
    with pytest.raises(ValidationError):
        append_security_events([], [{"control": "guard", "outcome": "blocked"}])
