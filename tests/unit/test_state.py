"""Tests for the agent state schema."""

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph

from src.state.schema import MAX_CHECKPOINT_MESSAGES, AgentState


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
