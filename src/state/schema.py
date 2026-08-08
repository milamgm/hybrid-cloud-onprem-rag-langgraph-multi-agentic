"""Define the shared state for agent workflows."""

from __future__ import annotations

from typing import Annotated

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """State shared by nodes in an agent workflow."""

    messages: Annotated[list[AnyMessage], add_messages]
