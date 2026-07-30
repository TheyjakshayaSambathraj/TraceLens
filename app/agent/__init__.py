"""LangGraph agent package: graph definition, state, nodes.

Provides the leave request decision agent with LangGraph workflow,
strongly-typed state, and workflow nodes.
"""

from app.agent.state import AgentState
from app.agent.graph import LeaveDecisionAgent, create_agent

__all__ = [
    "AgentState",
    "LeaveDecisionAgent",
    "create_agent",
]
