"""LangGraph workflow definition for the leave decision agent.

This module constructs the complete decision workflow as an explicit
state machine with clear transitions and error handling.

The workflow follows this structure:

    START
      |
      v
    retrieve_policy
      |
      v
    retrieve_employee
      |
      v
    make_decision
      |
      v
    generate_response
      |
      v
    END

Each node is independently testable and observable.
"""

from __future__ import annotations

import structlog
import uuid
from typing import Callable

from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableConfig

from app.agent.state import AgentState
from app.agent.nodes.nodes import (
    retrieve_policy_node,
    retrieve_employee_node,
    make_decision_node,
    generate_response_node,
)
from app.rag.retriever import RetrieverService
from app.services.employee import EmployeeService
from app.services.llm_provider import LLMProvider

logger = structlog.get_logger(__name__)


class LeaveDecisionAgent:
    """LangGraph-based leave request decision agent.

    This class constructs and manages the decision workflow graph.
    It coordinates between nodes and external services (RAG, employee DB, LLM).
    """

    def __init__(
        self,
        retriever: RetrieverService,
        employee_service: EmployeeService,
        llm_provider: LLMProvider,
    ):
        """Initialize the leave decision agent.

        Args:
            retriever: RetrieverService for policy document retrieval.
            employee_service: EmployeeService for employee information.
            llm_provider: LLMProvider for LLM access.
        """
        self.retriever = retriever
        self.employee_service = employee_service
        self.llm_provider = llm_provider

        self.graph = self._build_graph()
        logger.info("leave_decision_agent_initialized")

    def _build_graph(self):
        """Construct the LangGraph workflow.

        Returns:
            Compiled LangGraph StateGraph instance.
        """
        workflow = StateGraph(AgentState)

        # Add nodes to the graph
        workflow.add_node(
            "retrieve_policy",
            self._retrieve_policy_wrapper,
        )
        workflow.add_node(
            "retrieve_employee",
            self._retrieve_employee_wrapper,
        )
        workflow.add_node(
            "make_decision",
            self._make_decision_wrapper,
        )
        workflow.add_node(
            "generate_response",
            self._generate_response_wrapper,
        )

        # Define edges (transitions)
        workflow.add_edge("retrieve_policy", "retrieve_employee")
        workflow.add_edge("retrieve_employee", "make_decision")
        workflow.add_edge("make_decision", "generate_response")
        workflow.add_edge("generate_response", END)

        # Set the starting node
        workflow.set_entry_point("retrieve_policy")

        # Compile the graph
        compiled_graph = workflow.compile()
        logger.info("langgraph_workflow_compiled")

        return compiled_graph

    def _retrieve_policy_wrapper(self, state: AgentState) -> AgentState:
        """Wrapper for retrieve_policy_node to inject dependencies."""
        update = retrieve_policy_node(state, self.retriever)
        return {**state, **update}

    def _retrieve_employee_wrapper(self, state: AgentState) -> AgentState:
        """Wrapper for retrieve_employee_node to inject dependencies."""
        update = retrieve_employee_node(state, self.employee_service)
        return {**state, **update}

    def _make_decision_wrapper(self, state: AgentState) -> AgentState:
        """Wrapper for make_decision_node to inject dependencies."""
        update = make_decision_node(state, self.llm_provider)
        return {**state, **update}

    def _generate_response_wrapper(self, state: AgentState) -> AgentState:
        """Wrapper for generate_response_node."""
        update = generate_response_node(state)
        return {**state, **update}

    def decide(
        self,
        request: str,
        user_id: str = "USER-001",
        session_id: str | None = None,
    ) -> AgentState:
        """Execute the decision workflow.

        Args:
            request: The user's leave request (e.g., "Can employee EMP-001 take 15 days?").
            user_id: The user making the request.
            session_id: Session identifier for traceability. Generated if not provided.

        Returns:
            Final agent state after workflow completion.

        Raises:
            ValueError: If request validation fails.
            Exception: If any node fails during execution.
        """
        if not request or not request.strip():
            logger.error("empty_request_in_decide")
            raise ValueError("Request cannot be empty")

        if session_id is None:
            session_id = f"SESSION-{uuid.uuid4().hex[:12].upper()}"

        logger.info(
            "decision_workflow_started",
            session_id=session_id,
            user_id=user_id,
            request=request,
        )

        # Initialize state
        initial_state: AgentState = {
            "session_id": session_id,
            "user_id": user_id,
            "request": request,
            "metadata": {
                "created_at": str(__import__("datetime").datetime.utcnow()),
                "workflow_version": "phase-2",
            },
        }

        try:
            # Execute the graph
            final_state = self.graph.invoke(initial_state)

            decision_value = final_state.get("decision")
            if hasattr(decision_value, "decision"):
                decision_label = decision_value.decision.value
            elif isinstance(decision_value, dict):
                decision_label = decision_value.get("decision", "UNKNOWN")
            else:
                decision_label = "UNKNOWN"

            logger.info(
                "decision_workflow_completed",
                session_id=session_id,
                decision=decision_label,
            )

            return final_state

        except Exception as e:
            logger.error(
                "decision_workflow_failed",
                session_id=session_id,
                error=str(e),
                exc_info=True,
            )
            raise

    def get_graph(self):
        """Get the compiled LangGraph instance for inspection or testing.

        Returns:
            Compiled StateGraph instance.
        """
        return self.graph


def create_agent(
    retriever: RetrieverService,
    employee_service: EmployeeService,
    llm_provider: LLMProvider,
) -> LeaveDecisionAgent:
    """Factory function for creating a LeaveDecisionAgent.

    Args:
        retriever: RetrieverService instance.
        employee_service: EmployeeService instance.
        llm_provider: LLMProvider instance.

    Returns:
        Initialized LeaveDecisionAgent instance.
    """
    return LeaveDecisionAgent(retriever, employee_service, llm_provider)
