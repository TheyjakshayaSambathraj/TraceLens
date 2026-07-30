"""Typed agent state schema for LangGraph workflow.

The state is shared across all nodes and contains all information
accumulated during the decision process.
"""

from __future__ import annotations

from typing import TypedDict, Optional

from app.schemas.decision import DecisionResult
from app.schemas.employee import EmployeeInfo
from app.schemas.retrieval import RetrievalResult


class AgentState(TypedDict, total=False):
    """Strongly typed state shared across LangGraph nodes.

    This state flows through the entire decision workflow and accumulates
    information at each stage. All fields are optional to support partial
    initialization, but validation occurs at node boundaries.

    Attributes:
        session_id: Unique session identifier for traceability.
        user_id: The user making or requesting the decision.
        request: The original user request/question.
        retrieved_documents: Policy documents retrieved from the vector store.
        retrieved_context: Formatted context from retrieved documents.
        employee_data: Employee information retrieved from the service.
        decision: Structured decision result from the LLM.
        decision_reason: Internal LLM reasoning (for observability only).
        final_response: User-facing response message.
        metadata: Auxiliary metadata accumulated during execution.
    """

    session_id: str
    user_id: str
    request: str
    retrieved_documents: RetrievalResult
    retrieved_context: str
    employee_data: EmployeeInfo
    decision: DecisionResult
    decision_reason: str
    final_response: str
    metadata: dict[str, str]
