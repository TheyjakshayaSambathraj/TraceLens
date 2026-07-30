"""Instrumented agent execution wrapper.

This module provides an instrumentation layer around the existing
LangGraph agent without modifying its behavior.

The wrapper:
1. Establishes execution context
2. Publishes INPUT_RECEIVED event
3. Executes the agent (with LangSmith tracing if configured)
4. Emits events for meaningful execution stages
5. Captures failures with error handling
6. Maintains full transparency to the underlying agent

The wrapped agent behaves identically to the unwrapped agent,
semantically equivalent.
"""

from __future__ import annotations

import structlog
import time
from datetime import datetime, timezone
from typing import Callable, Any, Optional

from app.observability.execution_context import ExecutionContext
from app.observability.events import (
    ExecutionEvent,
    EventType,
    InputReceivedEvent,
    RetrievalStartedEvent,
    RetrievalCompletedEvent,
    ToolStartedEvent,
    ToolCompletedEvent,
    DecisionStartedEvent,
    DecisionCompletedEvent,
    OutputGeneratedEvent,
    ExecutionFailedEvent,
)
from app.observability.publisher import get_publisher, EventPublisher
from app.agent.graph import LeaveDecisionAgent
from app.agent.state import AgentState


logger = structlog.get_logger(__name__)


class InstrumentedAgent:
    """Instrumented wrapper around LeaveDecisionAgent.

    Adds observability without changing agent semantics or behavior.
    Events are published via the configured event bus.

    Attributes:
        agent: The underlying LeaveDecisionAgent instance.
        publisher: Event publisher for emitting execution events.
    """

    def __init__(
        self,
        agent: LeaveDecisionAgent,
        publisher: Optional[EventPublisher] = None,
    ):
        """Initialize the instrumented agent.

        Args:
            agent: LeaveDecisionAgent instance to wrap.
            publisher: EventPublisher for events. Defaults to global publisher.
        """
        self.agent = agent
        self.publisher = publisher or get_publisher()
        self._sequence = 0

    def _next_sequence(self) -> int:
        """Get the next sequence number for event correlation.

        Sequence numbers are monotonically increasing per execution
        context and enable proper ordering of events.

        Returns:
            Next sequence number.
        """
        self._sequence += 1
        return self._sequence

    def _reset_sequence(self) -> None:
        """Reset sequence counter for new execution."""
        self._sequence = 0

    def _emit_event(self, event: ExecutionEvent, context: ExecutionContext) -> None:
        """Emit an execution event with context and sequencing.

        Args:
            event: ExecutionEvent to emit.
            context: ExecutionContext for correlation.
        """
        # Populate event with context
        event.session_id = context.session_id
        event.user_id = context.user_id
        event.trace_id = context.trace_id
        event.sequence = self._next_sequence()

        logger.debug(
            "event_emitted",
            event_type=event.event_type.value,
            sequence=event.sequence,
            session_id=context.session_id,
        )

        self.publisher.publish(event)

    def decide(
        self,
        request: str,
        user_id: str = "USER-001",
        session_id: Optional[str] = None,
    ) -> AgentState:
        """Execute the decision workflow with instrumentation.

        This method is semantically identical to agent.decide() but adds:
        - Execution context creation
        - Event emission for observable stages
        - Failure tracking
        - Duration measurement

        The return value is unchanged from the underlying agent.

        Args:
            request: The user's leave request.
            user_id: The user making the request.
            session_id: Session identifier. Generated if not provided.

        Returns:
            Final agent state after workflow completion.

        Raises:
            ValueError: If request validation fails (same as unwrapped agent).
            Exception: If any node fails (same as unwrapped agent).
        """
        # Create execution context
        context = ExecutionContext.create(
            session_id=session_id,
            user_id=user_id,
        )
        self._reset_sequence()

        execution_start = time.time()
        execution_start_utc = datetime.now(timezone.utc)

        logger.info(
            "instrumented_execution_started",
            session_id=context.session_id,
            user_id=user_id,
            request=request,
        )

        try:
            # Emit INPUT_RECEIVED event
            input_event = InputReceivedEvent(
                request=request,
                metadata={"user_id": user_id},
            )
            self._emit_event(input_event, context)

            # Emit staged execution start events
            retrieval_started_event = RetrievalStartedEvent(query=request)
            self._emit_event(retrieval_started_event, context)

            tool_started_event = ToolStartedEvent(tool_name="employee_service")
            self._emit_event(tool_started_event, context)

            decision_started_event = DecisionStartedEvent(
                model=type(self.agent.llm_provider).__name__
            )
            self._emit_event(decision_started_event, context)

            original_decide = self.agent.decide

            # Execute the agent (this is where LangSmith tracing happens)
            final_state = original_decide(
                request=request,
                user_id=user_id,
                session_id=context.session_id,
            )

            # Extract information from final state for event emission
            self._emit_retrieval_event(final_state, context)
            self._emit_tool_event(final_state, context)
            self._emit_decision_event(final_state, context)
            self._emit_output_event(final_state, context)

            execution_duration = time.time() - execution_start

            logger.info(
                "instrumented_execution_completed",
                session_id=context.session_id,
                duration_sec=execution_duration,
            )

            return final_state

        except Exception as e:
            execution_duration = time.time() - execution_start

            # Emit EXECUTION_FAILED event
            failure_event = ExecutionFailedEvent(
                failure_category=self._categorize_failure(e),
                error_message=self._safe_error_message(e),
            )
            failure_event.duration_ms = execution_duration * 1000
            self._emit_event(failure_event, context)

            logger.error(
                "instrumented_execution_failed",
                session_id=context.session_id,
                error=str(e),
                duration_sec=execution_duration,
                exc_info=True,
            )

            # Re-raise the original exception unchanged
            raise

    def _emit_retrieval_event(
        self, final_state: AgentState, context: ExecutionContext
    ) -> None:
        """Emit retrieval events based on final state.

        Args:
            final_state: Final agent state.
            context: Execution context.
        """
        retrieved_docs = final_state.get("retrieved_documents")
        if retrieved_docs:
            documents = getattr(retrieved_docs, "documents", [])
            completion_event = RetrievalCompletedEvent(
                retrieved_count=len(documents),
                document_ids=[
                    getattr(doc, "chunk_id", "")
                    if not isinstance(doc, dict)
                    else doc.get("chunk_id", "")
                    for doc in documents
                ],
                source_names=[
                    getattr(doc, "source", "")
                    if not isinstance(doc, dict)
                    else doc.get("source", "")
                    for doc in documents
                ],
            )
            self._emit_event(completion_event, context)

    def _emit_tool_event(
        self, final_state: AgentState, context: ExecutionContext
    ) -> None:
        """Emit tool events based on final state.

        Args:
            final_state: Final agent state.
            context: Execution context.
        """
        employee_data = final_state.get("employee_data")
        if employee_data:
            employee_response = (
                employee_data.dict()
                if hasattr(employee_data, "dict")
                else dict(employee_data)
                if isinstance(employee_data, dict)
                else {}
            )
            tool_event = ToolCompletedEvent(
                tool_name="employee_service",
                status="success",
                response_keys=list(employee_response.keys()),
            )
            self._emit_event(tool_event, context)

    def _emit_decision_event(
        self, final_state: AgentState, context: ExecutionContext
    ) -> None:
        """Emit decision events based on final state.

        Args:
            final_state: Final agent state.
            context: Execution context.
        """
        decision = final_state.get("decision")
        if decision:
            if hasattr(decision, "decision"):
                decision_text = decision.decision.value
                decision_reason = getattr(decision, "reason", "")
                policy_references = getattr(decision, "policy_references", [])
            elif isinstance(decision, dict):
                decision_text = decision.get("decision", "UNKNOWN")
                decision_reason = decision.get("reason", "")
                policy_references = decision.get("policy_references", [])
            else:
                decision_text = "UNKNOWN"
                decision_reason = ""
                policy_references = []

            decision_event = DecisionCompletedEvent(
                decision=decision_text,
                decision_reason=decision_reason,
                policy_references=policy_references,
            )
            self._emit_event(decision_event, context)

    def _emit_output_event(
        self, final_state: AgentState, context: ExecutionContext
    ) -> None:
        """Emit output events based on final state.

        Args:
            final_state: Final agent state.
            context: Execution context.
        """
        final_response = final_state.get("final_response", "")
        if final_response:
            output_event = OutputGeneratedEvent(
                response_length=len(final_response),
            )
            self._emit_event(output_event, context)

    def _categorize_failure(self, error: Exception) -> str:
        """Categorize an exception into a failure category.

        Args:
            error: The exception that occurred.

        Returns:
            Failure category string.
        """
        error_type = type(error).__name__.lower()
        error_message = str(error).lower()

        if "retriev" in error_type or "retriev" in error_message or "vector" in error_type or "vector" in error_message:
            return "RETRIEVAL_ERROR"
        elif "tool" in error_type or "tool" in error_message or "employee" in error_type or "employee" in error_message:
            return "TOOL_ERROR"
        elif "llm" in error_type or "llm" in error_message or "gemini" in error_type or "gemini" in error_message:
            return "LLM_ERROR"
        else:
            return "EXECUTION_ERROR"

    def _safe_error_message(self, error: Exception) -> str:
        """Extract a safe error message without secrets or stack traces.

        Args:
            error: The exception.

        Returns:
            Safe error message string.
        """
        message = str(error)

        # Filter out common secrets
        sensitive_patterns = [
            "api_key",
            "api-key",
            "apikey",
            "token",
            "password",
            "secret",
            "credential",
        ]

        for pattern in sensitive_patterns:
            if pattern in message.lower():
                return f"{type(error).__name__}: Sensitive information filtered"

        # Limit message length to avoid excessive logging
        if len(message) > 500:
            message = message[:500] + "..."

        return message

    def get_agent(self) -> LeaveDecisionAgent:
        """Get the underlying agent.

        Enables access to agent properties if needed.

        Returns:
            The wrapped LeaveDecisionAgent instance.
        """
        return self.agent

    def get_graph(self):
        """Get the compiled LangGraph instance.

        Returns:
            Compiled StateGraph instance from underlying agent.
        """
        return self.agent.get_graph()


def create_instrumented_agent(
    agent: LeaveDecisionAgent,
    publisher: Optional[EventPublisher] = None,
) -> InstrumentedAgent:
    """Factory function for creating an instrumented agent.

    Args:
        agent: LeaveDecisionAgent to wrap.
        publisher: Optional EventPublisher. Uses global default if not provided.

    Returns:
        InstrumentedAgent wrapper.
    """
    return InstrumentedAgent(agent, publisher)
