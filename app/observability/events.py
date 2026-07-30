"""Execution event model for observability.

Events represent observable stages during agent execution. They are strongly
typed, chronologically ordered, and correlated via session_id.

Events are NOT the final audit records. That transformation happens in Phase 4.
Events are observational: they capture what happened during execution.
"""

from __future__ import annotations

from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Any
from uuid import uuid4


class EventType(str, Enum):
    """Enumeration of observable execution event types.

    Each event type represents a meaningful boundary in agent execution:
    - INPUT_RECEIVED: Request received and validation complete.
    - RETRIEVAL_STARTED: Beginning retrieval from vector store.
    - RETRIEVAL_COMPLETED: Retrieval finished with results.
    - TOOL_STARTED: External tool (e.g., employee service) invoked.
    - TOOL_COMPLETED: Tool returned results.
    - DECISION_STARTED: LLM decision generation started.
    - DECISION_COMPLETED: LLM decision generation completed.
    - OUTPUT_GENERATED: Final response formatted.
    - EXECUTION_FAILED: Execution terminated with error.
    """

    INPUT_RECEIVED = "INPUT_RECEIVED"
    RETRIEVAL_STARTED = "RETRIEVAL_STARTED"
    RETRIEVAL_COMPLETED = "RETRIEVAL_COMPLETED"
    TOOL_STARTED = "TOOL_STARTED"
    TOOL_COMPLETED = "TOOL_COMPLETED"
    DECISION_STARTED = "DECISION_STARTED"
    DECISION_COMPLETED = "DECISION_COMPLETED"
    OUTPUT_GENERATED = "OUTPUT_GENERATED"
    EXECUTION_FAILED = "EXECUTION_FAILED"


@dataclass
class ExecutionEvent:
    """Strongly typed execution event.

    Every event emitted during agent execution contains these core fields
    for correlation and timeline reconstruction.

    Attributes:
        event_id: Unique identifier for this event.
        event_type: Enumeration of what occurred.
        session_id: Application-level session identifier for correlation.
        user_id: User associated with the execution.
        trace_id: LangSmith trace/run ID if available.
        timestamp: UTC timestamp when event was recorded.
        sequence: Monotonically increasing sequence number within session.
        duration_ms: Duration of the operation (if applicable).
        metadata: Event-specific metadata and context.
    """

    event_id: str = field(default_factory=lambda: str(uuid4()))
    event_type: EventType = field(default=EventType.INPUT_RECEIVED)
    session_id: str = ""
    user_id: str = ""
    trace_id: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    sequence: int = 0
    duration_ms: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert event to dictionary representation.

        Returns:
            Dictionary suitable for JSON serialization.
        """
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "trace_id": self.trace_id,
            "timestamp": self.timestamp.isoformat(),
            "sequence": self.sequence,
            "duration_ms": self.duration_ms,
            "metadata": self.metadata,
        }


@dataclass
class InputReceivedEvent(ExecutionEvent):
    """Input received and request validated.

    Metadata:
        request: The user's request text.
    """

    event_type: EventType = field(default=EventType.INPUT_RECEIVED, init=False)
    request: str = ""

    def __post_init__(self):
        """Ensure request is in metadata."""
        if self.request and "request" not in self.metadata:
            self.metadata["request"] = self.request


@dataclass
class RetrievalStartedEvent(ExecutionEvent):
    """Retrieval operation starting.

    Metadata:
        query: The query sent to the vector store.
    """

    event_type: EventType = field(default=EventType.RETRIEVAL_STARTED, init=False)
    query: str = ""

    def __post_init__(self):
        """Ensure query is in metadata."""
        if self.query and "query" not in self.metadata:
            self.metadata["query"] = self.query


@dataclass
class RetrievalCompletedEvent(ExecutionEvent):
    """Retrieval operation completed successfully.

    Metadata:
        retrieved_count: Number of documents returned.
        document_ids: List of retrieved document identifiers.
        source_names: List of source file/chunk names.
        retrieval_duration_ms: Latency of retrieval operation.
    """

    event_type: EventType = field(default=EventType.RETRIEVAL_COMPLETED, init=False)
    retrieved_count: int = 0
    document_ids: list[str] = field(default_factory=list)
    source_names: list[str] = field(default_factory=list)

    def __post_init__(self):
        """Ensure retrieval metadata is populated."""
        if not self.metadata:
            self.metadata = {}
        self.metadata.update(
            {
                "retrieved_count": self.retrieved_count,
                "document_ids": self.document_ids,
                "source_names": self.source_names,
            }
        )


@dataclass
class ToolStartedEvent(ExecutionEvent):
    """External tool invocation starting.

    Metadata:
        tool_name: Name of the tool being invoked.
    """

    event_type: EventType = field(default=EventType.TOOL_STARTED, init=False)
    tool_name: str = ""

    def __post_init__(self):
        """Ensure tool_name is in metadata."""
        if self.tool_name and "tool_name" not in self.metadata:
            self.metadata["tool_name"] = self.tool_name


@dataclass
class ToolCompletedEvent(ExecutionEvent):
    """External tool invocation completed.

    Metadata:
        tool_name: Name of the tool.
        status: "success" or "error".
        response_keys: List of keys in response object (sanitized).
    """

    event_type: EventType = field(default=EventType.TOOL_COMPLETED, init=False)
    tool_name: str = ""
    status: str = "success"
    response_keys: list[str] = field(default_factory=list)

    def __post_init__(self):
        """Ensure tool metadata is populated."""
        if not self.metadata:
            self.metadata = {}
        self.metadata.update(
            {
                "tool_name": self.tool_name,
                "status": self.status,
                "response_keys": self.response_keys,
            }
        )


@dataclass
class DecisionStartedEvent(ExecutionEvent):
    """Decision generation (LLM invocation) starting.

    Metadata:
        model: LLM model identifier.
    """

    event_type: EventType = field(default=EventType.DECISION_STARTED, init=False)
    model: str = ""

    def __post_init__(self):
        """Ensure model is in metadata."""
        if self.model and "model" not in self.metadata:
            self.metadata["model"] = self.model


@dataclass
class DecisionCompletedEvent(ExecutionEvent):
    """Decision generation completed.

    Metadata:
        decision: The structured decision (APPROVED/REJECTED/NEEDS_REVIEW).
        decision_reason: Brief reason for decision.
        policy_references: List of policy chunks referenced.
    """

    event_type: EventType = field(default=EventType.DECISION_COMPLETED, init=False)
    decision: str = ""
    decision_reason: str = ""
    policy_references: list[str] = field(default_factory=list)

    def __post_init__(self):
        """Ensure decision metadata is populated."""
        if not self.metadata:
            self.metadata = {}
        self.metadata.update(
            {
                "decision": self.decision,
                "decision_reason": self.decision_reason,
                "policy_references": self.policy_references,
            }
        )


@dataclass
class OutputGeneratedEvent(ExecutionEvent):
    """Final response formatted and ready for user.

    Metadata:
        response_length: Character count of final response.
    """

    event_type: EventType = field(default=EventType.OUTPUT_GENERATED, init=False)
    response_length: int = 0

    def __post_init__(self):
        """Ensure response metadata is populated."""
        if not self.metadata:
            self.metadata = {}
        self.metadata["response_length"] = self.response_length


@dataclass
class ExecutionFailedEvent(ExecutionEvent):
    """Execution terminated with error.

    Metadata:
        failure_category: Classification of failure (e.g., RETRIEVAL_ERROR, LLM_ERROR).
        error_message: Safe error message (no stack traces or secrets).
    """

    event_type: EventType = field(default=EventType.EXECUTION_FAILED, init=False)
    failure_category: str = "UNKNOWN_ERROR"
    error_message: str = ""

    def __post_init__(self):
        """Ensure failure metadata is populated."""
        if not self.metadata:
            self.metadata = {}
        self.metadata.update(
            {
                "failure_category": self.failure_category,
                "error_message": self.error_message,
            }
        )
