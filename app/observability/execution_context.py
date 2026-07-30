"""Execution context for tracing agent invocations.

The execution context is a strongly-typed container that encapsulates
identifiers, timestamps, and metadata for a single agent execution.

It enables correlation of events across an entire execution lifecycle
without introducing global mutable state.
"""

from __future__ import annotations

from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Optional
import uuid


@dataclass(frozen=True)
class ExecutionContext:
    """Immutable execution context for a single agent invocation.

    The context is created at the start of execution and passed through
    instrumentation layers to enable proper tracing and correlation.

    Attributes:
        session_id: Application-level session identifier for this execution.
            Enables grouping of related events within TraceLens.
        user_id: The user ID associated with the execution.
        trace_id: LangSmith trace/run identifier when available.
            If unavailable at creation time, it will be populated after
            LangSmith initializes.
        request_id: Optional request identifier for correlation with HTTP layers.
        started_at: UTC timestamp when execution began.
        metadata: Arbitrary key-value metadata associated with execution.
    """

    session_id: str
    user_id: str
    trace_id: Optional[str] = None
    request_id: Optional[str] = None
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, str] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        session_id: Optional[str] = None,
        user_id: str = "ANONYMOUS",
        trace_id: Optional[str] = None,
        request_id: Optional[str] = None,
        metadata: Optional[dict[str, str]] = None,
    ) -> ExecutionContext:
        """Create a new execution context with defaults.

        Args:
            session_id: Application session ID. Generates UUID if not provided.
            user_id: User identifier. Defaults to ANONYMOUS.
            trace_id: LangSmith trace ID if available.
            request_id: Optional request correlation ID.
            metadata: Optional metadata dictionary.

        Returns:
            ExecutionContext instance.
        """
        if session_id is None:
            session_id = f"SESSION-{uuid.uuid4().hex[:12].upper()}"

        return cls(
            session_id=session_id,
            user_id=user_id,
            trace_id=trace_id,
            request_id=request_id,
            metadata=metadata or {},
        )

    def with_trace_id(self, trace_id: str) -> ExecutionContext:
        """Return a new context with the trace_id set.

        This is used after LangSmith is initialized and we obtain a run ID.

        Args:
            trace_id: The LangSmith trace/run ID.

        Returns:
            New ExecutionContext with trace_id populated.
        """
        return ExecutionContext(
            session_id=self.session_id,
            user_id=self.user_id,
            trace_id=trace_id,
            request_id=self.request_id,
            started_at=self.started_at,
            metadata=self.metadata,
        )

    def with_metadata(self, key: str, value: str) -> ExecutionContext:
        """Return a new context with additional metadata.

        Args:
            key: Metadata key.
            value: Metadata value.

        Returns:
            New ExecutionContext with metadata updated.
        """
        new_metadata = {**self.metadata, key: value}
        return ExecutionContext(
            session_id=self.session_id,
            user_id=self.user_id,
            trace_id=self.trace_id,
            request_id=self.request_id,
            started_at=self.started_at,
            metadata=new_metadata,
        )
