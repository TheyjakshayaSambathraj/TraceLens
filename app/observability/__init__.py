"""Observability and instrumentation for TraceLens.

This package provides:
- Execution context: Strongly-typed container for session/user/trace IDs
- Event model: Typed events for observable execution stages
- Event publisher: Lightweight event bus abstraction
- Instrumented agent: Non-invasive wrapper around LangGraph agent
- LangSmith config: Integration with LangSmith observability platform

The instrumentation layer is DECOUPLED from audit persistence,
PII redaction, and governance. Those concerns are handled in later phases.
"""

from app.observability.execution_context import ExecutionContext
from app.observability.events import (
    EventType,
    ExecutionEvent,
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
from app.observability.publisher import (
    EventPublisher,
    InProcessEventPublisher,
    get_publisher,
    set_publisher,
    reset_publisher,
)
from app.observability.config import LangSmithConfig, get_langsmith_config
from app.observability.instrumentation import InstrumentedAgent, create_instrumented_agent

__all__ = [
    "ExecutionContext",
    "EventType",
    "ExecutionEvent",
    "InputReceivedEvent",
    "RetrievalStartedEvent",
    "RetrievalCompletedEvent",
    "ToolStartedEvent",
    "ToolCompletedEvent",
    "DecisionStartedEvent",
    "DecisionCompletedEvent",
    "OutputGeneratedEvent",
    "ExecutionFailedEvent",
    "EventPublisher",
    "InProcessEventPublisher",
    "get_publisher",
    "set_publisher",
    "reset_publisher",
    "LangSmithConfig",
    "get_langsmith_config",
    "InstrumentedAgent",
    "create_instrumented_agent",
]
