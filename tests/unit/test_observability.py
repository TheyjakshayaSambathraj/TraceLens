"""Unit tests for observability instrumentation.

Tests cover:
- Execution context creation and immutability
- Event model validation and serialization
- Event publisher subscription and delivery
- Instrumented agent wrapper behavior
- LangSmith configuration
"""

from __future__ import annotations

import pytest
from datetime import datetime, timezone
from unittest.mock import Mock, MagicMock, patch
from io import StringIO

from app.observability.execution_context import ExecutionContext
from app.observability.events import (
    EventType,
    ExecutionEvent,
    InputReceivedEvent,
    RetrievalCompletedEvent,
    DecisionCompletedEvent,
    OutputGeneratedEvent,
    ExecutionFailedEvent,
)
from app.observability.publisher import (
    InProcessEventPublisher,
    get_publisher,
    set_publisher,
    reset_publisher,
)
from app.observability.config import LangSmithConfig
from app.observability.instrumentation import InstrumentedAgent


class TestExecutionContext:
    """Tests for ExecutionContext immutability and creation."""

    def test_create_with_defaults(self):
        """Test creating context with defaults."""
        ctx = ExecutionContext.create(user_id="USER-001")
        assert ctx.user_id == "USER-001"
        assert ctx.session_id.startswith("SESSION-")
        assert ctx.trace_id is None
        assert ctx.started_at is not None
        assert isinstance(ctx.started_at, datetime)

    def test_create_with_custom_session_id(self):
        """Test creating context with custom session ID."""
        session_id = "SESSION-CUSTOM-12345"
        ctx = ExecutionContext.create(
            session_id=session_id,
            user_id="USER-002",
        )
        assert ctx.session_id == session_id
        assert ctx.user_id == "USER-002"

    def test_context_is_frozen(self):
        """Test that ExecutionContext is immutable."""
        ctx = ExecutionContext.create()
        with pytest.raises(AttributeError):
            ctx.user_id = "MODIFIED"

    def test_with_trace_id(self):
        """Test adding trace_id to context."""
        ctx = ExecutionContext.create(user_id="USER-001")
        ctx_with_trace = ctx.with_trace_id("TRACE-12345")

        # Original unchanged
        assert ctx.trace_id is None
        # New context has trace
        assert ctx_with_trace.trace_id == "TRACE-12345"
        # Other fields preserved
        assert ctx_with_trace.user_id == ctx.user_id
        assert ctx_with_trace.session_id == ctx.session_id

    def test_with_metadata(self):
        """Test adding metadata to context."""
        ctx = ExecutionContext.create(user_id="USER-001")
        ctx_with_meta = ctx.with_metadata("request_source", "API")

        # Original unchanged
        assert ctx.metadata == {}
        # New context has metadata
        assert ctx_with_meta.metadata["request_source"] == "API"
        # Other fields preserved
        assert ctx_with_meta.user_id == ctx.user_id

    def test_context_timestamps_are_utc(self):
        """Test that timestamps are UTC."""
        ctx = ExecutionContext.create()
        assert ctx.started_at.tzinfo == timezone.utc


class TestEventModel:
    """Tests for event model validation and serialization."""

    def test_event_creation(self):
        """Test basic event creation."""
        event = ExecutionEvent()
        assert event.event_type == EventType.INPUT_RECEIVED
        assert event.session_id == ""
        assert event.timestamp is not None

    def test_event_to_dict(self):
        """Test event serialization to dict."""
        event = ExecutionEvent(
            session_id="SESSION-001",
            user_id="USER-001",
            event_type=EventType.INPUT_RECEIVED,
        )
        event_dict = event.to_dict()

        assert event_dict["session_id"] == "SESSION-001"
        assert event_dict["user_id"] == "USER-001"
        assert event_dict["event_type"] == "INPUT_RECEIVED"
        assert "timestamp" in event_dict

    def test_input_received_event(self):
        """Test InputReceivedEvent populates metadata."""
        event = InputReceivedEvent(
            request="Can employee EMP-001 take 15 days?",
            session_id="SESSION-001",
        )
        assert event.event_type == EventType.INPUT_RECEIVED
        assert event.metadata["request"] == "Can employee EMP-001 take 15 days?"

    def test_retrieval_completed_event(self):
        """Test RetrievalCompletedEvent."""
        event = RetrievalCompletedEvent(
            retrieved_count=3,
            document_ids=["DOC-1", "DOC-2", "DOC-3"],
            source_names=["leave_policy.md"],
        )
        assert event.event_type == EventType.RETRIEVAL_COMPLETED
        assert event.metadata["retrieved_count"] == 3
        assert event.metadata["document_ids"] == ["DOC-1", "DOC-2", "DOC-3"]

    def test_decision_completed_event(self):
        """Test DecisionCompletedEvent."""
        event = DecisionCompletedEvent(
            decision="APPROVED",
            decision_reason="Employee has sufficient leave balance",
            policy_references=["POLICY-1", "POLICY-2"],
        )
        assert event.event_type == EventType.DECISION_COMPLETED
        assert event.metadata["decision"] == "APPROVED"
        assert "sufficient" in event.metadata["decision_reason"]

    def test_failure_event(self):
        """Test ExecutionFailedEvent."""
        event = ExecutionFailedEvent(
            failure_category="RETRIEVAL_ERROR",
            error_message="Vector store not found",
        )
        assert event.event_type == EventType.EXECUTION_FAILED
        assert event.metadata["failure_category"] == "RETRIEVAL_ERROR"

    def test_event_sequence_number(self):
        """Test event sequence numbers."""
        event1 = ExecutionEvent(sequence=1)
        event2 = ExecutionEvent(sequence=2)
        event3 = ExecutionEvent(sequence=3)

        assert event1.sequence < event2.sequence < event3.sequence


class TestEventPublisher:
    """Tests for event publisher and subscription."""

    def test_publisher_creation(self):
        """Test creating an in-process publisher."""
        publisher = InProcessEventPublisher()
        assert publisher is not None

    def test_subscribe_and_publish(self):
        """Test subscribing to events and publishing."""
        publisher = InProcessEventPublisher()
        received_events = []

        def handler(event: ExecutionEvent):
            received_events.append(event)

        publisher.subscribe(EventType.INPUT_RECEIVED.value, handler)

        event = InputReceivedEvent(
            request="Test",
            session_id="SESSION-001",
        )
        publisher.publish(event)

        assert len(received_events) == 1
        assert received_events[0].event_type == EventType.INPUT_RECEIVED

    def test_wildcard_subscription(self):
        """Test subscribing to all events via wildcard."""
        publisher = InProcessEventPublisher()
        received_events = []

        def handler(event: ExecutionEvent):
            received_events.append(event)

        publisher.subscribe("*", handler)

        event1 = InputReceivedEvent(request="Test")
        event2 = RetrievalCompletedEvent()

        publisher.publish(event1)
        publisher.publish(event2)

        assert len(received_events) == 2

    def test_publisher_error_handling(self):
        """Test that publisher handles handler errors gracefully."""
        publisher = InProcessEventPublisher()

        def broken_handler(event: ExecutionEvent):
            raise RuntimeError("Handler error")

        def good_handler(event: ExecutionEvent):
            pass

        publisher.subscribe(EventType.INPUT_RECEIVED.value, broken_handler)
        publisher.subscribe(EventType.INPUT_RECEIVED.value, good_handler)

        event = InputReceivedEvent(request="Test")
        # Should not raise
        publisher.publish(event)

    def test_get_publisher_singleton(self):
        """Test that get_publisher returns same instance."""
        reset_publisher()
        pub1 = get_publisher()
        pub2 = get_publisher()
        assert pub1 is pub2

    def test_set_publisher_override(self):
        """Test overriding the default publisher."""
        reset_publisher()
        mock_publisher = Mock()
        set_publisher(mock_publisher)

        assert get_publisher() is mock_publisher


class TestLangSmithConfig:
    """Tests for LangSmith configuration."""

    def test_config_from_env_disabled(self):
        """Test loading config with tracing disabled."""
        from app.config.settings import get_settings
        with patch.dict("os.environ", {"LANGCHAIN_TRACING_V2": "false"}):
            get_settings.cache_clear()
            config = LangSmithConfig.from_env()
            assert config.enabled is False

    def test_config_from_env_enabled_without_key(self):
        """Test that enabled tracing without API key is disabled."""
        from app.config.settings import get_settings
        with patch.dict(
            "os.environ",
            {"LANGCHAIN_TRACING_V2": "true", "LANGCHAIN_API_KEY": ""},
            clear=False,
        ):
            get_settings.cache_clear()
            config = LangSmithConfig.from_env()
            # Should disable if no key
            assert not config.is_configured()

    def test_config_from_env_with_key(self):
        """Test loading config with API key."""
        from app.config.settings import get_settings
        with patch.dict(
            "os.environ",
            {
                "LANGCHAIN_TRACING_V2": "true",
                "LANGCHAIN_API_KEY": "sk-test-key-123",
                "LANGCHAIN_PROJECT": "test-project",
            },
        ):
            get_settings.cache_clear()
            config = LangSmithConfig.from_env()
            assert config.enabled is True
            assert config.api_key == "sk-test-key-123"
            assert config.project == "test-project"
            assert config.is_configured() is True

    def test_config_configure_langchain(self):
        """Test configuring LangChain environment."""
        config = LangSmithConfig(
            enabled=True,
            api_key="sk-test",
            project="test-project",
        )

        with patch.dict("os.environ", {}, clear=False):
            config.configure_langchain()
            assert __import__("os").environ.get("LANGCHAIN_TRACING_V2") == "true"
            assert __import__("os").environ.get("LANGCHAIN_PROJECT") == "test-project"


class TestInstrumentedAgent:
    """Tests for instrumented agent wrapper."""

    def test_instrumented_agent_creation(self):
        """Test creating an instrumented agent."""
        mock_agent = Mock()
        instrumented = InstrumentedAgent(mock_agent)

        assert instrumented.agent is mock_agent

    def test_instrumented_agent_decide_emits_events(self):
        """Test that decide() emits events."""
        # Create mock agent
        mock_agent = Mock()
        mock_agent.decide.return_value = {
            "session_id": "SESSION-001",
            "user_id": "USER-001",
            "request": "Can EMP-001 take 15 days?",
            "decision": {"decision": "APPROVED", "reason": "Balance available"},
            "final_response": "Leave approved.",
        }

        # Create mock publisher
        mock_publisher = Mock()
        received_events = []

        def capture_event(event):
            received_events.append(event)

        mock_publisher.publish.side_effect = capture_event

        # Create instrumented agent
        instrumented = InstrumentedAgent(mock_agent, mock_publisher)

        # Execute
        result = instrumented.decide(
            request="Can EMP-001 take 15 days?",
            user_id="USER-001",
        )

        # Verify result is unchanged
        assert result["decision"]["decision"] == "APPROVED"

        # Verify events were emitted
        assert len(received_events) > 0
        assert received_events[0].event_type == EventType.INPUT_RECEIVED
        assert any(e.event_type == EventType.RETRIEVAL_STARTED for e in received_events)
        assert any(e.event_type == EventType.TOOL_STARTED for e in received_events)
        assert any(e.event_type == EventType.DECISION_STARTED for e in received_events)

    def test_instrumented_agent_handles_typed_decision_result(self):
        """Test that instrumented agent handles typed DecisionResult state."""
        from app.schemas.decision import DecisionResult, DecisionStatus

        mock_agent = Mock()
        mock_agent.decide.return_value = {
            "session_id": "SESSION-001",
            "user_id": "USER-001",
            "request": "Can EMP-001 take 15 days?",
            "decision": DecisionResult(
                decision=DecisionStatus.APPROVED,
                reason="Balance available",
                policy_references=["Section 2.1"],
                evidence=["Leave balance sufficient"],
            ),
            "final_response": "Leave approved.",
        }

        mock_publisher = Mock()
        received_events = []

        def capture_event(event):
            received_events.append(event)

        mock_publisher.publish.side_effect = capture_event
        instrumented = InstrumentedAgent(mock_agent, mock_publisher)

        result = instrumented.decide(
            request="Can EMP-001 take 15 days?",
            user_id="USER-001",
        )

        assert result["decision"].decision == DecisionStatus.APPROVED
        assert any(
            e.event_type == EventType.DECISION_COMPLETED
            and e.metadata["decision"] == "APPROVED"
            for e in received_events
        )

    def test_instrumented_agent_failure_handling(self):
        """Test failure event emission on error."""
        # Create mock agent that fails
        mock_agent = Mock()
        mock_agent.decide.side_effect = ValueError("Invalid request")

        mock_publisher = Mock()
        received_events = []

        def capture_event(event):
            received_events.append(event)

        mock_publisher.publish.side_effect = capture_event

        instrumented = InstrumentedAgent(mock_agent, mock_publisher)

        # Execution should raise
        with pytest.raises(ValueError):
            instrumented.decide(request="Bad request", user_id="USER-001")

        # Failure event should be emitted
        failure_events = [e for e in received_events if e.event_type == EventType.EXECUTION_FAILED]
        assert len(failure_events) > 0

    def test_instrumented_agent_categorizes_failures(self):
        """Test failure categorization."""
        mock_agent = Mock()
        mock_publisher = Mock()
        instrumented = InstrumentedAgent(mock_agent, mock_publisher)

        # Test different error types
        assert instrumented._categorize_failure(RuntimeError("retrieval failed")) == "RETRIEVAL_ERROR"
        assert instrumented._categorize_failure(RuntimeError("tool error")) == "TOOL_ERROR"
        assert instrumented._categorize_failure(RuntimeError("generic error")) == "EXECUTION_ERROR"

    def test_instrumented_agent_safe_error_message(self):
        """Test that sensitive information is filtered from error messages."""
        mock_agent = Mock()
        instrumented = InstrumentedAgent(mock_agent)

        # Test secret filtering
        error_with_key = RuntimeError("api_key=sk-12345-secret")
        safe_msg = instrumented._safe_error_message(error_with_key)
        assert "sk-12345" not in safe_msg
        assert "Sensitive information" in safe_msg

        # Test normal errors
        normal_error = RuntimeError("Resource not found")
        safe_msg = instrumented._safe_error_message(normal_error)
        assert "Resource not found" in safe_msg

    def test_instrumented_agent_sequence_numbers(self):
        """Test that events have monotonic sequence numbers."""
        mock_agent = Mock()
        mock_agent.decide.return_value = {
            "decision": {"decision": "APPROVED"},
            "final_response": "Approved",
        }

        mock_publisher = Mock()
        received_events = []

        def capture_event(event):
            received_events.append(event)

        mock_publisher.publish.side_effect = capture_event
        instrumented = InstrumentedAgent(mock_agent, mock_publisher)

        instrumented.decide(request="Test", user_id="USER-001")

        # Verify sequence numbers are monotonic
        sequences = [e.sequence for e in received_events]
        assert sequences == sorted(sequences)
        assert sequences[0] == 1  # Start at 1

    def test_instrumented_agent_correlation(self):
        """Test that all events share the same session ID."""
        mock_agent = Mock()
        mock_agent.decide.return_value = {
            "decision": {"decision": "APPROVED"},
            "final_response": "Approved",
        }

        mock_publisher = Mock()
        received_events = []

        def capture_event(event):
            received_events.append(event)

        mock_publisher.publish.side_effect = capture_event
        instrumented = InstrumentedAgent(mock_agent, mock_publisher)

        session_id = "SESSION-CORRELATION-TEST"
        instrumented.decide(request="Test", user_id="USER-001", session_id=session_id)

        # All events should have the same session_id
        for event in received_events:
            assert event.session_id == session_id
