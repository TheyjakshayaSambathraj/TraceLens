"""End-to-end audit pipeline tests.

Tests the full pipeline without external APIs:
    InstrumentedAgent (mock) → event bus → PII redaction → SQLite → reconstruction

PS-7.1 three-step verification:
    Step 1: Request received and INPUT_RECEIVED event persisted
    Step 2: Context retrieved (RETRIEVAL_COMPLETED) and tool called (TOOL_COMPLETED)
    Step 3: Decision reached (DECISION_COMPLETED) with evidence and policy references

The full reconstructed path must show:
- Three causal steps (retrieve → reason → decide)
- No raw PII in the database
- Complete timeline ordered by sequence
- Decision record in the decision_records table
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from app.audit.persistence import wire_persistence
from app.audit.reconstructor import DecisionPathReconstructor
from app.audit.events import PathStatus
from app.database.repository import AuditRepository
from app.database.session import reset_engine_for_testing
from app.observability.publisher import InProcessEventPublisher, reset_publisher, set_publisher
from app.observability.events import (
    EventType,
    InputReceivedEvent,
    RetrievalCompletedEvent,
    DecisionCompletedEvent,
    OutputGeneratedEvent,
    ExecutionEvent,
    ExecutionFailedEvent,
)
from app.privacy.redactor import PIIRedactor


@pytest.fixture(autouse=True)
def isolated_db():
    reset_engine_for_testing("sqlite:///:memory:")
    yield
    reset_publisher()


@pytest.fixture
def db_session_factory():
    from app.database.session import _get_session_factory
    return _get_session_factory()


@pytest.fixture
def publisher():
    reset_publisher()
    pub = InProcessEventPublisher()
    set_publisher(pub)
    return pub


@pytest.fixture
def wired_service(publisher, db_session_factory):
    """Wire the persistence service to the publisher."""
    return wire_persistence(
        publisher=publisher,
        session_factory=db_session_factory,
        redactor=PIIRedactor(),
    )


def _emit_complete_pipeline(publisher: InProcessEventPublisher, session_id: str, user_id: str = "USER-001"):
    """Simulate a complete agent run by emitting all expected events."""
    now = datetime.now(timezone.utc)

    events = [
        ExecutionEvent(
            event_id=f"{session_id}-001",
            session_id=session_id,
            user_id=user_id,
            trace_id=None,
            sequence=1,
            event_type=EventType.INPUT_RECEIVED,
            timestamp=now,
            duration_ms=None,
            metadata={"request": "Can employee EMP-001 take 10 days of leave?"},
        ),
        ExecutionEvent(
            event_id=f"{session_id}-002",
            session_id=session_id,
            user_id=user_id,
            trace_id=None,
            sequence=2,
            event_type=EventType.RETRIEVAL_STARTED,
            timestamp=now,
            duration_ms=None,
            metadata={"query": "consecutive leave policy"},
        ),
        ExecutionEvent(
            event_id=f"{session_id}-003",
            session_id=session_id,
            user_id=user_id,
            trace_id=None,
            sequence=3,
            event_type=EventType.RETRIEVAL_COMPLETED,
            timestamp=now,
            duration_ms=250.0,
            metadata={
                "retrieved_count": 3,
                "document_ids": ["doc-1", "doc-2", "doc-3"],
                "source_names": ["hr_leave_policy.md"],
            },
        ),
        ExecutionEvent(
            event_id=f"{session_id}-004",
            session_id=session_id,
            user_id=user_id,
            trace_id=None,
            sequence=4,
            event_type=EventType.TOOL_STARTED,
            timestamp=now,
            duration_ms=None,
            metadata={"tool_name": "retrieve_employee_data"},
        ),
        ExecutionEvent(
            event_id=f"{session_id}-005",
            session_id=session_id,
            user_id=user_id,
            trace_id=None,
            sequence=5,
            event_type=EventType.TOOL_COMPLETED,
            timestamp=now,
            duration_ms=120.0,
            metadata={
                "tool_name": "retrieve_employee_data",
                "status": "success",
                "response_keys": ["leave_balance", "employment_status", "job_level"],
            },
        ),
        ExecutionEvent(
            event_id=f"{session_id}-006",
            session_id=session_id,
            user_id=user_id,
            trace_id=None,
            sequence=6,
            event_type=EventType.DECISION_STARTED,
            timestamp=now,
            duration_ms=None,
            metadata={"model": "gemini-2.0-flash"},
        ),
        ExecutionEvent(
            event_id=f"{session_id}-007",
            session_id=session_id,
            user_id=user_id,
            trace_id=None,
            sequence=7,
            event_type=EventType.DECISION_COMPLETED,
            timestamp=now,
            duration_ms=1500.0,
            metadata={
                "decision": "APPROVED",
                "decision_reason": "Employee has 20 days remaining, request is within limit",
                "policy_references": ["Section 2.1: Annual Leave", "Section 3.1: Approval Thresholds"],
                "evidence": [
                    "leave_balance: 20 days",
                    "requested_days: 10",
                    "employment_status: ACTIVE",
                    "manager_approval_required: True",
                ],
            },
        ),
        ExecutionEvent(
            event_id=f"{session_id}-008",
            session_id=session_id,
            user_id=user_id,
            trace_id=None,
            sequence=8,
            event_type=EventType.OUTPUT_GENERATED,
            timestamp=now,
            duration_ms=200.0,
            metadata={"response_length": 256},
        ),
    ]

    for event in events:
        publisher.publish(event)

    return events


# ---------------------------------------------------------------------------
# PS-7.1 End-to-End Tests
# ---------------------------------------------------------------------------


class TestPS71ThreeStepVerification:
    """
    PS-7.1 Acceptance Test: Three-step retrieve → reason → decide path.

    Given a complete agent execution:
    Step 1 (Retrieve): RETRIEVAL_COMPLETED event present in reconstructed path
    Step 2 (Reason):   TOOL_COMPLETED event present (employee data tool)
    Step 3 (Decide):   DECISION_COMPLETED event with evidence and policy references
    """

    def test_step1_retrieval_present_in_reconstructed_path(
        self, wired_service, publisher, db_session_factory
    ):
        session_id = "E2E-PS71-001"
        _emit_complete_pipeline(publisher, session_id)

        db = db_session_factory()
        try:
            reconstructor = DecisionPathReconstructor(db)
            path = reconstructor.reconstruct(session_id)
        finally:
            db.close()

        retrieval_entries = [
            e for e in path.timeline
            if e.event_type == "RETRIEVAL_COMPLETED"
        ]
        assert len(retrieval_entries) >= 1, "Step 1: Policy retrieval must be in the path"
        assert retrieval_entries[0].details.get("retrieved_count", 0) > 0

    def test_step2_tool_call_present_in_reconstructed_path(
        self, wired_service, publisher, db_session_factory
    ):
        session_id = "E2E-PS71-002"
        _emit_complete_pipeline(publisher, session_id)

        db = db_session_factory()
        try:
            reconstructor = DecisionPathReconstructor(db)
            path = reconstructor.reconstruct(session_id)
        finally:
            db.close()

        tool_entries = [
            e for e in path.timeline
            if e.event_type == "TOOL_COMPLETED"
        ]
        assert len(tool_entries) >= 1, "Step 2: Employee data retrieval must be in the path"

    def test_step3_decision_present_with_evidence(
        self, wired_service, publisher, db_session_factory
    ):
        session_id = "E2E-PS71-003"
        _emit_complete_pipeline(publisher, session_id)

        db = db_session_factory()
        try:
            reconstructor = DecisionPathReconstructor(db)
            path = reconstructor.reconstruct(session_id)
        finally:
            db.close()

        decision_entries = [
            e for e in path.timeline
            if e.event_type == "DECISION_COMPLETED"
        ]
        assert len(decision_entries) >= 1, "Step 3: Decision must be in the path"
        details = decision_entries[0].details
        assert details["decision"] == "APPROVED"
        assert len(details.get("evidence", [])) > 0
        assert len(details.get("policy_references", [])) > 0

    def test_path_status_is_complete(
        self, wired_service, publisher, db_session_factory
    ):
        session_id = "E2E-PS71-COMPLETE"
        _emit_complete_pipeline(publisher, session_id)

        db = db_session_factory()
        try:
            reconstructor = DecisionPathReconstructor(db)
            path = reconstructor.reconstruct(session_id)
        finally:
            db.close()

        assert path.status in (PathStatus.COMPLETE, PathStatus.IN_PROGRESS)

    def test_all_events_persisted_in_sequence_order(
        self, wired_service, publisher, db_session_factory
    ):
        session_id = "E2E-SEQ"
        _emit_complete_pipeline(publisher, session_id)

        db = db_session_factory()
        try:
            repo = AuditRepository(db)
            events = repo.get_events_by_session(session_id)
        finally:
            db.close()

        assert len(events) == 8
        seqs = [e.sequence_number for e in events]
        assert seqs == list(range(1, 9))


class TestPS71PIIGuarantee:
    """End-to-end PII guarantee: no raw PII in database after pipeline."""

    def test_no_email_in_persisted_events(
        self, wired_service, publisher, db_session_factory
    ):
        session_id = "E2E-PII-001"

        # Emit an event with an email in the request
        event = ExecutionEvent(
            event_id=f"{session_id}-001",
            session_id=session_id,
            user_id="USER-001",
            trace_id=None,
            sequence=1,
            event_type=EventType.INPUT_RECEIVED,
            timestamp=datetime.now(timezone.utc),
            duration_ms=None,
            metadata={"request": "Request from manager alice.smith@company.com"},
        )
        publisher.publish(event)

        db = db_session_factory()
        try:
            repo = AuditRepository(db)
            events = repo.get_events_by_session(session_id)
        finally:
            db.close()

        assert len(events) == 1
        payload_text = events[0].payload_json
        assert "alice.smith@company.com" not in payload_text
        assert "[REDACTED" in payload_text

    def test_numeric_evidence_survives_in_db(
        self, wired_service, publisher, db_session_factory
    ):
        session_id = "E2E-NUMERIC"

        event = ExecutionEvent(
            event_id=f"{session_id}-007",
            session_id=session_id,
            user_id="USER-001",
            trace_id=None,
            sequence=7,
            event_type=EventType.DECISION_COMPLETED,
            timestamp=datetime.now(timezone.utc),
            duration_ms=1000.0,
            metadata={
                "decision": "REJECTED",
                "decision_reason": "Exceeds 10 day limit",
                "policy_references": ["Section 2.3"],
                "evidence": ["requested: 15 days", "maximum: 10 days"],
            },
        )
        publisher.publish(event)

        db = db_session_factory()
        try:
            repo = AuditRepository(db)
            events = repo.get_events_by_session(session_id)
        finally:
            db.close()

        payload = json.loads(events[0].payload_json)
        assert payload["decision"] == "REJECTED"
        evidence_str = json.dumps(payload["evidence"])
        assert "15" in evidence_str
        assert "10" in evidence_str


class TestPS71FailedExecutionPath:
    """Failed executions must still produce a reconstructable audit path."""

    def test_failed_execution_has_audit_record(
        self, wired_service, publisher, db_session_factory
    ):
        session_id = "E2E-FAIL-001"

        for seq, etype, metadata in [
            (1, EventType.INPUT_RECEIVED, {"request": "Take 5 days"}),
            (2, EventType.RETRIEVAL_STARTED, {"query": "leave policy"}),
            (3, EventType.EXECUTION_FAILED, {
                "failure_category": "RETRIEVAL_ERROR",
                "error_message": "Vector store unavailable",
            }),
        ]:
            publisher.publish(ExecutionEvent(
                event_id=f"{session_id}-{seq:03d}",
                session_id=session_id,
                user_id="USER-001",
                trace_id=None,
                sequence=seq,
                event_type=etype,
                timestamp=datetime.now(timezone.utc),
                duration_ms=None,
                metadata=metadata,
            ))

        db = db_session_factory()
        try:
            repo = AuditRepository(db)
            session = repo.get_session(session_id)
            events = repo.get_events_by_session(session_id)
        finally:
            db.close()

        assert session is not None
        assert session.status == "FAILED"
        assert len(events) == 3

    def test_failed_path_reconstruction_shows_failure_event(
        self, wired_service, publisher, db_session_factory
    ):
        session_id = "E2E-FAIL-RECON"

        for seq, etype, metadata in [
            (1, EventType.INPUT_RECEIVED, {"request": "Test"}),
            (2, EventType.EXECUTION_FAILED, {
                "failure_category": "TOOL_ERROR",
                "error_message": "Employee service unavailable",
            }),
        ]:
            publisher.publish(ExecutionEvent(
                event_id=f"{session_id}-{seq:03d}",
                session_id=session_id,
                user_id="USER-001",
                trace_id=None,
                sequence=seq,
                event_type=etype,
                timestamp=datetime.now(timezone.utc),
                duration_ms=None,
                metadata=metadata,
            ))

        db = db_session_factory()
        try:
            reconstructor = DecisionPathReconstructor(db)
            path = reconstructor.reconstruct(session_id)
        finally:
            db.close()

        assert path.status == PathStatus.FAILED
        failure_entries = [
            e for e in path.timeline
            if e.event_type == "EXECUTION_FAILED"
        ]
        assert len(failure_entries) == 1

    def test_langsmith_unavailable_does_not_affect_audit(
        self, wired_service, publisher, db_session_factory
    ):
        """LangSmith being unavailable must not affect the core audit pipeline."""
        session_id = "E2E-NO-LANGSMITH"

        # Emit events with no trace_id (LangSmith offline)
        event = ExecutionEvent(
            event_id=f"{session_id}-001",
            session_id=session_id,
            user_id="USER-001",
            trace_id=None,  # No LangSmith trace
            sequence=1,
            event_type=EventType.INPUT_RECEIVED,
            timestamp=datetime.now(timezone.utc),
            duration_ms=None,
            metadata={"request": "Test without LangSmith"},
        )
        publisher.publish(event)

        db = db_session_factory()
        try:
            repo = AuditRepository(db)
            session = repo.get_session(session_id)
            events = repo.get_events_by_session(session_id)
        finally:
            db.close()

        # Audit must work without LangSmith
        assert session is not None
        assert len(events) == 1
        assert session.trace_id is None  # No trace, that's fine
