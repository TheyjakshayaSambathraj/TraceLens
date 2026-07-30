"""Unit tests for the PII redaction → persistence pipeline.

The core PS-7.1 invariant:
    Raw PII must never reach the audit database.
    The AuditPersistenceService is the enforcement point.

Tests verify:
- PII present in raw events is absent from the persisted record
- Non-PII decision evidence (numbers, policy refs) survives redaction
- DECISION_COMPLETED creates a decision_record row
- OUTPUT_GENERATED marks session as COMPLETED
- EXECUTION_FAILED marks session as FAILED
- Persistence errors do not propagate to the caller
- pii_redacted flag is always True on persisted events
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from app.audit.persistence import AuditPersistenceService, wire_persistence
from app.database.repository import AuditRepository
from app.database.session import reset_engine_for_testing
from app.observability.events import (
    EventType,
    ExecutionEvent,
    InputReceivedEvent,
    DecisionCompletedEvent,
    OutputGeneratedEvent,
    ExecutionFailedEvent,
)
from app.privacy.redactor import PIIRedactor


@pytest.fixture(autouse=True)
def isolated_db():
    reset_engine_for_testing("sqlite:///:memory:")
    yield


@pytest.fixture
def db_session_factory():
    """Return a factory that produces fresh sessions from the test DB."""
    from app.database.session import _get_session_factory
    return _get_session_factory()


@pytest.fixture
def persistence_service(db_session_factory):
    """Return a wired AuditPersistenceService using the real PIIRedactor."""
    return AuditPersistenceService(
        session_factory=db_session_factory,
        redactor=PIIRedactor(),
    )


@pytest.fixture
def repo_session():
    """Yield a session and committed AuditRepository for verification queries."""
    from app.database.session import _get_session_factory
    factory = _get_session_factory()
    session = factory()
    try:
        yield session
    finally:
        session.close()


def _make_event(
    event_type: EventType,
    session_id: str = "SESSION-PIPE-001",
    user_id: str = "USER-001",
    metadata: dict = None,
    sequence: int = 1,
) -> ExecutionEvent:
    """Construct a minimal ExecutionEvent for testing."""
    return ExecutionEvent(
        event_id=f"EVT-{event_type.value}-{sequence}",
        session_id=session_id,
        user_id=user_id,
        trace_id=None,
        sequence=sequence,
        event_type=event_type,
        timestamp=datetime.now(timezone.utc),
        duration_ms=None,
        metadata=metadata or {},
    )


# ---------------------------------------------------------------------------
# PS-7.1 Core Invariant: PII Redaction before Persistence
# ---------------------------------------------------------------------------


class TestPIIRedactionInvariant:
    """The critical PS-7.1 guarantee: raw PII never reaches SQLite."""

    def test_email_in_raw_event_absent_from_db(
        self, persistence_service: AuditPersistenceService, repo_session
    ):
        """Email in event metadata must be redacted before persistence."""
        event = _make_event(
            EventType.INPUT_RECEIVED,
            metadata={"request": "Send to employee at john.doe@company.com"},
        )
        persistence_service.handle_event(event)

        repo = AuditRepository(repo_session)
        events = repo.get_events_by_session("SESSION-PIPE-001")
        assert len(events) == 1

        payload = json.loads(events[0].payload_json)
        assert "john.doe@company.com" not in json.dumps(payload)
        assert "[REDACTED_EMAIL]" in json.dumps(payload)

    def test_pii_redacted_flag_always_true(
        self, persistence_service: AuditPersistenceService, repo_session
    ):
        """pii_redacted flag must always be True on persisted events."""
        event = _make_event(
            EventType.INPUT_RECEIVED,
            metadata={"request": "Normal leave request, no PII here"},
        )
        persistence_service.handle_event(event)

        repo = AuditRepository(repo_session)
        events = repo.get_events_by_session("SESSION-PIPE-001")
        assert len(events) == 1
        assert events[0].pii_redacted is True

    def test_phone_in_event_absent_from_db(
        self, persistence_service: AuditPersistenceService, repo_session
    ):
        """Phone number in event metadata must be redacted."""
        event = _make_event(
            EventType.INPUT_RECEIVED,
            metadata={"request": "Call employee at +44-7700-900123 for approval"},
        )
        persistence_service.handle_event(event)

        repo = AuditRepository(repo_session)
        events = repo.get_events_by_session("SESSION-PIPE-001")
        payload = json.loads(events[0].payload_json)
        assert "7700900123" not in json.dumps(payload)
        assert "[REDACTED_PHONE]" in json.dumps(payload)

    def test_numeric_evidence_preserved_after_redaction(
        self, persistence_service: AuditPersistenceService, repo_session
    ):
        """
        Numeric decision evidence must survive redaction unchanged.

        This is critical: if leave_balance or requested_days are
        redacted, the audit record becomes useless for governance review.
        """
        event = _make_event(
            EventType.DECISION_COMPLETED,
            metadata={
                "decision": "APPROVED",
                "decision_reason": "Sufficient leave balance",
                "policy_references": ["Section 2.1"],
                "evidence": [
                    "leave_balance: 20 days",
                    "requested_days: 10",
                    "employment_status: ACTIVE",
                ],
            },
            sequence=2,
        )
        persistence_service.handle_event(event)

        repo = AuditRepository(repo_session)
        events = repo.get_events_by_session("SESSION-PIPE-001")
        payload = json.loads(events[0].payload_json)

        # Decision outcome preserved
        assert payload["decision"] == "APPROVED"
        # Policy references preserved
        assert "Section 2.1" in payload["policy_references"]
        # Numeric evidence preserved (numbers are never PII)
        evidence_text = json.dumps(payload["evidence"])
        assert "20" in evidence_text
        assert "10" in evidence_text

    def test_employee_id_preserved_after_redaction(
        self, persistence_service: AuditPersistenceService, repo_session
    ):
        """EMP-001 is an organizational identifier, not personal PII."""
        event = _make_event(
            EventType.INPUT_RECEIVED,
            metadata={"request": "Can employee EMP-001 take 15 days?"},
        )
        persistence_service.handle_event(event)

        repo = AuditRepository(repo_session)
        events = repo.get_events_by_session("SESSION-PIPE-001")
        payload = json.loads(events[0].payload_json)
        assert "EMP-001" in payload["request"]


# ---------------------------------------------------------------------------
# Session lifecycle
# ---------------------------------------------------------------------------


class TestSessionLifecycle:
    """Verify session creation and status transitions via events."""

    def test_first_event_creates_session(
        self, persistence_service: AuditPersistenceService, repo_session
    ):
        event = _make_event(EventType.INPUT_RECEIVED)
        persistence_service.handle_event(event)

        repo = AuditRepository(repo_session)
        session = repo.get_session("SESSION-PIPE-001")
        assert session is not None
        assert session.user_id == "USER-001"
        assert session.status == "IN_PROGRESS"

    def test_output_generated_marks_completed(
        self, persistence_service: AuditPersistenceService, repo_session
    ):
        for seq, etype in [(1, EventType.INPUT_RECEIVED), (2, EventType.OUTPUT_GENERATED)]:
            persistence_service.handle_event(
                _make_event(etype, sequence=seq)
            )

        repo = AuditRepository(repo_session)
        session = repo.get_session("SESSION-PIPE-001")
        assert session.status == "COMPLETED"
        assert session.completed_at is not None

    def test_execution_failed_marks_failed(
        self, persistence_service: AuditPersistenceService, repo_session
    ):
        for seq, etype in [
            (1, EventType.INPUT_RECEIVED),
            (2, EventType.EXECUTION_FAILED),
        ]:
            persistence_service.handle_event(
                _make_event(
                    etype,
                    sequence=seq,
                    metadata={"failure_category": "RETRIEVAL_ERROR", "error_message": "Store unavailable"},
                )
            )

        repo = AuditRepository(repo_session)
        session = repo.get_session("SESSION-PIPE-001")
        assert session.status == "FAILED"


# ---------------------------------------------------------------------------
# Decision record creation
# ---------------------------------------------------------------------------


class TestDecisionRecordCreation:
    """DECISION_COMPLETED event creates the decision_record row."""

    def test_decision_completed_creates_decision_record(
        self, persistence_service: AuditPersistenceService, repo_session
    ):
        event = _make_event(
            EventType.DECISION_COMPLETED,
            metadata={
                "decision": "REJECTED",
                "decision_reason": "Exceeds max consecutive days",
                "policy_references": ["Section 2.3"],
                "evidence": ["requested: 15 days", "max: 10 days"],
            },
        )
        persistence_service.handle_event(event)

        repo = AuditRepository(repo_session)
        record = repo.get_decision_record("SESSION-PIPE-001")
        assert record is not None
        assert record.decision == "REJECTED"
        assert record.user_id == "USER-001"

    def test_decision_record_reason_stored(
        self, persistence_service: AuditPersistenceService, repo_session
    ):
        event = _make_event(
            EventType.DECISION_COMPLETED,
            metadata={
                "decision": "APPROVED",
                "decision_reason": "All criteria met",
                "policy_references": [],
                "evidence": [],
            },
        )
        persistence_service.handle_event(event)

        repo = AuditRepository(repo_session)
        record = repo.get_decision_record("SESSION-PIPE-001")
        assert record.decision_reason == "All criteria met"


# ---------------------------------------------------------------------------
# Error resilience
# ---------------------------------------------------------------------------


class TestPersistenceErrorResilience:
    """Persistence failures must not propagate to the caller."""

    def test_broken_session_factory_does_not_raise(self):
        """handle_event must not raise even if the DB is unavailable."""
        def broken_factory():
            raise RuntimeError("Database unreachable")

        service = AuditPersistenceService(
            session_factory=broken_factory,
            redactor=PIIRedactor(),
        )

        event = _make_event(EventType.INPUT_RECEIVED)
        # Must not raise
        service.handle_event(event)

    def test_wire_persistence_subscribes_to_wildcard(self, db_session_factory):
        """wire_persistence must register on the '*' channel."""
        from app.observability.publisher import InProcessEventPublisher

        publisher = InProcessEventPublisher()
        service = wire_persistence(
            publisher=publisher,
            session_factory=db_session_factory,
        )

        assert service is not None
        # Publisher should have at least one subscriber on '*'
        assert len(publisher._subscribers.get("*", [])) >= 1
