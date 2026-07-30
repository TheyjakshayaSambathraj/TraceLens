"""Unit tests for AuditRepository.

Tests cover:
- Session creation, retrieval, and status updates
- Event persistence and sequence ordering
- Decision record upsert (create and update)
- Paginated queries by user
- Search with multiple filters
- Missing session returns None (not exception)

All tests use in-memory SQLite via reset_engine_for_testing().
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database.models import Base
from app.database.repository import AuditRepository
from app.database.session import reset_engine_for_testing


@pytest.fixture(autouse=True)
def isolated_db():
    """Reset to a fresh in-memory SQLite DB for every test.

    Uses reset_engine_for_testing so that get_db() also uses the same DB.
    """
    reset_engine_for_testing("sqlite:///:memory:")
    yield


@pytest.fixture
def db_session():
    """Provide a SQLAlchemy session connected to the test in-memory DB."""
    from app.database.session import _get_session_factory
    factory = _get_session_factory()
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


@pytest.fixture
def repo(db_session):
    """Provide an AuditRepository backed by the test session."""
    return AuditRepository(db_session)


# ---------------------------------------------------------------------------
# Session tests
# ---------------------------------------------------------------------------


class TestAuditSessionCRUD:
    """Session lifecycle: create, read, update."""

    def test_create_session(self, repo: AuditRepository, db_session):
        s = repo.create_session(
            session_id="SESSION-001",
            user_id="USER-001",
        )
        db_session.commit()

        assert s.session_id == "SESSION-001"
        assert s.user_id == "USER-001"
        assert s.status == "IN_PROGRESS"
        assert s.started_at is not None

    def test_get_session_by_id(self, repo: AuditRepository, db_session):
        repo.create_session(session_id="SESSION-002", user_id="USER-001")
        db_session.commit()

        result = repo.get_session("SESSION-002")
        assert result is not None
        assert result.session_id == "SESSION-002"

    def test_get_session_not_found(self, repo: AuditRepository):
        result = repo.get_session("DOES-NOT-EXIST")
        assert result is None

    def test_upsert_session_creates_new(self, repo: AuditRepository, db_session):
        s = repo.upsert_session(session_id="SESSION-003", user_id="USER-002")
        db_session.commit()

        assert s.session_id == "SESSION-003"

    def test_upsert_session_returns_existing(self, repo: AuditRepository, db_session):
        s1 = repo.create_session(session_id="SESSION-004", user_id="USER-003")
        db_session.commit()

        s2 = repo.upsert_session(session_id="SESSION-004", user_id="USER-003")
        assert s1.id == s2.id

    def test_update_session_status_to_completed(self, repo: AuditRepository, db_session):
        repo.create_session(session_id="SESSION-005", user_id="USER-001")
        db_session.commit()

        updated = repo.update_session_status(
            session_id="SESSION-005",
            status="COMPLETED",
            completed_at=datetime.now(timezone.utc),
        )
        db_session.commit()

        assert updated.status == "COMPLETED"
        assert updated.completed_at is not None

    def test_update_session_status_to_failed(self, repo: AuditRepository, db_session):
        repo.create_session(session_id="SESSION-006", user_id="USER-001")
        db_session.commit()

        repo.update_session_status(session_id="SESSION-006", status="FAILED")
        db_session.commit()

        s = repo.get_session("SESSION-006")
        assert s.status == "FAILED"

    def test_update_session_not_found_returns_none(self, repo: AuditRepository):
        result = repo.update_session_status("NONEXISTENT", "COMPLETED")
        assert result is None

    def test_trace_id_stored(self, repo: AuditRepository, db_session):
        repo.create_session(
            session_id="SESSION-007",
            user_id="USER-001",
            trace_id="LANGSMITH-TRACE-123",
        )
        db_session.commit()

        s = repo.get_session("SESSION-007")
        assert s.trace_id == "LANGSMITH-TRACE-123"


# ---------------------------------------------------------------------------
# Event persistence tests
# ---------------------------------------------------------------------------


class TestEventPersistence:
    """Audit event storage and retrieval."""

    def _create_session(self, repo, db_session, session_id="SESSION-EVT-001", user_id="USER-001"):
        repo.create_session(session_id=session_id, user_id=user_id)
        db_session.commit()

    def test_persist_single_event(self, repo: AuditRepository, db_session):
        self._create_session(repo, db_session)

        evt = repo.persist_event(
            event_id="EVT-001",
            session_id="SESSION-EVT-001",
            user_id="USER-001",
            trace_id=None,
            sequence_number=1,
            event_type="INPUT_RECEIVED",
            timestamp=datetime.now(timezone.utc),
            duration_ms=None,
            payload={"request": "Can EMP-001 take 5 days?"},
        )
        db_session.commit()

        assert evt.event_id == "EVT-001"
        assert evt.event_type == "INPUT_RECEIVED"
        assert evt.pii_redacted is True  # invariant always set

    def test_persist_multiple_events_ordered_by_sequence(
        self, repo: AuditRepository, db_session
    ):
        self._create_session(repo, db_session)
        now = datetime.now(timezone.utc)

        for seq, etype in [
            (1, "INPUT_RECEIVED"),
            (2, "RETRIEVAL_COMPLETED"),
            (3, "TOOL_COMPLETED"),
            (4, "DECISION_COMPLETED"),
            (5, "OUTPUT_GENERATED"),
        ]:
            repo.persist_event(
                event_id=f"EVT-{seq:03d}",
                session_id="SESSION-EVT-001",
                user_id="USER-001",
                trace_id=None,
                sequence_number=seq,
                event_type=etype,
                timestamp=now,
                duration_ms=None,
                payload={"sequence": seq},
            )
        db_session.commit()

        events = repo.get_events_by_session("SESSION-EVT-001")
        assert len(events) == 5
        seqs = [e.sequence_number for e in events]
        assert seqs == [1, 2, 3, 4, 5]

    def test_payload_json_roundtrip(self, repo: AuditRepository, db_session):
        """Payload must survive JSON serialization round-trip."""
        self._create_session(repo, db_session)

        original_payload = {
            "decision": "APPROVED",
            "policy_references": ["Section 2.1", "Section 3.4"],
            "evidence": ["leave_balance: 20", "employment_status: ACTIVE"],
        }

        repo.persist_event(
            event_id="EVT-PAYLOAD",
            session_id="SESSION-EVT-001",
            user_id="USER-001",
            trace_id=None,
            sequence_number=1,
            event_type="DECISION_COMPLETED",
            timestamp=datetime.now(timezone.utc),
            duration_ms=125.0,
            payload=original_payload,
        )
        db_session.commit()

        events = repo.get_events_by_session("SESSION-EVT-001")
        assert len(events) == 1
        recovered = json.loads(events[0].payload_json)
        assert recovered["decision"] == "APPROVED"
        assert len(recovered["policy_references"]) == 2

    def test_events_for_missing_session_empty(self, repo: AuditRepository):
        events = repo.get_events_by_session("NONEXISTENT-SESSION")
        assert events == []


# ---------------------------------------------------------------------------
# Decision record tests
# ---------------------------------------------------------------------------


class TestDecisionRecord:
    """Decision record upsert and retrieval."""

    def test_create_decision_record(self, repo: AuditRepository, db_session):
        repo.create_session(session_id="SESSION-DEC-001", user_id="USER-001")

        rec = repo.upsert_decision_record(
            session_id="SESSION-DEC-001",
            user_id="USER-001",
            decision="APPROVED",
            decision_reason="Sufficient leave balance",
            policy_references=["Section 2.1"],
            evidence=["balance: 20 days"],
        )
        db_session.commit()

        assert rec.decision == "APPROVED"
        assert rec.session_id == "SESSION-DEC-001"

    def test_update_decision_record(self, repo: AuditRepository, db_session):
        repo.create_session(session_id="SESSION-DEC-002", user_id="USER-001")

        repo.upsert_decision_record(
            session_id="SESSION-DEC-002",
            user_id="USER-001",
            decision="NEEDS_REVIEW",
            decision_reason="Insufficient data",
            policy_references=[],
            evidence=[],
        )
        db_session.commit()

        # Update same session
        repo.upsert_decision_record(
            session_id="SESSION-DEC-002",
            user_id="USER-001",
            decision="APPROVED",
            decision_reason="Data retrieved",
            policy_references=["Section 3.1"],
            evidence=["balance ok"],
        )
        db_session.commit()

        rec = repo.get_decision_record("SESSION-DEC-002")
        assert rec.decision == "APPROVED"

    def test_get_decision_record_not_found(self, repo: AuditRepository):
        rec = repo.get_decision_record("NO-SESSION")
        assert rec is None

    def test_policy_references_stored_as_json(self, repo: AuditRepository, db_session):
        repo.create_session(session_id="SESSION-DEC-003", user_id="USER-001")
        repo.upsert_decision_record(
            session_id="SESSION-DEC-003",
            user_id="USER-001",
            decision="REJECTED",
            decision_reason="Exceeds limit",
            policy_references=["Sec 2.3", "Sec 5.1"],
            evidence=["Days: 15 > max 10"],
        )
        db_session.commit()

        rec = repo.get_decision_record("SESSION-DEC-003")
        refs = json.loads(rec.policy_references)
        assert len(refs) == 2
        assert "Sec 2.3" in refs


# ---------------------------------------------------------------------------
# Query tests
# ---------------------------------------------------------------------------


class TestSessionQueries:
    """Pagination and search queries."""

    def _make_sessions(self, repo, db_session, n=5, user_id="USER-QUERY"):
        from datetime import timedelta
        now = datetime.now(timezone.utc)
        for i in range(n):
            repo.create_session(
                session_id=f"SESSION-Q-{i:03d}",
                user_id=user_id,
                started_at=now - timedelta(hours=n - i),
            )
        db_session.commit()

    def test_get_sessions_by_user_returns_all(self, repo: AuditRepository, db_session):
        self._make_sessions(repo, db_session, n=3)
        sessions = repo.get_sessions_by_user("USER-QUERY", limit=10)
        assert len(sessions) == 3

    def test_get_sessions_by_user_pagination(self, repo: AuditRepository, db_session):
        self._make_sessions(repo, db_session, n=5)
        page1 = repo.get_sessions_by_user("USER-QUERY", limit=2, offset=0)
        page2 = repo.get_sessions_by_user("USER-QUERY", limit=2, offset=2)
        assert len(page1) == 2
        assert len(page2) == 2
        # No overlap
        ids_p1 = {s.session_id for s in page1}
        ids_p2 = {s.session_id for s in page2}
        assert ids_p1.isdisjoint(ids_p2)

    def test_search_sessions_by_user_id(self, repo: AuditRepository, db_session):
        self._make_sessions(repo, db_session, n=3, user_id="USER-A")
        repo.create_session(session_id="SESSION-B-001", user_id="USER-B")
        db_session.commit()

        results = repo.search_sessions(user_id="USER-A")
        assert len(results) == 3
        for s in results:
            assert s.user_id == "USER-A"

    def test_search_sessions_by_status(self, repo: AuditRepository, db_session):
        repo.create_session(session_id="SESSION-DONE", user_id="USER-001")
        repo.create_session(session_id="SESSION-FAIL", user_id="USER-001")
        db_session.commit()

        repo.update_session_status("SESSION-DONE", "COMPLETED")
        repo.update_session_status("SESSION-FAIL", "FAILED")
        db_session.commit()

        completed = repo.search_sessions(status="COMPLETED")
        failed = repo.search_sessions(status="FAILED")

        assert any(s.session_id == "SESSION-DONE" for s in completed)
        assert any(s.session_id == "SESSION-FAIL" for s in failed)

    def test_search_sessions_by_decision(self, repo: AuditRepository, db_session):
        repo.create_session(session_id="SESSION-APP", user_id="USER-001")
        repo.create_session(session_id="SESSION-REJ", user_id="USER-001")
        db_session.commit()

        repo.upsert_decision_record(
            "SESSION-APP", "USER-001", "APPROVED", "", [], []
        )
        repo.upsert_decision_record(
            "SESSION-REJ", "USER-001", "REJECTED", "", [], []
        )
        db_session.commit()

        approved = repo.search_sessions(decision="APPROVED")
        rejected = repo.search_sessions(decision="REJECTED")

        assert any(s.session_id == "SESSION-APP" for s in approved)
        assert any(s.session_id == "SESSION-REJ" for s in rejected)

    def test_search_empty_filters_returns_all(self, repo: AuditRepository, db_session):
        for i in range(3):
            repo.create_session(session_id=f"SESSION-ALLQ-{i}", user_id=f"USER-{i}")
        db_session.commit()

        results = repo.search_sessions()
        assert len(results) == 3
