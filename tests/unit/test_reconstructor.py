"""Unit tests for DecisionPathReconstructor and TimelineBuilder.

Tests cover:
- Complete path reconstruction from a full set of events
- Incomplete path with missing steps represented explicitly
- Failed execution path detection
- Sequence validation and ordering
- Timeline entry type mapping
- Summary generation per event type
- LangSmith URL construction
- Session not found raises ValueError
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

import pytest

from app.audit.events import PathStatus, TimelineEntryType
from app.audit.reconstructor import DecisionPathReconstructor, _EXPECTED_EVENTS
from app.audit.timeline import TimelineBuilder
from app.database.session import reset_engine_for_testing
from app.database.repository import AuditRepository


@pytest.fixture(autouse=True)
def isolated_db():
    reset_engine_for_testing("sqlite:///:memory:")
    yield


@pytest.fixture
def db_session():
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
    return AuditRepository(db_session)


@pytest.fixture
def reconstructor(db_session):
    return DecisionPathReconstructor(db_session)


def _ts(offset_seconds: float = 0) -> datetime:
    """Return a UTC datetime offset from now by offset_seconds."""
    return datetime.now(timezone.utc) + timedelta(seconds=offset_seconds)


def _seed_complete_session(repo, db_session, session_id="SESSION-COMPLETE", user_id="USER-001"):
    """Seed a complete successful session with all 5 expected event types."""
    repo.create_session(session_id=session_id, user_id=user_id)

    events = [
        (1, "INPUT_RECEIVED", {"request": "Can EMP-001 take 10 days?"}),
        (2, "RETRIEVAL_STARTED", {"query": "leave policy consecutive"}),
        (3, "RETRIEVAL_COMPLETED", {
            "retrieved_count": 3,
            "document_ids": ["DOC-1", "DOC-2", "DOC-3"],
            "source_names": ["hr_policy.md"],
        }),
        (4, "TOOL_STARTED", {"tool_name": "retrieve_employee_data"}),
        (5, "TOOL_COMPLETED", {
            "tool_name": "retrieve_employee_data",
            "status": "success",
            "response_keys": ["leave_balance", "employment_status"],
        }),
        (6, "DECISION_STARTED", {"model": "gemini-2.0-flash"}),
        (7, "DECISION_COMPLETED", {
            "decision": "APPROVED",
            "decision_reason": "Employee has sufficient leave balance",
            "policy_references": ["Section 2.1"],
            "evidence": ["leave_balance: 20 days", "employment_status: ACTIVE"],
        }),
        (8, "OUTPUT_GENERATED", {"response_length": 142}),
    ]

    for seq, event_type, payload in events:
        repo.persist_event(
            event_id=f"EVT-{seq:03d}",
            session_id=session_id,
            user_id=user_id,
            trace_id=None,
            sequence_number=seq,
            event_type=event_type,
            timestamp=_ts(seq),
            duration_ms=float(seq * 50),
            payload=payload,
        )

    repo.update_session_status(session_id, "COMPLETED", _ts(10))
    repo.upsert_decision_record(
        session_id=session_id,
        user_id=user_id,
        decision="APPROVED",
        decision_reason="Sufficient balance",
        policy_references=["Section 2.1"],
        evidence=["leave_balance: 20 days"],
    )
    db_session.commit()


# ---------------------------------------------------------------------------
# DecisionPathReconstructor tests
# ---------------------------------------------------------------------------


class TestReconstructorComplete:
    """Tests for a fully complete execution path."""

    def test_reconstruct_complete_path(
        self, reconstructor: DecisionPathReconstructor, repo, db_session
    ):
        _seed_complete_session(repo, db_session)
        path = reconstructor.reconstruct("SESSION-COMPLETE")

        assert path.session_id == "SESSION-COMPLETE"
        assert path.user_id == "USER-001"
        assert path.status == PathStatus.COMPLETE
        assert path.pii_redacted is True  # invariant

    def test_complete_path_has_no_missing_steps(
        self, reconstructor: DecisionPathReconstructor, repo, db_session
    ):
        _seed_complete_session(repo, db_session)
        path = reconstructor.reconstruct("SESSION-COMPLETE")

        assert path.missing_steps == []

    def test_timeline_has_all_events(
        self, reconstructor: DecisionPathReconstructor, repo, db_session
    ):
        _seed_complete_session(repo, db_session)
        path = reconstructor.reconstruct("SESSION-COMPLETE")

        # 8 events were seeded
        assert len(path.timeline) == 8

    def test_timeline_ordered_by_sequence(
        self, reconstructor: DecisionPathReconstructor, repo, db_session
    ):
        _seed_complete_session(repo, db_session)
        path = reconstructor.reconstruct("SESSION-COMPLETE")

        seqs = [e.sequence for e in path.timeline]
        assert seqs == sorted(seqs)
        assert seqs[0] == 1

    def test_decision_entry_present_with_correct_decision(
        self, reconstructor: DecisionPathReconstructor, repo, db_session
    ):
        _seed_complete_session(repo, db_session)
        path = reconstructor.reconstruct("SESSION-COMPLETE")

        decision_entries = [
            e for e in path.timeline if e.event_type == "DECISION_COMPLETED"
        ]
        assert len(decision_entries) == 1
        assert decision_entries[0].details["decision"] == "APPROVED"

    def test_started_at_and_completed_at_present(
        self, reconstructor: DecisionPathReconstructor, repo, db_session
    ):
        _seed_complete_session(repo, db_session)
        path = reconstructor.reconstruct("SESSION-COMPLETE")

        assert path.started_at is not None
        assert path.completed_at is not None


class TestReconstructorIncomplete:
    """Tests for incomplete path (missing events)."""

    def test_incomplete_path_status(
        self, reconstructor: DecisionPathReconstructor, repo, db_session
    ):
        # Seed only INPUT_RECEIVED (no retrieval, no decision, no output)
        repo.create_session(session_id="SESSION-INCOMPLETE", user_id="USER-001")
        repo.persist_event(
            event_id="EVT-001",
            session_id="SESSION-INCOMPLETE",
            user_id="USER-001",
            trace_id=None,
            sequence_number=1,
            event_type="INPUT_RECEIVED",
            timestamp=_ts(),
            duration_ms=None,
            payload={"request": "Test"},
        )
        repo.update_session_status("SESSION-INCOMPLETE", "COMPLETED")
        db_session.commit()

        path = reconstructor.reconstruct("SESSION-INCOMPLETE")
        assert path.status == PathStatus.INCOMPLETE

    def test_missing_steps_explicitly_listed(
        self, reconstructor: DecisionPathReconstructor, repo, db_session
    ):
        repo.create_session(session_id="SESSION-INCOMPLETE-2", user_id="USER-001")
        repo.persist_event(
            event_id="EVT-001",
            session_id="SESSION-INCOMPLETE-2",
            user_id="USER-001",
            trace_id=None,
            sequence_number=1,
            event_type="INPUT_RECEIVED",
            timestamp=_ts(),
            duration_ms=None,
            payload={"request": "Test"},
        )
        db_session.commit()

        path = reconstructor.reconstruct("SESSION-INCOMPLETE-2")
        # Must explicitly list what's missing
        assert len(path.missing_steps) > 0
        assert "DECISION_COMPLETED" in path.missing_steps
        assert "OUTPUT_GENERATED" in path.missing_steps

    def test_missing_steps_not_fabricated(
        self, reconstructor: DecisionPathReconstructor, repo, db_session
    ):
        """Missing steps must be listed, NOT invented in the timeline."""
        repo.create_session(session_id="SESSION-GAPS", user_id="USER-001")
        repo.persist_event(
            event_id="EVT-001",
            session_id="SESSION-GAPS",
            user_id="USER-001",
            trace_id=None,
            sequence_number=1,
            event_type="INPUT_RECEIVED",
            timestamp=_ts(),
            duration_ms=None,
            payload={"request": "Test"},
        )
        db_session.commit()

        path = reconstructor.reconstruct("SESSION-GAPS")
        # Timeline only contains actually recorded events
        timeline_types = {e.event_type for e in path.timeline}
        for missing in path.missing_steps:
            assert missing not in timeline_types


class TestReconstructorFailed:
    """Tests for failed execution paths."""

    def test_failed_path_status(
        self, reconstructor: DecisionPathReconstructor, repo, db_session
    ):
        repo.create_session(session_id="SESSION-FAIL", user_id="USER-001")
        repo.persist_event(
            event_id="EVT-001",
            session_id="SESSION-FAIL",
            user_id="USER-001",
            trace_id=None,
            sequence_number=1,
            event_type="INPUT_RECEIVED",
            timestamp=_ts(),
            duration_ms=None,
            payload={"request": "Test"},
        )
        repo.persist_event(
            event_id="EVT-002",
            session_id="SESSION-FAIL",
            user_id="USER-001",
            trace_id=None,
            sequence_number=2,
            event_type="EXECUTION_FAILED",
            timestamp=_ts(1),
            duration_ms=None,
            payload={
                "failure_category": "RETRIEVAL_ERROR",
                "error_message": "Vector store unavailable",
            },
        )
        repo.update_session_status("SESSION-FAIL", "FAILED")
        db_session.commit()

        path = reconstructor.reconstruct("SESSION-FAIL")
        assert path.status == PathStatus.FAILED

    def test_failed_path_has_failure_timeline_entry(
        self, reconstructor: DecisionPathReconstructor, repo, db_session
    ):
        repo.create_session(session_id="SESSION-FAIL-2", user_id="USER-001")
        repo.persist_event(
            event_id="EVT-FAIL",
            session_id="SESSION-FAIL-2",
            user_id="USER-001",
            trace_id=None,
            sequence_number=1,
            event_type="EXECUTION_FAILED",
            timestamp=_ts(),
            duration_ms=None,
            payload={"failure_category": "TOOL_ERROR", "error_message": "Tool failed"},
        )
        repo.update_session_status("SESSION-FAIL-2", "FAILED")
        db_session.commit()

        path = reconstructor.reconstruct("SESSION-FAIL-2")
        failure_entries = [
            e for e in path.timeline if e.event_type == "EXECUTION_FAILED"
        ]
        assert len(failure_entries) == 1
        assert failure_entries[0].timeline_type == TimelineEntryType.FAILURE


class TestReconstructorErrors:
    """Error handling."""

    def test_session_not_found_raises_value_error(
        self, reconstructor: DecisionPathReconstructor
    ):
        with pytest.raises(ValueError, match="Session not found"):
            reconstructor.reconstruct("SESSION-DOES-NOT-EXIST")

    def test_get_session_record_not_found_raises(
        self, reconstructor: DecisionPathReconstructor
    ):
        with pytest.raises(ValueError):
            reconstructor.get_session_record("SESSION-MISSING")


class TestReconstructorLangSmithURL:
    """LangSmith URL construction."""

    def test_langsmith_url_with_trace_id(
        self, reconstructor: DecisionPathReconstructor, repo, db_session
    ):
        repo.create_session(
            session_id="SESSION-LS",
            user_id="USER-001",
            trace_id="TRACE-ABCD-1234",
        )
        db_session.commit()

        path = reconstructor.reconstruct("SESSION-LS")
        assert path.langsmith_url is not None
        assert "TRACE-ABCD-1234" in path.langsmith_url
        assert "smith.langchain.com" in path.langsmith_url

    def test_no_trace_id_no_url(
        self, reconstructor: DecisionPathReconstructor, repo, db_session
    ):
        repo.create_session(
            session_id="SESSION-NO-TRACE",
            user_id="USER-001",
            trace_id=None,
        )
        db_session.commit()

        path = reconstructor.reconstruct("SESSION-NO-TRACE")
        assert path.langsmith_url is None


# ---------------------------------------------------------------------------
# TimelineBuilder tests
# ---------------------------------------------------------------------------


class TestTimelineBuilder:
    """Unit tests for the TimelineBuilder."""

    @pytest.fixture
    def builder(self):
        return TimelineBuilder()

    def _make_event(self, seq: int, event_type: str, payload: dict = None):
        from app.audit.events import AuditEventRecord
        return AuditEventRecord(
            event_id=f"EVT-{seq:03d}",
            session_id="SESSION-TL",
            user_id="USER-001",
            trace_id=None,
            sequence_number=seq,
            event_type=event_type,
            timestamp=_ts(seq),
            duration_ms=float(seq * 10),
            payload=payload or {},
            pii_redacted=True,
        )

    def test_input_received_maps_to_input_type(self, builder: TimelineBuilder):
        events = [self._make_event(1, "INPUT_RECEIVED", {"request": "Test request"})]
        entries = builder.build(events)
        assert entries[0].timeline_type == TimelineEntryType.INPUT

    def test_decision_completed_maps_to_decision_type(self, builder: TimelineBuilder):
        events = [self._make_event(1, "DECISION_COMPLETED", {"decision": "APPROVED"})]
        entries = builder.build(events)
        assert entries[0].timeline_type == TimelineEntryType.DECISION

    def test_execution_failed_maps_to_failure_type(self, builder: TimelineBuilder):
        events = [self._make_event(1, "EXECUTION_FAILED", {"failure_category": "ERR"})]
        entries = builder.build(events)
        assert entries[0].timeline_type == TimelineEntryType.FAILURE

    def test_empty_events_returns_empty_list(self, builder: TimelineBuilder):
        assert builder.build([]) == []

    def test_decision_summary_includes_outcome(self, builder: TimelineBuilder):
        events = [
            self._make_event(1, "DECISION_COMPLETED", {
                "decision": "REJECTED",
                "decision_reason": "Exceeds max limit",
            })
        ]
        entries = builder.build(events)
        assert "REJECTED" in entries[0].summary

    def test_retrieval_summary_includes_count(self, builder: TimelineBuilder):
        events = [
            self._make_event(1, "RETRIEVAL_COMPLETED", {
                "retrieved_count": 5,
                "source_names": ["policy.md"],
            })
        ]
        entries = builder.build(events)
        assert "5" in entries[0].summary

    def test_duration_ms_preserved(self, builder: TimelineBuilder):
        events = [self._make_event(1, "INPUT_RECEIVED", {})]
        entries = builder.build(events)
        assert entries[0].duration_ms == 10.0

    def test_multiple_events_all_converted(self, builder: TimelineBuilder):
        events = [
            self._make_event(1, "INPUT_RECEIVED"),
            self._make_event(2, "RETRIEVAL_COMPLETED"),
            self._make_event(3, "DECISION_COMPLETED"),
        ]
        entries = builder.build(events)
        assert len(entries) == 3
