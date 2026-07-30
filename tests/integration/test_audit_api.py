"""Integration tests for all audit API endpoints.

Tests cover:
- GET /audit/sessions/{session_id} — 200 and 404
- GET /audit/sessions/{session_id}/decision-path — timeline structure
- GET /audit/sessions/{session_id}/summary — summary fields
- GET /audit/sessions/{session_id}/challenge-response — reference number present
- GET /audit/users/{user_id}/sessions — user listing
- GET /audit/search — search with filters
- All endpoints return proper HTTP codes

Strategy: Each test class gets its own in-memory SQLite database via env var
patching. The get_settings() cache is cleared between setups to ensure the
test database URL is picked up by the lifespan init_db() call.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.database.repository import AuditRepository
from app.database.session import get_db


def _make_test_app():
    """Create a test app with in-memory database and cleared settings cache."""
    # Clear lru_cache so get_settings() picks up patched env
    from app.config.settings import get_settings
    get_settings.cache_clear()

    # Also reset the database engine singleton
    from app.database.session import reset_engine_for_testing
    reset_engine_for_testing("sqlite:///:memory:")

    from app.main import create_app
    app = create_app()

    # Override get_db to use the test engine's session factory
    def override_get_db():
        from app.database.session import _get_session_factory
        db = _get_session_factory()()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    return app


def _seed_test_data():
    """Seed the test in-memory database with known sessions and events.

    Must be called AFTER the app's lifespan startup (init_db) has run,
    which happens when TestClient is used as a context manager.

    Returns:
        dict of named session IDs for tests to reference.
    """
    from app.database.session import _get_session_factory
    factory = _get_session_factory()
    db = factory()

    try:
        repo = AuditRepository(db)
        now = datetime.now(timezone.utc)

        # Session 1: Complete APPROVED decision
        repo.create_session("SESSION-API-001", "USER-TEST-001", started_at=now)
        for seq, etype, payload in [
            (1, "INPUT_RECEIVED", {"request": "Can EMP-001 take 10 days?"}),
            (2, "RETRIEVAL_COMPLETED", {
                "retrieved_count": 3,
                "source_names": ["policy.md"],
                "document_ids": ["D1", "D2", "D3"],
            }),
            (3, "TOOL_COMPLETED", {
                "tool_name": "retrieve_employee_data",
                "status": "success",
                "response_keys": ["leave_balance", "status"],
            }),
            (4, "DECISION_COMPLETED", {
                "decision": "APPROVED",
                "decision_reason": "Sufficient leave balance",
                "policy_references": ["Section 2.1"],
                "evidence": ["leave_balance: 20 days"],
            }),
            (5, "OUTPUT_GENERATED", {"response_length": 120}),
        ]:
            repo.persist_event(
                event_id=f"EVT-API-001-{seq:03d}",
                session_id="SESSION-API-001",
                user_id="USER-TEST-001",
                trace_id=None,
                sequence_number=seq,
                event_type=etype,
                timestamp=now,
                duration_ms=float(seq * 100),
                payload=payload,
            )
        repo.update_session_status("SESSION-API-001", "COMPLETED", now)
        repo.upsert_decision_record(
            "SESSION-API-001", "USER-TEST-001", "APPROVED",
            "Sufficient leave balance", ["Section 2.1"], ["leave_balance: 20 days"]
        )

        # Session 2: Failed execution
        repo.create_session("SESSION-API-002", "USER-TEST-002", started_at=now)
        repo.persist_event(
            event_id="EVT-API-002-001",
            session_id="SESSION-API-002",
            user_id="USER-TEST-002",
            trace_id=None,
            sequence_number=1,
            event_type="INPUT_RECEIVED",
            timestamp=now,
            duration_ms=None,
            payload={"request": "Take 5 days"},
        )
        repo.persist_event(
            event_id="EVT-API-002-002",
            session_id="SESSION-API-002",
            user_id="USER-TEST-002",
            trace_id=None,
            sequence_number=2,
            event_type="EXECUTION_FAILED",
            timestamp=now,
            duration_ms=None,
            payload={"failure_category": "RETRIEVAL_ERROR", "error_message": "Store down"},
        )
        repo.update_session_status("SESSION-API-002", "FAILED", now)

        # Session 3: REJECTED decision (same user as 001 for user listing test)
        repo.create_session("SESSION-API-003", "USER-TEST-001", started_at=now)
        repo.persist_event(
            event_id="EVT-API-003-001",
            session_id="SESSION-API-003",
            user_id="USER-TEST-001",
            trace_id=None,
            sequence_number=1,
            event_type="DECISION_COMPLETED",
            timestamp=now,
            duration_ms=None,
            payload={
                "decision": "REJECTED",
                "decision_reason": "Exceeds limit",
                "policy_references": ["Section 2.3"],
                "evidence": ["requested: 15 > max: 10"],
            },
        )
        repo.update_session_status("SESSION-API-003", "COMPLETED", now)
        repo.upsert_decision_record(
            "SESSION-API-003", "USER-TEST-001", "REJECTED",
            "Exceeds limit", ["Section 2.3"], ["requested: 15 > max: 10"]
        )

        db.commit()
    finally:
        db.close()

    return {
        "approved_session": "SESSION-API-001",
        "failed_session": "SESSION-API-002",
        "rejected_session": "SESSION-API-003",
        "user_a": "USER-TEST-001",
        "user_b": "USER-TEST-002",
    }


@pytest.fixture(scope="module")
def test_env():
    """Patch DATABASE_URL to in-memory SQLite for the entire test module."""
    with patch.dict(os.environ, {"DATABASE_URL": "sqlite:///:memory:"}):
        yield


@pytest.fixture(scope="module")
def seeded_client(test_env):
    """Create a seeded test client for the entire module."""
    app = _make_test_app()

    with TestClient(app) as c:
        # init_db() has run, now seed data
        ids = _seed_test_data()
        yield c, ids


# ---------------------------------------------------------------------------
# GET /audit/sessions/{session_id}
# ---------------------------------------------------------------------------


class TestGetSession:
    def test_get_existing_session_200(self, seeded_client):
        client, ids = seeded_client
        r = client.get(f"/api/v1/audit/sessions/{ids['approved_session']}")
        assert r.status_code == 200
        data = r.json()
        assert data["session_id"] == ids["approved_session"]
        assert data["user_id"] == ids["user_a"]
        assert data["status"] == "COMPLETED"

    def test_get_missing_session_404(self, seeded_client):
        client, ids = seeded_client
        r = client.get("/api/v1/audit/sessions/SESSION-DOES-NOT-EXIST")
        assert r.status_code == 404
        assert "not found" in r.json()["detail"].lower()

    def test_get_failed_session(self, seeded_client):
        client, ids = seeded_client
        r = client.get(f"/api/v1/audit/sessions/{ids['failed_session']}")
        assert r.status_code == 200
        assert r.json()["status"] == "FAILED"


# ---------------------------------------------------------------------------
# GET /audit/sessions/{session_id}/decision-path
# ---------------------------------------------------------------------------


class TestGetDecisionPath:
    def test_complete_path_has_timeline(self, seeded_client):
        client, ids = seeded_client
        r = client.get(f"/api/v1/audit/sessions/{ids['approved_session']}/decision-path")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "COMPLETE"
        assert len(data["timeline"]) > 0
        assert data["pii_redacted"] is True

    def test_timeline_has_input_and_decision(self, seeded_client):
        client, ids = seeded_client
        r = client.get(f"/api/v1/audit/sessions/{ids['approved_session']}/decision-path")
        assert r.status_code == 200
        types = [e["event_type"] for e in r.json()["timeline"]]
        assert "INPUT_RECEIVED" in types
        assert "DECISION_COMPLETED" in types

    def test_timeline_ordered_by_sequence(self, seeded_client):
        client, ids = seeded_client
        r = client.get(f"/api/v1/audit/sessions/{ids['approved_session']}/decision-path")
        seqs = [e["sequence"] for e in r.json()["timeline"]]
        assert seqs == sorted(seqs)

    def test_failed_path_status_is_failed(self, seeded_client):
        client, ids = seeded_client
        r = client.get(f"/api/v1/audit/sessions/{ids['failed_session']}/decision-path")
        assert r.status_code == 200
        assert r.json()["status"] == "FAILED"

    def test_decision_path_not_found_404(self, seeded_client):
        client, _ = seeded_client
        r = client.get("/api/v1/audit/sessions/NONEXISTENT/decision-path")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# GET /audit/sessions/{session_id}/summary
# ---------------------------------------------------------------------------


class TestGetDecisionSummary:
    def test_summary_has_required_fields(self, seeded_client):
        client, ids = seeded_client
        r = client.get(f"/api/v1/audit/sessions/{ids['approved_session']}/summary")
        assert r.status_code == 200
        data = r.json()
        for field in ("session_id", "decision", "summary", "confidence", "generated_at"):
            assert field in data, f"Missing field: {field}"

    def test_summary_decision_matches_seeded_decision(self, seeded_client):
        client, ids = seeded_client
        r = client.get(f"/api/v1/audit/sessions/{ids['approved_session']}/summary")
        assert r.status_code == 200
        assert r.json()["decision"] == "APPROVED"

    def test_summary_not_found_404(self, seeded_client):
        client, _ = seeded_client
        r = client.get("/api/v1/audit/sessions/NONEXISTENT/summary")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# GET /audit/sessions/{session_id}/challenge-response
# ---------------------------------------------------------------------------


class TestGetChallengeResponse:
    def test_challenge_response_200(self, seeded_client):
        client, ids = seeded_client
        r = client.get(f"/api/v1/audit/sessions/{ids['approved_session']}/challenge-response")
        assert r.status_code == 200
        data = r.json()
        assert "reference_number" in data
        assert "full_response" in data
        assert "decision_outcome" in data

    def test_challenge_reference_number_format(self, seeded_client):
        client, ids = seeded_client
        r = client.get(f"/api/v1/audit/sessions/{ids['approved_session']}/challenge-response")
        assert r.json()["reference_number"].startswith("TL-")

    def test_challenge_response_not_found_404(self, seeded_client):
        client, _ = seeded_client
        r = client.get("/api/v1/audit/sessions/NONEXISTENT/challenge-response")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# GET /audit/users/{user_id}/sessions
# ---------------------------------------------------------------------------


class TestGetUserSessions:
    def test_user_sessions_correct_count(self, seeded_client):
        client, ids = seeded_client
        r = client.get(f"/api/v1/audit/users/{ids['user_a']}/sessions")
        assert r.status_code == 200
        data = r.json()
        assert "sessions" in data
        assert len(data["sessions"]) >= 2
        for s in data["sessions"]:
            assert s["user_id"] == ids["user_a"]

    def test_user_sessions_pagination(self, seeded_client):
        client, ids = seeded_client
        r = client.get(
            f"/api/v1/audit/users/{ids['user_a']}/sessions",
            params={"limit": 1, "offset": 0},
        )
        assert r.status_code == 200
        assert len(r.json()["sessions"]) == 1

    def test_user_with_no_sessions_returns_empty(self, seeded_client):
        client, _ = seeded_client
        r = client.get("/api/v1/audit/users/USER-NOBODY/sessions")
        assert r.status_code == 200
        assert r.json()["sessions"] == []


# ---------------------------------------------------------------------------
# GET /audit/search
# ---------------------------------------------------------------------------


class TestSearchSessions:
    def test_search_all_no_filters(self, seeded_client):
        client, _ = seeded_client
        r = client.get("/api/v1/audit/search")
        assert r.status_code == 200
        assert len(r.json()["sessions"]) >= 3

    def test_search_by_user_id(self, seeded_client):
        client, ids = seeded_client
        r = client.get("/api/v1/audit/search", params={"user_id": ids["user_b"]})
        assert r.status_code == 200
        for s in r.json()["sessions"]:
            assert s["user_id"] == ids["user_b"]

    def test_search_by_decision_approved(self, seeded_client):
        client, ids = seeded_client
        r = client.get("/api/v1/audit/search", params={"decision": "APPROVED"})
        assert r.status_code == 200
        assert any(s["session_id"] == ids["approved_session"] for s in r.json()["sessions"])

    def test_search_by_decision_rejected(self, seeded_client):
        client, ids = seeded_client
        r = client.get("/api/v1/audit/search", params={"decision": "REJECTED"})
        assert r.status_code == 200
        assert any(s["session_id"] == ids["rejected_session"] for s in r.json()["sessions"])

    def test_search_by_status_failed(self, seeded_client):
        client, ids = seeded_client
        r = client.get("/api/v1/audit/search", params={"status": "FAILED"})
        assert r.status_code == 200
        assert any(s["session_id"] == ids["failed_session"] for s in r.json()["sessions"])

    def test_search_pagination_limit(self, seeded_client):
        client, _ = seeded_client
        r = client.get("/api/v1/audit/search", params={"limit": 1})
        assert r.status_code == 200
        assert len(r.json()["sessions"]) == 1
