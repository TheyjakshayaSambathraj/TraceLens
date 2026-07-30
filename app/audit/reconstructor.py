"""Decision path reconstructor.

Given a session_id, the DecisionPathReconstructor:
1. Retrieves all sanitized audit events from the repository.
2. Validates session consistency and event ordering.
3. Checks for expected events and records missing ones explicitly.
4. Constructs a structured DecisionPath with a chronological timeline.

Reconstruction rules
--------------------
- Events are ordered by sequence_number (primary) then timestamp (secondary).
- Missing events are represented explicitly in missing_steps.
- No data is invented or inferred. Absent = absent.
- The status reflects whether the path is COMPLETE, INCOMPLETE, or FAILED.

PS-7.1 compliance
-----------------
The reconstructed path represents the three-step retrieve → reason → decide
execution chain captured in the stored events.
"""

from __future__ import annotations

import structlog
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy.orm import Session

from app.audit.events import (
    AuditEventRecord,
    AuditSessionRecord,
    DecisionPath,
    PathStatus,
    TimelineEntry,
    TimelineEntryType,
)
from app.audit.timeline import TimelineBuilder
from app.database.repository import AuditRepository
from app.config.settings import get_settings

logger = structlog.get_logger(__name__)

# Expected events for a complete three-step execution
_EXPECTED_EVENTS = {
    "INPUT_RECEIVED",
    "RETRIEVAL_COMPLETED",
    "TOOL_COMPLETED",
    "DECISION_COMPLETED",
    "OUTPUT_GENERATED",
}


class DecisionPathReconstructor:
    """Reconstructs a structured decision path from stored audit events.

    This is the primary component for PS-7.1 decision path reconstruction.
    It reads from the repository (sanitized, PII-redacted events) and
    constructs a human-readable, structured audit trail.

    Args:
        db: SQLAlchemy Session.
    """

    def __init__(self, db: Session) -> None:
        self._db = db
        self._repo = AuditRepository(db)
        self._timeline_builder = TimelineBuilder()

    def reconstruct(self, session_id: str) -> DecisionPath:
        """Reconstruct the complete decision path for a session.

        Args:
            session_id: The session to reconstruct.

        Returns:
            DecisionPath with timeline, status, and missing_steps.

        Raises:
            ValueError: If the session does not exist.
        """
        logger.info("decision_path_reconstruction_started", session_id=session_id)

        # Load session record
        session_model = self._repo.get_session(session_id)
        if not session_model:
            raise ValueError(f"Session not found: {session_id}")

        # Load all events ordered by sequence
        event_models = self._repo.get_events_by_session(session_id)

        # Convert ORM → Pydantic
        events: list[AuditEventRecord] = [
            AuditEventRecord.from_orm_model(m) for m in event_models
        ]

        # Validate ordering
        self._validate_sequence(events)

        # Determine what's present and what's missing
        present_types = {e.event_type for e in events}
        missing_steps = [
            et for et in sorted(_EXPECTED_EVENTS)
            if et not in present_types
        ]

        # Determine path status
        path_status = self._determine_status(
            session_status=session_model.status,
            present_types=present_types,
            missing_steps=missing_steps,
        )

        # Build timeline
        timeline = self._timeline_builder.build(events)

        # Build LangSmith URL if trace_id is available
        langsmith_url = self._build_langsmith_url(session_model.trace_id)

        path = DecisionPath(
            session_id=session_id,
            user_id=session_model.user_id,
            trace_id=session_model.trace_id,
            status=path_status,
            started_at=session_model.started_at,
            completed_at=session_model.completed_at,
            timeline=timeline,
            missing_steps=missing_steps,
            pii_redacted=True,
            langsmith_url=langsmith_url,
        )

        logger.info(
            "decision_path_reconstruction_completed",
            session_id=session_id,
            status=path_status.value,
            event_count=len(events),
            missing_steps=missing_steps,
        )

        return path

    def _validate_sequence(self, events: list[AuditEventRecord]) -> None:
        """Validate that events have consistent and ordered sequence numbers.

        Logs warnings for ordering issues but does not raise — a partial
        reconstruction with explicit gaps is more useful than no reconstruction.

        Args:
            events: Events to validate.
        """
        if not events:
            return

        seen_seqs = set()
        prev_seq = -1

        for event in events:
            if event.sequence_number in seen_seqs:
                logger.warning(
                    "duplicate_sequence_number",
                    sequence=event.sequence_number,
                    event_type=event.event_type,
                )
            seen_seqs.add(event.sequence_number)

            if event.sequence_number <= prev_seq:
                logger.warning(
                    "out_of_order_sequence",
                    sequence=event.sequence_number,
                    prev_sequence=prev_seq,
                )
            prev_seq = event.sequence_number

    def _determine_status(
        self,
        session_status: str,
        present_types: set[str],
        missing_steps: list[str],
    ) -> PathStatus:
        """Determine the overall path status.

        Args:
            session_status: Status string from the audit_sessions row.
            present_types: Set of event type strings found.
            missing_steps: Expected event types that are missing.

        Returns:
            PathStatus enum value.
        """
        if session_status == "FAILED" or "EXECUTION_FAILED" in present_types:
            return PathStatus.FAILED

        if session_status == "IN_PROGRESS":
            return PathStatus.IN_PROGRESS

        if missing_steps:
            return PathStatus.INCOMPLETE

        return PathStatus.COMPLETE

    def _build_langsmith_url(self, trace_id: Optional[str]) -> Optional[str]:
        """Build a LangSmith trace URL from a trace ID.

        Args:
            trace_id: LangSmith run/trace ID.

        Returns:
            URL string or None if trace_id is absent.
        """
        if not trace_id:
            return None

        settings = get_settings()
        project = settings.langchain_project
        # Standard LangSmith URL format
        return f"https://smith.langchain.com/o/default/projects/p/{project}/r/{trace_id}"

    def get_session_record(self, session_id: str) -> AuditSessionRecord:
        """Get a session record without full reconstruction.

        Args:
            session_id: Session to retrieve.

        Returns:
            AuditSessionRecord.

        Raises:
            ValueError: If session not found.
        """
        session_model = self._repo.get_session(session_id)
        if not session_model:
            raise ValueError(f"Session not found: {session_id}")

        # Attach decision if available
        decision_record = self._repo.get_decision_record(session_id)
        decision = decision_record.decision if decision_record else None

        return AuditSessionRecord(
            session_id=session_model.session_id,
            user_id=session_model.user_id,
            trace_id=session_model.trace_id,
            status=session_model.status,
            started_at=session_model.started_at,
            completed_at=session_model.completed_at,
            created_at=session_model.created_at,
            decision=decision,
        )
