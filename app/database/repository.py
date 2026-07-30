"""Repository pattern implementation for the TraceLens audit store.

The AuditRepository is the ONLY component that directly touches SQLAlchemy
ORM models. All other layers (services, API routes) depend on this abstraction.

Architectural invariant:
    Raw PII must never reach this layer.
    The AuditPersistenceService enforces PII redaction BEFORE calling here.
    This repository assumes all data it receives is already sanitized.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Optional

import structlog
from sqlalchemy import and_, desc
from sqlalchemy.orm import Session

from app.database.models import (
    AuditEventModel,
    AuditSessionModel,
    DecisionRecordModel,
)

logger = structlog.get_logger(__name__)

# Default page size for paginated queries
DEFAULT_PAGE_SIZE = 20
MAX_PAGE_SIZE = 100


class AuditRepository:
    """Repository abstraction over the audit database.

    All database access for audit data goes through this class.
    Methods use explicit session injection for testability.

    Args:
        db: SQLAlchemy Session instance.
    """

    def __init__(self, db: Session) -> None:
        self._db = db

    # -------------------------------------------------------------------------
    # Session management
    # -------------------------------------------------------------------------

    def create_session(
        self,
        session_id: str,
        user_id: str,
        trace_id: Optional[str] = None,
        started_at: Optional[datetime] = None,
        status: str = "IN_PROGRESS",
    ) -> AuditSessionModel:
        """Create a new audit session record.

        Args:
            session_id: Unique session identifier.
            user_id: User associated with the session.
            trace_id: LangSmith trace ID if available.
            started_at: UTC timestamp when execution started.
            status: Initial status string.

        Returns:
            Persisted AuditSessionModel.
        """
        session = AuditSessionModel(
            session_id=session_id,
            user_id=user_id,
            trace_id=trace_id,
            started_at=started_at or datetime.now(timezone.utc),
            status=status,
        )
        self._db.add(session)
        self._db.flush()

        logger.info(
            "audit_session_created",
            session_id=session_id,
            user_id=user_id,
        )
        return session

    def get_session(self, session_id: str) -> Optional[AuditSessionModel]:
        """Retrieve an audit session by session_id.

        Args:
            session_id: Session to look up.

        Returns:
            AuditSessionModel or None.
        """
        return (
            self._db.query(AuditSessionModel)
            .filter(AuditSessionModel.session_id == session_id)
            .first()
        )

    def upsert_session(
        self,
        session_id: str,
        user_id: str,
        trace_id: Optional[str] = None,
        started_at: Optional[datetime] = None,
        status: str = "IN_PROGRESS",
    ) -> AuditSessionModel:
        """Get or create an audit session record.

        Args:
            session_id: Session identifier.
            user_id: User identifier.
            trace_id: LangSmith trace ID.
            started_at: Session start time.
            status: Session status.

        Returns:
            Existing or newly created AuditSessionModel.
        """
        existing = self.get_session(session_id)
        if existing:
            return existing
        return self.create_session(
            session_id=session_id,
            user_id=user_id,
            trace_id=trace_id,
            started_at=started_at,
            status=status,
        )

    def update_session_status(
        self,
        session_id: str,
        status: str,
        completed_at: Optional[datetime] = None,
        trace_id: Optional[str] = None,
    ) -> Optional[AuditSessionModel]:
        """Update session status and completion time.

        Args:
            session_id: Session to update.
            status: New status value.
            completed_at: UTC completion timestamp.
            trace_id: LangSmith trace ID (set if newly available).

        Returns:
            Updated session or None if not found.
        """
        session = self.get_session(session_id)
        if not session:
            logger.warning("update_session_not_found", session_id=session_id)
            return None

        session.status = status
        if completed_at:
            session.completed_at = completed_at
        if trace_id and not session.trace_id:
            session.trace_id = trace_id

        self._db.flush()
        return session

    def get_sessions_by_user(
        self,
        user_id: str,
        limit: int = DEFAULT_PAGE_SIZE,
        offset: int = 0,
    ) -> list[AuditSessionModel]:
        """Retrieve all sessions for a user, newest first.

        Args:
            user_id: User identifier.
            limit: Maximum records to return.
            offset: Pagination offset.

        Returns:
            List of AuditSessionModel ordered by started_at descending.
        """
        limit = min(limit, MAX_PAGE_SIZE)
        return (
            self._db.query(AuditSessionModel)
            .filter(AuditSessionModel.user_id == user_id)
            .order_by(desc(AuditSessionModel.started_at))
            .limit(limit)
            .offset(offset)
            .all()
        )

    def search_sessions(
        self,
        user_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        decision: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = DEFAULT_PAGE_SIZE,
        offset: int = 0,
    ) -> list[AuditSessionModel]:
        """Search sessions with optional filters.

        Args:
            user_id: Filter by user.
            start_time: Filter sessions started at or after this UTC time.
            end_time: Filter sessions started at or before this UTC time.
            decision: Filter by decision outcome (via decision_records join).
            status: Filter by session status.
            limit: Maximum records.
            offset: Pagination offset.

        Returns:
            Matching AuditSessionModel instances.
        """
        limit = min(limit, MAX_PAGE_SIZE)
        query = self._db.query(AuditSessionModel)

        if user_id:
            query = query.filter(AuditSessionModel.user_id == user_id)
        if start_time:
            query = query.filter(AuditSessionModel.started_at >= start_time)
        if end_time:
            query = query.filter(AuditSessionModel.started_at <= end_time)
        if status:
            query = query.filter(AuditSessionModel.status == status)

        if decision:
            # Join with decision_records to filter by outcome
            query = query.join(
                DecisionRecordModel,
                AuditSessionModel.session_id == DecisionRecordModel.session_id,
            ).filter(DecisionRecordModel.decision == decision)

        return (
            query.order_by(desc(AuditSessionModel.started_at))
            .limit(limit)
            .offset(offset)
            .all()
        )

    # -------------------------------------------------------------------------
    # Event persistence
    # -------------------------------------------------------------------------

    def persist_event(
        self,
        event_id: str,
        session_id: str,
        user_id: str,
        trace_id: Optional[str],
        sequence_number: int,
        event_type: str,
        timestamp: datetime,
        duration_ms: Optional[float],
        payload: dict,
        event_version: str = "1.0",
    ) -> AuditEventModel:
        """Persist a sanitized audit event.

        PRECONDITION: payload must already be PII-redacted.
        The pii_redacted flag is always set to True here.

        Args:
            event_id: Unique event identifier.
            session_id: Session this event belongs to.
            user_id: User associated with the event.
            trace_id: LangSmith trace ID if available.
            sequence_number: Monotonic sequence within session.
            event_type: EventType string.
            timestamp: UTC timestamp of the event.
            duration_ms: Operation duration in milliseconds.
            payload: PII-redacted event payload dict.
            event_version: Schema version string.

        Returns:
            Persisted AuditEventModel.
        """
        audit_event = AuditEventModel(
            event_id=event_id,
            session_id=session_id,
            user_id=user_id,
            trace_id=trace_id,
            sequence_number=sequence_number,
            event_type=event_type,
            timestamp=timestamp,
            duration_ms=duration_ms,
            payload_json=json.dumps(payload, default=str),
            event_version=event_version,
            pii_redacted=True,  # Enforced: only sanitized events reach here
        )
        self._db.add(audit_event)
        self._db.flush()

        logger.debug(
            "audit_event_persisted",
            event_id=event_id,
            session_id=session_id,
            event_type=event_type,
            sequence_number=sequence_number,
        )
        return audit_event

    def get_events_by_session(
        self, session_id: str
    ) -> list[AuditEventModel]:
        """Retrieve all events for a session in sequence order.

        Args:
            session_id: Session to retrieve events for.

        Returns:
            Events ordered by sequence_number ascending.
        """
        return (
            self._db.query(AuditEventModel)
            .filter(AuditEventModel.session_id == session_id)
            .order_by(AuditEventModel.sequence_number)
            .all()
        )

    # -------------------------------------------------------------------------
    # Decision records
    # -------------------------------------------------------------------------

    def upsert_decision_record(
        self,
        session_id: str,
        user_id: str,
        decision: Optional[str],
        decision_reason: Optional[str],
        policy_references: Optional[list[str]],
        evidence: Optional[list[str]],
    ) -> DecisionRecordModel:
        """Create or update the decision record for a session.

        Args:
            session_id: Session identifier.
            user_id: User identifier.
            decision: Decision outcome string.
            decision_reason: Human-readable reason.
            policy_references: Policy sections cited.
            evidence: Evidence facts considered.

        Returns:
            Persisted DecisionRecordModel.
        """
        existing = (
            self._db.query(DecisionRecordModel)
            .filter(DecisionRecordModel.session_id == session_id)
            .first()
        )

        if existing:
            existing.decision = decision
            existing.decision_reason = decision_reason
            existing.policy_references = json.dumps(policy_references or [])
            existing.evidence = json.dumps(evidence or [])
            self._db.flush()
            return existing

        record = DecisionRecordModel(
            session_id=session_id,
            user_id=user_id,
            decision=decision,
            decision_reason=decision_reason,
            policy_references=json.dumps(policy_references or []),
            evidence=json.dumps(evidence or []),
        )
        self._db.add(record)
        self._db.flush()

        logger.info(
            "decision_record_upserted",
            session_id=session_id,
            decision=decision,
        )
        return record

    def get_decision_record(
        self, session_id: str
    ) -> Optional[DecisionRecordModel]:
        """Retrieve the decision record for a session.

        Args:
            session_id: Session to look up.

        Returns:
            DecisionRecordModel or None.
        """
        return (
            self._db.query(DecisionRecordModel)
            .filter(DecisionRecordModel.session_id == session_id)
            .first()
        )
