"""SQLAlchemy ORM models for the TraceLens audit store.

Tables
------
audit_sessions    — one row per agent invocation (session lifecycle).
audit_events      — ordered event log, one row per execution event.
decision_records  — denormalized summary record for fast decision queries.

Design notes:
- payload_json is a TEXT column storing the sanitized (PII-redacted) event
  payload as JSON. Deliberately not over-normalized to maintain queryability
  and flexibility.
- pii_redacted flag guarantees at-rest auditing of the redaction guarantee.
- All timestamps are UTC.
- Indexes are created on the queryability-required fields: session_id,
  user_id, timestamp.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Shared declarative base for all TraceLens ORM models."""
    pass


class AuditSessionModel(Base):
    """Lifecycle record for a single agent execution session.

    Created when the first event for a session_id arrives.
    Updated when the session completes or fails.
    """

    __tablename__ = "audit_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)
    user_id: Mapped[str] = mapped_column(String(128), nullable=False)
    trace_id: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    started_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="IN_PROGRESS"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    __table_args__ = (
        Index("ix_audit_sessions_session_id", "session_id"),
        Index("ix_audit_sessions_user_id", "user_id"),
        Index("ix_audit_sessions_started_at", "started_at"),
        Index("ix_audit_sessions_status", "status"),
    )

    def __repr__(self) -> str:
        return (
            f"<AuditSessionModel session_id={self.session_id!r} "
            f"status={self.status!r}>"
        )


class AuditEventModel(Base):
    """Single execution event in the audit trail.

    Events are stored in sequence_number order per session.
    payload_json contains the PII-redacted event payload.
    The pii_redacted flag records that the redaction layer was applied.

    CRITICAL: payload_json must NEVER contain raw PII.
    The AuditPersistenceService enforces this invariant.
    """

    __tablename__ = "audit_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_id: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    session_id: Mapped[str] = mapped_column(String(128), nullable=False)
    user_id: Mapped[str] = mapped_column(String(128), nullable=False)
    trace_id: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    sequence_number: Mapped[int] = mapped_column(Integer, nullable=False)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    duration_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    payload_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    event_version: Mapped[str] = mapped_column(
        String(16), nullable=False, default="1.0"
    )
    pii_redacted: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    __table_args__ = (
        Index("ix_audit_events_session_id", "session_id"),
        Index("ix_audit_events_user_id", "user_id"),
        Index("ix_audit_events_timestamp", "timestamp"),
        Index("ix_audit_events_event_type", "event_type"),
        # Composite index for the most common query: events for a session in order
        Index("ix_audit_events_session_seq", "session_id", "sequence_number"),
    )

    def __repr__(self) -> str:
        return (
            f"<AuditEventModel session_id={self.session_id!r} "
            f"seq={self.sequence_number} type={self.event_type!r}>"
        )


class DecisionRecordModel(Base):
    """Denormalized decision record for fast governance queries.

    Created/updated when a DECISION_COMPLETED event is processed.
    Enables quick decision-level queries without full event reconstruction.
    """

    __tablename__ = "decision_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)
    user_id: Mapped[str] = mapped_column(String(128), nullable=False)
    decision: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    decision_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    policy_references: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    evidence: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    __table_args__ = (
        Index("ix_decision_records_session_id", "session_id"),
        Index("ix_decision_records_user_id", "user_id"),
        Index("ix_decision_records_decision", "decision"),
    )

    def __repr__(self) -> str:
        return (
            f"<DecisionRecordModel session_id={self.session_id!r} "
            f"decision={self.decision!r}>"
        )
