"""Audit persistence service.

This module bridges the observability event bus and the audit database.

Pipeline (strictly enforced)
----------------------------
ExecutionEvent (raw)
    ↓
PIIRedactor.redact()
    ↓
Sanitized payload
    ↓
AuditRepository.persist_event()
    ↓
SQLite audit_events table

CRITICAL INVARIANT:
    Raw events must NEVER reach the repository.
    This service is the ONLY consumer of ExecutionEvents that writes to
    the audit database. The redaction step is mandatory and cannot be bypassed.

Usage
-----
Call ``wire_persistence(publisher, db_session_factory)`` once at application
startup. It subscribes to the wildcard event channel and processes every event
emitted by the InstrumentedAgent.
"""

from __future__ import annotations

import structlog
from datetime import datetime, timezone
from typing import Callable, Optional

from sqlalchemy.orm import Session

from app.observability.events import (
    ExecutionEvent,
    EventType,
    DecisionCompletedEvent,
    OutputGeneratedEvent,
    ExecutionFailedEvent,
)
from app.observability.publisher import EventPublisher
from app.database.repository import AuditRepository
from app.privacy.redactor import PIIRedactor, get_redactor

logger = structlog.get_logger(__name__)


class AuditPersistenceService:
    """Subscribes to the event bus and persists sanitized audit events.

    This service is the enforcement point for the PII redaction guarantee.
    Every event that passes through this service is redacted before any
    database write occurs.

    Attributes:
        _redactor: PIIRedactor instance.
        _session_factory: Callable that returns a SQLAlchemy Session.
    """

    def __init__(
        self,
        session_factory: Callable[[], Session],
        redactor: Optional[PIIRedactor] = None,
    ) -> None:
        """Initialize the persistence service.

        Args:
            session_factory: Zero-arg callable returning a SQLAlchemy Session.
            redactor: PIIRedactor to use. Defaults to global singleton.
        """
        self._session_factory = session_factory
        self._redactor = redactor or get_redactor()
        logger.info("audit_persistence_service_initialized")

    def handle_event(self, event: ExecutionEvent) -> None:
        """Process an execution event: redact PII, then persist.

        This is the subscriber callback registered with the event publisher.

        Pipeline:
            1. Build the raw payload from the event.
            2. Apply PII redaction to the entire payload.
            3. Ensure audit session exists (upsert).
            4. Persist the sanitized event.
            5. If DECISION_COMPLETED, upsert the decision record.
            6. If OUTPUT_GENERATED or EXECUTION_FAILED, close the session.

        Args:
            event: Raw ExecutionEvent from the agent instrumentation.
        """
        try:
            # ----------------------------------------------------------------
            # Step 1: Build raw payload
            # ----------------------------------------------------------------
            raw_payload = self._build_payload(event)

            # ----------------------------------------------------------------
            # Step 2: PII REDACTION (mandatory — cannot be bypassed)
            # ----------------------------------------------------------------
            sanitized_payload = self._redactor.redact(raw_payload)

            # ----------------------------------------------------------------
            # Steps 3-6: Persist to database
            # ----------------------------------------------------------------
            with self._get_db_session() as db:
                repo = AuditRepository(db)

                # Ensure session exists
                repo.upsert_session(
                    session_id=event.session_id,
                    user_id=event.user_id,
                    trace_id=event.trace_id,
                    started_at=event.timestamp,
                )

                # Persist the sanitized event
                repo.persist_event(
                    event_id=event.event_id,
                    session_id=event.session_id,
                    user_id=event.user_id,
                    trace_id=event.trace_id,
                    sequence_number=event.sequence,
                    event_type=event.event_type.value,
                    timestamp=event.timestamp,
                    duration_ms=event.duration_ms,
                    payload=sanitized_payload,
                    event_version="1.0",
                )

                # Handle decision completion: upsert decision record
                if event.event_type == EventType.DECISION_COMPLETED:
                    self._handle_decision_completed(event, sanitized_payload, repo)

                # Handle terminal events: close the session
                if event.event_type in (
                    EventType.OUTPUT_GENERATED,
                    EventType.EXECUTION_FAILED,
                ):
                    self._finalize_session(event, repo)

        except Exception as exc:
            # Never let persistence errors propagate to the agent.
            # Log clearly but swallow so agent execution is unaffected.
            logger.error(
                "audit_persistence_failed",
                event_type=event.event_type.value,
                session_id=event.session_id,
                error=str(exc),
                exc_info=True,
            )

    def _build_payload(self, event: ExecutionEvent) -> dict:
        """Construct the payload dict from an event.

        Includes common fields plus event-specific metadata.
        The payload is the raw form BEFORE redaction.

        Args:
            event: ExecutionEvent to extract payload from.

        Returns:
            Raw payload dict (may contain PII).
        """
        payload = {
            "event_type": event.event_type.value,
            "sequence": event.sequence,
        }
        # Include event-specific metadata
        if event.metadata:
            payload.update(event.metadata)
        return payload

    def _handle_decision_completed(
        self,
        event: ExecutionEvent,
        sanitized_payload: dict,
        repo: AuditRepository,
    ) -> None:
        """Create/update the decision record from a DECISION_COMPLETED event.

        Args:
            event: The DECISION_COMPLETED event.
            sanitized_payload: Already-redacted payload.
            repo: AuditRepository instance.
        """
        decision = sanitized_payload.get("decision", "UNKNOWN")
        decision_reason = sanitized_payload.get("decision_reason", "")
        policy_references = sanitized_payload.get("policy_references", [])

        # Ensure policy_references is a list
        if isinstance(policy_references, str):
            policy_references = [policy_references]

        repo.upsert_decision_record(
            session_id=event.session_id,
            user_id=event.user_id,
            decision=decision,
            decision_reason=decision_reason,
            policy_references=policy_references,
            evidence=sanitized_payload.get("evidence", []),
        )

    def _finalize_session(
        self,
        event: ExecutionEvent,
        repo: AuditRepository,
    ) -> None:
        """Mark the audit session as complete or failed.

        Args:
            event: Terminal event (OUTPUT_GENERATED or EXECUTION_FAILED).
            repo: AuditRepository instance.
        """
        status = (
            "FAILED"
            if event.event_type == EventType.EXECUTION_FAILED
            else "COMPLETED"
        )
        repo.update_session_status(
            session_id=event.session_id,
            status=status,
            completed_at=datetime.now(timezone.utc),
            trace_id=event.trace_id,
        )

    def _get_db_session(self):
        """Return a context-managed database session.

        Returns:
            Context manager yielding a SQLAlchemy Session with auto-commit.
        """
        from contextlib import contextmanager

        @contextmanager
        def _session_context():
            db = self._session_factory()
            try:
                yield db
                db.commit()
            except Exception:
                db.rollback()
                raise
            finally:
                db.close()

        return _session_context()


def wire_persistence(
    publisher: EventPublisher,
    session_factory: Callable[[], Session],
    redactor: Optional[PIIRedactor] = None,
) -> AuditPersistenceService:
    """Wire the persistence service to the event publisher.

    This function should be called once at application startup.
    It creates an AuditPersistenceService and subscribes it to ALL events
    via the wildcard channel.

    Args:
        publisher: The InProcessEventPublisher (or any EventPublisher).
        session_factory: Callable returning a SQLAlchemy Session.
        redactor: Optional PIIRedactor. Defaults to global singleton.

    Returns:
        The wired AuditPersistenceService instance.
    """
    service = AuditPersistenceService(
        session_factory=session_factory,
        redactor=redactor,
    )

    # Subscribe to ALL events via wildcard
    publisher.subscribe("*", service.handle_event)

    logger.info("audit_persistence_wired_to_publisher")
    return service
