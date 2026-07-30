"""Timeline construction from raw audit events.

The TimelineBuilder converts a sequence of AuditEventRecord objects into
a structured list of TimelineEntry items suitable for display and reconstruction.

Each event type maps to a specific TimelineEntryType with a generated summary
and structured details. Unknown event types are represented transparently rather
than silently dropped.
"""

from __future__ import annotations

import structlog
from typing import Any

from app.audit.events import (
    AuditEventRecord,
    TimelineEntry,
    TimelineEntryType,
)

logger = structlog.get_logger(__name__)

# Mapping from raw event type strings to display types
_EVENT_TYPE_MAP: dict[str, TimelineEntryType] = {
    "INPUT_RECEIVED": TimelineEntryType.INPUT,
    "RETRIEVAL_STARTED": TimelineEntryType.CONTEXT_RETRIEVED,
    "RETRIEVAL_COMPLETED": TimelineEntryType.CONTEXT_RETRIEVED,
    "TOOL_STARTED": TimelineEntryType.TOOL_CALL,
    "TOOL_COMPLETED": TimelineEntryType.TOOL_RESPONSE,
    "DECISION_STARTED": TimelineEntryType.DECISION,
    "DECISION_COMPLETED": TimelineEntryType.DECISION,
    "OUTPUT_GENERATED": TimelineEntryType.OUTPUT,
    "EXECUTION_FAILED": TimelineEntryType.FAILURE,
}


class TimelineBuilder:
    """Converts a sequence of AuditEventRecord into TimelineEntry objects.

    Each entry is self-contained with its type, summary, and structured details.
    The builder never invents information — if a field is missing from an event's
    payload, it is represented as absent (not fabricated).
    """

    def build(self, events: list[AuditEventRecord]) -> list[TimelineEntry]:
        """Convert ordered events into timeline entries.

        Args:
            events: Audit events ordered by sequence_number.

        Returns:
            List of TimelineEntry, one per input event.
        """
        entries = []
        for event in events:
            try:
                entry = self._build_entry(event)
                entries.append(entry)
            except Exception as exc:
                logger.error(
                    "timeline_entry_build_failed",
                    event_id=event.event_id,
                    event_type=event.event_type,
                    error=str(exc),
                )
                # Represent the failure as an entry rather than dropping it
                entries.append(self._build_error_entry(event, exc))
        return entries

    def _build_entry(self, event: AuditEventRecord) -> TimelineEntry:
        """Build a single timeline entry from an audit event.

        Args:
            event: AuditEventRecord to convert.

        Returns:
            TimelineEntry with type, summary, and details.
        """
        timeline_type = _EVENT_TYPE_MAP.get(
            event.event_type, TimelineEntryType.OUTPUT
        )
        summary = self._build_summary(event)
        details = self._build_details(event)

        return TimelineEntry(
            sequence=event.sequence_number,
            timestamp=event.timestamp,
            event_type=event.event_type,
            timeline_type=timeline_type,
            summary=summary,
            details=details,
            duration_ms=event.duration_ms,
        )

    def _build_summary(self, event: AuditEventRecord) -> str:
        """Generate a one-line human-readable summary.

        Args:
            event: AuditEventRecord.

        Returns:
            Summary string.
        """
        payload = event.payload

        if event.event_type == "INPUT_RECEIVED":
            request = payload.get("request", "(no request captured)")
            return f"Request received: {request[:120]}"

        elif event.event_type == "RETRIEVAL_STARTED":
            query = payload.get("query", "")
            return f"Policy retrieval started{': ' + query[:80] if query else ''}"

        elif event.event_type == "RETRIEVAL_COMPLETED":
            count = payload.get("retrieved_count", 0)
            sources = payload.get("source_names", [])
            source_str = f" from {sources[0]}" if sources else ""
            return f"Retrieved {count} policy document(s){source_str}"

        elif event.event_type == "TOOL_STARTED":
            tool = payload.get("tool_name", "unknown tool")
            return f"Tool invoked: {tool}"

        elif event.event_type == "TOOL_COMPLETED":
            tool = payload.get("tool_name", "unknown tool")
            status = payload.get("status", "unknown")
            return f"Tool completed: {tool} [{status}]"

        elif event.event_type == "DECISION_STARTED":
            model = payload.get("model", "LLM")
            return f"Decision generation started using {model}"

        elif event.event_type == "DECISION_COMPLETED":
            decision = payload.get("decision", "UNKNOWN")
            reason = payload.get("decision_reason", "")
            summary = f"Decision: {decision}"
            if reason:
                summary += f" — {reason[:100]}"
            return summary

        elif event.event_type == "OUTPUT_GENERATED":
            length = payload.get("response_length", 0)
            return f"Final response generated ({length} characters)"

        elif event.event_type == "EXECUTION_FAILED":
            category = payload.get("failure_category", "UNKNOWN_ERROR")
            msg = payload.get("error_message", "")
            return f"Execution failed [{category}]{': ' + msg[:80] if msg else ''}"

        else:
            return f"Event: {event.event_type}"

    def _build_details(self, event: AuditEventRecord) -> dict[str, Any]:
        """Extract structured display details from an event payload.

        Only returns fields relevant to each event type.
        Never returns raw PII — payload is already sanitized.

        Args:
            event: AuditEventRecord.

        Returns:
            Dict of structured details for display.
        """
        payload = event.payload
        event_type = event.event_type

        if event_type == "INPUT_RECEIVED":
            return {
                "request": payload.get("request", ""),
                "user_id": event.user_id,
            }

        elif event_type == "RETRIEVAL_STARTED":
            return {
                "query": payload.get("query", ""),
            }

        elif event_type == "RETRIEVAL_COMPLETED":
            return {
                "retrieved_count": payload.get("retrieved_count", 0),
                "document_ids": payload.get("document_ids", []),
                "source_names": payload.get("source_names", []),
            }

        elif event_type in ("TOOL_STARTED", "TOOL_COMPLETED"):
            return {
                "tool_name": payload.get("tool_name", ""),
                "status": payload.get("status", ""),
                "response_keys": payload.get("response_keys", []),
            }

        elif event_type == "DECISION_STARTED":
            return {
                "model": payload.get("model", ""),
            }

        elif event_type == "DECISION_COMPLETED":
            return {
                "decision": payload.get("decision", ""),
                "decision_reason": payload.get("decision_reason", ""),
                "policy_references": payload.get("policy_references", []),
                "evidence": payload.get("evidence", []),
            }

        elif event_type == "OUTPUT_GENERATED":
            return {
                "response_length": payload.get("response_length", 0),
            }

        elif event_type == "EXECUTION_FAILED":
            return {
                "failure_category": payload.get("failure_category", ""),
                "error_message": payload.get("error_message", ""),
            }

        return dict(payload)

    def _build_error_entry(
        self, event: AuditEventRecord, exc: Exception
    ) -> TimelineEntry:
        """Build a fallback entry when normal construction fails.

        Args:
            event: The event that caused the error.
            exc: The exception raised.

        Returns:
            TimelineEntry indicating the reconstruction error.
        """
        return TimelineEntry(
            sequence=event.sequence_number,
            timestamp=event.timestamp,
            event_type=event.event_type,
            timeline_type=TimelineEntryType.FAILURE,
            summary=f"[Timeline reconstruction error for {event.event_type}]",
            details={"error": str(exc)},
            duration_ms=None,
        )
