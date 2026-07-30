"""Pydantic models for the audit retrieval and reconstruction layer.

These models represent the OUTPUT of the audit system — the structured
data returned by the API and consumed by the Streamlit dashboard.

They are distinct from the raw ExecutionEvent dataclasses (observability layer)
and from the SQLAlchemy ORM models (persistence layer).

All models here contain ONLY sanitized (PII-redacted) data.
"""

from __future__ import annotations

import json
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class PathStatus(str, Enum):
    """Status of a reconstructed decision path.

    COMPLETE   — all expected events are present.
    INCOMPLETE — some expected events are missing (explicitly represented).
    FAILED     — execution terminated with an error.
    IN_PROGRESS — execution is still ongoing.
    """

    COMPLETE = "COMPLETE"
    INCOMPLETE = "INCOMPLETE"
    FAILED = "FAILED"
    IN_PROGRESS = "IN_PROGRESS"


class TimelineEntryType(str, Enum):
    """Human-readable classification for timeline display."""

    INPUT = "INPUT"
    CONTEXT_RETRIEVED = "CONTEXT_RETRIEVED"
    TOOL_CALL = "TOOL_CALL"
    TOOL_RESPONSE = "TOOL_RESPONSE"
    DECISION = "DECISION"
    OUTPUT = "OUTPUT"
    FAILURE = "FAILURE"


class TimelineEntry(BaseModel):
    """A single entry in the reconstructed decision timeline.

    Attributes:
        sequence: Monotonic sequence number for ordering.
        timestamp: UTC timestamp of the event.
        event_type: Raw event type string.
        timeline_type: Human-readable classification.
        summary: One-line human-readable summary of what happened.
        details: Structured details relevant to this event type.
        duration_ms: Operation duration in milliseconds if available.
    """

    sequence: int = Field(..., description="Event sequence number")
    timestamp: datetime = Field(..., description="UTC event timestamp")
    event_type: str = Field(..., description="Raw event type")
    timeline_type: TimelineEntryType = Field(..., description="Display classification")
    summary: str = Field(..., description="One-line human-readable summary")
    details: dict[str, Any] = Field(
        default_factory=dict, description="Structured event details"
    )
    duration_ms: Optional[float] = Field(
        None, description="Operation duration in milliseconds"
    )


class DecisionPath(BaseModel):
    """Complete reconstructed decision path for a session.

    This is the primary output of the DecisionPathReconstructor.
    It represents the causal chain from INPUT → OUTPUT as a structured
    timeline with explicit representation of missing steps.

    Attributes:
        session_id: The session this path belongs to.
        user_id: User who triggered the decision.
        trace_id: LangSmith trace ID if available (for correlation).
        status: Completeness/success status of the path.
        started_at: Session start time.
        completed_at: Session completion time.
        timeline: Ordered list of timeline entries.
        missing_steps: Event types expected but not found.
        pii_redacted: Always True — confirms PII protection.
        langsmith_url: Link to LangSmith trace if available.
    """

    session_id: str = Field(..., description="Session identifier")
    user_id: str = Field(..., description="User identifier")
    trace_id: Optional[str] = Field(None, description="LangSmith trace ID")
    status: PathStatus = Field(..., description="Path completeness status")
    started_at: Optional[datetime] = Field(None, description="Session start time")
    completed_at: Optional[datetime] = Field(None, description="Session end time")
    timeline: list[TimelineEntry] = Field(
        default_factory=list, description="Ordered timeline entries"
    )
    missing_steps: list[str] = Field(
        default_factory=list,
        description="Expected event types not found (honest about gaps)",
    )
    pii_redacted: bool = Field(
        True,
        description="Confirms PII redaction was applied before persistence",
    )
    langsmith_url: Optional[str] = Field(
        None, description="Link to LangSmith trace if configured"
    )

    model_config = {"from_attributes": True}


class AuditSessionRecord(BaseModel):
    """API-level audit session record.

    Returned by GET /audit/sessions/{session_id}.
    """

    session_id: str
    user_id: str
    trace_id: Optional[str] = None
    status: str
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    created_at: datetime
    decision: Optional[str] = None  # Populated from decision_records join

    model_config = {"from_attributes": True}


class AuditEventRecord(BaseModel):
    """API-level audit event record.

    Each event includes the sanitized payload parsed from stored JSON.
    """

    event_id: str
    session_id: str
    user_id: str
    trace_id: Optional[str] = None
    sequence_number: int
    event_type: str
    timestamp: datetime
    duration_ms: Optional[float] = None
    payload: dict[str, Any] = Field(default_factory=dict)
    event_version: str = "1.0"
    pii_redacted: bool = True

    model_config = {"from_attributes": True}

    @classmethod
    def from_orm_model(cls, model) -> AuditEventRecord:
        """Construct from an AuditEventModel ORM instance.

        Parses payload_json back into a dict.
        """
        try:
            payload = json.loads(model.payload_json) if model.payload_json else {}
        except (json.JSONDecodeError, TypeError):
            payload = {}

        return cls(
            event_id=model.event_id,
            session_id=model.session_id,
            user_id=model.user_id,
            trace_id=model.trace_id,
            sequence_number=model.sequence_number,
            event_type=model.event_type,
            timestamp=model.timestamp,
            duration_ms=model.duration_ms,
            payload=payload,
            event_version=model.event_version,
            pii_redacted=model.pii_redacted,
        )


class DecisionSummary(BaseModel):
    """Human-readable decision summary for non-technical reviewers.

    Generated by DecisionSummaryService using the DecisionPath as input.
    Never exposes hidden chain-of-thought.
    """

    session_id: str
    decision: str
    summary: str = Field(..., description="Plain-English narrative summary")
    evidence_considered: list[str] = Field(
        default_factory=list,
        description="Evidence facts observed in the decision path",
    )
    policy_basis: list[str] = Field(
        default_factory=list, description="Policy references cited"
    )
    confidence: str = Field(
        ..., description="Qualitative confidence (NOT mathematical certainty)"
    )
    limitations: list[str] = Field(
        default_factory=list,
        description="Acknowledged gaps or uncertainties",
    )
    generated_at: datetime = Field(
        ..., description="UTC timestamp of summary generation"
    )

    model_config = {"from_attributes": True}


class RegulatoryChallengeResponse(BaseModel):
    """Draft response to a regulatory challenge.

    Generated by RegulatoryChallengegenerator using the DecisionPath.
    Grounded exclusively in stored audit evidence.
    """

    session_id: str
    reference_number: str = Field(..., description="Internal reference ID")
    generated_at: datetime
    decision_summary: str
    data_considered: list[str] = Field(
        default_factory=list, description="Data sources used in the decision"
    )
    policy_basis: list[str] = Field(
        default_factory=list, description="Policy framework applied"
    )
    tools_used: list[str] = Field(
        default_factory=list, description="Tools/services consulted"
    )
    decision_outcome: str
    reasoning_basis: str = Field(
        ..., description="How evidence led to the decision"
    )
    limitations: list[str] = Field(
        default_factory=list, description="Acknowledged limitations"
    )
    full_response: str = Field(
        ..., description="Complete formatted regulatory response text"
    )

    model_config = {"from_attributes": True}


class SessionListResponse(BaseModel):
    """Paginated list of audit sessions."""

    sessions: list[AuditSessionRecord]
    total: int
    limit: int
    offset: int


class AgentDecideRequest(BaseModel):
    """Request model for POST /agent/decide."""

    request: str = Field(..., description="The leave request text")
    user_id: str = Field(default="USER-001", description="User making the request")
    session_id: Optional[str] = Field(
        None, description="Optional custom session ID"
    )


class AgentDecideResponse(BaseModel):
    """Response model for POST /agent/decide."""

    session_id: str
    decision: str
    reason: str
    final_response: str
    trace_id: Optional[str] = None
    audit_url: str = Field(..., description="URL to retrieve the audit record")
