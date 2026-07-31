"""Audit and agent API routes.

Implements the PS-7.1 queryability and governance API:

Agent execution:
    POST /agent/decide          — Run the leave decision agent

Audit retrieval:
    GET /audit/sessions/{session_id}                    — Session record
    GET /audit/sessions/{session_id}/decision-path      — Reconstructed path
    GET /audit/sessions/{session_id}/summary            — LLM summary
    GET /audit/sessions/{session_id}/challenge-response — Regulatory response
    GET /audit/users/{user_id}/sessions                 — User sessions
    GET /audit/search                                   — Search with filters

Security principles:
    - Input validation via Pydantic/FastAPI
    - ORM usage prevents SQL injection
    - No raw stack traces in error responses
    - Sanitized audit records only (PII already redacted at persistence)
    - No API keys or secrets returned
"""

from __future__ import annotations

import structlog
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.audit.events import (
    AgentDecideRequest,
    AgentDecideResponse,
    AuditSessionRecord,
    DecisionPath,
    DecisionSummary,
    RegulatoryChallengeResponse,
    SessionListResponse,
)
from app.audit.reconstructor import DecisionPathReconstructor
from app.audit.summary import DecisionSummaryService
from app.audit.challenge import RegulatoryChallengegenerator
from app.api.dependencies import (
    get_reconstructor,
    get_summary_service,
    get_challenge_generator,
    get_repository,
    get_instrumented_agent,
)
from app.database.repository import AuditRepository, DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE

logger = structlog.get_logger(__name__)

router = APIRouter(tags=["audit"])

# ---------------------------------------------------------------------------
# Agent execution
# ---------------------------------------------------------------------------


@router.post(
    "/agent/decide",
    response_model=AgentDecideResponse,
    summary="Run the leave decision agent",
    description=(
        "Execute the leave decision workflow. Returns the session ID that "
        "can be used to query the audit trail after execution."
    ),
)
def decide(
    request: AgentDecideRequest,
    agent=Depends(get_instrumented_agent),
):
    """Execute the leave decision agent and return the result.

    The agent runs the full retrieve → reason → decide workflow.
    Events are captured, PII-redacted, and persisted automatically.

    Args:
        request: AgentDecideRequest with the leave request text.
        agent: InstrumentedAgent from DI.

    Returns:
        AgentDecideResponse with session_id, decision, and audit URL.
    """
    logger.info(
        "agent_decide_request",
        user_id=request.user_id,
        session_id=request.session_id,
    )

    try:
        final_state = agent.decide(
            request=request.request,
            user_id=request.user_id,
            session_id=request.session_id,
        )

        decision_value = final_state.get("decision")
        if hasattr(decision_value, "decision"):
            decision_str = decision_value.decision.value
            reason_str = getattr(decision_value, "reason", "")
        elif isinstance(decision_value, dict):
            decision_str = decision_value.get("decision", "UNKNOWN")
            reason_str = decision_value.get("reason", "")
        else:
            decision_str = "UNKNOWN"
            reason_str = ""

        session_id = final_state.get("session_id", request.session_id or "UNKNOWN")
        final_response = final_state.get("final_response", "")

        return AgentDecideResponse(
            session_id=session_id,
            decision=decision_str,
            reason=reason_str,
            final_response=final_response,
            trace_id=None,  # trace_id available via audit session
            audit_url=f"/api/v1/audit/sessions/{session_id}",
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )
    except RuntimeError as exc:
        logger.error("agent_decide_failed", error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent service unavailable. Check configuration.",
        )
    except Exception as exc:
        logger.error("agent_decide_unexpected_error", error=str(exc), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Decision failed: {str(exc)}",
        )


# ---------------------------------------------------------------------------
# Session retrieval
# ---------------------------------------------------------------------------


@router.get(
    "/audit/sessions/{session_id}",
    response_model=AuditSessionRecord,
    summary="Get audit session record",
)
def get_session(
    session_id: str,
    reconstructor: DecisionPathReconstructor = Depends(get_reconstructor),
):
    """Retrieve an audit session record by session ID.

    Args:
        session_id: Session to retrieve.
        reconstructor: Injected DecisionPathReconstructor.

    Returns:
        AuditSessionRecord.
    """
    try:
        return reconstructor.get_session_record(session_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found: {session_id}",
        )


@router.get(
    "/audit/sessions/{session_id}/decision-path",
    response_model=DecisionPath,
    summary="Get the reconstructed decision path",
    description=(
        "Returns the complete decision timeline for a session: input, "
        "context retrieved, tools called, decision evidence, and final output."
    ),
)
def get_decision_path(
    session_id: str,
    reconstructor: DecisionPathReconstructor = Depends(get_reconstructor),
):
    """Reconstruct and return the complete decision path.

    Args:
        session_id: Session to reconstruct.
        reconstructor: Injected DecisionPathReconstructor.

    Returns:
        DecisionPath with complete timeline.
    """
    try:
        return reconstructor.reconstruct(session_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found: {session_id}",
        )


@router.get(
    "/audit/sessions/{session_id}/summary",
    response_model=DecisionSummary,
    summary="Get a plain-English decision summary",
    description=(
        "Generate a human-readable summary suitable for non-technical reviewers. "
        "Grounded only in the stored audit evidence. Never exposes chain-of-thought."
    ),
)
def get_decision_summary(
    session_id: str,
    reconstructor: DecisionPathReconstructor = Depends(get_reconstructor),
    summary_service: DecisionSummaryService = Depends(get_summary_service),
):
    """Generate and return a decision summary.

    Args:
        session_id: Session to summarize.
        reconstructor: Injected reconstructor.
        summary_service: Injected summary service.

    Returns:
        DecisionSummary.
    """
    try:
        path = reconstructor.reconstruct(session_id)
        return summary_service.generate(path)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found: {session_id}",
        )
    except Exception as exc:
        logger.error("summary_generation_failed", session_id=session_id, error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Summary generation failed.",
        )


@router.get(
    "/audit/sessions/{session_id}/challenge-response",
    response_model=RegulatoryChallengeResponse,
    summary="Generate a regulatory challenge response",
    description=(
        "Generate a professional draft response to a regulatory challenge, "
        "grounded exclusively in the stored audit path."
    ),
)
def get_challenge_response(
    session_id: str,
    reconstructor: DecisionPathReconstructor = Depends(get_reconstructor),
    challenge_generator: RegulatoryChallengegenerator = Depends(get_challenge_generator),
):
    """Generate and return a regulatory challenge response.

    Args:
        session_id: Session to generate response for.
        reconstructor: Injected reconstructor.
        challenge_generator: Injected challenge generator.

    Returns:
        RegulatoryChallengeResponse.
    """
    try:
        path = reconstructor.reconstruct(session_id)
        return challenge_generator.generate(path)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session not found: {session_id}",
        )
    except Exception as exc:
        logger.error("challenge_response_failed", session_id=session_id, error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Challenge response generation failed.",
        )


# ---------------------------------------------------------------------------
# User and search queries
# ---------------------------------------------------------------------------


@router.get(
    "/audit/users/{user_id}/sessions",
    response_model=SessionListResponse,
    summary="List all sessions for a user",
)
def get_user_sessions(
    user_id: str,
    limit: int = Query(DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE),
    offset: int = Query(0, ge=0),
    repo: AuditRepository = Depends(get_repository),
):
    """Retrieve all audit sessions for a given user ID.

    Args:
        user_id: User to query.
        limit: Page size (max 100).
        offset: Pagination offset.
        repo: AuditRepository from DI.

    Returns:
        SessionListResponse with paginated results.
    """
    sessions = repo.get_sessions_by_user(user_id=user_id, limit=limit, offset=offset)

    return SessionListResponse(
        sessions=[
            AuditSessionRecord(
                session_id=s.session_id,
                user_id=s.user_id,
                trace_id=s.trace_id,
                status=s.status,
                started_at=s.started_at,
                completed_at=s.completed_at,
                created_at=s.created_at,
            )
            for s in sessions
        ],
        total=len(sessions),
        limit=limit,
        offset=offset,
    )


@router.get(
    "/audit/search",
    response_model=SessionListResponse,
    summary="Search audit sessions",
    description=(
        "Search sessions by user ID, time range, decision outcome, or status. "
        "All parameters are optional. Results are paginated (max 100 per page)."
    ),
)
def search_sessions(
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    start_time: Optional[datetime] = Query(
        None, description="Filter sessions started at or after this UTC time"
    ),
    end_time: Optional[datetime] = Query(
        None, description="Filter sessions started at or before this UTC time"
    ),
    decision: Optional[str] = Query(
        None,
        description="Filter by decision outcome (APPROVED, REJECTED, NEEDS_REVIEW)",
    ),
    session_status: Optional[str] = Query(
        None, alias="status", description="Filter by session status"
    ),
    limit: int = Query(DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE),
    offset: int = Query(0, ge=0),
    repo: AuditRepository = Depends(get_repository),
):
    """Search audit sessions with optional filters.

    Args:
        user_id: Optional user ID filter.
        start_time: Optional start of time range.
        end_time: Optional end of time range.
        decision: Optional decision outcome filter.
        session_status: Optional session status filter.
        limit: Page size.
        offset: Pagination offset.
        repo: AuditRepository from DI.

    Returns:
        SessionListResponse with matching sessions.
    """
    sessions = repo.search_sessions(
        user_id=user_id,
        start_time=start_time,
        end_time=end_time,
        decision=decision,
        status=session_status,
        limit=limit,
        offset=offset,
    )

    return SessionListResponse(
        sessions=[
            AuditSessionRecord(
                session_id=s.session_id,
                user_id=s.user_id,
                trace_id=s.trace_id,
                status=s.status,
                started_at=s.started_at,
                completed_at=s.completed_at,
                created_at=s.created_at,
            )
            for s in sessions
        ],
        total=len(sessions),
        limit=limit,
        offset=offset,
    )
