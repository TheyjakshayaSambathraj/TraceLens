"""FastAPI dependency injection for audit infrastructure.

All shared resources (database sessions, services, agent) are wired here.
FastAPI routes depend on these functions, never on module-level globals.

This keeps the application testable: tests can override any dependency via
app.dependency_overrides without touching production code.
"""

from __future__ import annotations

from typing import Generator, Optional

import structlog
from fastapi import Depends
from sqlalchemy.orm import Session

from app.config.settings import Settings, get_settings


def get_settings_dependency() -> Settings:
    """Provide application settings as a FastAPI dependency.

    Returns:
        Cached Settings instance.
    """
    return get_settings()

from app.database.session import get_db
from app.database.repository import AuditRepository
from app.privacy.redactor import PIIRedactor, get_redactor
from app.audit.reconstructor import DecisionPathReconstructor
from app.audit.summary import DecisionSummaryService
from app.audit.challenge import RegulatoryChallengegenerator

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

def get_repository(db: Session = Depends(get_db)) -> AuditRepository:
    """Provide an AuditRepository for a request.

    Args:
        db: SQLAlchemy session from get_db dependency.

    Returns:
        AuditRepository instance.
    """
    return AuditRepository(db)


# ---------------------------------------------------------------------------
# Privacy
# ---------------------------------------------------------------------------

def get_pii_redactor() -> PIIRedactor:
    """Provide the PIIRedactor singleton.

    Returns:
        PIIRedactor instance.
    """
    return get_redactor()


# ---------------------------------------------------------------------------
# Audit services
# ---------------------------------------------------------------------------

def get_reconstructor(db: Session = Depends(get_db)) -> DecisionPathReconstructor:
    """Provide a DecisionPathReconstructor for a request.

    Args:
        db: SQLAlchemy session.

    Returns:
        DecisionPathReconstructor instance.
    """
    return DecisionPathReconstructor(db)


def get_summary_service() -> DecisionSummaryService:
    """Provide a DecisionSummaryService.

    The LLM provider is optionally injected. If not configured (no API key),
    the service uses the structured fallback generator.

    Returns:
        DecisionSummaryService instance.
    """
    try:
        from app.services.llm_provider import create_llm_provider
        from app.config.settings import get_settings
        settings = get_settings()
        if settings.google_api_key or settings.groq_api_key:
            provider = create_llm_provider(settings)
            return DecisionSummaryService(llm_provider=provider)
    except Exception as exc:
        logger.warning("summary_service_llm_unavailable", error=str(exc))

    return DecisionSummaryService(llm_provider=None)


def get_challenge_generator() -> RegulatoryChallengegenerator:
    """Provide a RegulatoryChallengegenerator.

    Returns:
        RegulatoryChallengegenerator instance.
    """
    try:
        from app.services.llm_provider import create_llm_provider
        from app.config.settings import get_settings
        settings = get_settings()
        if settings.google_api_key or settings.groq_api_key:
            provider = create_llm_provider(settings)
            return RegulatoryChallengegenerator(llm_provider=provider)
    except Exception as exc:
        logger.warning("challenge_generator_llm_unavailable", error=str(exc))

    return RegulatoryChallengegenerator(llm_provider=None)


# ---------------------------------------------------------------------------
# Agent (lazy initialization to avoid startup cost in tests)
# ---------------------------------------------------------------------------

_agent_instance = None
_instrumented_agent_instance = None


def get_instrumented_agent():
    """Provide the InstrumentedAgent singleton.

    Lazy-initializes the agent on first request. This avoids loading the
    embedding model during test collection.

    Returns:
        InstrumentedAgent instance.

    Raises:
        RuntimeError: If the agent cannot be initialized (e.g., missing config).
    """
    global _agent_instance, _instrumented_agent_instance

    if _instrumented_agent_instance is not None:
        return _instrumented_agent_instance

    try:
        from app.config.settings import get_settings
        from app.rag.retriever import RetrieverService
        from app.rag.ingest import DocumentIngestionPipeline
        from app.services.employee import EmployeeService
        from app.services.llm_provider import create_llm_provider
        from app.agent.graph import LeaveDecisionAgent
        from app.observability.instrumentation import InstrumentedAgent
        from app.observability.publisher import get_publisher
        from pathlib import Path

        settings = get_settings()

        # Build or load vector store
        data_dir = Path("data")
        vector_store_path = data_dir / ".vectorstore"

        ingester = DocumentIngestionPipeline(vector_store_path=str(vector_store_path))
        retriever_service = RetrieverService(vector_store_path=str(vector_store_path))

        if vector_store_path.exists():
            retriever_service.load(str(vector_store_path))
        else:
            policy_dir = data_dir / "policies"
            if policy_dir.exists():
                vector_store = ingester.ingest(str(policy_dir))
                ingester.save_vector_store(vector_store)
                retriever_service.load(str(vector_store_path))

        employee_service = EmployeeService()
        llm_provider = create_llm_provider(settings)
        agent = LeaveDecisionAgent(retriever_service, employee_service, llm_provider)

        _instrumented_agent_instance = InstrumentedAgent(
            agent=agent,
            publisher=get_publisher(),
        )

        logger.info("instrumented_agent_initialized")
        return _instrumented_agent_instance

    except Exception as exc:
        logger.error("instrumented_agent_initialization_failed", error=str(exc))
        raise RuntimeError(f"Agent initialization failed: {exc}") from exc
