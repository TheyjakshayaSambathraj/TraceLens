"""TraceLens FastAPI application entry point.

Composition root: configures logging, builds the FastAPI app, registers
middleware, mounts routers, and wires the audit persistence pipeline on startup.

Run locally with:

    uvicorn app.main:app --reload

or via the provided Docker image (see ``docker/Dockerfile``).
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from app.api.middleware.logging_middleware import RequestContextLoggingMiddleware
from app.api.routes.health import router as health_router
from app.api.routes.audit import router as audit_router
from app.config.logging_config import configure_logging, get_logger
from app.config.settings import get_settings

settings = get_settings()
configure_logging(settings)
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage application startup and shutdown events.

    Startup sequence:
    1. Initialize database tables (idempotent).
    2. Wire the audit persistence service to the event publisher.
       Events emitted by InstrumentedAgent will be automatically captured,
       PII-redacted, and persisted to SQLite.
    3. Configure LangSmith if credentials are available.

    The persistence pipeline must be wired BEFORE any agent executions.

    Args:
        app: The FastAPI application instance.

    Yields:
        Control back to FastAPI for the lifetime of the application.
    """
    logger.info(
        "application_startup",
        app_name=settings.app_name,
        app_version=settings.app_version,
        app_env=settings.app_env,
    )

    # --- Database initialization ---
    from app.database.session import init_db, _get_session_factory
    init_db()
    logger.info("database_initialized")

    # --- Audit persistence pipeline wiring ---
    from app.observability.publisher import get_publisher
    from app.audit.persistence import wire_persistence

    publisher = get_publisher()
    session_factory = _get_session_factory()
    wire_persistence(publisher=publisher, session_factory=session_factory)
    logger.info("audit_persistence_pipeline_wired")

    # --- LangSmith configuration (optional) ---
    try:
        from app.observability.config import get_langsmith_config
        ls_config = get_langsmith_config()
        if ls_config.is_configured():
            logger.info(
                "langsmith_tracing_active",
                project=ls_config.project,
            )
        else:
            logger.info("langsmith_tracing_not_configured")
    except Exception as exc:
        # LangSmith unavailability must never block startup
        logger.warning("langsmith_config_failed", error=str(exc))

    yield

    logger.info("application_shutdown")


def create_app() -> FastAPI:
    """Build and configure the FastAPI application.

    Using an application factory keeps the app independently constructible
    in tests and keeps configuration explicit and centralized.

    Returns:
        A fully configured FastAPI application instance.
    """
    application = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "TraceLens — Enterprise AI Decision Governance Platform. "
            "Implements AIVER PS-7.1: Decision Path Auditor."
        ),
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    application.add_middleware(RequestContextLoggingMiddleware)

    if settings.cors_origins_list:
        application.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins_list,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # Health check (Phase 1)
    application.include_router(health_router, prefix=settings.api_prefix)

    # Audit + agent routes (PS-7.1)
    application.include_router(audit_router, prefix=settings.api_prefix)

    return application


app = create_app()
