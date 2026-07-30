"""SQLAlchemy engine and session factory management.

Responsibilities:
- Create the engine from settings.database_url.
- Create all schema tables on first connection (create_all).
- Expose a session factory for dependency injection.
- Expose a context-manager ``get_db()`` for use in FastAPI dependencies.

The engine is created once as a module-level singleton (lazy, on first call)
to avoid connection overhead on import.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

import structlog
from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.config.settings import get_settings
from app.database.models import Base

logger = structlog.get_logger(__name__)

_engine: Engine | None = None
_SessionLocal: sessionmaker | None = None


def _get_engine() -> Engine:
    """Build or return the cached SQLAlchemy engine.

    SQLite specific settings:
    - ``check_same_thread=False`` is required for SQLite when multiple threads
      share the same connection (FastAPI request lifecycle).
    - WAL journal mode for better concurrent read performance.
    - Foreign key enforcement is not enabled by default in SQLite; we enable it
      explicitly for referential integrity.
    """
    global _engine
    if _engine is None:
        settings = get_settings()
        connect_args = {}
        if "sqlite" in settings.database_url:
            connect_args["check_same_thread"] = False

        _engine = create_engine(
            settings.database_url,
            connect_args=connect_args,
            # Connection pool size for SQLite should be 1 in production,
            # but StaticPool is better for in-memory test DBs.
            echo=False,
        )

        # Enable SQLite WAL mode and FK support on each new connection
        if "sqlite" in settings.database_url:
            @event.listens_for(_engine, "connect")
            def _set_sqlite_pragmas(dbapi_conn, connection_record):
                cursor = dbapi_conn.cursor()
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.close()

        logger.info(
            "database_engine_created",
            url=settings.database_url.split("///")[-1],  # Safe: no credentials
        )

    return _engine


def _get_session_factory() -> sessionmaker:
    """Return the cached session factory."""
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(
            bind=_get_engine(),
            autocommit=False,
            autoflush=False,
        )
    return _SessionLocal


def init_db() -> None:
    """Create all database tables if they do not already exist.

    Idempotent — safe to call on every startup.
    Must be called before the first request is handled.
    """
    engine = _get_engine()
    Base.metadata.create_all(bind=engine)
    logger.info("database_tables_initialized")


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency that provides a transactional database session.

    Yields a Session within a try/finally block so the session is always
    closed even if the request fails. Commits are the caller's responsibility.

    Yields:
        SQLAlchemy Session.
    """
    factory = _get_session_factory()
    db = factory()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def db_session() -> Generator[Session, None, None]:
    """Context manager providing a database session with auto-commit/rollback.

    Use this outside of FastAPI dependency injection (e.g., in background
    tasks, startup hooks, or tests).

    Yields:
        SQLAlchemy Session.
    """
    factory = _get_session_factory()
    db = factory()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def reset_engine_for_testing(url: str = "sqlite:///:memory:") -> None:
    """Replace the engine with a fresh in-memory database.

    ONLY for use in tests. Creates all tables on the new engine.

    For SQLite in-memory databases, uses StaticPool to ensure all
    connections share the same database instance (otherwise each
    new connection gets a separate empty database).

    Args:
        url: SQLAlchemy connection string for the test database.
    """
    global _engine, _SessionLocal

    if url == "sqlite:///:memory:":
        from sqlalchemy.pool import StaticPool
        _engine = create_engine(
            url,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
    else:
        connect_args = {"check_same_thread": False} if "sqlite" in url else {}
        _engine = create_engine(url, connect_args=connect_args)

    if "sqlite" in url:
        @event.listens_for(_engine, "connect")
        def _set_pragmas(dbapi_conn, connection_record):
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

    Base.metadata.create_all(bind=_engine)
    _SessionLocal = sessionmaker(bind=_engine, autocommit=False, autoflush=False)
    logger.debug("test_database_engine_reset", url=url)
