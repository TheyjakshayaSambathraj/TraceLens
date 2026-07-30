"""Structured logging configuration.

TraceLens uses ``structlog`` exclusively for logging. No module in this
codebase should ever call ``print()`` or use the stdlib ``logging`` module
directly for application logs -- doing so bypasses the structured,
JSON-capable pipeline configured here and breaks correlation of
``request_id`` / ``session_id`` / ``trace_id`` across log lines.

Call :func:`configure_logging` exactly once, during application startup,
before any other module emits a log record.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog

from app.config.settings import Settings


def configure_logging(settings: Settings) -> None:
    """Configure structlog (and stdlib logging as its backend) for the process.

    This wires structlog's processor pipeline on top of a stdlib
    ``logging`` handler, so that both structlog loggers and any
    third-party library that still uses stdlib ``logging`` (e.g.
    uvicorn) are rendered consistently.

    Args:
        settings: Application settings. Controls the minimum emitted log
            level and whether output is rendered as JSON or as
            human-readable console text.
    """
    log_level = getattr(logging, settings.log_level, logging.INFO)

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    renderer: Any = (
        structlog.processors.JSONRenderer()
        if settings.log_json
        else structlog.dev.ConsoleRenderer(colors=True)
    )

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers = [handler]
    root_logger.setLevel(log_level)

    # Route common noisy/duplicate third-party loggers through the same
    # handler instead of letting them configure their own.
    for noisy_logger in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logging.getLogger(noisy_logger).handlers = [handler]
        logging.getLogger(noisy_logger).propagate = False


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a structlog bound logger.

    Args:
        name: Optional logger name, conventionally the calling module's
            ``__name__``. Helps trace which component emitted a record.

    Returns:
        A structlog bound logger ready for use. Bind request-scoped
        context (``request_id``, ``session_id``, ``trace_id``) via
        ``structlog.contextvars.bind_contextvars`` rather than passing
        it manually on every call.
    """
    return structlog.get_logger(name)
