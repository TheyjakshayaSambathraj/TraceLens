"""Request correlation and structured access logging middleware.

Binds ``request_id``, ``session_id``, and ``trace_id`` into structlog's
context variables for the lifetime of a single request, so that every
log line emitted anywhere during that request -- including from deeply
nested services -- is automatically tagged with them, without those
services needing to know about HTTP at all.

``request_id`` is always generated server-side. ``session_id`` and
``trace_id`` are read from inbound headers when present (so a caller or
upstream gateway can supply them for cross-service correlation) and
generated otherwise.
"""

from __future__ import annotations

import time
import uuid
from typing import Awaitable, Callable

import structlog
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

logger = structlog.get_logger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"
SESSION_ID_HEADER = "X-Session-ID"
TRACE_ID_HEADER = "X-Trace-ID"


class RequestContextLoggingMiddleware(BaseHTTPMiddleware):
    """Binds correlation IDs to structlog context and logs access records."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        """Process a single request/response cycle.

        Args:
            request: The incoming Starlette request.
            call_next: The next handler in the middleware chain.

        Returns:
            The response produced by downstream handlers, with
            correlation ID headers attached.
        """
        request_id = request.headers.get(REQUEST_ID_HEADER, str(uuid.uuid4()))
        session_id = request.headers.get(SESSION_ID_HEADER, str(uuid.uuid4()))
        trace_id = request.headers.get(TRACE_ID_HEADER, str(uuid.uuid4()))

        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            request_id=request_id,
            session_id=session_id,
            trace_id=trace_id,
        )

        start_time = time.perf_counter()
        try:
            logger.info(
                "http_request_started",
                method=request.method,
                path=request.url.path,
            )
        except Exception:
            pass  # Never let logging crash the request

        response = await call_next(request)

        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)
        try:
            logger.info(
                "http_request_completed",
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                duration_ms=duration_ms,
            )
        except Exception:
            pass  # Never let logging crash the request

        response.headers[REQUEST_ID_HEADER] = request_id
        response.headers[SESSION_ID_HEADER] = session_id
        response.headers[TRACE_ID_HEADER] = trace_id
        return response
