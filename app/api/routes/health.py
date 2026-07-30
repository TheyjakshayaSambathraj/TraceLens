"""Health check endpoint.

Used by orchestrators (Docker, Render, Kubernetes) and uptime monitors
to determine whether the process is alive and correctly configured.
Phase 1 reports liveness only; later phases may extend this with
readiness checks against the database and vector store.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.api.dependencies import get_settings_dependency
from app.config.settings import Settings

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    """Response payload for the health check endpoint.

    Attributes:
        status: Literal health indicator, ``"ok"`` when the service is up.
        app_name: The running application's name.
        app_version: The running application's semantic version.
        app_env: The deployment environment the process is running in.
    """

    status: str = Field(default="ok")
    app_name: str
    app_version: str
    app_env: str


@router.get("/health", response_model=HealthResponse, summary="Liveness check")
async def get_health(
    settings: Annotated[Settings, Depends(get_settings_dependency)],
) -> HealthResponse:
    """Report basic liveness information about the running service.

    Args:
        settings: Application settings, injected.

    Returns:
        A :class:`HealthResponse` confirming the process is up.
    """
    return HealthResponse(
        app_name=settings.app_name,
        app_version=settings.app_version,
        app_env=settings.app_env,
    )
