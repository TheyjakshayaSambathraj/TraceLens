"""Tests for the /health endpoint."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.config.settings import get_settings


def test_health_returns_200(client: TestClient) -> None:
    """The health endpoint should respond with HTTP 200."""
    settings = get_settings()
    response = client.get(f"{settings.api_prefix}/health")
    assert response.status_code == 200


def test_health_returns_expected_payload(client: TestClient) -> None:
    """The health endpoint should report status and app metadata."""
    settings = get_settings()
    response = client.get(f"{settings.api_prefix}/health")
    payload = response.json()

    assert payload["status"] == "ok"
    assert payload["app_name"] == settings.app_name
    assert payload["app_version"] == settings.app_version
    assert payload["app_env"] == settings.app_env


def test_health_sets_correlation_id_headers(client: TestClient) -> None:
    """The health endpoint response should carry correlation ID headers."""
    settings = get_settings()
    response = client.get(f"{settings.api_prefix}/health")

    assert "X-Request-ID" in response.headers
    assert "X-Session-ID" in response.headers
    assert "X-Trace-ID" in response.headers
