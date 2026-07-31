"""Application settings.

All runtime configuration for TraceLens is centralized here using
``pydantic-settings``. Nothing in the rest of the codebase should read
environment variables directly -- every module that needs configuration
depends on a :class:`Settings` instance obtained via :func:`get_settings`.

This keeps configuration:
    * Typed -- invalid or missing values fail fast at startup.
    * Testable -- settings can be constructed explicitly in unit tests
      without touching the real environment.
    * Centralized -- there is exactly one place that knows about
      environment variable names.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Strongly typed application configuration.

    Values are loaded from environment variables and, if present, a
    ``.env`` file at the project root. See ``.env.example`` for the full
    list of supported variables and their intended purpose.

    Attributes:
        app_name: Human readable application name, used in logs and docs.
        app_env: Deployment environment. Controls behaviors such as
            reload-on-change and verbose error responses.
        app_version: Semantic version of the running application.
        api_prefix: URL prefix under which all versioned API routes are
            mounted.
        host: Interface the ASGI server binds to.
        port: Port the ASGI server listens on.
        log_level: Minimum severity of log records that are emitted.
        log_json: Whether logs are emitted as JSON (recommended for
            production/log-aggregation) or as human-readable console
            output (recommended for local development).
        google_api_key: API key for Google Gemini. Required once the
            agent layer is implemented; not required for Phase 1.
        langchain_api_key: API key for LangSmith tracing.
        langchain_project: LangSmith project name that traces are
            grouped under.
        langchain_tracing_v2: Whether LangSmith tracing is enabled.
        database_url: SQLAlchemy connection string for persistence.
        model_name: Identifier of the LLM used by the agent layer.
        cors_allow_origins: Comma-separated list of origins allowed to
            call the API from a browser. Defaults to no origins.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Application metadata -------------------------------------------------
    app_name: str = Field(default="TraceLens")
    app_env: Literal["local", "development", "staging", "production"] = Field(
        default="local"
    )
    app_version: str = Field(default="0.1.0")
    api_prefix: str = Field(default="/api/v1")

    # --- Server -----------------------------------------------------------------
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)

    # --- Logging ------------------------------------------------------------------
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO"
    )
    log_json: bool = Field(default=True)

    # --- Third-party integrations (consumed starting in later phases) ------------
    google_api_key: str | None = Field(default=None)
    groq_api_key: str | None = Field(default=None)
    langchain_api_key: str | None = Field(default=None)
    langchain_project: str = Field(default="tracelens")
    langchain_tracing_v2: bool = Field(default=False)
    langchain_endpoint: str | None = Field(default=None)

    # --- Persistence (consumed starting in the database phase) -------------------
    database_url: str = Field(default="sqlite:///./tracelens.db")

    # --- Model configuration -------------------------------------------------------
    llm_provider: str = Field(default="gemini")
    model_name: str = Field(default="gemini-2.5-flash")

    # --- CORS ------------------------------------------------------------------------
    cors_allow_origins: str = Field(default="")

    @field_validator("log_level", mode="before")
    @classmethod
    def _normalize_log_level(cls, value: str) -> str:
        """Uppercase log level so ``info`` and ``INFO`` are both accepted."""
        return value.upper() if isinstance(value, str) else value

    @property
    def cors_origins_list(self) -> list[str]:
        """Parse :attr:`cors_allow_origins` into a list of origins.

        Returns:
            A list of origin strings. Empty entries produced by trailing
            commas or blank configuration are filtered out.
        """
        return [origin.strip() for origin in self.cors_allow_origins.split(",") if origin.strip()]

    @property
    def is_production(self) -> bool:
        """Whether the application is running in the production environment."""
        return self.app_env == "production"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide cached :class:`Settings` instance.

    Using ``lru_cache`` gives us a cheap singleton without introducing
    global mutable state: the settings object is constructed once, is
    immutable in practice, and can still be overridden in tests via
    FastAPI's dependency override mechanism (``app.dependency_overrides``).

    Returns:
        The cached application settings.
    """
    return Settings()
