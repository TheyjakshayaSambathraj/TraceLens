"""LangSmith tracing configuration and initialization.

LangSmith is the primary observability platform for TraceLens.
This module configures LangSmith integration using the official SDK
mechanisms and environment-based configuration.

Configuration is loaded from:
- LANGCHAIN_TRACING_V2: Enable/disable tracing
- LANGCHAIN_API_KEY: LangSmith API key
- LANGCHAIN_PROJECT: Project name for grouping traces
- LANGCHAIN_ENDPOINT: Custom endpoint (optional)

Do NOT hardcode credentials.
"""

from __future__ import annotations

import os
import structlog
from typing import Optional
from functools import lru_cache

from app.config.settings import get_settings


logger = structlog.get_logger(__name__)


class LangSmithConfig:
    """LangSmith tracing configuration.

    Attributes:
        enabled: Whether LangSmith tracing is active.
        api_key: LangSmith API key (loaded from env).
        project: Project name for grouping traces.
        endpoint: Custom LangSmith endpoint (optional).
    """

    def __init__(
        self,
        enabled: bool = False,
        api_key: Optional[str] = None,
        project: str = "tracelens",
        endpoint: Optional[str] = None,
    ):
        """Initialize LangSmith configuration.

        Args:
            enabled: Whether tracing is enabled.
            api_key: LangSmith API key.
            project: Project name for traces.
            endpoint: Custom endpoint URL.
        """
        self.enabled = enabled
        self.api_key = api_key
        self.project = project
        self.endpoint = endpoint

    @classmethod
    def from_env(cls) -> LangSmithConfig:
        """Load configuration from environment variables.

        Reads:
        - LANGCHAIN_TRACING_V2: "true" or "false"
        - LANGCHAIN_API_KEY: API key string
        - LANGCHAIN_PROJECT: Project name
        - LANGCHAIN_ENDPOINT: Custom endpoint

        Returns:
            LangSmithConfig instance with environment values.
        """
        settings = get_settings()
        enabled = settings.langchain_tracing_v2
        api_key = settings.langchain_api_key
        project = settings.langchain_project
        endpoint = settings.langchain_endpoint

        config = cls(
            enabled=enabled,
            api_key=api_key,
            project=project,
            endpoint=endpoint,
        )

        if config.enabled:
            if not config.api_key:
                logger.warning(
                    "langsmith_tracing_enabled_but_no_api_key",
                    hint="Set LANGCHAIN_API_KEY environment variable to enable tracing",
                )
                config.enabled = False

        logger.info(
            "langsmith_config_loaded",
            enabled=config.enabled,
            project=config.project,
            has_api_key=bool(config.api_key),
        )

        return config

    def configure_langchain(self) -> None:
        """Apply configuration to LangChain/LangSmith.

        Sets environment variables that the LangChain SDK respects.
        This should be called early in application initialization.
        """
        if self.enabled and self.api_key:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_PROJECT"] = self.project
            os.environ["LANGCHAIN_API_KEY"] = self.api_key
            if self.endpoint:
                os.environ["LANGCHAIN_ENDPOINT"] = self.endpoint
            logger.info("langsmith_configured_for_langchain")
        else:
            # Ensure tracing is disabled if no API key
            os.environ["LANGCHAIN_TRACING_V2"] = "false"
            logger.info("langsmith_tracing_disabled")

    def is_configured(self) -> bool:
        """Check if LangSmith is properly configured.

        Returns:
            True if enabled and has an API key.
        """
        return self.enabled and bool(self.api_key)


@lru_cache(maxsize=1)
def get_langsmith_config() -> LangSmithConfig:
    """Get the cached LangSmith configuration.

    Configuration is loaded once from environment at first call.
    In testing, this can be mocked via dependency injection.

    Returns:
        LangSmithConfig instance.
    """
    config = LangSmithConfig.from_env()
    config.configure_langchain()
    return config
