"""LLM provider factory.

Provides access to language models with configuration-driven setup.
Supports testability through dependency injection.
"""

from __future__ import annotations

import structlog
from typing import Protocol

from langchain_groq import ChatGroq
from langchain_core.language_models import BaseLanguageModel

from app.config.settings import Settings

logger = structlog.get_logger(__name__)


class LLMProvider(Protocol):
    """Protocol for LLM providers (supports alternative implementations)."""

    def get_model(self) -> BaseLanguageModel:
        """Get the configured LLM instance."""
        ...


class GeminiProvider:
    """Google Gemini LLM provider.

    Constructs a ChatGoogleGenerativeAI instance using application
    configuration. The API key and model name are sourced from settings
    and can be overridden for testing.
    """

    def __init__(self, settings: Settings):
        """Initialize the Gemini provider.

        Args:
            settings: Application settings containing API key and model name.

        Raises:
            ValueError: If the required configuration is missing.
        """
        if not settings.google_api_key:
            logger.error("missing_google_api_key")
            raise ValueError(
                "GOOGLE_API_KEY environment variable is required "
                "for the Gemini LLM provider"
            )

        if not settings.model_name:
            logger.error("missing_model_name")
            raise ValueError("MODEL_NAME configuration is required")

        self.settings = settings
        self.model_name = settings.model_name
        self._model: BaseLanguageModel | None = None

    def get_model(self) -> BaseLanguageModel:
        """Get or create the ChatGoogleGenerativeAI instance.

        Uses lazy initialization to defer API client creation until
        the model is actually needed.

        Returns:
            ChatGoogleGenerativeAI LLM instance.
        """
        if self._model is None:
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
            except ImportError as exc:
                raise ImportError(
                    "langchain-google-genai package is required for GeminiProvider. "
                    "Install it via pip install langchain-google-genai."
                ) from exc

            logger.info(
                "initializing_gemini_model",
                model=self.model_name,
            )
            self._model = ChatGoogleGenerativeAI(
                model=self.model_name,
                google_api_key=self.settings.google_api_key,
                temperature=0.7,
            )

        return self._model


class GroqProvider:
    """Groq LLM provider.

    Constructs a ChatGroq instance using application
    configuration. The API key and model name are sourced from settings.
    """

    def __init__(self, settings: Settings):
        """Initialize the Groq provider.

        Args:
            settings: Application settings containing API key and model name.

        Raises:
            ValueError: If the required configuration is missing.
        """
        if not settings.groq_api_key:
            logger.error("missing_groq_api_key")
            raise ValueError(
                "GROQ_API_KEY environment variable is required "
                "for the Groq LLM provider"
            )

        if not settings.model_name:
            logger.error("missing_model_name")
            raise ValueError("MODEL_NAME configuration is required")

        self.settings = settings
        if not settings.model_name or settings.model_name.startswith("gemini"):
            self.model_name = "llama-3.3-70b-versatile"
        else:
            self.model_name = settings.model_name
        self._model: BaseLanguageModel | None = None

    def get_model(self) -> BaseLanguageModel:
        """Get or create the ChatGroq instance.

        Returns:
            ChatGroq LLM instance.
        """
        if self._model is None:
            logger.info(
                "initializing_groq_model",
                model=self.model_name,
            )
            self._model = ChatGroq(
                model=self.model_name,
                groq_api_key=self.settings.groq_api_key,
                temperature=0.7,
            )

        return self._model


class MockLLMProvider:
    """Mock LLM provider for testing.

    Returns a mock LLM that doesn't require API credentials and
    produces deterministic responses. Useful for unit tests and
    demonstrations without network access.
    """

    def __init__(self, model_name: str = "mock"):
        """Initialize the mock provider.

        Args:
            model_name: Display name for the mock model.
        """
        self.model_name = model_name
        logger.info("mock_llm_provider_initialized", model=model_name)

    def get_model(self):
        """Get the mock LLM instance.

        Returns:
            Mock LLM instance (FakeListLLM).
        """
        from langchain_community.llms.fake import FakeListLLM

        # Pre-defined responses for testing
        responses = [
            '{"decision": "APPROVED", "reason": "Request is within policy", "policy_references": ["Section 2.1"], "evidence": ["Leave balance sufficient"]}',
            '{"decision": "REJECTED", "reason": "Exceeds maximum", "policy_references": ["Section 2.3"], "evidence": ["15 days > 10 days max"]}',
            '{"decision": "NEEDS_REVIEW", "reason": "Insufficient information", "policy_references": [], "evidence": []}',
        ]

        return FakeListLLM(responses=responses)


def create_llm_provider(settings: Settings, use_mock: bool = False) -> LLMProvider:
    """Factory function for creating LLM providers.

    Args:
        settings: Application settings.
        use_mock: If True, returns a mock provider instead of Gemini.

    Returns:
        LLMProvider instance (Gemini or Mock).

    Raises:
        ValueError: If configuration is invalid (for real provider).
    """
    if use_mock:
        logger.info("using_mock_llm_provider")
        return MockLLMProvider()

    provider_name = settings.llm_provider.lower()
    if provider_name == "groq" or (not settings.google_api_key and settings.groq_api_key):
        logger.info("using_groq_llm_provider")
        return GroqProvider(settings)

    logger.info("using_gemini_llm_provider")
    return GeminiProvider(settings)
