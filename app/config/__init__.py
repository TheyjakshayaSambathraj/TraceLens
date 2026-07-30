"""Configuration package.

Exposes the application settings singleton, structured logging setup,
and the dependency injection container used throughout TraceLens.
"""

from app.config.settings import Settings, get_settings

__all__ = ["Settings", "get_settings"]
