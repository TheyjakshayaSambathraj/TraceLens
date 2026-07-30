"""Event publisher abstraction for execution instrumentation.

The publisher is a lightweight in-process event bus that decouples
event emission from event consumption.

Future phases may replace the in-process implementation with a
distributed message broker (Kafka, Redis Streams, etc.).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, List
import structlog

from app.observability.events import ExecutionEvent


logger = structlog.get_logger(__name__)


class EventPublisher(ABC):
    """Abstract base class for event publication.

    Implementations may be in-process, asynchronous, or distributed.
    """

    @abstractmethod
    def publish(self, event: ExecutionEvent) -> None:
        """Publish an execution event.

        Args:
            event: ExecutionEvent to publish.
        """
        pass

    @abstractmethod
    def subscribe(
        self, event_type: str, handler: Callable[[ExecutionEvent], None]
    ) -> None:
        """Subscribe to events of a specific type.

        Args:
            event_type: EventType string to subscribe to.
            handler: Callable that processes the event.
        """
        pass


class InProcessEventPublisher(EventPublisher):
    """In-process event publisher with simple subscription model.

    Events are published immediately to all subscribed handlers.
    This implementation is suitable for development and Phase 3.

    Attributes:
        _subscribers: Mapping from event_type to list of handlers.
    """

    def __init__(self):
        """Initialize the in-process publisher."""
        self._subscribers: dict[str, List[Callable[[ExecutionEvent], None]]] = {}

    def publish(self, event: ExecutionEvent) -> None:
        """Publish an execution event.

        Immediately calls all handlers subscribed to the event type.

        Args:
            event: ExecutionEvent to publish.
        """
        event_type = event.event_type.value

        logger.debug(
            "publishing_event",
            event_type=event_type,
            session_id=event.session_id,
            event_id=event.event_id,
        )

        # Publish to type-specific handlers
        handlers = self._subscribers.get(event_type, [])
        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                # Never let subscriber errors propagate
                logger.error(
                    "event_handler_failed",
                    event_type=event_type,
                    error=str(e),
                    exc_info=True,
                )

        # Also publish to wildcard handlers (event_type="*")
        wildcard_handlers = self._subscribers.get("*", [])
        for handler in wildcard_handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error(
                    "wildcard_handler_failed",
                    event_type=event_type,
                    error=str(e),
                    exc_info=True,
                )

    def subscribe(
        self, event_type: str, handler: Callable[[ExecutionEvent], None]
    ) -> None:
        """Subscribe to events.

        Args:
            event_type: EventType string to subscribe to. Use "*" for all events.
            handler: Callable that processes the event.
        """
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []

        self._subscribers[event_type].append(handler)
        logger.debug("event_handler_subscribed", event_type=event_type)


# Global publisher instance for use throughout the application.
# In testing, this can be mocked or replaced via dependency injection.
_default_publisher: EventPublisher | None = None


def get_publisher() -> EventPublisher:
    """Get the default event publisher instance.

    Returns:
        InProcessEventPublisher instance. In testing, this can be overridden.
    """
    global _default_publisher
    if _default_publisher is None:
        _default_publisher = InProcessEventPublisher()
    return _default_publisher


def set_publisher(publisher: EventPublisher) -> None:
    """Set the global event publisher instance.

    Used in testing to inject a mock publisher or alternate implementation.

    Args:
        publisher: EventPublisher instance to use globally.
    """
    global _default_publisher
    _default_publisher = publisher


def reset_publisher() -> None:
    """Reset the global event publisher to None.

    Used in test cleanup.
    """
    global _default_publisher
    _default_publisher = None
