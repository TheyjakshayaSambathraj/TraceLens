"""Application dependency injection container.

TraceLens favors explicit, constructor-based dependency injection over a
"magic" service locator. :class:`Container` is the single composition
root: it is responsible for constructing shared, expensive-to-create
objects (settings, and -- starting in later phases -- the DB session
factory, vector store, LangSmith client, event bus, etc.) exactly once
and handing out references to them.

FastAPI route handlers never instantiate services directly; they declare
a dependency on a provider function (see ``app/api/dependencies.py``)
which in turn pulls from the container. This keeps business logic
decoupled from *how* a dependency is constructed, which is what makes
each service independently testable -- tests can construct a
:class:`Container` (or a subset of providers) with fakes/mocks instead
of real infrastructure.

Phase 1 only wires application settings. Later phases will extend this
container with providers for the database session factory, the vector
store, the event bus, and the LangGraph agent -- without changing how
existing consumers obtain their dependencies.
"""

from __future__ import annotations

from functools import lru_cache

from app.config.settings import Settings, get_settings


class Container:
    """Composition root holding process-wide singleton dependencies.

    Attributes:
        settings: The application's typed configuration.
    """

    def __init__(self, settings: Settings) -> None:
        """Initialize the container.

        Args:
            settings: Application settings to expose to consumers and to
                use when constructing other singletons in later phases.
        """
        self.settings = settings

    # Future phases add provider methods here, e.g.:
    #
    #     def database_session_factory(self) -> sessionmaker: ...
    #     def event_bus(self) -> EventBus: ...
    #     def vector_store(self) -> VectorStore: ...
    #
    # Each provider should be responsible for constructing exactly one
    # collaborator and its own dependencies, following the Factory
    # pattern, so route-level dependencies stay one-line lookups.


@lru_cache(maxsize=1)
def get_container() -> Container:
    """Return the process-wide cached :class:`Container` instance.

    Returns:
        The cached dependency injection container.
    """
    return Container(settings=get_settings())
