# TraceLens Architecture

This document will hold the detailed architecture reference (component
diagrams, event schema, and audit data model) as later phases implement
the agent, audit, RAG, and persistence layers.

## Phase 1 scope

Phase 1 establishes the project skeleton only:

- Configuration (`app/config/settings.py`)
- Structured logging (`app/config/logging_config.py`)
- Dependency injection container (`app/config/container.py`)
- FastAPI application bootstrap (`app/main.py`)
- Liveness health endpoint (`GET /api/v1/health`)
- Docker packaging

No agent, audit, RAG, or database logic is implemented yet.

## High-level architecture (target state)

```
Frontend
   |
FastAPI REST API
   |
Decision Audit Service
   |
Instrumentation Layer
   |
LangGraph Agent
   |
Retriever
   |
Tools
   |
LLM
   |
LangSmith
   |
SQLite
```

The agent never communicates with the database directly. The agent only
emits execution events onto the event bus; the audit package and
persistence layer are independent consumers of those events.
