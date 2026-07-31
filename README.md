# TraceLens

**Enterprise AI Decision Governance Platform — AIVER PS-7.1**

TraceLens audits the decisions made by AI agents. It captures every step an agent takes — context retrieval, tool calls, reasoning, and final output — reconstructs the decision path, and makes it inspectable, summarizable, and challengeable for governance, compliance, and regulatory review.

> **Status: Complete Implementation (Milestones A–J)**
> All PS-7.1 Decision Path Auditor capabilities are implemented and tested.

---

## What It Does

Given a session ID, TraceLens answers the six PS-7.1 governance questions:

| Question | How TraceLens Answers |
|---|---|
| What request was received? | `INPUT_RECEIVED` event — exact request text |
| What context was retrieved? | `RETRIEVAL_COMPLETED` event — document count, sources |
| What employee data was consulted? | `TOOL_COMPLETED` event — tool name, response field names |
| What decision was reached? | `DECISION_COMPLETED` event — outcome, reason, policy refs |
| What evidence supported it? | Evidence list stored in the decision record |
| Why that decision? | Policy references + evidential basis |

And generates:
- A **plain-English summary** for non-technical reviewers
- A **regulatory challenge response** for compliance submissions

---

## Architecture

```
User Request
     │
     ▼
InstrumentedAgent          ← wraps LeaveDecisionAgent
     │  emits ExecutionEvents
     ▼
InProcessEventPublisher    ← in-process event bus
     │  subscribers:
     │   └─ AuditPersistenceService
     │         │  1. PII redaction (PIIRedactor)
     │         │  2. SQLite persistence (AuditRepository)
     │         ▼
     │       SQLite ─── tracelens.db
     │
     ▼ (separate read path)
FastAPI REST API
     │
     ├─ GET /audit/sessions/{id}
     ├─ GET /audit/sessions/{id}/decision-path   ← DecisionPathReconstructor
     ├─ GET /audit/sessions/{id}/summary         ← DecisionSummaryService
     ├─ GET /audit/sessions/{id}/challenge-response  ← RegulatoryChallengegenerator
     ├─ GET /audit/users/{id}/sessions
     └─ GET /audit/search
     │
     ▼
Streamlit Dashboard
(http://localhost:8501)
```

**Key design invariants:**
- The agent emits events. It does **not** write to the database directly.
- PII is redacted **before** any data reaches SQLite.
- LangSmith is an observability integration, **not** the system of record.
- The audit store is queryable independently of LangSmith availability.
- Hidden chain-of-thought is **never** captured or exposed.

---

## Project Layout

```
tracelens/
├── app/
│   ├── api/
│   │   ├── routes/
│   │   │   ├── audit.py           # All PS-7.1 audit + agent endpoints
│   │   │   └── health.py          # Liveness check
│   │   ├── middleware/            # Correlation ID + access logging
│   │   └── dependencies.py        # FastAPI DI providers
│   ├── agent/                     # LangGraph decision agent
│   ├── audit/
│   │   ├── events.py              # Pydantic models (DecisionPath, etc.)
│   │   ├── timeline.py            # TimelineBuilder
│   │   ├── reconstructor.py       # DecisionPathReconstructor
│   │   ├── summary.py             # DecisionSummaryService
│   │   ├── challenge.py           # RegulatoryChallengegenerator
│   │   └── persistence.py         # AuditPersistenceService (event bus → SQLite)
│   ├── database/
│   │   ├── models.py              # SQLAlchemy ORM models
│   │   ├── repository.py          # AuditRepository (all DB access)
│   │   └── session.py             # Engine singleton + get_db()
│   ├── observability/
│   │   ├── events.py              # Typed ExecutionEvent classes
│   │   ├── publisher.py           # InProcessEventPublisher
│   │   └── instrumentation.py     # InstrumentedAgent wrapper
│   ├── privacy/
│   │   └── redactor.py            # PIIRedactor (email, phone, IP, gov-ID, name)
│   ├── rag/                       # Ingestion + FAISS retriever
│   ├── services/                  # Employee service, LLM provider
│   ├── config/                    # Settings, logging
│   └── main.py                    # FastAPI application + lifespan
├── dashboard/
│   └── app.py                     # Streamlit governance dashboard
├── tests/
│   ├── unit/
│   │   ├── test_pii_redactor.py
│   │   ├── test_audit_repository.py
│   │   ├── test_reconstructor.py
│   │   └── test_persistence_pipeline.py
│   └── integration/
│       ├── test_audit_api.py
│       └── test_e2e_audit.py
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── data/
│   └── policies/                  # HR policy documents (Markdown)
├── requirements.txt
├── pyproject.toml
├── .env.example
└── README.md
```

---

## Getting Started

### Prerequisites

- Python 3.12+
- Docker (optional)
- Google Gemini API key (for live agent execution)
- LangSmith API key (optional, for observability)

### Local Setup

```bash
# 1. Create virtual environment
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate # Linux/Mac

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env and set GOOGLE_API_KEY at minimum

# 4. Start the API
uvicorn app.main:app --reload

# 5. Verify it's running
curl http://localhost:8000/api/v1/health

# 6. Start the Streamlit dashboard (separate terminal)
streamlit run dashboard/app.py
```

### Running with Docker

```bash
cp .env.example .env
# Edit .env to set GOOGLE_API_KEY

docker compose -f docker/docker-compose.yml up --build
```

- **API**: http://localhost:8000
- **Dashboard**: http://localhost:8501
- **API Docs**: http://localhost:8000/docs

### Running Tests

```bash
# Full test suite
pytest

# New audit tests only
pytest tests/unit/test_pii_redactor.py \
       tests/unit/test_audit_repository.py \
       tests/unit/test_reconstructor.py \
       tests/unit/test_persistence_pipeline.py \
       tests/integration/test_audit_api.py \
       tests/integration/test_e2e_audit.py \
       -v
```

---

## API Reference

### Run the Leave Decision Agent

```http
POST /api/v1/agent/decide
Content-Type: application/json

{
  "request": "Can employee EMP-001 take 15 consecutive days of leave?",
  "user_id": "USER-001",
  "session_id": null
}
```

Response includes `session_id` and `audit_url` for follow-up queries.

### Query the Audit Trail

```http
# Session record
GET /api/v1/audit/sessions/{session_id}

# Full decision timeline
GET /api/v1/audit/sessions/{session_id}/decision-path

# Plain-English summary
GET /api/v1/audit/sessions/{session_id}/summary

# Regulatory challenge response
GET /api/v1/audit/sessions/{session_id}/challenge-response

# User session history
GET /api/v1/audit/users/{user_id}/sessions?limit=20&offset=0

# Search with filters
GET /api/v1/audit/search?user_id=USER-001&decision=APPROVED&status=COMPLETED
```

---

## Configuration

All configuration is environment-variable driven. See `.env.example` for the full list.

| Variable | Purpose | Required |
|---|---|:---:|
| `GOOGLE_API_KEY` | Gemini API key (for agent execution) | Optional (Required if using Gemini) |
| `GROQ_API_KEY` | Groq API key (for alternative LLM provider) | Optional (Required if using Groq) |
| `LLM_PROVIDER` | LLM provider (`gemini` or `groq`) | No (default: `groq` or `gemini`) |
| `MODEL_NAME` | LLM model identifier | No (default: `gemini-2.5-flash` or `llama-3.3-70b-versatile`) |
| `LANGCHAIN_API_KEY` | LangSmith API key | No |
| `LANGCHAIN_PROJECT` | LangSmith project name | No |
| `LANGCHAIN_TRACING_V2` | Enable LangSmith tracing | No |
| `DATABASE_URL` | SQLAlchemy connection string | No (default: SQLite) |
| `APP_ENV` | `local` / `development` / `staging` / `production` | No |
| `LOG_LEVEL` | `DEBUG` / `INFO` / `WARNING` / `ERROR` | No |
| `LOG_JSON` | JSON logs (true) vs console logs (false) | No |
| `CORS_ALLOW_ORIGINS` | Comma-separated allowed origins | No |

---

## Cloud Deployment (Render Blueprint)

TraceLens includes a production-ready `render.yaml` infrastructure blueprint for deploying both the FastAPI backend and Streamlit dashboard to **Render**.

1. Connect your repository (`https://github.com/TheyjakshayaSambathraj/TraceLens.git`) on [Render](https://render.com).
2. Render will automatically detect `render.yaml` and configure:
   - `tracelens-api`: FastAPI web service with persistent disk for SQLite database `/srv/app/data/tracelens.db`.
   - `tracelens-dashboard`: Streamlit frontend service.
3. Set environment variables `GOOGLE_API_KEY` (or `GROQ_API_KEY`) and `TRACELENS_API_URL` in the Render dashboard.

---

## PS-7.1 Acceptance Criteria Coverage

| Criterion | Implementation |
|---|---|
| Instrumented agent wrapper | `InstrumentedAgent` wraps `LeaveDecisionAgent` |
| Events for all pipeline steps | `EventType` enum: INPUT, RETRIEVAL, TOOL, DECISION, OUTPUT, FAILURE |
| PII redacted before persistence | `PIIRedactor` applied in `AuditPersistenceService` before every DB write |
| No raw PII in audit store | Enforced at persistence layer; `pii_redacted=True` flag on all events |
| Decision path reconstruction | `DecisionPathReconstructor.reconstruct(session_id)` |
| Missing steps explicitly reported | `path.missing_steps` list — no fabrication |
| Plain-English summary | `DecisionSummaryService.generate(path)` |
| Regulatory challenge response | `RegulatoryChallengegenerator.generate(path)` |
| Query by session ID | `GET /audit/sessions/{session_id}/decision-path` |
| Query by user ID | `GET /audit/users/{user_id}/sessions` |
| Query by time range | `GET /audit/search?start_time=...&end_time=...` |
| Query by decision outcome | `GET /audit/search?decision=APPROVED` |
| LangSmith trace correlation | `trace_id` stored per session; `langsmith_url` in response |
| LangSmith outage resilience | `trace_id=None` gracefully handled; audit works independently |
| No chain-of-thought exposure | LLM internal reasoning never captured or stored |
| Governance dashboard | Streamlit app at `dashboard/app.py` |
| Audit is the system of record | SQLite is the source of truth; LangSmith is supplementary |

---

## Design Principles

- **Event-driven audit** — agent emits events; persistence is a decoupled consumer
- **Repository pattern** — all SQLite access through `AuditRepository`
- **Dependency injection** — all services via FastAPI `Depends()`, testable via overrides
- **Conservative PII redaction** — false negative (missed PII) is safer than false positive (destroyed evidence)
- **Structured logging** — `structlog` throughout; no `print()` calls
- **No hardcoded secrets** — all configuration via environment variables

---

## License

Proprietary — internal project.
