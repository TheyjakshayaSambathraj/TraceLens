"""TraceLens audit package.

Provides the complete audit infrastructure for PS-7.1 compliance:
- Event models (Pydantic, API-layer)
- Persistence service (event bus → PII redaction → SQLite)
- Timeline builder
- Decision path reconstructor
- Decision summary generator
- Regulatory challenge response generator
"""

from app.audit.events import (
    DecisionPath,
    DecisionSummary,
    PathStatus,
    RegulatoryChallengeResponse,
    TimelineEntry,
    TimelineEntryType,
    AuditSessionRecord,
    AuditEventRecord,
    SessionListResponse,
    AgentDecideRequest,
    AgentDecideResponse,
)
from app.audit.timeline import TimelineBuilder
from app.audit.reconstructor import DecisionPathReconstructor
from app.audit.summary import DecisionSummaryService
from app.audit.challenge import RegulatoryChallengegenerator
from app.audit.persistence import AuditPersistenceService, wire_persistence

__all__ = [
    "DecisionPath",
    "DecisionSummary",
    "PathStatus",
    "RegulatoryChallengeResponse",
    "TimelineEntry",
    "TimelineEntryType",
    "AuditSessionRecord",
    "AuditEventRecord",
    "SessionListResponse",
    "AgentDecideRequest",
    "AgentDecideResponse",
    "TimelineBuilder",
    "DecisionPathReconstructor",
    "DecisionSummaryService",
    "RegulatoryChallengegenerator",
    "AuditPersistenceService",
    "wire_persistence",
]
