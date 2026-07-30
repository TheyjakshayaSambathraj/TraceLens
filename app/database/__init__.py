"""TraceLens database package.

Exports the public surface of the database layer:
- Base: SQLAlchemy declarative base
- ORM models: AuditSessionModel, AuditEventModel, DecisionRecordModel
- Session management: init_db, get_db, db_session
- Repository: AuditRepository
"""

from app.database.models import (
    AuditEventModel,
    AuditSessionModel,
    Base,
    DecisionRecordModel,
)
from app.database.repository import AuditRepository
from app.database.session import db_session, get_db, init_db

__all__ = [
    "Base",
    "AuditSessionModel",
    "AuditEventModel",
    "DecisionRecordModel",
    "AuditRepository",
    "init_db",
    "get_db",
    "db_session",
]
