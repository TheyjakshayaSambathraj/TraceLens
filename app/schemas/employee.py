"""Employee information schema.

Only includes fields relevant to leave request decisions.
Does not expose sensitive PII.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class EmploymentStatus(str, Enum):
    """Employment status types."""

    ACTIVE = "ACTIVE"
    ON_LEAVE = "ON_LEAVE"
    TERMINATED = "TERMINATED"
    SUSPENDED = "SUSPENDED"


class EmployeeInfo(BaseModel):
    """Employee information relevant to leave approval decisions.

    Attributes:
        employee_id: Unique employee identifier.
        department: Employee's department.
        employment_status: Current employment status.
        leave_balance: Number of leave days remaining.
        manager_approval_required: Whether manager approval is required.
        job_level: Job level/seniority for approval routing.
    """

    employee_id: str = Field(..., description="Unique employee identifier")
    department: str = Field(..., description="Employee's department")
    employment_status: EmploymentStatus = Field(
        ..., description="Current employment status"
    )
    leave_balance: int = Field(..., description="Remaining leave days")
    manager_approval_required: bool = Field(
        ..., description="Whether manager approval is required"
    )
    job_level: str = Field(
        default="standard", description="Job level for approval routing"
    )

    class Config:
        """Pydantic configuration."""

        json_schema_extra = {
            "example": {
                "employee_id": "EMP-001",
                "department": "Engineering",
                "employment_status": "ACTIVE",
                "leave_balance": 20,
                "manager_approval_required": True,
                "job_level": "senior",
            }
        }
