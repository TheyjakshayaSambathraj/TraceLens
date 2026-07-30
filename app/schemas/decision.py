"""Structured decision schema.

The LLM must not return arbitrary free-form text internally.
All decisions must conform to this schema and be validated.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class DecisionStatus(str, Enum):
    """Decision outcome status."""

    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    NEEDS_REVIEW = "NEEDS_REVIEW"


class DecisionResult(BaseModel):
    """Structured decision output from the LLM.

    Attributes:
        decision: The outcome of the decision logic.
        reason: Concise explanation of the reasoning.
        policy_references: Specific policy sections cited to support the decision.
        evidence: List of facts or evidence considered in making the decision.
    """

    decision: DecisionStatus = Field(
        ..., description="Decision outcome: APPROVED, REJECTED, or NEEDS_REVIEW"
    )
    reason: str = Field(..., description="Concise explanation of the decision")
    policy_references: list[str] = Field(
        default_factory=list, description="Specific policy sections referenced"
    )
    evidence: list[str] = Field(
        default_factory=list, description="Facts and evidence considered"
    )

    class Config:
        """Pydantic configuration."""

        json_schema_extra = {
            "example": {
                "decision": "REJECTED",
                "reason": "Requested period exceeds maximum consecutive leave",
                "policy_references": ["Section 2.3: Maximum Consecutive Leave"],
                "evidence": [
                    "Requested: 15 consecutive days",
                    "Maximum allowed: 10 consecutive days",
                    "Employee leave balance: 20 days",
                ],
            }
        }
