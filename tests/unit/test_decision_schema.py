"""Tests for decision schema and validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from app.schemas.decision import DecisionResult, DecisionStatus


class TestDecisionResult:
    """Tests for DecisionResult schema."""

    def test_valid_decision_approved(self):
        """Test creating a valid APPROVED decision."""
        decision = DecisionResult(
            decision=DecisionStatus.APPROVED,
            reason="Request is within policy limits",
            policy_references=["Section 2.1: Annual Leave Allowance"],
            evidence=[
                "Leave balance: 20 days",
                "Requested: 5 days",
                "Within policy limits",
            ],
        )

        assert decision.decision == DecisionStatus.APPROVED
        assert decision.reason == "Request is within policy limits"
        assert len(decision.policy_references) == 1
        assert len(decision.evidence) == 3

    def test_valid_decision_rejected(self):
        """Test creating a valid REJECTED decision."""
        decision = DecisionResult(
            decision=DecisionStatus.REJECTED,
            reason="Exceeds maximum consecutive leave",
            policy_references=["Section 2.3: Maximum Consecutive Leave"],
            evidence=["Requested: 15 days", "Maximum allowed: 10 days"],
        )

        assert decision.decision == DecisionStatus.REJECTED
        assert "Exceeds" in decision.reason

    def test_valid_decision_needs_review(self):
        """Test creating a valid NEEDS_REVIEW decision."""
        decision = DecisionResult(
            decision=DecisionStatus.NEEDS_REVIEW,
            reason="Insufficient information to make decision",
            policy_references=[],
            evidence=["Cannot determine leave balance"],
        )

        assert decision.decision == DecisionStatus.NEEDS_REVIEW

    def test_decision_with_empty_lists(self):
        """Test decision with empty policy_references and evidence lists."""
        decision = DecisionResult(
            decision=DecisionStatus.APPROVED,
            reason="Auto-approved",
            policy_references=[],
            evidence=[],
        )

        assert decision.policy_references == []
        assert decision.evidence == []

    def test_decision_with_missing_optional_fields(self):
        """Test that optional fields use defaults."""
        decision = DecisionResult(
            decision=DecisionStatus.APPROVED,
            reason="Approved",
        )

        assert decision.policy_references == []
        assert decision.evidence == []

    def test_invalid_decision_status(self):
        """Test that invalid decision status raises ValidationError."""
        with pytest.raises(ValidationError):
            DecisionResult(
                decision="INVALID",  # type: ignore
                reason="Test",
            )

    def test_decision_serialization(self):
        """Test that decision can be serialized to JSON."""
        decision = DecisionResult(
            decision=DecisionStatus.APPROVED,
            reason="Approved",
            policy_references=["Policy X"],
            evidence=["Fact 1"],
        )

        json_data = decision.model_dump_json()
        assert "APPROVED" in json_data
        assert "Approved" in json_data

    def test_decision_deserialization(self):
        """Test that decision can be deserialized from dict."""
        data = {
            "decision": "REJECTED",
            "reason": "Policy violation",
            "policy_references": ["Section 5"],
            "evidence": ["Evidence 1", "Evidence 2"],
        }

        decision = DecisionResult(**data)

        assert decision.decision == DecisionStatus.REJECTED
        assert len(decision.evidence) == 2

    def test_decision_status_enum_values(self):
        """Test that DecisionStatus enum has expected values."""
        assert DecisionStatus.APPROVED.value == "APPROVED"
        assert DecisionStatus.REJECTED.value == "REJECTED"
        assert DecisionStatus.NEEDS_REVIEW.value == "NEEDS_REVIEW"
