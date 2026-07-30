"""Integration tests for the leave decision agent.

These tests require real embeddings and may make external API calls.
Run with: pytest -m integration
"""

from __future__ import annotations

import pytest
import tempfile
import shutil
from pathlib import Path

from app.agent.graph import LeaveDecisionAgent
from app.rag.ingest import DocumentIngestionPipeline
from app.rag.retriever import RetrieverService
from app.services.employee import EmployeeService
from app.services.llm_provider import MockLLMProvider, create_llm_provider
from app.config.settings import Settings


# Mark all tests in this file as integration tests
pytestmark = pytest.mark.integration


@pytest.fixture
def temp_policy_dir():
    """Create a temporary directory with test policy."""
    temp_dir = tempfile.mkdtemp()
    policy_path = Path(temp_dir) / "test_policy.md"
    policy_path.write_text(
        """# HR Leave Policy

## Annual Leave Entitlements

- Standard employees: 20 working days per year
- Senior employees: 25 working days per year

## Maximum Consecutive Leave

- Standard employees: Maximum 10 consecutive days
- Senior/Management: Maximum 15 consecutive days

## Approval Requirements

- Manager approval required for all leave
- Exceptional approval for periods exceeding limits
"""
    )
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def temp_vector_store():
    """Create a temporary directory for vector store."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def retriever(temp_policy_dir, temp_vector_store):
    """Create an ingested retriever (uses real embeddings)."""
    pipeline = DocumentIngestionPipeline(vector_store_path=temp_vector_store)
    vector_store = pipeline.ingest(temp_policy_dir)
    pipeline.save_vector_store(vector_store)
    return RetrieverService(vector_store_path=temp_vector_store)


@pytest.fixture
def agent(retriever):
    """Create a LeaveDecisionAgent with mocked LLM."""
    employee_service = EmployeeService()
    llm_provider = MockLLMProvider()
    return LeaveDecisionAgent(retriever, employee_service, llm_provider)


class TestLeaveDecisionAgent:
    """Integration tests for LeaveDecisionAgent."""

    def test_agent_initialization(self, agent):
        """Test that agent initializes correctly."""
        assert agent is not None
        assert agent.graph is not None

    def test_agent_decide_basic_flow(self, agent):
        """Test basic decision workflow execution."""
        result = agent.decide(
            request="Can employee EMP-001 take 5 consecutive days of leave?",
            user_id="USER-001",
            session_id="TEST-SESSION-001",
        )

        assert result is not None
        assert result.get("session_id") == "TEST-SESSION-001"
        assert result.get("user_id") == "USER-001"
        assert result.get("request") is not None

    def test_agent_workflow_has_all_stages(self, agent):
        """Test that workflow completes all stages."""
        result = agent.decide(
            request="Can employee EMP-001 take 10 days of leave?",
            session_id="TEST-002",
        )

        # Check that all workflow stages completed
        assert result.get("retrieved_documents") is not None
        assert result.get("retrieved_context") is not None
        assert result.get("employee_data") is not None
        assert result.get("decision") is not None
        assert result.get("final_response") is not None

    def test_agent_decision_is_structured(self, agent):
        """Test that decision output is structured."""
        result = agent.decide(
            request="Can employee EMP-002 take 8 days starting next Monday?",
            session_id="TEST-003",
        )

        decision = result.get("decision")
        assert decision is not None
        assert hasattr(decision, "decision")
        assert hasattr(decision, "reason")
        assert hasattr(decision, "policy_references")
        assert hasattr(decision, "evidence")

    def test_agent_retrieves_policy(self, agent):
        """Test that agent retrieves policy documents."""
        result = agent.decide(
            request="Can employee EMP-001 take 15 consecutive days?",
            session_id="TEST-004",
        )

        retrieval_result = result.get("retrieved_documents")
        assert retrieval_result is not None
        assert len(retrieval_result.documents) > 0

    def test_agent_retrieves_employee_info(self, agent):
        """Test that agent retrieves employee information."""
        result = agent.decide(
            request="Can employee EMP-003 take leave?",
            session_id="TEST-005",
        )

        employee_data = result.get("employee_data")
        assert employee_data is not None
        assert employee_data.employee_id == "EMP-003"

    def test_agent_generates_user_response(self, agent):
        """Test that agent generates a user-facing response."""
        result = agent.decide(
            request="Can employee EMP-001 take 5 days of leave?",
            session_id="TEST-006",
        )

        final_response = result.get("final_response")
        assert final_response is not None
        assert len(final_response) > 0
        assert "leave request" in final_response.lower()

    def test_agent_invalid_request_raises_error(self, agent):
        """Test that empty request raises ValueError."""
        with pytest.raises(ValueError):
            agent.decide(request="", session_id="TEST-007")

    def test_agent_missing_employee_raises_error(self, agent):
        """Test that missing employee ID in request raises error."""
        with pytest.raises(ValueError):
            agent.decide(
                request="Can someone take 5 days of leave?",
                session_id="TEST-008",
            )

    def test_agent_generates_session_id_if_not_provided(self, agent):
        """Test that agent generates session ID if not provided."""
        result = agent.decide(
            request="Can employee EMP-001 take 3 days of leave?",
        )

        session_id = result.get("session_id")
        assert session_id is not None
        assert session_id.startswith("SESSION-")

    def test_agent_preserves_metadata(self, agent):
        """Test that agent preserves and augments metadata."""
        result = agent.decide(
            request="Can employee EMP-001 take 2 days of leave?",
            session_id="TEST-009",
            user_id="USER-TEST",
        )

        assert result.get("metadata") is not None
        assert result.get("metadata").get("workflow_version") == "phase-2"

    def test_agent_with_different_employees(self, agent):
        """Test agent with different employee IDs."""
        for emp_id in ["EMP-001", "EMP-002", "EMP-003"]:
            result = agent.decide(
                request=f"Can employee {emp_id} take 5 days of leave?",
                session_id=f"TEST-{emp_id}",
            )

            assert result.get("employee_data").employee_id == emp_id
