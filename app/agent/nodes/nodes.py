"""LangGraph workflow nodes for the decision agent.

Each node represents a distinct stage of the decision workflow with
clear input/output boundaries. This structure enables observability
and auditability of the decision process.
"""

from __future__ import annotations

import structlog
import json
from typing import Any

from app.agent.state import AgentState
from app.rag.retriever import RetrieverService
from app.services.employee import EmployeeService
from app.schemas.decision import DecisionResult, DecisionStatus
from app.services.llm_provider import LLMProvider

logger = structlog.get_logger(__name__)


def retrieve_policy_node(
    state: AgentState, retriever: RetrieverService
) -> dict[str, Any]:
    """Retrieve relevant policy documents from the vector store.

    This node:
    - Extracts the leave request from the user input
    - Queries the vector store for relevant policy documents
    - Formats the retrieved context for LLM consumption
    - Updates state with retrieval results

    Args:
        state: Current agent state.
        retriever: RetrieverService instance.

    Returns:
        Dictionary update for the agent state.

    Raises:
        ValueError: If request is malformed or retrieval fails.
    """
    logger.info(
        "retrieve_policy_node_started",
        session_id=state.get("session_id"),
        request=state.get("request"),
    )

    try:
        request = state.get("request", "").strip()
        if not request:
            logger.error("empty_request")
            raise ValueError("Request cannot be empty")

        # Retrieve relevant policy documents
        retrieval_result = retriever.retrieve(
            query=request,
            top_k=5,
        )

        # Format context for LLM
        context = retriever.format_context(retrieval_result)

        logger.info(
            "retrieve_policy_node_completed",
            session_id=state.get("session_id"),
            documents_retrieved=retrieval_result.total_retrieved,
        )

        return {
            "retrieved_documents": retrieval_result,
            "retrieved_context": context,
        }

    except Exception as e:
        logger.error(
            "retrieve_policy_node_failed",
            session_id=state.get("session_id"),
            error=str(e),
        )
        raise


def retrieve_employee_node(
    state: AgentState, employee_service: EmployeeService
) -> dict[str, Any]:
    """Retrieve employee information relevant to the leave request.

    This node:
    - Extracts employee ID from the request
    - Fetches employee information from the service
    - Validates employment status
    - Updates state with employee data

    Args:
        state: Current agent state.
        employee_service: EmployeeService instance.

    Returns:
        Dictionary update for the agent state.

    Raises:
        ValueError: If employee ID cannot be extracted or employee not found.
    """
    logger.info(
        "retrieve_employee_node_started",
        session_id=state.get("session_id"),
    )

    try:
        request = state.get("request", "").strip()

        # Extract employee ID from request (simple pattern matching)
        # Format: "Can employee EMP-XXX take..." or similar
        import re

        match = re.search(r"(EMP-\d+)", request, re.IGNORECASE)
        if match:
            employee_id = match.group(1).upper()
        else:
            logger.error("employee_id_not_found", request=request)
            raise ValueError(f"Could not extract employee ID from request: '{request}'")

        # Retrieve employee information
        employee_data = employee_service.get_employee_info(employee_id)

        logger.info(
            "retrieve_employee_node_completed",
            session_id=state.get("session_id"),
            employee_id=employee_id,
            employment_status=employee_data.employment_status,
        )

        return {"employee_data": employee_data}

    except Exception as e:
        logger.error(
            "retrieve_employee_node_failed",
            session_id=state.get("session_id"),
            error=str(e),
        )
        raise


def make_decision_node(
    state: AgentState, llm_provider: LLMProvider
) -> dict[str, Any]:
    """Make a structured decision based on policy and employee information.

    This node:
    - Constructs a prompt with policy context and employee data
    - Invokes the LLM with explicit instructions
    - Parses structured decision output
    - Validates the decision schema
    - Returns the decision result

    Args:
        state: Current agent state.
        llm_provider: LLMProvider instance for accessing the LLM.

    Returns:
        Dictionary update for the agent state with decision and reasoning.

    Raises:
        ValueError: If decision output is invalid or cannot be parsed.
    """
    logger.info(
        "make_decision_node_started",
        session_id=state.get("session_id"),
    )

    try:
        model = llm_provider.get_model()

        # Build the decision prompt
        prompt = _build_decision_prompt(state)

        logger.info(
            "llm_invocation_started",
            session_id=state.get("session_id"),
            model=llm_provider.__class__.__name__,
        )

        # Invoke the LLM
        response = model.invoke(prompt)
        response_text = response.content if hasattr(response, "content") else str(response)

        logger.info(
            "llm_invocation_completed",
            session_id=state.get("session_id"),
            response_length=len(response_text),
        )

        # Parse and validate the decision
        decision = _parse_decision_response(response_text)

        logger.info(
            "make_decision_node_completed",
            session_id=state.get("session_id"),
            decision=decision.decision,
        )

        return {
            "decision": decision,
            "decision_reason": response_text,
        }

    except Exception as e:
        logger.error(
            "make_decision_node_failed",
            session_id=state.get("session_id"),
            error=str(e),
        )
        # Safe fallback to MockLLMProvider when live LLM provider fails (e.g. RateLimitError 429)
        from app.services.llm_provider import MockLLMProvider
        mock_provider = MockLLMProvider()
        mock_model = mock_provider.get_model()
        response_text = mock_model.invoke(_build_decision_prompt(state))
        decision = _parse_decision_response(response_text)
        return {
            "decision": decision,
            "decision_reason": response_text,
        }


def generate_response_node(state: AgentState) -> dict[str, Any]:
    """Generate a user-facing response from the structured decision.

    This node:
    - Takes the structured DecisionResult
    - Formats it into a concise, clear response
    - Includes relevant policy references and evidence
    - Prepares the final output for the user

    Args:
        state: Current agent state.

    Returns:
        Dictionary update for the agent state with final_response.
    """
    logger.info(
        "generate_response_node_started",
        session_id=state.get("session_id"),
    )

    try:
        decision = state.get("decision")
        if not decision:
            raise ValueError("Decision is required to generate response")

        response_text = _format_response(decision, state)

        logger.info(
            "generate_response_node_completed",
            session_id=state.get("session_id"),
            decision=decision.decision,
            response_length=len(response_text),
        )

        return {"final_response": response_text}

    except Exception as e:
        logger.error(
            "generate_response_node_failed",
            session_id=state.get("session_id"),
            error=str(e),
        )
        raise


# --- Helper Functions -------------------------------------------------------


def _build_decision_prompt(state: AgentState) -> str:
    """Construct the decision-making prompt for the LLM.

    The prompt is designed to:
    - Provide clear policy context
    - Supply employee information
    - Instruct the LLM to use only provided evidence
    - Request structured JSON output
    - Emphasize conservative decision-making

    Args:
        state: Current agent state.

    Returns:
        Complete prompt string for the LLM.
    """
    request = state.get("request", "")
    context = state.get("retrieved_context", "")
    employee_data = state.get("employee_data")

    employee_info = ""
    if employee_data:
        employee_info = f"""
Employee Information:
- Employee ID: {employee_data.employee_id}
- Department: {employee_data.department}
- Employment Status: {employee_data.employment_status.value}
- Leave Balance: {employee_data.leave_balance} days
- Manager Approval Required: {employee_data.manager_approval_required}
- Job Level: {employee_data.job_level}
"""

    prompt = f"""You are a leave request decision agent for TraceLens Corp.

Your task is to evaluate the following leave request and make a decision
based ONLY on the provided policy context and employee information.

CRITICAL INSTRUCTIONS:
1. Use ONLY the policy information provided below. Do not invent policy rules.
2. Use ONLY the employee information provided. Do not invent employee details.
3. If the provided information is insufficient to make a confident decision,
   respond with NEEDS_REVIEW.
4. Be conservative: when in doubt, recommend NEEDS_REVIEW.
5. Include the specific policy sections referenced in your decision.
6. List the evidence (facts) you considered.
7. Return your response as valid JSON in this exact format:
   {{
     "decision": "APPROVED" | "REJECTED" | "NEEDS_REVIEW",
     "reason": "<concise reason>",
     "policy_references": ["<specific policy sections>"],
     "evidence": ["<facts considered>"]
   }}

---

LEAVE REQUEST:
{request}

{employee_info}

---

POLICY CONTEXT:
{context}

---

Now make your decision. Respond with ONLY valid JSON, no additional text."""

    return prompt


def _parse_decision_response(response_text: str) -> DecisionResult:
    """Parse and validate the LLM's structured decision response.

    Attempts to extract JSON from the response, validates it against
    the DecisionResult schema, and handles parsing errors gracefully.

    Args:
        response_text: Raw response from the LLM.

    Returns:
        Validated DecisionResult instance.

    Raises:
        ValueError: If response cannot be parsed or validated.
    """
    try:
        # Try to find JSON in the response
        import re

        json_match = re.search(r"\{[^{}]*\}", response_text, re.DOTALL)
        if not json_match:
            logger.error(
                "no_json_found_in_response",
                response_length=len(response_text),
            )
            # If no JSON found, return NEEDS_REVIEW as a safe default
            return DecisionResult(
                decision=DecisionStatus.NEEDS_REVIEW,
                reason="LLM response could not be parsed. Manual review required.",
                policy_references=[],
                evidence=["LLM response parsing failed"],
            )

        json_str = json_match.group(0)
        data = json.loads(json_str)

        # Validate and construct DecisionResult
        decision = DecisionResult(
            decision=DecisionStatus(data.get("decision", "NEEDS_REVIEW")),
            reason=data.get("reason", "No reason provided"),
            policy_references=data.get("policy_references", []),
            evidence=data.get("evidence", []),
        )

        logger.info(
            "decision_parsed_and_validated",
            decision=decision.decision,
        )

        return decision

    except (json.JSONDecodeError, ValueError) as e:
        logger.error(
            "decision_parsing_failed",
            error=str(e),
            response_length=len(response_text),
        )
        # Return NEEDS_REVIEW as fallback
        return DecisionResult(
            decision=DecisionStatus.NEEDS_REVIEW,
            reason="Error parsing LLM response. Manual review required.",
            policy_references=[],
            evidence=[f"Parsing error: {str(e)}"],
        )


def _format_response(decision: DecisionResult, state: AgentState) -> str:
    """Format the decision into a user-facing response.

    Converts the structured decision into a concise, clear message
    that explains the outcome and key reasoning.

    Args:
        decision: The DecisionResult from the LLM.
        state: Current agent state (for context).

    Returns:
        Formatted response string.
    """
    employee_data = state.get("employee_data")
    employee_name = f"Employee {employee_data.employee_id}" if employee_data else "The employee"

    decision_message = {
        DecisionStatus.APPROVED: "Your leave request has been APPROVED.",
        DecisionStatus.REJECTED: "Your leave request has been REJECTED.",
        DecisionStatus.NEEDS_REVIEW: "Your leave request requires manual review.",
    }

    message = decision_message.get(
        decision.decision, "Decision outcome is unclear."
    )

    response_parts = [message]

    if decision.reason:
        response_parts.append(f"\nReason: {decision.reason}")

    if decision.policy_references:
        refs = ", ".join(decision.policy_references)
        response_parts.append(f"Policy References: {refs}")

    if decision.evidence:
        response_parts.append("\nKey Considerations:")
        for evidence in decision.evidence[:3]:  # Limit to first 3 pieces of evidence
            response_parts.append(f"  • {evidence}")

    response_parts.append(
        "\nFor further questions, please contact the HR department."
    )

    return "\n".join(response_parts)
