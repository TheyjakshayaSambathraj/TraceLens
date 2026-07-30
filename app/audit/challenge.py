"""Regulatory challenge response generator.

PS-7.1 Bonus capability: Given a reconstructed decision path, generate
a professional draft response to a hypothetical regulatory challenge
explaining the AI decision in terms of the specific data considered.

Design principles
-----------------
- Response is grounded ONLY in the stored, sanitized audit path.
- No invented evidence, policy, or context.
- Professional regulatory language appropriate for compliance submissions.
- LLM-assisted with deterministic fallback.
- Never exposes hidden chain-of-thought.
"""

from __future__ import annotations

import json
import re
import structlog
import uuid
from datetime import datetime, timezone
from typing import Optional

from app.audit.events import DecisionPath, RegulatoryChallengeResponse

logger = structlog.get_logger(__name__)


class RegulatoryChallengegenerator:
    """Generates a regulatory challenge response from a decision path.

    The response explains the AI decision in terms that would satisfy a
    regulatory inquiry — citing the specific data, policy, and tools
    that contributed to the outcome.

    Args:
        llm_provider: Optional LLMProvider. Uses rule-based fallback if None.
    """

    def __init__(self, llm_provider=None) -> None:
        self._llm_provider = llm_provider

    def generate(self, path: DecisionPath) -> RegulatoryChallengeResponse:
        """Generate a regulatory challenge response.

        Args:
            path: Reconstructed DecisionPath containing sanitized evidence.

        Returns:
            RegulatoryChallengeResponse with professional draft language.
        """
        logger.info(
            "regulatory_challenge_response_generation_started",
            session_id=path.session_id,
        )

        facts = self._extract_facts(path)
        reference_number = f"TL-{path.session_id[-8:].upper()}-{datetime.now(timezone.utc).strftime('%Y%m%d')}"

        if self._llm_provider is not None:
            try:
                response = self._generate_with_llm(path, facts, reference_number)
                logger.info(
                    "regulatory_challenge_response_generated_with_llm",
                    session_id=path.session_id,
                )
                return response
            except Exception as exc:
                logger.warning(
                    "regulatory_challenge_llm_failed_using_fallback",
                    session_id=path.session_id,
                    error=str(exc),
                )

        return self._generate_fallback(path, facts, reference_number)

    def _extract_facts(self, path: DecisionPath) -> dict:
        """Extract structured facts from a decision path for challenge response.

        Args:
            path: DecisionPath to inspect.

        Returns:
            Dict of categorized facts.
        """
        facts = {
            "request": "",
            "user_id": path.user_id,
            "retrieved_count": 0,
            "sources": [],
            "tools_used": [],
            "tool_response_keys": [],
            "decision": "UNKNOWN",
            "decision_reason": "",
            "policy_references": [],
            "evidence": [],
            "failed": path.status.value == "FAILED",
            "missing_steps": path.missing_steps,
        }

        for entry in path.timeline:
            et = entry.event_type
            d = entry.details

            if et == "INPUT_RECEIVED":
                facts["request"] = d.get("request", "")
            elif et == "RETRIEVAL_COMPLETED":
                facts["retrieved_count"] = d.get("retrieved_count", 0)
                facts["sources"] = d.get("source_names", [])
            elif et == "TOOL_COMPLETED":
                tool_name = d.get("tool_name", "")
                if tool_name and tool_name not in facts["tools_used"]:
                    facts["tools_used"].append(tool_name)
                facts["tool_response_keys"] = d.get("response_keys", [])
            elif et == "DECISION_COMPLETED":
                facts["decision"] = d.get("decision", "UNKNOWN")
                facts["decision_reason"] = d.get("decision_reason", "")
                facts["policy_references"] = d.get("policy_references", [])
                facts["evidence"] = d.get("evidence", [])

        return facts

    def _generate_with_llm(
        self,
        path: DecisionPath,
        facts: dict,
        reference_number: str,
    ) -> RegulatoryChallengeResponse:
        """Generate response using the configured LLM.

        Args:
            path: DecisionPath.
            facts: Pre-extracted facts.
            reference_number: Internal reference ID.

        Returns:
            RegulatoryChallengeResponse from LLM.
        """
        prompt = self._build_challenge_prompt(path, facts, reference_number)
        model = self._llm_provider.get_model()
        response = model.invoke(prompt)
        response_text = (
            response.content if hasattr(response, "content") else str(response)
        )

        # Try to parse structured JSON from LLM
        try:
            json_match = re.search(r"\{[\s\S]*\}", response_text)
            if json_match:
                data = json.loads(json_match.group(0))
                return RegulatoryChallengeResponse(
                    session_id=path.session_id,
                    reference_number=reference_number,
                    generated_at=datetime.now(timezone.utc),
                    decision_summary=data.get("decision_summary", ""),
                    data_considered=data.get("data_considered", facts["evidence"]),
                    policy_basis=data.get("policy_basis", facts["policy_references"]),
                    tools_used=data.get("tools_used", facts["tools_used"]),
                    decision_outcome=facts["decision"],
                    reasoning_basis=data.get("reasoning_basis", facts["decision_reason"]),
                    limitations=data.get("limitations", facts["missing_steps"]),
                    full_response=data.get("full_response", response_text),
                )
        except (json.JSONDecodeError, KeyError):
            pass

        # Return raw LLM response as full_response
        return self._generate_fallback(path, facts, reference_number, llm_supplement=response_text)

    def _build_challenge_prompt(
        self, path: DecisionPath, facts: dict, reference_number: str
    ) -> str:
        """Construct the regulatory challenge response prompt.

        Args:
            path: DecisionPath.
            facts: Extracted facts.
            reference_number: Reference ID.

        Returns:
            Prompt string.
        """
        timeline_text = "\n".join(
            f"  [{e.sequence:02d}] {e.event_type}: {e.summary}"
            for e in path.timeline
        )

        return f"""You are a compliance officer drafting a formal response to a regulatory inquiry about an AI-assisted decision.

CRITICAL INSTRUCTIONS:
1. Base your response ONLY on the documented decision path below.
2. Do NOT invent evidence, policy citations, or data not present in the audit trail.
3. Do NOT expose internal reasoning steps, model weights, or chain-of-thought.
4. Use formal, professional regulatory language.
5. Be transparent about limitations and missing information.
6. Return valid JSON in the exact format specified.

AUDIT RECORD:
Reference: {reference_number}
Session: {path.session_id}
User: {path.user_id}
Decision: {facts['decision']}
Status: {path.status.value}

Decision Path Timeline:
{timeline_text}

Documented Evidence:
- Request: {facts['request']}
- Policy documents consulted: {facts['retrieved_count']} document(s)
- Policy sources: {', '.join(facts['sources']) or 'not recorded'}
- Tools/services used: {', '.join(facts['tools_used']) or 'none'}
- Decision outcome: {facts['decision']}
- Stated reason: {facts['decision_reason']}
- Policy references cited: {json.dumps(facts['policy_references'])}
- Evidence factors: {json.dumps(facts['evidence'])}
- Missing audit steps: {facts['missing_steps'] or 'none'}

Return ONLY this JSON:
{{
  "decision_summary": "<1-2 sentence summary of the decision>",
  "data_considered": ["<data item 1>", "<data item 2>"],
  "policy_basis": ["<policy reference>"],
  "tools_used": ["<tool name>"],
  "reasoning_basis": "<how the evidence led to the decision>",
  "limitations": ["<limitation or gap>"],
  "full_response": "<complete formal regulatory response letter text, 3-5 paragraphs>"
}}"""

    def _generate_fallback(
        self,
        path: DecisionPath,
        facts: dict,
        reference_number: str,
        llm_supplement: Optional[str] = None,
    ) -> RegulatoryChallengeResponse:
        """Generate a rule-based regulatory response without the LLM.

        Args:
            path: DecisionPath.
            facts: Extracted facts.
            reference_number: Internal reference.
            llm_supplement: Optional partial LLM text to incorporate.

        Returns:
            RegulatoryChallengeResponse.
        """
        now = datetime.now(timezone.utc)
        date_str = now.strftime("%B %d, %Y")

        decision = facts["decision"]
        request = facts["request"] or "(not recorded)"
        policy_refs = facts["policy_references"]
        evidence = facts["evidence"]
        tools = facts["tools_used"]
        sources = facts["sources"]
        reason = facts["decision_reason"] or "See decision evidence."

        # Build the full formal response letter
        policy_section = (
            "\n".join(f"  - {ref}" for ref in policy_refs)
            if policy_refs
            else "  - Policy documentation was retrieved but specific sections were not recorded in this audit trail."
        )

        evidence_section = (
            "\n".join(f"  - {ev}" for ev in evidence)
            if evidence
            else "  - Evidence details are documented in the full audit event log."
        )

        tools_section = (
            "\n".join(f"  - {t}" for t in tools)
            if tools
            else "  - No external tools were recorded in this audit trail."
        )

        limitations_list = list(facts["missing_steps"])
        if not policy_refs:
            limitations_list.append("Specific policy section citations were not captured in the audit log.")
        if not evidence:
            limitations_list.append("Detailed evidence list was not captured in the audit log.")

        limitations_section = (
            "\n".join(f"  - {lim}" for lim in limitations_list)
            if limitations_list
            else "  - No significant limitations identified."
        )

        full_response = f"""REGULATORY INQUIRY RESPONSE
Reference: {reference_number}
Date: {date_str}
Session ID: {path.session_id}

Dear Regulatory Authority,

We write in response to your inquiry regarding an AI-assisted decision made
on {path.started_at.strftime('%B %d, %Y') if path.started_at else date_str}
(Session Reference: {path.session_id}).

1. SUMMARY OF REQUEST AND DECISION

The system received the following request:
  "{request}"

Following evaluation against applicable policies and relevant employee data,
the system reached the decision: {decision}.
Stated basis: {reason}

2. DATA AND INFORMATION CONSIDERED

The following information was considered in reaching the decision:

Policy Documentation:
{policy_section}

Factual Evidence:
{evidence_section}

Data Sources Consulted:
  - Vector store retrieval from: {', '.join(sources) if sources else 'policy document store'}
  - {facts['retrieved_count']} relevant policy document(s) retrieved

3. TOOLS AND SYSTEMS USED

The following automated tools and services were consulted:
{tools_section}

4. DECISION OUTCOME AND BASIS

Decision: {decision}
The decision was reached by evaluating the retrieved policy documentation
against the employee-specific data obtained from the HR information system.
{reason}

5. AUDIT TRAIL AND GOVERNANCE

This decision was captured by the TraceLens AI Governance Platform.
A complete, sanitized audit trail is available for review (Reference: {reference_number}).
All personally identifiable information has been redacted from this record
in compliance with data protection requirements.

Audit status: {path.status.value}
LangSmith Trace: {path.trace_id or 'Not available (LangSmith not configured)'}

6. LIMITATIONS AND DISCLOSURES

{limitations_section}

We remain available to provide further information as required.

Sincerely,
AI Governance Office
TraceLens Platform
"""

        return RegulatoryChallengeResponse(
            session_id=path.session_id,
            reference_number=reference_number,
            generated_at=now,
            decision_summary=f"A leave request was evaluated and the decision reached was: {decision}. {reason}",
            data_considered=evidence or ["Employee data retrieved from HR system", "Policy documentation retrieved from vector store"],
            policy_basis=policy_refs,
            tools_used=tools,
            decision_outcome=decision,
            reasoning_basis=reason,
            limitations=limitations_list,
            full_response=llm_supplement or full_response,
        )
