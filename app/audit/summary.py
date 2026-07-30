"""Decision summary service.

Generates a plain-English summary of a decision path suitable for:
- Non-technical reviewers (HR managers, affected employees)
- Governance auditors
- Regulatory review

Critical guardrails
-------------------
- The summary is grounded ONLY in the reconstructed DecisionPath.
- The LLM is explicitly instructed NOT to invent evidence or policy.
- Hidden chain-of-thought is NEVER exposed or stored.
- If the LLM is unavailable, a structured fallback is returned.

The generated summary captures:
1. What request was made?
2. What information was considered?
3. What policy/context was retrieved?
4. What tool information was used?
5. What decision was reached?
6. Why was that decision reached (based on observed evidence)?
"""

from __future__ import annotations

import json
import structlog
from datetime import datetime, timezone
from typing import Optional

from app.audit.events import DecisionPath, DecisionSummary, TimelineEntryType

logger = structlog.get_logger(__name__)


class DecisionSummaryService:
    """Generates human-readable summaries of reconstructed decision paths.

    Uses the configured LLM when available, with a structured rule-based
    fallback when the LLM is not configured or fails.

    Args:
        llm_provider: Optional LLMProvider instance. If None, uses fallback.
    """

    def __init__(self, llm_provider=None) -> None:
        self._llm_provider = llm_provider

    def generate(self, path: DecisionPath) -> DecisionSummary:
        """Generate a decision summary from a reconstructed path.

        Args:
            path: The reconstructed DecisionPath.

        Returns:
            DecisionSummary with plain-English narrative.
        """
        logger.info(
            "decision_summary_generation_started",
            session_id=path.session_id,
            has_llm=self._llm_provider is not None,
        )

        # Extract structured facts from the path
        facts = self._extract_facts(path)

        # Try LLM-based summary first
        if self._llm_provider is not None:
            try:
                summary = self._generate_with_llm(path, facts)
                logger.info(
                    "decision_summary_generated_with_llm",
                    session_id=path.session_id,
                )
                return summary
            except Exception as exc:
                logger.warning(
                    "decision_summary_llm_failed_using_fallback",
                    session_id=path.session_id,
                    error=str(exc),
                )

        # Fallback: structured rule-based summary
        return self._generate_fallback(path, facts)

    def _extract_facts(self, path: DecisionPath) -> dict:
        """Extract structured facts from the timeline for prompt/fallback use.

        Args:
            path: DecisionPath to extract from.

        Returns:
            Dict of extracted facts keyed by category.
        """
        facts = {
            "request": "",
            "retrieved_count": 0,
            "sources": [],
            "tools_used": [],
            "tool_response_keys": [],
            "decision": path.status.value if not path.timeline else "UNKNOWN",
            "decision_reason": "",
            "policy_references": [],
            "evidence": [],
            "output_generated": False,
            "failed": path.status.value == "FAILED",
        }

        for entry in path.timeline:
            et = entry.event_type
            details = entry.details

            if et == "INPUT_RECEIVED":
                facts["request"] = details.get("request", "")

            elif et == "RETRIEVAL_COMPLETED":
                facts["retrieved_count"] = details.get("retrieved_count", 0)
                facts["sources"] = details.get("source_names", [])

            elif et == "TOOL_COMPLETED":
                tool_name = details.get("tool_name", "")
                if tool_name and tool_name not in facts["tools_used"]:
                    facts["tools_used"].append(tool_name)
                facts["tool_response_keys"] = details.get("response_keys", [])

            elif et == "DECISION_COMPLETED":
                facts["decision"] = details.get("decision", "UNKNOWN")
                facts["decision_reason"] = details.get("decision_reason", "")
                facts["policy_references"] = details.get("policy_references", [])
                facts["evidence"] = details.get("evidence", [])

            elif et == "OUTPUT_GENERATED":
                facts["output_generated"] = True

        return facts

    def _generate_with_llm(
        self, path: DecisionPath, facts: dict
    ) -> DecisionSummary:
        """Generate summary using the configured LLM.

        Strict prompt instructions prevent chain-of-thought leakage and
        hallucination of evidence not present in the decision path.

        Args:
            path: The DecisionPath.
            facts: Pre-extracted facts dict.

        Returns:
            DecisionSummary from LLM output.
        """
        prompt = self._build_summary_prompt(path, facts)
        model = self._llm_provider.get_model()
        response = model.invoke(prompt)
        response_text = (
            response.content if hasattr(response, "content") else str(response)
        )

        # Try to parse structured JSON response
        try:
            import re
            json_match = re.search(r"\{[\s\S]*\}", response_text)
            if json_match:
                data = json.loads(json_match.group(0))
                return DecisionSummary(
                    session_id=path.session_id,
                    decision=facts["decision"],
                    summary=data.get("summary", response_text[:500]),
                    evidence_considered=data.get("evidence_considered", facts["evidence"]),
                    policy_basis=data.get("policy_basis", facts["policy_references"]),
                    confidence=data.get("confidence", "moderate"),
                    limitations=data.get("limitations", path.missing_steps),
                    generated_at=datetime.now(timezone.utc),
                )
        except (json.JSONDecodeError, KeyError):
            pass

        # If JSON parsing fails, use raw response as summary
        return DecisionSummary(
            session_id=path.session_id,
            decision=facts["decision"],
            summary=response_text[:1000],
            evidence_considered=facts["evidence"],
            policy_basis=facts["policy_references"],
            confidence="moderate",
            limitations=path.missing_steps,
            generated_at=datetime.now(timezone.utc),
        )

    def _build_summary_prompt(self, path: DecisionPath, facts: dict) -> str:
        """Construct a strictly grounded prompt for the LLM.

        The prompt instructs the LLM to summarize ONLY what is present
        in the decision path — never to invent evidence or policy.

        Args:
            path: DecisionPath.
            facts: Pre-extracted facts.

        Returns:
            Prompt string.
        """
        timeline_text = self._format_timeline_for_prompt(path)
        return f"""You are an AI governance auditor summarizing a decision for a non-technical reviewer.

CRITICAL INSTRUCTIONS:
1. Summarize ONLY what is documented in the decision path below.
2. Do NOT invent evidence, policy, or tool results not present in the path.
3. Do NOT expose internal reasoning steps or chain-of-thought.
4. If information is missing from the path, explicitly state it is not available.
5. Distinguish clearly between observed facts and the decision conclusion.
6. Use plain, accessible language suitable for a non-technical person.
7. Return valid JSON in the exact format below.

DECISION PATH:
Session: {path.session_id}
User: {path.user_id}
Status: {path.status.value}
Missing steps: {path.missing_steps or 'None'}

Timeline:
{timeline_text}

Key facts observed:
- Request: {facts['request']}
- Policy documents retrieved: {facts['retrieved_count']}
- Policy sources: {', '.join(facts['sources']) or 'not recorded'}
- Tools used: {', '.join(facts['tools_used']) or 'none'}
- Decision: {facts['decision']}
- Decision reason: {facts['decision_reason']}
- Policy references: {', '.join(facts['policy_references']) or 'none cited'}
- Evidence: {json.dumps(facts['evidence'])}

Return ONLY this JSON (no other text):
{{
  "summary": "<2-3 sentence plain-English narrative>",
  "evidence_considered": ["<evidence fact 1>", "<evidence fact 2>"],
  "policy_basis": ["<policy reference 1>"],
  "confidence": "<high|moderate|low>",
  "limitations": ["<limitation if any>"]
}}"""

    def _format_timeline_for_prompt(self, path: DecisionPath) -> str:
        """Format timeline entries as readable text for the prompt.

        Args:
            path: DecisionPath with timeline.

        Returns:
            Multi-line string of timeline steps.
        """
        lines = []
        for entry in path.timeline:
            lines.append(f"  [{entry.sequence:02d}] {entry.event_type}: {entry.summary}")
        return "\n".join(lines) or "  (no timeline entries recorded)"

    def _generate_fallback(
        self, path: DecisionPath, facts: dict
    ) -> DecisionSummary:
        """Generate a structured summary without the LLM.

        Constructs a human-readable summary purely from the extracted facts.
        Always available regardless of LLM configuration.

        Args:
            path: DecisionPath.
            facts: Pre-extracted facts.

        Returns:
            Rule-based DecisionSummary.
        """
        request = facts["request"] or "(request not recorded)"
        decision = facts["decision"]
        decision_reason = facts["decision_reason"] or "No reason recorded."
        retrieved_count = facts["retrieved_count"]
        tools = facts["tools_used"]
        sources = facts["sources"]

        # Build narrative
        parts = [
            f"A leave request was submitted: \"{request}\".",
        ]

        if retrieved_count:
            src_str = f" from {', '.join(sources)}" if sources else ""
            parts.append(
                f"The system retrieved {retrieved_count} relevant policy "
                f"document(s){src_str} to evaluate the request."
            )

        if tools:
            parts.append(
                f"Employee data was retrieved via: {', '.join(tools)}."
            )

        if decision == "APPROVED":
            parts.append(
                f"The decision reached was APPROVED. {decision_reason}"
            )
        elif decision == "REJECTED":
            parts.append(
                f"The decision reached was REJECTED. {decision_reason}"
            )
        elif decision == "NEEDS_REVIEW":
            parts.append(
                f"The decision reached was NEEDS_REVIEW — human review is required. "
                f"{decision_reason}"
            )
        elif facts["failed"]:
            parts.append(
                "Execution encountered an error. The decision could not be completed."
            )
        else:
            parts.append(f"Decision outcome: {decision}.")

        summary_text = " ".join(parts)

        # Confidence heuristic
        if path.missing_steps or facts["failed"]:
            confidence = "low"
        elif not facts["policy_references"]:
            confidence = "moderate"
        else:
            confidence = "high"

        limitations = list(path.missing_steps)
        if not facts["policy_references"]:
            limitations.append("No specific policy references were recorded.")
        if not facts["evidence"]:
            limitations.append("No detailed evidence list was captured.")

        return DecisionSummary(
            session_id=path.session_id,
            decision=decision,
            summary=summary_text,
            evidence_considered=facts["evidence"],
            policy_basis=facts["policy_references"],
            confidence=confidence,
            limitations=limitations,
            generated_at=datetime.now(timezone.utc),
        )
