"""Real-User UI Acceptance Test Runner for TraceLens.

Simulates the exact interaction flow performed by the Streamlit dashboard
for all 8 user acceptance scenarios.
"""

import json
import requests
import sys

API_BASE = "http://localhost:8000/api/v1"

SCENARIOS = [
    {
        "id": "Scenario 1",
        "name": "Normal Leave (Standard Employee)",
        "query": "Can employee EMP-001 take 5 consecutive days of annual leave starting next Monday?",
        "user_id": "USER-UI-001"
    },
    {
        "id": "Scenario 2",
        "name": "Excessive Consecutive Leave",
        "query": "Can employee EMP-001 take 15 consecutive days of annual leave starting next Monday?",
        "user_id": "USER-UI-002"
    },
    {
        "id": "Scenario 3",
        "name": "Advance Notice Violation",
        "query": "Can employee EMP-001 take annual leave starting tomorrow?",
        "user_id": "USER-UI-003"
    },
    {
        "id": "Scenario 4",
        "name": "Carry-Forward Policy",
        "query": "Can employee EMP-001 carry forward 8 unused annual leave days into the next calendar year?",
        "user_id": "USER-UI-004"
    },
    {
        "id": "Scenario 5",
        "name": "Probationary Employee",
        "query": "Can a probationary employee take 10 days of annual leave during their first three months?",
        "user_id": "USER-UI-005"
    },
    {
        "id": "Scenario 6",
        "name": "Unrelated / Unsupported Request",
        "query": "Can EMP-001 receive a company-funded laptop upgrade next month?",
        "user_id": "USER-UI-006"
    },
    {
        "id": "Scenario 7",
        "name": "PII Handling",
        "query": "Employee John Smith, email john.smith@example.com, phone +1-555-123-4567, wants to know whether he can take 5 days of annual leave next month.",
        "user_id": "USER-UI-007"
    }
]

def run_scenario(sc):
    print("=" * 80)
    print(f"RUNNING: {sc['id']} — {sc['name']}")
    print(f"User Request: {sc['query']}")
    print(f"User ID: {sc['user_id']}")
    print("-" * 80)

    # 1. Submit decision request (Streamlit Sidebar action)
    post_url = f"{API_BASE}/agent/decide"
    payload = {
        "request": sc["query"],
        "user_id": sc["user_id"],
        "session_id": None
    }

    try:
        resp = requests.post(post_url, json=payload, timeout=60)
        if resp.status_code != 200:
            print(f"❌ DECISION SUBMISSION FAILED: HTTP {resp.status_code} — {resp.text}")
            return None

        result = resp.json()
        session_id = result.get("session_id")
        decision = result.get("decision")
        reason = result.get("reason")
        final_response = result.get("final_response")

        print(f"✅ Decision Generated: {decision}")
        print(f"Session ID: {session_id}")
        print(f"Reason: {reason}")
        print(f"Final Response:\n{final_response}\n")

        # 2. Perform Session Lookup & Load Audit Detail (Streamlit Detail View action)
        print("--- SESSION LOOKUP & DECISION PATH DETAIL ---")
        session_rec = requests.get(f"{API_BASE}/audit/sessions/{session_id}").json()
        print(f"Session Record Status: {session_rec.get('status')}")

        path = requests.get(f"{API_BASE}/audit/sessions/{session_id}/decision-path").json()
        print(f"Path Status: {path.get('status')}")
        print(f"PII Redacted Flag: {path.get('pii_redacted')}")
        print(f"Missing Steps: {path.get('missing_steps')}")

        timeline = path.get("timeline", [])
        print(f"Timeline Step Count: {len(timeline)}")
        for step in timeline:
            print(f"  [{step.get('sequence'):02d}] {step.get('event_type')}: {step.get('summary')}")
            details = step.get("details", {})
            if step.get("event_type") == "DECISION_COMPLETED":
                print(f"      Policy Refs: {details.get('policy_references')}")
                print(f"      Evidence: {details.get('evidence')}")

        # 3. Generate Decision Summary
        print("\n--- PLAIN-ENGLISH DECISION SUMMARY ---")
        summary = requests.get(f"{API_BASE}/audit/sessions/{session_id}/summary").json()
        print(f"Summary Decision: {summary.get('decision')}")
        print(f"Confidence: {summary.get('confidence')}")
        print(f"Summary Narrative:\n{summary.get('summary')}")
        print(f"Evidence: {summary.get('evidence_considered')}")
        print(f"Policy Basis: {summary.get('policy_basis')}")

        # 4. Generate Regulatory Challenge Response
        print("\n--- REGULATORY CHALLENGE RESPONSE ---")
        challenge = requests.get(f"{API_BASE}/audit/sessions/{session_id}/challenge-response").json()
        print(f"Reference: {challenge.get('reference_number')}")
        print(f"Data Considered: {challenge.get('data_considered')}")
        print(f"Full Response Letter:\n{challenge.get('full_response')}\n")

        return {
            "scenario": sc,
            "session_id": session_id,
            "decision": decision,
            "reason": reason,
            "path": path,
            "summary": summary,
            "challenge": challenge
        }

    except Exception as e:
        print(f"❌ SCENARIO FAILED WITH EXCEPTION: {e}")
        return None

def main():
    results = []
    for sc in SCENARIOS:
        res = run_scenario(sc)
        if res:
            results.append(res)

    print("=" * 80)
    print("ALL SCENARIOS COMPLETED.")
    print("=" * 80)

if __name__ == "__main__":
    main()
