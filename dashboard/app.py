"""TraceLens Governance Dashboard.

Enterprise AI Decision Governance Platform — PS-7.1 Decision Path Auditor.

This dashboard consumes the TraceLens FastAPI and presents a governance-grade
view of AI decision audit trails. It does NOT access SQLite directly.

Features:
- Decision search (by session ID, user, time range, outcome)
- Complete decision timeline viewer
- Decision evidence and policy reference viewer
- PII protection status indicator
- Plain-English decision summary
- Regulatory challenge response generator
- LangSmith trace correlation link
- One-click demo scenario execution
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE = os.environ.get("TRACELENS_API_URL", "http://localhost:8000/api/v1")
REQUEST_TIMEOUT = 30

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="TraceLens — AI Governance",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "TraceLens — Enterprise AI Decision Governance Platform (PS-7.1)",
    },
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown(
    """
<style>
    /* Global font */
    html, body, [class*="css"] {
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #0f1117;
        border-right: 1px solid #1e2130;
    }
    section[data-testid="stSidebar"] * {
        color: #c9d1d9 !important;
    }

    /* Main area */
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    /* Decision status badges */
    .badge-approved {
        background: linear-gradient(135deg, #0d6e2e, #1a9e43);
        color: white; padding: 6px 18px; border-radius: 20px;
        font-weight: 700; font-size: 14px; display: inline-block;
        letter-spacing: 0.5px;
    }
    .badge-rejected {
        background: linear-gradient(135deg, #8b1a1a, #c0392b);
        color: white; padding: 6px 18px; border-radius: 20px;
        font-weight: 700; font-size: 14px; display: inline-block;
        letter-spacing: 0.5px;
    }
    .badge-needs-review {
        background: linear-gradient(135deg, #7d4f00, #e67e22);
        color: white; padding: 6px 18px; border-radius: 20px;
        font-weight: 700; font-size: 14px; display: inline-block;
        letter-spacing: 0.5px;
    }
    .badge-unknown {
        background: linear-gradient(135deg, #2d3748, #4a5568);
        color: white; padding: 6px 18px; border-radius: 20px;
        font-weight: 700; font-size: 14px; display: inline-block;
        letter-spacing: 0.5px;
    }

    /* Timeline */
    .timeline-item {
        border-left: 3px solid #1f6feb;
        padding: 12px 16px;
        margin: 8px 0;
        background: #161b22;
        border-radius: 0 8px 8px 0;
    }
    .timeline-item-failure {
        border-left: 3px solid #c0392b;
        padding: 12px 16px;
        margin: 8px 0;
        background: #1c1010;
        border-radius: 0 8px 8px 0;
    }
    .timeline-item-decision {
        border-left: 3px solid #f39c12;
        padding: 12px 16px;
        margin: 8px 0;
        background: #1a1610;
        border-radius: 0 8px 8px 0;
    }
    .timeline-step {
        color: #58a6ff; font-size: 11px; font-weight: 700;
        text-transform: uppercase; letter-spacing: 1px;
    }
    .timeline-summary {
        color: #e6edf3; font-size: 14px; margin: 4px 0;
    }
    .timeline-meta {
        color: #6e7681; font-size: 11px;
    }

    /* Cards */
    .metric-card {
        background: #161b22; border: 1px solid #30363d;
        border-radius: 10px; padding: 16px; margin: 4px 0;
    }
    .metric-label {
        color: #8b949e; font-size: 11px; font-weight: 600;
        text-transform: uppercase; letter-spacing: 1px;
    }
    .metric-value {
        color: #e6edf3; font-size: 20px; font-weight: 700; margin-top: 4px;
    }

    /* PII Shield */
    .pii-shield {
        background: linear-gradient(135deg, #0d4f2e, #1a6e3e);
        border: 1px solid #238636; border-radius: 8px;
        padding: 10px 16px; display: inline-flex;
        align-items: center; gap: 8px;
    }
    .pii-shield-text { color: #3fb950; font-weight: 600; font-size: 13px; }

    /* Evidence list */
    .evidence-item {
        background: #0d1117; border: 1px solid #21262d;
        border-radius: 6px; padding: 8px 12px; margin: 4px 0;
        color: #c9d1d9; font-size: 13px;
    }

    /* Section headers */
    .section-header {
        color: #8b949e; font-size: 11px; font-weight: 700;
        text-transform: uppercase; letter-spacing: 2px;
        border-bottom: 1px solid #21262d; padding-bottom: 6px;
        margin: 20px 0 12px 0;
    }

    /* Status indicators */
    .status-complete { color: #3fb950; }
    .status-incomplete { color: #e67e22; }
    .status-failed { color: #f85149; }
    .status-in-progress { color: #58a6ff; }

    /* Regulatory response */
    .reg-response {
        background: #0d1117; border: 1px solid #30363d;
        border-radius: 8px; padding: 20px;
        font-family: 'Courier New', monospace;
        font-size: 12px; color: #c9d1d9;
        line-height: 1.7; white-space: pre-wrap;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------


def api_get(path: str, params: dict = None) -> Optional[dict]:
    """Make a GET request to the TraceLens API.

    Args:
        path: API path (without base URL).
        params: Optional query parameters.

    Returns:
        Response JSON dict or None on error.
    """
    try:
        url = f"{API_BASE}{path}"
        response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 404:
            return None
        else:
            st.error(f"API error {response.status_code}: {response.text[:200]}")
            return None
    except requests.ConnectionError:
        st.error(
            f"Cannot connect to TraceLens API at `{API_BASE}`. "
            "Ensure the API server is running."
        )
        return None
    except requests.Timeout:
        st.error("API request timed out.")
        return None
    except Exception as exc:
        st.error(f"Request failed: {exc}")
        return None


def api_post(path: str, payload: dict) -> Optional[dict]:
    """Make a POST request to the TraceLens API.

    Args:
        path: API path.
        payload: Request body.

    Returns:
        Response JSON dict or None on error.
    """
    try:
        url = f"{API_BASE}{path}"
        response = requests.post(
            url, json=payload, timeout=REQUEST_TIMEOUT
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API error {response.status_code}: {response.text[:300]}")
            return None
    except requests.ConnectionError:
        st.error(f"Cannot connect to TraceLens API at `{API_BASE}`.")
        return None
    except Exception as exc:
        st.error(f"Request failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------


def decision_badge(decision: str) -> str:
    """Return an HTML badge for a decision status.

    Args:
        decision: Decision string (APPROVED, REJECTED, NEEDS_REVIEW).

    Returns:
        HTML badge string.
    """
    css_map = {
        "APPROVED": "badge-approved",
        "REJECTED": "badge-rejected",
        "NEEDS_REVIEW": "badge-needs-review",
    }
    css = css_map.get(decision, "badge-unknown")
    return f'<span class="{css}">{decision}</span>'


def status_color(status: str) -> str:
    """Map session/path status to CSS class.

    Args:
        status: Status string.

    Returns:
        CSS class string.
    """
    return {
        "COMPLETE": "status-complete",
        "COMPLETED": "status-complete",
        "INCOMPLETE": "status-incomplete",
        "FAILED": "status-failed",
        "IN_PROGRESS": "status-in-progress",
    }.get(status, "status-in-progress")


def format_ts(ts_str: Optional[str]) -> str:
    """Format an ISO timestamp for display.

    Args:
        ts_str: ISO 8601 timestamp string or None.

    Returns:
        Formatted string.
    """
    if not ts_str:
        return "—"
    try:
        dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    except Exception:
        return ts_str


def duration_str(duration_ms: Optional[float]) -> str:
    """Format a duration for display.

    Args:
        duration_ms: Duration in milliseconds.

    Returns:
        Human-readable string.
    """
    if duration_ms is None:
        return ""
    if duration_ms < 1000:
        return f"{duration_ms:.0f} ms"
    return f"{duration_ms / 1000:.2f} s"


# ---------------------------------------------------------------------------
# Timeline rendering
# ---------------------------------------------------------------------------

TIMELINE_ICONS = {
    "INPUT_RECEIVED": "📥",
    "RETRIEVAL_STARTED": "🔍",
    "RETRIEVAL_COMPLETED": "📚",
    "TOOL_STARTED": "⚙️",
    "TOOL_COMPLETED": "✅",
    "DECISION_STARTED": "🧠",
    "DECISION_COMPLETED": "⚖️",
    "OUTPUT_GENERATED": "📤",
    "EXECUTION_FAILED": "❌",
}

TIMELINE_LABELS = {
    "INPUT_RECEIVED": "Input Received",
    "RETRIEVAL_STARTED": "Policy Retrieval Started",
    "RETRIEVAL_COMPLETED": "Policy Context Retrieved",
    "TOOL_STARTED": "Employee Data Query",
    "TOOL_COMPLETED": "Employee Data Retrieved",
    "DECISION_STARTED": "Decision Generation Started",
    "DECISION_COMPLETED": "Decision Generated",
    "OUTPUT_GENERATED": "Final Response Generated",
    "EXECUTION_FAILED": "Execution Failed",
}


def render_timeline(timeline: list[dict]) -> None:
    """Render the decision timeline as an expandable card sequence.

    Args:
        timeline: List of timeline entry dicts from the API.
    """
    st.markdown('<div class="section-header">🕐 Decision Timeline</div>', unsafe_allow_html=True)

    if not timeline:
        st.info("No timeline entries recorded for this session.")
        return

    for entry in timeline:
        event_type = entry.get("event_type", "UNKNOWN")
        icon = TIMELINE_ICONS.get(event_type, "📌")
        label = TIMELINE_LABELS.get(event_type, event_type)
        summary = entry.get("summary", "")
        seq = entry.get("sequence", 0)
        ts = format_ts(entry.get("timestamp"))
        dur = duration_str(entry.get("duration_ms"))
        details = entry.get("details", {})

        is_failure = event_type == "EXECUTION_FAILED"
        is_decision = event_type in ("DECISION_COMPLETED", "DECISION_STARTED")

        css_class = (
            "timeline-item-failure" if is_failure
            else "timeline-item-decision" if is_decision
            else "timeline-item"
        )

        with st.expander(
            f"{icon} **{seq:02d}. {label}** — {summary[:80]}{'…' if len(summary) > 80 else ''}",
            expanded=(event_type in ("INPUT_RECEIVED", "DECISION_COMPLETED", "OUTPUT_GENERATED")),
        ):
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown(f"**{summary}**")
            with col2:
                st.markdown(f"`{ts}`{' · ' + dur if dur else ''}")

            if details:
                # Render details in a structured way
                _render_event_details(event_type, details)


def _render_event_details(event_type: str, details: dict) -> None:
    """Render structured details for a specific event type.

    Args:
        event_type: The event type string.
        details: Details dict from the timeline entry.
    """
    if event_type == "INPUT_RECEIVED":
        if details.get("request"):
            st.markdown("**Request:**")
            st.code(details["request"], language=None)

    elif event_type == "RETRIEVAL_COMPLETED":
        cols = st.columns(3)
        with cols[0]:
            st.metric("Documents", details.get("retrieved_count", 0))
        with cols[1]:
            sources = details.get("source_names", [])
            if sources:
                st.markdown("**Sources:**")
                for s in sources:
                    st.markdown(f"  - `{s}`")

    elif event_type in ("TOOL_STARTED", "TOOL_COMPLETED"):
        cols = st.columns(2)
        with cols[0]:
            st.markdown(f"**Tool:** `{details.get('tool_name', 'N/A')}`")
        with cols[1]:
            status = details.get("status", "")
            if status:
                st.markdown(f"**Status:** `{status}`")
        keys = details.get("response_keys", [])
        if keys:
            st.markdown(f"**Data fields retrieved:** `{', '.join(keys)}`")

    elif event_type == "DECISION_COMPLETED":
        decision = details.get("decision", "UNKNOWN")
        st.markdown(
            f"**Decision:** {decision_badge(decision)}",
            unsafe_allow_html=True,
        )
        reason = details.get("decision_reason", "")
        if reason:
            st.markdown(f"**Reason:** {reason}")

        policy_refs = details.get("policy_references", [])
        if policy_refs:
            st.markdown("**Policy References:**")
            for ref in policy_refs:
                st.markdown(
                    f'<div class="evidence-item">📋 {ref}</div>',
                    unsafe_allow_html=True,
                )

        evidence = details.get("evidence", [])
        if evidence:
            st.markdown("**Evidence Considered:**")
            for ev in evidence:
                st.markdown(
                    f'<div class="evidence-item">• {ev}</div>',
                    unsafe_allow_html=True,
                )

    elif event_type == "OUTPUT_GENERATED":
        length = details.get("response_length", 0)
        st.markdown(f"**Response length:** {length} characters")

    elif event_type == "EXECUTION_FAILED":
        st.error(
            f"**Category:** {details.get('failure_category', 'UNKNOWN')}  \n"
            f"**Message:** {details.get('error_message', 'No message recorded')}"
        )

    else:
        if details:
            st.json(details)


# ---------------------------------------------------------------------------
# Main views
# ---------------------------------------------------------------------------


def render_session_detail(session_id: str) -> None:
    """Render the full detail view for a session.

    Args:
        session_id: Session to display.
    """
    # Fetch all data in parallel (sequential for simplicity)
    session = api_get(f"/audit/sessions/{session_id}")
    if session is None:
        st.error(f"Session `{session_id}` not found in the audit store.")
        return

    path = api_get(f"/audit/sessions/{session_id}/decision-path")

    # -----------------------------------------------------------------------
    # Header
    # -----------------------------------------------------------------------
    decision = session.get("decision", "UNKNOWN") or "UNKNOWN"
    path_status = path.get("status", "UNKNOWN") if path else "UNKNOWN"
    status_css = status_color(path_status)

    col_title, col_status = st.columns([3, 1])
    with col_title:
        st.markdown(f"## 🔍 Audit Session")
        st.markdown(f"`{session_id}`")
    with col_status:
        st.markdown(
            f"<br><span class='{status_css}' style='font-size:18px; font-weight:700;'>● {path_status}</span>",
            unsafe_allow_html=True,
        )

    # -----------------------------------------------------------------------
    # Overview metrics
    # -----------------------------------------------------------------------
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Decision</div>'
            f'<div class="metric-value">{decision_badge(decision)}</div></div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">User</div>'
            f'<div class="metric-value" style="font-size:16px;">{session.get("user_id", "—")}</div></div>',
            unsafe_allow_html=True,
        )
    with col3:
        started = format_ts(session.get("started_at"))
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Started</div>'
            f'<div class="metric-value" style="font-size:13px;">{started}</div></div>',
            unsafe_allow_html=True,
        )
    with col4:
        # PII shield
        st.markdown(
            f'<div class="metric-card"><div class="metric-label">Privacy</div>'
            f'<div class="pii-shield" style="margin-top:6px;">'
            f'<span style="font-size:18px;">🛡️</span>'
            f'<span class="pii-shield-text">PII Protected</span>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # -----------------------------------------------------------------------
    # LangSmith trace link
    # -----------------------------------------------------------------------
    trace_id = session.get("trace_id") or (path.get("trace_id") if path else None)
    langsmith_url = path.get("langsmith_url") if path else None
    if trace_id:
        if langsmith_url:
            st.info(
                f"🔗 **LangSmith Trace:** [{trace_id}]({langsmith_url})  \n"
                f"The TraceLens audit record is correlated with this external trace."
            )
        else:
            st.info(f"🔗 **LangSmith Trace ID:** `{trace_id}`")
    else:
        st.info("ℹ️ **LangSmith:** Not configured — TraceLens maintains its own audit record independently.")

    # -----------------------------------------------------------------------
    # Missing steps indicator
    # -----------------------------------------------------------------------
    if path and path.get("missing_steps"):
        missing = path["missing_steps"]
        st.warning(
            f"⚠️ **Incomplete path detected.** "
            f"The following expected events were not recorded: `{', '.join(missing)}`"
        )

    # -----------------------------------------------------------------------
    # Timeline (centerpiece)
    # -----------------------------------------------------------------------
    if path and path.get("timeline"):
        render_timeline(path["timeline"])
    else:
        st.warning("No timeline data available for this session.")

    st.markdown("---")

    # -----------------------------------------------------------------------
    # Decision Summary tab
    # -----------------------------------------------------------------------
    tab_summary, tab_challenge = st.tabs([
        "📋 Decision Summary",
        "⚖️ Regulatory Challenge Response",
    ])

    with tab_summary:
        st.markdown(
            "<div class='section-header'>Plain-English Decision Summary</div>",
            unsafe_allow_html=True,
        )
        with st.spinner("Generating decision summary…"):
            summary = api_get(f"/audit/sessions/{session_id}/summary")

        if summary:
            decision_val = summary.get("decision", "UNKNOWN")
            confidence = summary.get("confidence", "unknown")

            col_s1, col_s2 = st.columns([2, 1])
            with col_s1:
                st.markdown(
                    f"{decision_badge(decision_val)}",
                    unsafe_allow_html=True,
                )
                st.markdown(f"**Confidence:** {confidence.upper()}")
            with col_s2:
                st.markdown(
                    f"*Generated: {format_ts(summary.get('generated_at'))}*"
                )

            st.markdown(f"\n{summary.get('summary', '')}")

            evidence = summary.get("evidence_considered", [])
            policy_basis = summary.get("policy_basis", [])
            limitations = summary.get("limitations", [])

            if evidence:
                st.markdown("**Evidence Considered:**")
                for ev in evidence:
                    st.markdown(
                        f'<div class="evidence-item">• {ev}</div>',
                        unsafe_allow_html=True,
                    )
            if policy_basis:
                st.markdown("**Policy Basis:**")
                for pb in policy_basis:
                    st.markdown(
                        f'<div class="evidence-item">📋 {pb}</div>',
                        unsafe_allow_html=True,
                    )
            if limitations:
                st.markdown("**⚠️ Limitations:**")
                for lim in limitations:
                    st.markdown(f"  - {lim}")
        else:
            st.warning("Could not generate summary for this session.")

    with tab_challenge:
        st.markdown(
            "<div class='section-header'>Regulatory Challenge Response (Draft)</div>",
            unsafe_allow_html=True,
        )
        st.info(
            "This is an automatically generated draft response to a hypothetical "
            "regulatory inquiry. Grounded exclusively in the stored audit evidence."
        )

        with st.spinner("Generating regulatory response…"):
            challenge = api_get(
                f"/audit/sessions/{session_id}/challenge-response"
            )

        if challenge:
            col_r1, col_r2 = st.columns(2)
            with col_r1:
                st.markdown(f"**Reference:** `{challenge.get('reference_number', '—')}`")
            with col_r2:
                st.markdown(f"**Generated:** {format_ts(challenge.get('generated_at'))}")

            # Key fields
            st.markdown(f"**Decision Outcome:** {challenge.get('decision_outcome', '—')}")

            data_considered = challenge.get("data_considered", [])
            if data_considered:
                st.markdown("**Data Considered:**")
                for d in data_considered:
                    st.markdown(
                        f'<div class="evidence-item">• {d}</div>',
                        unsafe_allow_html=True,
                    )

            limitations = challenge.get("limitations", [])
            if limitations:
                st.markdown("**Limitations:**")
                for lim in limitations:
                    st.markdown(f"  - {lim}")

            # Full response letter
            st.markdown("**Full Response:**")
            st.markdown(
                f'<div class="reg-response">{challenge.get("full_response", "")}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.warning("Could not generate regulatory response for this session.")


def render_session_list(sessions: list[dict]) -> None:
    """Render a list of sessions as a search results table.

    Args:
        sessions: List of session dicts from the API.
    """
    if not sessions:
        st.info("No sessions found matching the search criteria.")
        return

    st.markdown(f"**{len(sessions)} session(s) found**")

    for s in sessions:
        decision = s.get("decision", "—") or "—"
        status = s.get("status", "—")
        session_id = s.get("session_id", "—")
        user_id = s.get("user_id", "—")
        started = format_ts(s.get("started_at"))

        col1, col2, col3, col4, col5 = st.columns([3, 2, 2, 1, 1])
        with col1:
            st.markdown(f"`{session_id}`")
        with col2:
            st.markdown(user_id)
        with col3:
            st.markdown(started)
        with col4:
            st.markdown(
                decision_badge(decision) if decision != "—" else "—",
                unsafe_allow_html=True,
            )
        with col5:
            if st.button("View", key=f"view_{session_id}"):
                st.session_state["selected_session_id"] = session_id
                st.rerun()

        st.divider()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def render_sidebar() -> tuple[Optional[str], Optional[str], Optional[datetime], Optional[datetime], Optional[str]]:
    """Render the sidebar controls.

    Returns:
        Tuple of (search_session_id, user_id, start_time, end_time, decision_filter).
    """
    with st.sidebar:
        st.markdown(
            """
            <div style="text-align:center; padding: 16px 0;">
                <div style="font-size: 32px;">🔍</div>
                <div style="font-size: 18px; font-weight: 700; color: #e6edf3;">TraceLens</div>
                <div style="font-size: 11px; color: #6e7681; margin-top: 4px;">
                    AI Decision Governance<br>PS-7.1 Audit Platform
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # Direct session lookup
        st.markdown("#### 🎯 Session Lookup")
        session_id_input = st.text_input(
            "Session ID",
            value=st.session_state.get("selected_session_id", ""),
            placeholder="SESSION-XXXXXXXXXXXX",
            key="session_id_input",
        )

        if session_id_input:
            if st.button("🔍 Load Session", use_container_width=True, type="primary"):
                st.session_state["selected_session_id"] = session_id_input.strip()
                st.rerun()

        st.markdown("---")

        # Search filters
        st.markdown("#### 🔎 Search Filters")
        user_id = st.text_input(
            "User ID",
            placeholder="USER-001",
            key="filter_user_id",
        )

        decision_filter = st.selectbox(
            "Decision Outcome",
            options=["(any)", "APPROVED", "REJECTED", "NEEDS_REVIEW"],
            key="filter_decision",
        )
        decision_filter = None if decision_filter == "(any)" else decision_filter

        st.markdown("**Time Range**")
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            start_date = st.date_input("From", value=None, key="filter_start")
        with col_d2:
            end_date = st.date_input("To", value=None, key="filter_end")

        start_time = (
            datetime(start_date.year, start_date.month, start_date.day, tzinfo=timezone.utc)
            if start_date else None
        )
        end_time = (
            datetime(end_date.year, end_date.month, end_date.day, 23, 59, 59, tzinfo=timezone.utc)
            if end_date else None
        )

        if st.button("🔎 Search", use_container_width=True):
            st.session_state["search_triggered"] = True
            st.session_state["selected_session_id"] = None
            st.rerun()

        st.markdown("---")

        # Demo scenario
        st.markdown("#### 🎬 Demo Scenario")
        st.markdown(
            '<span style="font-size:12px; color:#6e7681;">Run the PS-7.1 demo: '
            'EMP-001 leave request</span>',
            unsafe_allow_html=True,
        )
        demo_request = st.text_area(
            "Request",
            value="Can employee EMP-001 take 15 consecutive days of leave?",
            height=80,
            key="demo_request",
        )
        demo_user = st.text_input("User ID", value="USER-001", key="demo_user")

        if st.button("▶ Run Demo", use_container_width=True, type="primary"):
            with st.spinner("Running agent…"):
                result = api_post(
                    "/agent/decide",
                    {
                        "request": demo_request,
                        "user_id": demo_user,
                        "session_id": None,
                    },
                )
            if result:
                new_session_id = result.get("session_id", "")
                st.session_state["selected_session_id"] = new_session_id
                st.success(f"✅ Agent completed — Decision: **{result.get('decision')}**")
                st.info(f"Session: `{new_session_id}`")
                st.rerun()

        st.markdown("---")

        # API status
        st.markdown("#### 🟢 API Status")
        health = api_get("/health")
        if health:
            st.success(f"API Online — v{health.get('app_version', '?')}")
            st.caption(f"Env: {health.get('app_env', '?')}")
        else:
            st.error("API Offline")

    return (
        session_id_input or None,
        user_id or None,
        start_time,
        end_time,
        decision_filter,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Main dashboard application entry point."""

    # Initialize session state
    if "selected_session_id" not in st.session_state:
        st.session_state["selected_session_id"] = None
    if "search_triggered" not in st.session_state:
        st.session_state["search_triggered"] = False

    # Render sidebar and get filters
    session_id_input, user_id, start_time, end_time, decision_filter = render_sidebar()

    # -----------------------------------------------------------------------
    # Main content
    # -----------------------------------------------------------------------
    selected_session = st.session_state.get("selected_session_id")

    if selected_session:
        # Show detail view
        if st.button("← Back to Search"):
            st.session_state["selected_session_id"] = None
            st.session_state["search_triggered"] = False
            st.rerun()
        render_session_detail(selected_session)

    elif st.session_state.get("search_triggered") or user_id or decision_filter or start_time:
        # Show search results
        st.markdown("## 🔎 Search Results")

        params = {}
        if user_id:
            params["user_id"] = user_id
        if decision_filter:
            params["decision"] = decision_filter
        if start_time:
            params["start_time"] = start_time.isoformat()
        if end_time:
            params["end_time"] = end_time.isoformat()

        with st.spinner("Searching audit records…"):
            result = api_get("/audit/search", params=params)

        if result:
            # Table header
            col1, col2, col3, col4, col5 = st.columns([3, 2, 2, 1, 1])
            with col1:
                st.markdown("**Session ID**")
            with col2:
                st.markdown("**User**")
            with col3:
                st.markdown("**Started**")
            with col4:
                st.markdown("**Decision**")
            with col5:
                st.markdown("**Action**")
            st.divider()

            render_session_list(result.get("sessions", []))
        else:
            st.info("No results or API unavailable.")

    else:
        # Landing page
        st.markdown(
            """
            <div style="text-align: center; padding: 60px 20px;">
                <div style="font-size: 64px; margin-bottom: 16px;">🔍</div>
                <h1 style="font-size: 32px; font-weight: 800; color: #e6edf3;">TraceLens</h1>
                <p style="font-size: 16px; color: #8b949e; max-width: 600px; margin: 0 auto 32px;">
                    Enterprise AI Decision Governance Platform<br>
                    AIVER PS-7.1 — Decision Path Auditor
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(
                """
                <div class="metric-card">
                    <div style="font-size: 28px; margin-bottom: 8px;">🔍</div>
                    <div class="metric-label">Session Lookup</div>
                    <div style="color: #8b949e; font-size: 13px; margin-top: 8px;">
                        Enter a session ID in the sidebar to view the complete decision audit trail.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                """
                <div class="metric-card">
                    <div style="font-size: 28px; margin-bottom: 8px;">🎬</div>
                    <div class="metric-label">Demo Scenario</div>
                    <div style="color: #8b949e; font-size: 13px; margin-top: 8px;">
                        Use the sidebar demo to run a live leave decision
                        and see the complete audit trail reconstructed.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col3:
            st.markdown(
                """
                <div class="metric-card">
                    <div style="font-size: 28px; margin-bottom: 8px;">🛡️</div>
                    <div class="metric-label">PII Protected</div>
                    <div style="color: #8b949e; font-size: 13px; margin-top: 8px;">
                        All audit records are PII-redacted before persistence.
                        Only sanitized evidence is displayed.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # PS-7.1 compliance summary
        st.markdown("### ✅ PS-7.1 Compliance Coverage")
        compliance_items = [
            ("Instrumented Agent Wrapper", "Captures input, tools, retrieval, decision, output events"),
            ("Decision Path Reconstruction", "Given session_id → complete structured timeline"),
            ("Decision Summary Generator", "Plain-English summary grounded in audit evidence"),
            ("PII Redaction Layer", "Applied before persistence — zero raw PII in audit store"),
            ("Queryability", "By session ID, user ID, time range, decision outcome"),
            ("Regulatory Challenge Response", "Draft formal response grounded in audit path"),
            ("LangSmith Correlation", "trace_id links TraceLens record to LangSmith observability"),
        ]
        for item, desc in compliance_items:
            st.markdown(f"✅ **{item}** — {desc}")


if __name__ == "__main__":
    main()
