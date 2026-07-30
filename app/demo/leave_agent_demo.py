"""Demo script for the TraceLens leave decision agent with observability.

This script demonstrates the complete Phase 3 workflow:
1. LangSmith configuration
2. Document ingestion
3. RAG retrieval
4. Employee lookup
5. LLM-based decision making
6. Response generation
7. Execution event observability

Run with:
    python -m app.demo.leave_agent_demo

Requires:
    - GOOGLE_API_KEY environment variable (for real Gemini)
    - Or runs with mock LLM if key is not provided

Optional:
    - LANGCHAIN_TRACING_V2=true
    - LANGCHAIN_API_KEY=<your-langsmith-key>
    - LANGCHAIN_PROJECT=<project-name>
"""

from __future__ import annotations

import sys
import structlog
from pathlib import Path

from app.config.settings import get_settings
from app.config.logging_config import configure_logging
from app.observability.config import get_langsmith_config
from app.observability.instrumentation import create_instrumented_agent
from app.observability.publisher import get_publisher
from app.observability.events import ExecutionEvent
from app.rag.ingest import DocumentIngestionPipeline
from app.rag.retriever import RetrieverService
from app.services.employee import EmployeeService
from app.services.llm_provider import create_llm_provider
from app.agent.graph import LeaveDecisionAgent

logger = structlog.get_logger(__name__)


def run_demo():
    """Run the complete demo workflow with instrumentation."""
    settings = get_settings()
    configure_logging(settings)

    logger.info("=== TraceLens Leave Decision Agent Demo ===")
    logger.info("Phase 3: LangGraph Workflow + RAG + Tools + Observability")

    # Initialize LangSmith configuration
    logger.info("\n[CONFIG] Configuring LangSmith observability...")
    langsmith_config = get_langsmith_config()
    if langsmith_config.is_configured():
        logger.info("✓ LangSmith tracing configured", project=langsmith_config.project)
    else:
        logger.info("ℹ LangSmith tracing disabled (no API key configured)")
        logger.info("  To enable: set LANGCHAIN_API_KEY environment variable")

    # Setup event listener to show observable events
    publisher = get_publisher()
    
    def log_event(event: ExecutionEvent) -> None:
        """Log execution events for demo visibility."""
        logger.info(
            "EVENT",
            event_type=event.event_type.value,
            sequence=event.sequence,
            session_id=event.session_id,
        )
    
    publisher.subscribe("*", log_event)

    # Step 1: Ingest policy documents
    logger.info("\n[STEP 1] Ingesting policy documents...")
    pipeline = DocumentIngestionPipeline(
        vector_store_path="./data/.vectorstore",
    )

    try:
        vector_store = pipeline.ingest(policy_directory="./data/policies")
        pipeline.save_vector_store(vector_store)
        logger.info("✓ Policy documents ingested successfully")
    except Exception as e:
        logger.error("✗ Failed to ingest policy documents", error=str(e))
        sys.exit(1)

    # Step 2: Initialize services
    logger.info("\n[STEP 2] Initializing services...")
    try:
        retriever = RetrieverService(vector_store_path="./data/.vectorstore")
        logger.info("✓ Retriever initialized")

        employee_service = EmployeeService()
        logger.info("✓ Employee service initialized")

        # Use mock LLM if no API key is provided (for demo)
        use_mock = not settings.google_api_key
        llm_provider = create_llm_provider(settings, use_mock=use_mock)
        logger.info(
            f"✓ LLM provider initialized (mock={use_mock})"
        )

    except Exception as e:
        logger.error("✗ Failed to initialize services", error=str(e))
        sys.exit(1)

    # Step 3: Create agent with instrumentation
    logger.info("\n[STEP 3] Creating instrumented agent...")
    try:
        agent = LeaveDecisionAgent(
            retriever=retriever,
            employee_service=employee_service,
            llm_provider=llm_provider,
        )
        logger.info("✓ Base agent created")

        # Wrap with instrumentation
        instrumented_agent = create_instrumented_agent(agent)
        logger.info("✓ Agent wrapped with observability instrumentation")
    except Exception as e:
        logger.error("✗ Failed to create agent", error=str(e))
        sys.exit(1)

    # Step 4: Run sample requests
    logger.info("\n[STEP 4] Running sample leave requests with observability...")

    test_requests = [
        {
            "request": "Can employee EMP-001 take 15 consecutive days of leave starting next Monday?",
            "description": "Senior engineer - requesting 15 days (within senior limit)",
        },
        {
            "request": "Can employee EMP-002 take 8 days of leave?",
            "description": "Finance employee - requesting 8 days",
        },
        {
            "request": "Can employee EMP-003 take 25 days of leave?",
            "description": "Junior HR - requesting 25 days (exceeds leave balance)",
        },
    ]

    for i, test_case in enumerate(test_requests, 1):
        logger.info(f"\n--- Test Case {i}: {test_case['description']} ---")
        logger.info(f"Request: {test_case['request']}")

        try:
            # Execute using instrumented agent
            # This will emit observable events through the publisher
            result = instrumented_agent.decide(
                request=test_case["request"],
                user_id=f"USER-DEMO-{i}",
                session_id=f"SESSION-DEMO-{i:03d}",
            )

            # Display results
            logger.info(
                f"Session: {result.get('session_id')}"
            )
            logger.info(
                f"Employee: {result.get('employee_data').employee_id}"
            )

            decision = result.get("decision")
            logger.info(f"Decision: {decision.decision.value}")
            logger.info(f"Reason: {decision.reason}")

            if decision.policy_references:
                logger.info(f"Policy References: {', '.join(decision.policy_references)}")

            logger.info("\nFinal Response:")
            logger.info(result.get("final_response"))

        except Exception as e:
            logger.error(f"✗ Test case {i} failed", error=str(e))
            continue

    logger.info("\n=== Demo Complete ===")
    logger.info("Events were published to the event bus throughout execution.")
    logger.info("In Phase 4, these events will be persisted as governance audit records.")


if __name__ == "__main__":
    run_demo()
