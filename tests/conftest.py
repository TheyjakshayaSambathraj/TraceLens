"""Shared pytest fixtures and configuration.

Provides shared fixtures for testing, including mocked embeddings,
test data, and isolated temporary directories for tests.
"""

from __future__ import annotations

from typing import Iterator, Any
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.main import create_app


@pytest.fixture()
def client() -> Iterator[TestClient]:
    """Provide a FastAPI test client backed by a freshly constructed app.

    Yields:
        A ``TestClient`` for issuing requests against the application
        without running a real ASGI server.
    """
    application = create_app()
    with TestClient(application) as test_client:
        yield test_client


@pytest.fixture
def mock_embeddings():
    """Create a mock embeddings model that returns deterministic vectors.
    
    This avoids downloading the real sentence-transformers model during tests.
    Returns a 384-dimensional vector (matching all-MiniLM-L6-v2 dimensions).
    """
    mock = MagicMock()
    
    def embed_documents(texts: list[str]) -> list[list[float]]:
        """Return deterministic embeddings based on text hash."""
        embeddings = []
        for text in texts:
            # Create a deterministic 384-dim vector based on text hash
            hash_val = hash(text) % 1000
            vector = [float(hash_val % 384) / 384.0 for _ in range(384)]
            embeddings.append(vector)
        return embeddings
    
    def embed_query(text: str) -> list[float]:
        """Return deterministic embedding for a query."""
        hash_val = hash(text) % 1000
        return [float(hash_val % 384) / 384.0 for _ in range(384)]
    
    mock.embed_documents = embed_documents
    mock.embed_query = embed_query
    return mock


@pytest.fixture
def temp_policy_dir():
    """Create a temporary directory with a test policy document."""
    temp_dir = tempfile.mkdtemp()
    policy_path = Path(temp_dir) / "test_policy.md"
    policy_path.write_text(
        """# Test HR Leave Policy

## Annual Leave Entitlements

- Standard employees: 20 working days per year
- Senior employees: 25 working days per year
- Manager-level: 30 working days per year

## Maximum Consecutive Leave

- Standard employees: Maximum 10 consecutive days without special approval
- Senior/Management: Maximum 15 consecutive days
- Restricted periods: November-December and June-July require manager approval

## Approval Requirements

- Manager approval required for leave exceeding 5 consecutive days
- HR approval required for leave exceeding 10 consecutive days
- Executive approval required for leave exceeding 15 consecutive days

## Carry-forward and Expiry

- Up to 5 days can be carried forward to next year
- Carried-over leave must be used by June 30 or expires
- No carry-forward is permitted for leave not used by December 31
"""
    )
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def temp_vector_store():
    """Create a temporary directory for the FAISS vector store."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)
