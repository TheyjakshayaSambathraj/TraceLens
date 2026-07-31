"""Tests for the RAG retriever service.

Unit tests use mocked embeddings and vector stores for speed and determinism.
Integration tests with real embeddings are marked with @pytest.mark.integration.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock
from app.schemas.retrieval import RetrievedDocument, RetrievalResult


class TestRetrieverServiceUnit:
    """Fast unit tests for retriever logic using mocked components."""

    def test_format_context_joins_documents(self):
        """Test that format_context properly joins retrieved documents."""
        from app.rag.retriever import RetrieverService
        
        retriever = RetrieverService.__new__(RetrieverService)  # Create without __init__
        
        mock_doc1 = RetrievedDocument(
            content="Policy section 1",
            source="policy.md",
            chunk_id="chunk_1",
            document_id="doc_1",
            metadata={}
        )
        mock_doc2 = RetrievedDocument(
            content="Policy section 2",
            source="policy.md",
            chunk_id="chunk_2",
            document_id="doc_2",
            metadata={}
        )
        
        mock_result = RetrievalResult(
            documents=[mock_doc1, mock_doc2],
            query="leave policy",
            total_retrieved=2
        )
        
        context = retriever.format_context(mock_result)
        
        assert isinstance(context, str)
        assert "Policy section 1" in context
        assert "Policy section 2" in context

    def test_retrieved_document_schema(self):
        """Test RetrievedDocument schema validation."""
        doc = RetrievedDocument(
            content="Test content",
            source="test.md",
            chunk_id="chunk_1",
            document_id="doc_1",
            metadata={"key": "value"}
        )
        
        assert doc.content == "Test content"
        assert doc.source == "test.md"
        assert doc.chunk_id == "chunk_1"
        assert doc.metadata == {"key": "value"}

    def test_retrieval_result_schema(self):
        """Test RetrievalResult schema validation."""
        doc1 = RetrievedDocument(
            content="Content 1",
            source="doc.md",
            chunk_id="c1",
            document_id="d1",
            metadata={}
        )
        doc2 = RetrievedDocument(
            content="Content 2",
            source="doc.md",
            chunk_id="c2",
            document_id="d2",
            metadata={}
        )
        
        result = RetrievalResult(
            documents=[doc1, doc2],
            query="test query",
            total_retrieved=2
        )
        
        assert result.total_retrieved == 2
        assert len(result.documents) == 2
        assert result.query == "test query"


# Integration tests marked for optional execution
@pytest.mark.integration
class TestRetrieverServiceIntegration:
    """Integration tests using real FAISS vector store and embeddings.
    
    Run with: pytest -m integration
    """

    @pytest.fixture
    def ingested_retriever(self, temp_policy_dir, temp_vector_store):
        """Create a retriever with a real ingested policy (slow setup).
        
        This downloads the embedding model and builds the FAISS index.
        Only used for integration tests marked with @pytest.mark.integration.
        """
        from app.rag.ingest import DocumentIngestionPipeline
        from app.rag.retriever import RetrieverService
        
        pipeline = DocumentIngestionPipeline(vector_store_path=temp_vector_store)
        vector_store = pipeline.ingest(temp_policy_dir)
        pipeline.save_vector_store(vector_store)
        
        retriever = RetrieverService(vector_store_path=temp_vector_store)
        return retriever

    def test_retrieve_with_valid_query(self, ingested_retriever):
        """Test retrieval with real embeddings."""
        result = ingested_retriever.retrieve("maximum consecutive leave")
        
        assert result.documents is not None
        assert len(result.documents) > 0
        assert result.query == "maximum consecutive leave"
        assert result.total_retrieved > 0

    def test_retrieve_documents_have_metadata(self, ingested_retriever):
        """Test retrieved documents have all metadata with real embeddings."""
        result = ingested_retriever.retrieve("leave balance")
        
        for doc in result.documents:
            assert doc.content
            assert doc.source
            assert doc.chunk_id
            assert doc.document_id
            assert doc.metadata is not None

    def test_retrieve_with_empty_query_raises_error(self, ingested_retriever):
        """Test that empty queries raise ValueError."""
        with pytest.raises(ValueError, match="Query cannot be empty"):
            ingested_retriever.retrieve("")

    def test_retrieve_with_whitespace_query_raises_error(self, ingested_retriever):
        """Test that whitespace-only queries raise ValueError."""
        with pytest.raises(ValueError, match="Query cannot be empty"):
            ingested_retriever.retrieve("   ")

    def test_retrieve_with_different_queries(self, ingested_retriever):
        """Test retrieval with different queries."""
        queries = ["leave policy", "manager approval", "consecutive days"]
        
        for query in queries:
            result = ingested_retriever.retrieve(query)
            assert result.total_retrieved > 0
            assert result.query == query

    def test_retrieve_with_whitespace_query_raises_error(self, ingested_retriever):
        """Test that whitespace-only queries raise ValueError."""
        with pytest.raises(ValueError):
            ingested_retriever.retrieve("   ")

    def test_retrieve_respects_top_k(self, ingested_retriever):
        """Test that retrieve respects the top_k parameter."""
        result = ingested_retriever.retrieve("leave", top_k=2)
        assert result.total_retrieved <= 2

    def test_format_context(self, ingested_retriever):
        """Test context formatting for LLM."""
        result = ingested_retriever.retrieve("manager approval")
        context = ingested_retriever.format_context(result)

        assert "Retrieved Policy Context" in context
        assert "test_policy.md" in context
        assert len(context) > 0

    def test_missing_vector_store_raises_error(self, temp_vector_store):
        """Test that missing vector store raises FileNotFoundError."""
        from app.rag.retriever import RetrieverService
        with pytest.raises(FileNotFoundError):
            RetrieverService(vector_store_path="/nonexistent/path")
