"""RAG retrieval service.

Handles querying the FAISS vector store and returning structured
retrieval results with metadata.
"""

from __future__ import annotations

import os
import structlog
from app.rag.vector_store import DeterministicVectorStore

from app.schemas.retrieval import RetrievedDocument, RetrievalResult

logger = structlog.get_logger(__name__)


class RetrieverService:
    """Service for retrieving relevant policy documents from the vector store.

    This service handles:
    - Loading the index
    - Executing similarity searches
    - Formatting results into structured objects
    - Including metadata for auditability
    """

    def __init__(
        self,
        vector_store_path: str = "./data/.vectorstore",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        top_k: int = 5,
    ):
        """Initialize the retriever service.

        Args:
            vector_store_path: Path to the persisted index.
            embedding_model: Embedding model identifier (kept for compatibility).
            top_k: Number of documents to retrieve per query.

        Raises:
            FileNotFoundError: If the vector store has not been created yet.
        """
        if not os.path.exists(vector_store_path):
            logger.error("vector_store_missing", path=vector_store_path)
            raise FileNotFoundError(
                f"Vector store not found at {vector_store_path}. "
                "Run ingestion first with: python -m app.rag.ingest"
            )

        self.vector_store_path = vector_store_path
        self.embedding_model = embedding_model
        self.top_k = top_k

        self._embeddings = None
        self.vector_store = None

        logger.info("retriever_initialized", vector_store_path=vector_store_path)

    def load(self, vector_store_path: str | None = None) -> None:
        """Load a persisted vector store from disk."""
        if vector_store_path is not None:
            self.vector_store_path = vector_store_path

        if not os.path.exists(self.vector_store_path):
            logger.error("vector_store_missing", path=self.vector_store_path)
            raise FileNotFoundError(
                f"Vector store not found at {self.vector_store_path}. "
                "Run ingestion first with: python -m app.rag.ingest"
            )

        self.vector_store = DeterministicVectorStore.load_local(
            self.vector_store_path
        )
        logger.info("vector_store_loaded", path=self.vector_store_path)

    def retrieve(self, query: str, top_k: int | None = None) -> RetrievalResult:
        """Retrieve relevant documents for a query.

        Args:
            query: The query string.
            top_k: Number of documents to retrieve. Uses instance default if None.

        Returns:
            RetrievalResult with structured documents and metadata.

        Raises:
            ValueError: If the query is empty or retrieval fails.
        """
        if not query or not query.strip():
            logger.error("empty_query", query=query)
            raise ValueError("Query cannot be empty")

        k = top_k or self.top_k
        logger.info("retrieval_started", query=query, top_k=k)

        try:
            if self.vector_store is None:
                self.load()

            # Perform similarity search with scores
            results = self.vector_store.similarity_search_with_score(query, k=k)

            documents = []
            for doc, score in results:
                retrieved_doc = RetrievedDocument(
                    content=doc.page_content,
                    source=doc.metadata.get("source", "unknown"),
                    chunk_id=doc.metadata.get("chunk_id", "unknown"),
                    document_id=doc.metadata.get("document_id", "unknown"),
                    metadata={
                        "similarity_score": str(score),
                        **{
                            k: v
                            for k, v in doc.metadata.items()
                            if k not in ["source", "chunk_id", "document_id"]
                        },
                    },
                )
                documents.append(retrieved_doc)

            result = RetrievalResult(
                documents=documents, query=query, total_retrieved=len(documents)
            )

            logger.info(
                "retrieval_completed",
                query=query,
                total_retrieved=len(documents),
            )

            return result

        except Exception as e:
            logger.error("retrieval_failed", query=query, error=str(e))
            raise

    def format_context(self, retrieval_result: RetrievalResult) -> str:
        """Format retrieved documents into a context string for the LLM.

        Args:
            retrieval_result: RetrievalResult from retrieve().

        Returns:
            Formatted context string ready for LLM consumption.
        """
        context_parts = [
            f"Retrieved Policy Context ({len(retrieval_result.documents)} documents):\n"
        ]

        for i, doc in enumerate(retrieval_result.documents, 1):
            context_parts.append(f"\n--- Document {i}: {doc.source} ---")
            context_parts.append(f"Section: {doc.metadata.get('section', 'N/A')}")
            context_parts.append(f"Content:\n{doc.content}")

        return "\n".join(context_parts)
