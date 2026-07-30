"""Retrieval-augmented generation package: ingestion, retrieval, vector store.

Provides document ingestion pipeline and retrieval service for policy documents.
"""

from app.rag.ingest import DocumentIngestionPipeline, run_ingestion
from app.rag.retriever import RetrieverService

__all__ = [
    "DocumentIngestionPipeline",
    "RetrieverService",
    "run_ingestion",
]
