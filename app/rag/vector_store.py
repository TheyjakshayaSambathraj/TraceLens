"""Lightweight, deterministic vector store replacement for policy retrieval.

Replaces heavy PyTorch / SentenceTransformers / FAISS dependencies with a
lightweight, zero-overhead term-matching and keyword-relevance retriever.
Preserves FAISS-compatible interface (from_documents, load_local, save_local,
similarity_search_with_score) while consuming <1MB RAM.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import structlog
from langchain_core.documents import Document

logger = structlog.get_logger(__name__)


class DeterministicVectorStore:
    """Lightweight in-memory document store with deterministic text search.

    Mimics FAISS interface without requiring PyTorch, sentence-transformers, or FAISS binaries.
    """

    def __init__(self, documents: list[Document] | None = None):
        self.documents: list[Document] = documents or []

    @classmethod
    def from_documents(
        cls, documents: list[Document], embeddings: Any = None
    ) -> DeterministicVectorStore:
        """Construct store from a list of LangChain Document objects."""
        return cls(documents=list(documents))

    def save_local(self, folder_path: str, index_name: str = "index") -> None:
        """Persist documents and metadata to disk as JSON."""
        os.makedirs(folder_path, exist_ok=True)
        file_path = Path(folder_path) / f"{index_name}.json"

        serialized = []
        for doc in self.documents:
            serialized.append(
                {
                    "page_content": doc.page_content,
                    "metadata": doc.metadata,
                }
            )

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(serialized, f, indent=2, ensure_ascii=False)

        logger.info("vector_store_saved", path=str(file_path), count=len(serialized))

    @classmethod
    def load_local(
        cls,
        folder_path: str,
        embeddings: Any = None,
        index_name: str = "index",
        allow_dangerous_deserialization: bool = True,
    ) -> DeterministicVectorStore:
        """Load persisted documents from disk JSON."""
        folder = Path(folder_path)
        file_path = folder / f"{index_name}.json"

        if not file_path.exists():
            json_files = list(folder.glob("*.json"))
            if json_files:
                file_path = json_files[0]
            else:
                raise FileNotFoundError(f"Vector store index not found at {folder_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        docs = [
            Document(
                page_content=item["page_content"],
                metadata=item.get("metadata", {}),
            )
            for item in raw_data
        ]
        logger.info("vector_store_loaded", path=str(file_path), count=len(docs))
        return cls(documents=docs)

    def similarity_search_with_score(
        self, query: str, k: int = 5
    ) -> list[tuple[Document, float]]:
        """Perform deterministic relevance search for query terms against documents.

        Returns list of (Document, score) tuples ordered by descending score.
        """
        if not query or not query.strip():
            return []

        query_terms = [
            term.lower()
            for term in re.findall(r"\w+", query)
            if len(term) > 1
        ]

        scored_docs: list[tuple[Document, float, int]] = []

        for idx, doc in enumerate(self.documents):
            content_lower = doc.page_content.lower()

            matches = 0
            for term in query_terms:
                if term in content_lower:
                    matches += 1

            if query_terms:
                match_score = matches / len(query_terms)
            else:
                match_score = 0.0

            if query.strip().lower() in content_lower:
                match_score += 0.5

            if match_score > 0:
                final_score = round(min(0.99, 0.5 + match_score * 0.49), 4)
            else:
                final_score = round(max(0.01, 0.1 - (idx * 0.001)), 4)

            scored_docs.append((doc, final_score, idx))

        scored_docs.sort(key=lambda item: (-item[1], item[2]))

        return [(doc, score) for doc, score, _ in scored_docs[:k]]
