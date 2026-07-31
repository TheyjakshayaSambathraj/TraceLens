"""RAG document ingestion pipeline.

Loads policy documents, chunks them, creates embeddings, and stores
them in a FAISS vector store for later retrieval.

The ingestion process is idempotent and can be run multiple times
to rebuild the index.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import structlog
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from app.rag.vector_store import DeterministicVectorStore

logger = structlog.get_logger(__name__)


class DocumentIngestionPipeline:
    """Pipeline for ingesting policy documents into a vector store.

    This class handles the complete ingestion workflow:
    - Loading documents from the file system
    - Splitting documents into overlapping chunks
    - Creating embeddings using a sentence transformer
    - Storing embeddings in FAISS
    - Persisting the index to disk
    """

    def __init__(
        self,
        vector_store_path: str = "./data/.vectorstore",
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        """Initialize the ingestion pipeline.

        Args:
            vector_store_path: Directory where the FAISS index will be stored.
            chunk_size: Number of characters per chunk.
            chunk_overlap: Number of overlapping characters between chunks.
            embedding_model: HuggingFace embedding model identifier.
        """
        self.vector_store_path = vector_store_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = embedding_model
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )

    def load_documents_from_directory(self, directory: str) -> list:
        """Load all markdown files from a directory.

        Args:
            directory: Path to directory containing policy documents.

        Returns:
            List of LangChain Document objects.

        Raises:
            FileNotFoundError: If the directory does not exist.
        """
        directory_path = Path(directory)
        if not directory_path.exists():
            logger.error("directory_not_found", directory=directory)
            raise FileNotFoundError(f"Directory not found: {directory}")

        documents = []
        for file_path in directory_path.glob("*.md"):
            logger.info("loading_document", file=file_path.name)
            try:
                loader = TextLoader(str(file_path), encoding="utf-8")
                docs = loader.load()
                for doc in docs:
                    doc.metadata["source"] = file_path.name
                    doc.metadata["document_id"] = file_path.stem
                documents.extend(docs)
            except Exception as e:
                logger.error(
                    "document_load_failed", file=file_path.name, error=str(e)
                )
                raise

        logger.info("documents_loaded", count=len(documents))
        return documents

    def chunk_documents(self, documents: list) -> list:
        """Split documents into overlapping chunks.

        Args:
            documents: List of LangChain Document objects.

        Returns:
            List of chunked Document objects.
        """
        chunks = self.text_splitter.split_documents(documents)
        logger.info("documents_chunked", total_chunks=len(chunks))

        # Assign chunk IDs
        for idx, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = f"chunk_{idx:04d}"

        return chunks

    def create_embeddings_and_store(
        self, chunks: list, use_gpu: bool = False
    ) -> DeterministicVectorStore:
        """Create document store for chunks.

        Args:
            chunks: List of chunked Document objects.
            use_gpu: Parameter kept for signature compatibility.

        Returns:
            DeterministicVectorStore instance.
        """
        logger.info(
            "creating_document_store", count=len(chunks)
        )

        vector_store = DeterministicVectorStore.from_documents(chunks)
        logger.info("vector_store_created", num_documents=len(chunks))

        return vector_store

    def ingest(self, policy_directory: str) -> DeterministicVectorStore:
        """Run the complete ingestion pipeline.

        Args:
            policy_directory: Path to directory containing policy documents.

        Returns:
            DeterministicVectorStore instance.

        Raises:
            FileNotFoundError: If policy directory does not exist.
            Exception: If ingestion fails at any stage.
        """
        logger.info("ingestion_started", directory=policy_directory)

        documents = self.load_documents_from_directory(policy_directory)
        chunks = self.chunk_documents(documents)
        vector_store = self.create_embeddings_and_store(chunks)

        logger.info("ingestion_completed", total_chunks=len(chunks))

        return vector_store

    def save_vector_store(self, vector_store: DeterministicVectorStore) -> None:
        """Persist the index to disk.

        Args:
            vector_store: DeterministicVectorStore to persist.
        """
        os.makedirs(self.vector_store_path, exist_ok=True)
        vector_store.save_local(self.vector_store_path)
        logger.info("vector_store_saved", path=self.vector_store_path)

    def load_vector_store(self) -> DeterministicVectorStore:
        """Load a persisted index from disk.

        Returns:
            DeterministicVectorStore instance.

        Raises:
            FileNotFoundError: If the vector store has not been created yet.
        """
        if not os.path.exists(self.vector_store_path):
            logger.error(
                "vector_store_not_found", path=self.vector_store_path
            )
            raise FileNotFoundError(
                f"Vector store not found at {self.vector_store_path}. "
                "Run ingestion first."
            )

        vector_store = DeterministicVectorStore.load_local(
            self.vector_store_path
        )
        logger.info("vector_store_loaded", path=self.vector_store_path)

        return vector_store


def run_ingestion(
    policy_directory: str = "./data/policies",
    vector_store_path: str = "./data/.vectorstore",
) -> None:
    """Standalone ingestion runner.

    Can be invoked as:
        python -m app.rag.ingest

    Args:
        policy_directory: Path to policy documents directory.
        vector_store_path: Path where FAISS index will be saved.
    """
    pipeline = DocumentIngestionPipeline(vector_store_path=vector_store_path)
    vector_store = pipeline.ingest(policy_directory)
    pipeline.save_vector_store(vector_store)
    logger.info(
        "ingestion_script_completed",
        policy_directory=policy_directory,
        vector_store_path=vector_store_path,
    )


if __name__ == "__main__":
    from app.config.logging_config import configure_logging
    from app.config.settings import get_settings

    settings = get_settings()
    configure_logging(settings)

    run_ingestion()
