"""Schemas for RAG retrieval results.

Retrieved documents include metadata for auditability and traceability.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class RetrievedDocument(BaseModel):
    """A single retrieved document chunk.

    Attributes:
        content: The text content of the document chunk.
        source: The source file or document identifier.
        chunk_id: Unique identifier for this chunk within the document.
        document_id: Unique identifier for the full document.
        metadata: Additional metadata about the chunk.
    """

    content: str = Field(..., description="Text content of the retrieved chunk")
    source: str = Field(..., description="Source document or file identifier")
    chunk_id: str = Field(..., description="Unique chunk identifier")
    document_id: str = Field(..., description="Full document identifier")
    metadata: dict[str, str] = Field(
        default_factory=dict, description="Additional chunk metadata"
    )

    class Config:
        """Pydantic configuration."""

        json_schema_extra = {
            "example": {
                "content": "Employees may carry forward a maximum of 5 days...",
                "source": "hr_leave_policy.md",
                "chunk_id": "chunk_0003",
                "document_id": "doc_hr_leave_policy",
                "metadata": {"section": "Carry-forward Rules", "page": "2"},
            }
        }


class RetrievalResult(BaseModel):
    """Results from a vector store retrieval query.

    Attributes:
        documents: List of retrieved document chunks.
        query: The query that was executed.
        total_retrieved: Number of documents retrieved.
    """

    documents: list[RetrievedDocument] = Field(..., description="Retrieved documents")
    query: str = Field(..., description="The retrieval query")
    total_retrieved: int = Field(..., description="Number of documents retrieved")

    class Config:
        """Pydantic configuration."""

        json_schema_extra = {
            "example": {
                "documents": [
                    {
                        "content": "...",
                        "source": "hr_leave_policy.md",
                        "chunk_id": "chunk_0001",
                        "document_id": "doc_hr_leave_policy",
                        "metadata": {},
                    }
                ],
                "query": "maximum consecutive leave",
                "total_retrieved": 1,
            }
        }
