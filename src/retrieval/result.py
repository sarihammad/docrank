"""Shared RetrievalResult dataclass used by all retrievers."""

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class RetrievalResult:
    """A single retrieved passage with its score and provenance.

    Attributes:
        chunk_id: Unique ID of the source chunk.
        doc_id: ID of the parent document.
        score: Retrieval score (BM25, cosine similarity, or RRF).
        text: Text content of the chunk.
        title: Title of the parent document.
        rank: Zero-based rank position within the result list.
        metadata: Arbitrary extra metadata inherited from the chunk.
    """

    chunk_id: str
    doc_id: str
    score: float
    text: str
    title: str = ""
    rank: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        preview = self.text[:60].replace("\n", " ")
        return (
            f"RetrievalResult(rank={self.rank}, score={self.score:.4f}, "
            f"chunk_id={self.chunk_id!r}, text={preview!r}...)"
        )
