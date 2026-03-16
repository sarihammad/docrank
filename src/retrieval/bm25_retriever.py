"""BM25 sparse retriever backed by rank_bm25."""

import logging
import pickle
import re
from pathlib import Path
from typing import List, Set

from rank_bm25 import BM25Okapi  # type: ignore

from ..ingestion.document import Chunk
from .result import RetrievalResult

logger = logging.getLogger(__name__)

# Common English stopwords (lightweight, no NLTK dependency).
_STOPWORDS: Set[str] = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "up", "about", "into", "through", "during",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "could", "should", "may", "might",
    "it", "its", "this", "that", "these", "those", "i", "we", "you", "he",
    "she", "they", "what", "which", "who", "whom", "not", "no", "nor",
}


def _tokenize(text: str) -> List[str]:
    """Lowercase, strip punctuation, remove stopwords."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


class BM25Retriever:
    """Sparse BM25 retriever using BM25Okapi from rank_bm25.

    Build the index once from a list of Chunks, then call search() for
    ranked retrieval.  The index can be persisted to disk and reloaded.
    """

    def __init__(self) -> None:
        self._bm25: BM25Okapi | None = None
        self._chunks: List[Chunk] = []
        self._tokenized_corpus: List[List[str]] = []

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def build_index(self, chunks: List[Chunk]) -> None:
        """Build a BM25 index from a list of text chunks.

        Args:
            chunks: Chunked documents to index.
        """
        if not chunks:
            raise ValueError("Cannot build BM25 index from empty chunk list.")

        logger.info("Building BM25 index over %d chunks…", len(chunks))
        self._chunks = list(chunks)
        self._tokenized_corpus = [_tokenize(c.text) for c in self._chunks]
        self._bm25 = BM25Okapi(self._tokenized_corpus)
        logger.info("BM25 index built.")

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, k: int) -> List[RetrievalResult]:
        """Retrieve the top-k chunks by BM25 score.

        Args:
            query: Natural-language query string.
            k: Number of results to return.

        Returns:
            List of RetrievalResult sorted by descending BM25 score.

        Raises:
            RuntimeError: If the index has not been built yet.
        """
        if self._bm25 is None:
            raise RuntimeError(
                "BM25 index is not built. Call build_index() first."
            )

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)
        k = min(k, len(self._chunks))

        # Partial sort — O(n log k) instead of O(n log n).
        import numpy as np

        top_indices = np.argpartition(scores, -k)[-k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        results: List[RetrievalResult] = []
        for rank, idx in enumerate(top_indices):
            chunk = self._chunks[int(idx)]
            results.append(
                RetrievalResult(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    score=float(scores[int(idx)]),
                    text=chunk.text,
                    title=chunk.title,
                    rank=rank,
                    metadata=dict(chunk.metadata),
                )
            )
        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Serialize the BM25 index and chunk list to disk.

        Args:
            path: File path (typically ending in .pkl).
        """
        if self._bm25 is None:
            raise RuntimeError("Nothing to save — index has not been built.")

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "bm25": self._bm25,
            "chunks": self._chunks,
            "tokenized_corpus": self._tokenized_corpus,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("BM25 index saved to %s.", path)

    def load(self, path: str) -> None:
        """Load a previously saved BM25 index from disk.

        Args:
            path: File path produced by save().

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        if not Path(path).is_file():
            raise FileNotFoundError(f"BM25 index file not found: {path}")

        with open(path, "rb") as f:
            payload = pickle.load(f)

        self._bm25 = payload["bm25"]
        self._chunks = payload["chunks"]
        self._tokenized_corpus = payload.get("tokenized_corpus", [])
        logger.info(
            "BM25 index loaded from %s (%d chunks).",
            path,
            len(self._chunks),
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_built(self) -> bool:
        return self._bm25 is not None

    @property
    def num_chunks(self) -> int:
        return len(self._chunks)
