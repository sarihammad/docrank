"""Cross-encoder reranker using sentence-transformers CrossEncoder."""

import logging
from typing import List

import numpy as np
from sentence_transformers import CrossEncoder  # type: ignore

from ..retrieval.result import RetrievalResult

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Reranks retrieval candidates using a cross-encoder model.

    Unlike bi-encoders, a cross-encoder reads the query and passage
    concatenated together through the same transformer stack, giving it
    full attention across both inputs.  This produces significantly more
    accurate relevance scores at the cost of O(n) forward passes.

    The standard usage pattern is:
      1. Retrieve top-100 candidates cheaply with BM25 / FAISS.
      2. Rerank with the cross-encoder to surface the best top-20.
      3. Pass the top-5 to the LLM for generation.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        batch_size: int = 32,
    ) -> None:
        """
        Args:
            model_name: HuggingFace model ID for the cross-encoder.
            batch_size: Number of (query, passage) pairs to score per batch.
                        Reduce this if running out of GPU/CPU memory.
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self._model: CrossEncoder | None = None

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    @property
    def model(self) -> CrossEncoder:
        if self._model is None:
            logger.info("Loading cross-encoder model: %s", self.model_name)
            self._model = CrossEncoder(self.model_name)
        return self._model

    # ------------------------------------------------------------------
    # Reranking
    # ------------------------------------------------------------------

    def rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: int,
    ) -> List[RetrievalResult]:
        """Score all (query, passage) pairs and return the top_k results.

        Args:
            query: The original user query.
            results: Candidate results from the retrieval stage.
            top_k: Number of results to return after reranking.

        Returns:
            Up to top_k RetrievalResult objects sorted by descending
            cross-encoder score.  The `score` field is overwritten with
            the cross-encoder logit, and `rank` is reset.
        """
        if not results:
            return []

        top_k = min(top_k, len(results))

        # Build input pairs for the cross-encoder.
        pairs = [(query, r.text) for r in results]

        # Score in batches.
        all_scores: List[float] = []
        for i in range(0, len(pairs), self.batch_size):
            batch = pairs[i : i + self.batch_size]
            batch_scores = self.model.predict(batch)
            if isinstance(batch_scores, np.ndarray):
                all_scores.extend(batch_scores.tolist())
            else:
                all_scores.extend(list(batch_scores))

        # Sort by descending score and take top_k.
        scored = sorted(
            zip(all_scores, results), key=lambda x: x[0], reverse=True
        )

        reranked: List[RetrievalResult] = []
        for rank, (score, result) in enumerate(scored[:top_k]):
            reranked.append(
                RetrievalResult(
                    chunk_id=result.chunk_id,
                    doc_id=result.doc_id,
                    score=float(score),
                    text=result.text,
                    title=result.title,
                    rank=rank,
                    metadata=dict(result.metadata),
                )
            )

        return reranked

    def predict_relevance(self, query: str, text: str) -> float:
        """Score a single (query, passage) pair.

        Args:
            query: Query string.
            text: Passage text.

        Returns:
            Cross-encoder relevance logit (higher = more relevant).
        """
        score = self.model.predict([(query, text)])
        if isinstance(score, np.ndarray):
            return float(score[0])
        return float(score)
