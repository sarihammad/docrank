"""Hybrid retriever combining BM25 and dense search via Reciprocal Rank Fusion."""

import logging
from typing import Dict, List

from .bm25_retriever import BM25Retriever
from .dense_retriever import DenseRetriever
from .result import RetrievalResult

logger = logging.getLogger(__name__)

# Standard RRF constant.  k=60 is the value used in the original Cormack et al.
# paper and remains the de-facto default.
_RRF_K = 60


class HybridRetriever:
    """Combines BM25 and dense retrieval via Reciprocal Rank Fusion (RRF).

    RRF score for a document d given N ranked lists:
        score(d) = Σ_i  weight_i / (k + rank_i(d))

    where rank_i(d) is the 1-based position of d in list i, and k=60 dampens
    the influence of very high ranks.  Weights are applied per-list and allow
    tuning the relative contribution of sparse vs. dense signals.
    """

    def __init__(
        self,
        bm25: BM25Retriever,
        dense: DenseRetriever,
        bm25_weight: float = 0.5,
        dense_weight: float = 0.5,
    ) -> None:
        """
        Args:
            bm25: Initialised BM25Retriever.
            dense: Initialised DenseRetriever.
            bm25_weight: Weight applied to BM25 RRF scores.
            dense_weight: Weight applied to dense RRF scores.
        """
        self.bm25 = bm25
        self.dense = dense
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight

    # ------------------------------------------------------------------
    # Hybrid search
    # ------------------------------------------------------------------

    def search(self, query: str, k: int) -> List[RetrievalResult]:
        """Retrieve top-k results via weighted Reciprocal Rank Fusion.

        Args:
            query: Natural-language query.
            k: Number of merged results to return.

        Returns:
            Fused and re-ranked list of RetrievalResult objects.
        """
        # Retrieve a generous pool from each system before fusion.
        retrieval_k = max(k * 2, 100)

        bm25_results = self.bm25.search(query, retrieval_k)
        dense_results = self.dense.search(query, retrieval_k)

        fused = self._reciprocal_rank_fusion(
            ranked_lists=[bm25_results, dense_results],
            weights=[self.bm25_weight, self.dense_weight],
        )
        return fused[:k]

    def search_bm25_only(self, query: str, k: int) -> List[RetrievalResult]:
        """BM25-only retrieval (for ablation studies).

        Args:
            query: Natural-language query.
            k: Number of results to return.

        Returns:
            BM25 results with ranks reset from 0.
        """
        results = self.bm25.search(query, k)
        for i, r in enumerate(results):
            r.rank = i
        return results

    def search_dense_only(self, query: str, k: int) -> List[RetrievalResult]:
        """Dense-only retrieval (for ablation studies).

        Args:
            query: Natural-language query.
            k: Number of results to return.

        Returns:
            Dense results with ranks reset from 0.
        """
        results = self.dense.search(query, k)
        for i, r in enumerate(results):
            r.rank = i
        return results

    # ------------------------------------------------------------------
    # RRF implementation
    # ------------------------------------------------------------------

    @staticmethod
    def _reciprocal_rank_fusion(
        ranked_lists: List[List[RetrievalResult]],
        weights: List[float],
    ) -> List[RetrievalResult]:
        """Compute weighted RRF scores and merge multiple ranked lists.

        Args:
            ranked_lists: Each element is a list of RetrievalResult objects
                sorted by descending score (rank 0 = best).
            weights: Per-list weight multipliers, same length as ranked_lists.

        Returns:
            Merged list sorted by descending fused RRF score.
        """
        # chunk_id -> accumulated RRF score
        rrf_scores: Dict[str, float] = {}
        # chunk_id -> best RetrievalResult reference (for text/metadata)
        result_map: Dict[str, RetrievalResult] = {}

        for results, weight in zip(ranked_lists, weights):
            for rank, result in enumerate(results):
                cid = result.chunk_id
                # 1-based rank in the RRF formula.
                rrf_score = weight / (_RRF_K + rank + 1)
                rrf_scores[cid] = rrf_scores.get(cid, 0.0) + rrf_score
                if cid not in result_map:
                    result_map[cid] = result

        fused: List[RetrievalResult] = []
        for rank, (cid, score) in enumerate(
            sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        ):
            ref = result_map[cid]
            fused.append(
                RetrievalResult(
                    chunk_id=ref.chunk_id,
                    doc_id=ref.doc_id,
                    score=score,
                    text=ref.text,
                    title=ref.title,
                    rank=rank,
                    metadata=dict(ref.metadata),
                )
            )

        return fused
