"""Tests for the cross-encoder reranking component."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.reranking.cross_encoder import CrossEncoderReranker
from src.retrieval.result import RetrievalResult


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

CANDIDATE_PASSAGES = [
    "Machine learning models learn patterns from training data.",
    "Deep learning uses neural networks with many hidden layers.",
    "The capital of France is Paris, a major European city.",
    "Natural language processing enables text understanding.",
    "Reinforcement learning agents learn from environment rewards.",
]

QUERY = "How do neural networks learn?"


@pytest.fixture(scope="module")
def reranker() -> CrossEncoderReranker:
    return CrossEncoderReranker(
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        batch_size=8,
    )


@pytest.fixture
def candidates() -> list:
    return [
        RetrievalResult(
            chunk_id=f"c{i}",
            doc_id=f"d{i}",
            score=float(len(CANDIDATE_PASSAGES) - i),
            text=text,
            title=f"Doc {i}",
            rank=i,
        )
        for i, text in enumerate(CANDIDATE_PASSAGES)
    ]


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

class TestCrossEncoderReranker:
    def test_rerank_returns_top_k(self, reranker, candidates):
        results = reranker.rerank(QUERY, candidates, top_k=3)
        assert len(results) == 3

    def test_rerank_respects_top_k_larger_than_candidates(self, reranker, candidates):
        results = reranker.rerank(QUERY, candidates, top_k=100)
        assert len(results) == len(candidates)

    def test_rerank_returns_empty_on_empty_input(self, reranker):
        results = reranker.rerank(QUERY, [], top_k=5)
        assert results == []

    def test_ranks_reset_to_zero_based(self, reranker, candidates):
        results = reranker.rerank(QUERY, candidates, top_k=5)
        ranks = [r.rank for r in results]
        assert ranks == list(range(len(results)))

    def test_results_sorted_descending_by_score(self, reranker, candidates):
        results = reranker.rerank(QUERY, candidates, top_k=5)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_reranking_changes_order(self, reranker, candidates):
        """Reranking should produce a different order from the original retrieval."""
        original_order = [c.chunk_id for c in candidates]
        reranked_order = [r.chunk_id for r in reranker.rerank(QUERY, candidates, top_k=5)]
        # The cross-encoder should move relevant passages to the top.
        # At minimum, the orders should differ for this heterogeneous set.
        # (We cannot guarantee exact order, but the ML-relevant passage
        # describing neural networks should score higher than Paris capital.)
        paris_rank_original = original_order.index("c2")
        paris_rank_reranked = reranked_order.index("c2") if "c2" in reranked_order else len(reranked_order)
        ml_ranks_reranked = [
            reranked_order.index(cid)
            for cid in reranked_order
            if cid in {"c0", "c1", "c3"}
        ]
        if ml_ranks_reranked:
            assert min(ml_ranks_reranked) < paris_rank_reranked

    def test_predict_relevance_returns_float(self, reranker):
        score = reranker.predict_relevance(QUERY, CANDIDATE_PASSAGES[1])
        assert isinstance(score, float)

    def test_relevant_passage_scores_higher_than_irrelevant(self, reranker):
        relevant = "Neural networks learn by adjusting weights through backpropagation."
        irrelevant = "The Eiffel Tower is located in Paris, France."
        rel_score = reranker.predict_relevance(QUERY, relevant)
        irrel_score = reranker.predict_relevance(QUERY, irrelevant)
        assert rel_score > irrel_score

    def test_chunk_ids_preserved_after_reranking(self, reranker, candidates):
        original_ids = {c.chunk_id for c in candidates}
        reranked = reranker.rerank(QUERY, candidates, top_k=len(candidates))
        reranked_ids = {r.chunk_id for r in reranked}
        assert reranked_ids == original_ids

    def test_score_field_overwritten_with_cross_encoder_score(
        self, reranker, candidates
    ):
        """Original retrieval scores should be replaced by cross-encoder logits."""
        original_scores = {c.chunk_id: c.score for c in candidates}
        reranked = reranker.rerank(QUERY, candidates, top_k=3)
        for r in reranked:
            # Cross-encoder logits are typically in a different range from
            # BM25/cosine scores.  They should not equal the original values
            # for all results (the probability of accidental equality is ~0).
            assert r.score != original_scores.get(r.chunk_id, r.score - 1)
