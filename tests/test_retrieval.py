"""Tests for BM25, dense, and hybrid retrieval components."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.document import Chunk, Document
from src.ingestion.chunker import RecursiveChunker
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid import HybridRetriever, _RRF_K
from src.retrieval.result import RetrievalResult


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

SAMPLE_DOCS = [
    Document(
        doc_id="d1",
        text=(
            "Machine learning algorithms learn from data to make predictions. "
            "Supervised learning uses labelled training examples."
        ),
        title="Machine Learning Overview",
    ),
    Document(
        doc_id="d2",
        text=(
            "Neural networks are inspired by biological neurons. "
            "Deep learning uses many layers to learn representations."
        ),
        title="Deep Learning Basics",
    ),
    Document(
        doc_id="d3",
        text=(
            "Natural language processing enables computers to understand text. "
            "Transformers use self-attention mechanisms."
        ),
        title="NLP and Transformers",
    ),
    Document(
        doc_id="d4",
        text=(
            "Reinforcement learning agents learn by interacting with environments. "
            "Reward signals guide the learning process."
        ),
        title="Reinforcement Learning",
    ),
    Document(
        doc_id="d5",
        text=(
            "Information retrieval systems rank documents by relevance. "
            "BM25 is a classic probabilistic retrieval model."
        ),
        title="Information Retrieval",
    ),
]


@pytest.fixture
def chunks() -> list:
    chunker = RecursiveChunker(chunk_size=512, chunk_overlap=64)
    return chunker.chunk_documents(SAMPLE_DOCS)


@pytest.fixture
def bm25_retriever(chunks) -> BM25Retriever:
    r = BM25Retriever()
    r.build_index(chunks)
    return r


@pytest.fixture
def dense_retriever(chunks) -> DenseRetriever:
    r = DenseRetriever(model_name="all-MiniLM-L6-v2")
    r.build_index(chunks, batch_size=8)
    return r


@pytest.fixture
def hybrid_retriever(bm25_retriever, dense_retriever) -> HybridRetriever:
    return HybridRetriever(
        bm25=bm25_retriever,
        dense=dense_retriever,
        bm25_weight=0.5,
        dense_weight=0.5,
    )


# ------------------------------------------------------------------
# BM25 tests
# ------------------------------------------------------------------

class TestBM25Retriever:
    def test_build_index_fails_on_empty(self):
        with pytest.raises(ValueError):
            BM25Retriever().build_index([])

    def test_search_returns_k_results(self, bm25_retriever):
        results = bm25_retriever.search("machine learning", k=3)
        assert len(results) == 3

    def test_search_returns_at_most_n_chunks(self, bm25_retriever):
        """Requesting more results than chunks should return all chunks."""
        results = bm25_retriever.search("learning", k=1000)
        assert len(results) <= bm25_retriever.num_chunks

    def test_scores_are_non_negative(self, bm25_retriever):
        results = bm25_retriever.search("neural networks", k=5)
        for r in results:
            assert r.score >= 0.0

    def test_results_sorted_descending(self, bm25_retriever):
        results = bm25_retriever.search("deep learning", k=5)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_ranks_start_at_zero(self, bm25_retriever):
        results = bm25_retriever.search("transformers", k=3)
        assert results[0].rank == 0
        assert results[1].rank == 1

    def test_result_has_correct_fields(self, bm25_retriever):
        results = bm25_retriever.search("retrieval", k=1)
        r = results[0]
        assert r.chunk_id
        assert r.doc_id
        assert r.text
        assert isinstance(r.score, float)

    def test_relevant_document_ranks_high(self, bm25_retriever):
        """A targeted query for BM25 should find the IR document near the top."""
        results = bm25_retriever.search("BM25 retrieval documents", k=5)
        doc_ids = [r.doc_id for r in results]
        assert "d5" in doc_ids

    def test_save_and_load(self, bm25_retriever, tmp_path):
        path = str(tmp_path / "bm25.pkl")
        bm25_retriever.save(path)

        loaded = BM25Retriever()
        loaded.load(path)
        assert loaded.num_chunks == bm25_retriever.num_chunks

        original = bm25_retriever.search("machine learning", k=3)
        reloaded = loaded.search("machine learning", k=3)
        assert [r.chunk_id for r in original] == [r.chunk_id for r in reloaded]

    def test_load_raises_on_missing_file(self):
        with pytest.raises(FileNotFoundError):
            BM25Retriever().load("/nonexistent/bm25.pkl")


# ------------------------------------------------------------------
# Dense retriever tests
# ------------------------------------------------------------------

class TestDenseRetriever:
    def test_search_returns_k_results(self, dense_retriever):
        results = dense_retriever.search("deep neural network", k=3)
        assert len(results) == 3

    def test_scores_are_bounded(self, dense_retriever):
        """Cosine similarity for L2-normalised vectors is in [-1, 1]."""
        results = dense_retriever.search("attention mechanism", k=5)
        for r in results:
            assert -1.01 <= r.score <= 1.01

    def test_results_sorted_descending(self, dense_retriever):
        results = dense_retriever.search("language model", k=5)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_encode_batch_normalised(self, dense_retriever):
        texts = ["hello world", "machine learning"]
        embeddings = dense_retriever.encode_batch(texts, show_progress=False)
        norms = np.linalg.norm(embeddings, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_encode_query_shape(self, dense_retriever):
        emb = dense_retriever.encode_query("test query")
        assert emb.ndim == 2
        assert emb.shape[0] == 1

    def test_save_and_load(self, dense_retriever, tmp_path):
        index_path = str(tmp_path / "faiss.index")
        meta_path = index_path + ".meta.json"
        dense_retriever.save(index_path, meta_path)

        loaded = DenseRetriever()
        loaded.load(index_path, meta_path)
        assert loaded.num_chunks == dense_retriever.num_chunks

        original = dense_retriever.search("reinforcement learning", k=3)
        reloaded = loaded.search("reinforcement learning", k=3)
        assert [r.chunk_id for r in original] == [r.chunk_id for r in reloaded]

    def test_load_raises_on_missing_index(self):
        with pytest.raises(FileNotFoundError):
            DenseRetriever().load("/nonexistent/faiss.index")


# ------------------------------------------------------------------
# Hybrid retriever / RRF tests
# ------------------------------------------------------------------

class TestHybridRetriever:
    def test_search_returns_k_results(self, hybrid_retriever):
        results = hybrid_retriever.search("machine learning", k=5)
        assert len(results) == 5

    def test_results_sorted_descending(self, hybrid_retriever):
        results = hybrid_retriever.search("neural network", k=5)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_rrf_scores_positive(self, hybrid_retriever):
        results = hybrid_retriever.search("transformer attention", k=3)
        for r in results:
            assert r.score > 0.0

    def test_bm25_only_search(self, hybrid_retriever):
        results = hybrid_retriever.search_bm25_only("retrieval system", k=4)
        assert len(results) == 4
        assert all(isinstance(r, RetrievalResult) for r in results)

    def test_dense_only_search(self, hybrid_retriever):
        results = hybrid_retriever.search_dense_only("language models", k=4)
        assert len(results) == 4

    def test_rrf_formula_correctness(self):
        """Verify the RRF score formula: sum(w / (k + rank))."""
        # Create two simple ranked lists with one common document.
        list_a = [
            RetrievalResult("c1", "d1", 1.0, "text a", rank=0),
            RetrievalResult("c2", "d2", 0.8, "text b", rank=1),
        ]
        list_b = [
            RetrievalResult("c2", "d2", 0.9, "text b", rank=0),
            RetrievalResult("c1", "d1", 0.7, "text a", rank=1),
        ]
        weights = [1.0, 1.0]
        fused = HybridRetriever._reciprocal_rank_fusion([list_a, list_b], weights)

        # c1 appears at rank 0 in list_a and rank 1 in list_b.
        # c2 appears at rank 1 in list_a and rank 0 in list_b.
        expected_c1 = 1.0 / (_RRF_K + 1) + 1.0 / (_RRF_K + 2)
        expected_c2 = 1.0 / (_RRF_K + 2) + 1.0 / (_RRF_K + 1)

        score_map = {r.chunk_id: r.score for r in fused}
        assert abs(score_map["c1"] - expected_c1) < 1e-8
        assert abs(score_map["c2"] - expected_c2) < 1e-8

    def test_rrf_weights_affect_scores(self):
        """Higher BM25 weight should boost a BM25-top-1 document."""
        list_a = [RetrievalResult("c1", "d1", 1.0, "text", rank=0)]
        list_b = [RetrievalResult("c2", "d2", 1.0, "text", rank=0)]

        fused_equal = HybridRetriever._reciprocal_rank_fusion(
            [list_a, list_b], [0.5, 0.5]
        )
        fused_bm25_heavy = HybridRetriever._reciprocal_rank_fusion(
            [list_a, list_b], [0.9, 0.1]
        )

        score_c1_equal = next(r.score for r in fused_equal if r.chunk_id == "c1")
        score_c1_heavy = next(r.score for r in fused_bm25_heavy if r.chunk_id == "c1")
        assert score_c1_heavy > score_c1_equal

    def test_no_duplicate_chunk_ids(self, hybrid_retriever):
        results = hybrid_retriever.search("learning algorithm", k=10)
        chunk_ids = [r.chunk_id for r in results]
        assert len(chunk_ids) == len(set(chunk_ids))
