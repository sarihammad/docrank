"""Tests for the full RAG pipeline and document ingestion components."""

import sys
from pathlib import Path
from typing import Dict, List
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.chunker import RecursiveChunker
from src.ingestion.document import Chunk, Document
from src.ingestion.loader import CorpusLoader
from src.generation.context_builder import ContextBuilder
from src.generation.llm import GenerationResult, LLMClient
from src.pipeline.rag import RAGPipeline, RAGResult
from src.reranking.cross_encoder import CrossEncoderReranker
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.result import RetrievalResult


# ------------------------------------------------------------------
# Test documents
# ------------------------------------------------------------------

DOCS = [
    Document(
        doc_id="d1",
        text=(
            "Machine learning is a subset of artificial intelligence. "
            "It enables systems to learn from data without explicit programming. "
            "Supervised learning, unsupervised learning, and reinforcement "
            "learning are the main paradigms."
        ),
        title="Machine Learning Fundamentals",
    ),
    Document(
        doc_id="d2",
        text=(
            "Transformer models revolutionised natural language processing. "
            "The attention mechanism allows the model to focus on relevant parts "
            "of the input sequence. BERT and GPT are widely-used transformer architectures."
        ),
        title="Transformers in NLP",
    ),
    Document(
        doc_id="d3",
        text=(
            "Retrieval-augmented generation combines a retrieval system with a "
            "generative language model. The retriever finds relevant documents, "
            "which are then used as context for the generator."
        ),
        title="RAG Systems",
    ),
]


# ------------------------------------------------------------------
# Chunker tests
# ------------------------------------------------------------------

class TestRecursiveChunker:
    def test_chunk_produces_correct_ids(self):
        chunker = RecursiveChunker(chunk_size=200, chunk_overlap=20)
        chunks = chunker.chunk_documents(DOCS[:1])
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_id == f"d1__chunk_{i}"

    def test_chunk_doc_id_matches_parent(self):
        chunker = RecursiveChunker(chunk_size=200, chunk_overlap=20)
        chunks = chunker.chunk_documents(DOCS)
        for chunk in chunks:
            assert any(chunk.doc_id == doc.doc_id for doc in DOCS)

    def test_chunk_text_within_size(self):
        chunk_size = 150
        chunker = RecursiveChunker(chunk_size=chunk_size, chunk_overlap=20)
        chunks = chunker.chunk_documents(DOCS)
        oversized = [c for c in chunks if len(c.text) > chunk_size * 1.5]
        # Allow minor overrun from hard-split fallback.
        assert len(oversized) == 0

    def test_char_positions_are_monotone(self):
        chunker = RecursiveChunker(chunk_size=200, chunk_overlap=20)
        doc = DOCS[0]
        chunks = chunker.chunk_document(doc)
        for chunk in chunks:
            assert chunk.start_char >= 0
            assert chunk.end_char > chunk.start_char
            assert chunk.end_char <= len(doc.text) + 5  # small tolerance

    def test_no_empty_chunks(self):
        chunker = RecursiveChunker(chunk_size=200, chunk_overlap=20)
        chunks = chunker.chunk_documents(DOCS)
        assert all(chunk.text.strip() for chunk in chunks)

    def test_overlap_raises_on_invalid(self):
        with pytest.raises(ValueError):
            RecursiveChunker(chunk_size=100, chunk_overlap=100)

    def test_empty_document_produces_no_chunks(self):
        chunker = RecursiveChunker(chunk_size=200, chunk_overlap=20)
        empty_doc = Document(doc_id="empty", text="   ")
        chunks = chunker.chunk_document(empty_doc)
        assert chunks == []

    def test_title_inherited(self):
        chunker = RecursiveChunker(chunk_size=200, chunk_overlap=20)
        chunks = chunker.chunk_document(DOCS[0])
        for chunk in chunks:
            assert chunk.title == DOCS[0].title


# ------------------------------------------------------------------
# CorpusLoader tests
# ------------------------------------------------------------------

class TestCorpusLoader:
    def test_load_from_directory(self, tmp_path):
        (tmp_path / "doc1.txt").write_text("Hello world, this is document one.")
        (tmp_path / "doc2.md").write_text("# Heading\nThis is document two.")
        (tmp_path / "ignore.csv").write_text("col1,col2\n1,2")

        loader = CorpusLoader()
        docs = loader.load_from_directory(str(tmp_path))
        assert len(docs) == 2
        filenames = {d.doc_id for d in docs}
        assert any("doc1.txt" in fid for fid in filenames)
        assert any("doc2.md" in fid for fid in filenames)

    def test_load_from_directory_raises_on_missing(self):
        with pytest.raises(FileNotFoundError):
            CorpusLoader().load_from_directory("/nonexistent/path")

    def test_load_from_jsonl(self, tmp_path):
        jsonl_content = (
            '{"id": "1", "title": "Doc One", "text": "Content of document one."}\n'
            '{"id": "2", "text": "Content of document two."}\n'
            '{"id": "3", "title": "Empty text"}\n'  # no text — should be skipped
        )
        jsonl_file = tmp_path / "corpus.jsonl"
        jsonl_file.write_text(jsonl_content, encoding="utf-8")

        loader = CorpusLoader()
        docs = loader.load_from_jsonl(str(jsonl_file))
        assert len(docs) == 2
        assert docs[0].doc_id == "1"
        assert docs[0].title == "Doc One"
        assert docs[1].doc_id == "2"

    def test_load_from_jsonl_raises_on_missing(self):
        with pytest.raises(FileNotFoundError):
            CorpusLoader().load_from_jsonl("/nonexistent/corpus.jsonl")


# ------------------------------------------------------------------
# ContextBuilder tests
# ------------------------------------------------------------------

class TestContextBuilder:
    def test_build_context_respects_token_budget(self):
        cb = ContextBuilder(max_tokens=100)
        chunks = [
            RetrievalResult(
                chunk_id=f"c{i}",
                doc_id=f"d{i}",
                score=1.0,
                text="a " * 50,
                title=f"Doc {i}",
            )
            for i in range(10)
        ]
        context = cb.build_context(chunks)
        # Rough check: at 4 chars/token, 100 tokens ~ 400 chars.
        assert len(context) < 600

    def test_build_context_formats_sources(self):
        cb = ContextBuilder(max_tokens=3000)
        chunks = [
            RetrievalResult(
                chunk_id="c0",
                doc_id="d0",
                score=1.0,
                text="Some relevant text.",
                title="My Title",
            )
        ]
        context = cb.build_context(chunks)
        assert "[Source 1: My Title]" in context
        assert "Some relevant text." in context

    def test_build_prompt_returns_two_messages(self):
        cb = ContextBuilder()
        messages = cb.build_prompt("What is RAG?", "context text here")
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "What is RAG?" in messages[1]["content"]
        assert "context text here" in messages[1]["content"]

    def test_extract_citations_returns_cited_chunks(self):
        cb = ContextBuilder()
        response = (
            "Machine learning algorithms are used to learn from data. "
            "Transformer models use attention mechanisms."
        )
        chunks = [
            RetrievalResult(
                chunk_id="c0",
                doc_id="d0",
                score=1.0,
                text="machine learning algorithms learn from training data",
                title="ML",
            ),
            RetrievalResult(
                chunk_id="c1",
                doc_id="d1",
                score=0.9,
                text="transformer models use attention and self-attention mechanisms",
                title="NLP",
            ),
            RetrievalResult(
                chunk_id="c2",
                doc_id="d2",
                score=0.5,
                text="the history of the french revolution in europe",
                title="History",
            ),
        ]
        cited = cb.extract_citations(response, chunks)
        assert "c0" in cited
        assert "c1" in cited
        assert "c2" not in cited


# ------------------------------------------------------------------
# LLM client tests
# ------------------------------------------------------------------

class TestLLMClient:
    def test_mock_response_when_no_api_key(self):
        client = LLMClient(api_key="")
        messages = [{"role": "user", "content": "What is AI?"}]
        result = client.generate(messages)
        assert "OPENAI_API_KEY" in result or "LLM not configured" in result

    def test_generate_with_metadata_returns_result(self):
        client = LLMClient(api_key="")
        messages = [{"role": "user", "content": "test"}]
        result = client.generate_with_metadata(messages)
        assert isinstance(result, GenerationResult)
        assert result.model == "none"
        assert result.tokens_used == 0


# ------------------------------------------------------------------
# Full RAG pipeline tests (with mocked LLM)
# ------------------------------------------------------------------

class TestRAGPipeline:
    @pytest.fixture
    def pipeline(self):
        """Build a minimal pipeline with real retrieval but mocked LLM."""
        chunker = RecursiveChunker(chunk_size=300, chunk_overlap=30)
        chunks = chunker.chunk_documents(DOCS)

        bm25 = BM25Retriever()
        bm25.build_index(chunks)

        dense = DenseRetriever(model_name="all-MiniLM-L6-v2")
        dense.build_index(chunks, batch_size=8)

        hybrid = HybridRetriever(bm25=bm25, dense=dense)

        reranker = CrossEncoderReranker(
            model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

        # Mock LLM to avoid needing an API key in tests.
        mock_llm = MagicMock(spec=LLMClient)
        mock_llm.generate_with_metadata.return_value = GenerationResult(
            text="RAG combines retrieval and generation.",
            tokens_used=15,
            latency_ms=100.0,
            model="mock",
        )

        ctx_builder = ContextBuilder(max_tokens=2000)

        return RAGPipeline(
            hybrid_retriever=hybrid,
            reranker=reranker,
            llm=mock_llm,
            context_builder=ctx_builder,
        )

    def test_query_returns_rag_result(self, pipeline):
        result = pipeline.query("What is RAG?", k_retrieve=5, k_rerank=3, k_context=2)
        assert isinstance(result, RAGResult)

    def test_query_returns_answer(self, pipeline):
        result = pipeline.query("How do transformers work?", k_retrieve=5, k_rerank=3, k_context=2)
        assert result.answer == "RAG combines retrieval and generation."

    def test_query_returns_sources(self, pipeline):
        result = pipeline.query("What is machine learning?", k_retrieve=5, k_rerank=3, k_context=2)
        assert len(result.sources) > 0
        assert all(isinstance(s, RetrievalResult) for s in result.sources)

    def test_query_has_latency_breakdown(self, pipeline):
        result = pipeline.query("explain transformers", k_retrieve=5, k_rerank=3, k_context=2)
        assert result.latency.retrieval_ms > 0
        assert result.latency.reranking_ms > 0
        assert result.latency.total_ms > 0

    def test_batch_query_length_matches_input(self, pipeline):
        questions = ["What is ML?", "What is NLP?", "What is RAG?"]
        results = pipeline.batch_query(questions, k_retrieve=5, k_rerank=3, k_context=2)
        assert len(results) == len(questions)

    def test_ablation_returns_four_result_sets(self, pipeline):
        ablation = pipeline.query_ablation("attention mechanism", k=3)
        assert len(ablation.bm25_only) > 0
        assert len(ablation.dense_only) > 0
        assert len(ablation.hybrid) > 0
        assert len(ablation.hybrid_reranked) > 0

    def test_rag_result_to_dict(self, pipeline):
        result = pipeline.query("What is RAG?", k_retrieve=5, k_rerank=3, k_context=2)
        d = result.to_dict()
        assert "question" in d
        assert "answer" in d
        assert "sources" in d
        assert "latency" in d
        assert isinstance(d["sources"], list)
