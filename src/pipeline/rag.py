"""Full RAG inference pipeline with hybrid retrieval and cross-encoder reranking."""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..generation.context_builder import ContextBuilder
from ..generation.llm import GenerationResult, LLMClient
from ..reranking.cross_encoder import CrossEncoderReranker
from ..retrieval.hybrid import HybridRetriever
from ..retrieval.result import RetrievalResult

logger = logging.getLogger(__name__)


@dataclass
class LatencyBreakdown:
    """Per-stage latency measurements in milliseconds."""

    retrieval_ms: float = 0.0
    reranking_ms: float = 0.0
    context_ms: float = 0.0
    generation_ms: float = 0.0
    total_ms: float = 0.0


@dataclass
class RAGResult:
    """Output of a single RAG pipeline query.

    Attributes:
        question: The original user question.
        answer: Generated answer text.
        sources: List of RetrievalResults used as LLM context.
        retrieval_results: All results returned by the hybrid retriever.
        reranked_results: Results after cross-encoder reranking.
        cited_chunks: Chunk IDs cited in the answer.
        generation_metadata: Token usage and model details.
        latency: Per-stage timing breakdown.
    """

    question: str
    answer: str
    sources: List[RetrievalResult] = field(default_factory=list)
    retrieval_results: List[RetrievalResult] = field(default_factory=list)
    reranked_results: List[RetrievalResult] = field(default_factory=list)
    cited_chunks: List[str] = field(default_factory=list)
    generation_metadata: Optional[GenerationResult] = None
    latency: LatencyBreakdown = field(default_factory=LatencyBreakdown)

    def to_dict(self) -> Dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "sources": [
                {
                    "chunk_id": r.chunk_id,
                    "doc_id": r.doc_id,
                    "title": r.title,
                    "text": r.text[:300],
                    "score": r.score,
                    "rank": r.rank,
                }
                for r in self.sources
            ],
            "cited_chunks": self.cited_chunks,
            "latency": {
                "retrieval_ms": self.latency.retrieval_ms,
                "reranking_ms": self.latency.reranking_ms,
                "context_ms": self.latency.context_ms,
                "generation_ms": self.latency.generation_ms,
                "total_ms": self.latency.total_ms,
            },
            "tokens_used": (
                self.generation_metadata.tokens_used
                if self.generation_metadata
                else 0
            ),
        }


@dataclass
class AblationResult:
    """Comparison of retrieval strategies for a single query."""

    question: str
    bm25_only: List[RetrievalResult] = field(default_factory=list)
    dense_only: List[RetrievalResult] = field(default_factory=list)
    hybrid: List[RetrievalResult] = field(default_factory=list)
    hybrid_reranked: List[RetrievalResult] = field(default_factory=list)


class RAGPipeline:
    """End-to-end RAG pipeline.

    Query flow:
      1. Hybrid retrieval (BM25 + FAISS via RRF) → top-100
      2. Cross-encoder reranking → top-20
      3. Token-budget context selection → top-5
      4. LLM generation with cited sources
    """

    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        reranker: CrossEncoderReranker,
        llm: LLMClient,
        context_builder: ContextBuilder,
    ) -> None:
        """
        Args:
            hybrid_retriever: Initialised HybridRetriever.
            reranker: Initialised CrossEncoderReranker.
            llm: Initialised LLMClient.
            context_builder: Initialised ContextBuilder.
        """
        self.retriever = hybrid_retriever
        self.reranker = reranker
        self.llm = llm
        self.context_builder = context_builder

    # ------------------------------------------------------------------
    # Single query
    # ------------------------------------------------------------------

    def query(
        self,
        question: str,
        k_retrieve: int = 100,
        k_rerank: int = 20,
        k_context: int = 5,
    ) -> RAGResult:
        """Execute the full RAG pipeline for a single question.

        Args:
            question: Natural-language question.
            k_retrieve: Number of candidates to retrieve before reranking.
            k_rerank: Number of results to keep after reranking.
            k_context: Number of context chunks to pass to the LLM.

        Returns:
            RAGResult with answer, sources, and latency breakdown.
        """
        pipeline_start = time.perf_counter()
        latency = LatencyBreakdown()

        # Stage 1: Hybrid retrieval.
        t0 = time.perf_counter()
        retrieval_results = self.retriever.search(question, k=k_retrieve)
        latency.retrieval_ms = (time.perf_counter() - t0) * 1000.0
        logger.debug(
            "Retrieval returned %d results in %.1fms.",
            len(retrieval_results),
            latency.retrieval_ms,
        )

        # Stage 2: Cross-encoder reranking.
        t0 = time.perf_counter()
        reranked = self.reranker.rerank(question, retrieval_results, top_k=k_rerank)
        latency.reranking_ms = (time.perf_counter() - t0) * 1000.0
        logger.debug(
            "Reranking produced %d results in %.1fms.",
            len(reranked),
            latency.reranking_ms,
        )

        # Stage 3: Context selection.
        t0 = time.perf_counter()
        context_chunks = reranked[:k_context]
        context = self.context_builder.build_context(context_chunks)
        messages = self.context_builder.build_prompt(question, context)
        latency.context_ms = (time.perf_counter() - t0) * 1000.0

        # Stage 4: LLM generation.
        t0 = time.perf_counter()
        gen_result = self.llm.generate_with_metadata(
            messages, temperature=0.1, max_tokens=512
        )
        latency.generation_ms = (time.perf_counter() - t0) * 1000.0

        latency.total_ms = (time.perf_counter() - pipeline_start) * 1000.0

        # Extract citations.
        cited = self.context_builder.extract_citations(
            gen_result.text, context_chunks
        )

        return RAGResult(
            question=question,
            answer=gen_result.text,
            sources=context_chunks,
            retrieval_results=retrieval_results,
            reranked_results=reranked,
            cited_chunks=cited,
            generation_metadata=gen_result,
            latency=latency,
        )

    # ------------------------------------------------------------------
    # Batch query
    # ------------------------------------------------------------------

    def batch_query(
        self,
        questions: List[str],
        k_retrieve: int = 100,
        k_rerank: int = 20,
        k_context: int = 5,
    ) -> List[RAGResult]:
        """Run the RAG pipeline over a list of questions.

        Args:
            questions: List of natural-language questions.
            k_retrieve: Retrieval cutoff.
            k_rerank: Reranking cutoff.
            k_context: Context selection cutoff.

        Returns:
            List of RAGResult objects in the same order as the input.
        """
        results: List[RAGResult] = []
        for i, question in enumerate(questions):
            logger.info("Processing query %d/%d: %s", i + 1, len(questions), question[:80])
            result = self.query(
                question,
                k_retrieve=k_retrieve,
                k_rerank=k_rerank,
                k_context=k_context,
            )
            results.append(result)
        return results

    # ------------------------------------------------------------------
    # Ablation study
    # ------------------------------------------------------------------

    def query_ablation(
        self,
        question: str,
        k: int = 10,
    ) -> AblationResult:
        """Compare four retrieval strategies on the same question.

        Strategies compared:
          1. BM25 only
          2. Dense (FAISS) only
          3. Hybrid (BM25 + dense via RRF)
          4. Hybrid + cross-encoder reranking

        Args:
            question: Natural-language question.
            k: Number of results per strategy.

        Returns:
            AblationResult containing ranked lists from all four strategies.
        """
        bm25_results = self.retriever.search_bm25_only(question, k=k)
        dense_results = self.retriever.search_dense_only(question, k=k)
        hybrid_results = self.retriever.search(question, k=k)
        reranked_results = self.reranker.rerank(
            question, hybrid_results, top_k=min(k, len(hybrid_results))
        )

        return AblationResult(
            question=question,
            bm25_only=bm25_results,
            dense_only=dense_results,
            hybrid=hybrid_results,
            hybrid_reranked=reranked_results,
        )
