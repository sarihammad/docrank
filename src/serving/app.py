"""FastAPI application for the DocRank RAG platform."""

import logging
import os
from contextlib import asynccontextmanager
from typing import Dict, List, Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from pydantic import BaseModel, Field
from starlette.responses import Response

from ..config import settings
from ..generation.context_builder import ContextBuilder
from ..generation.llm import LLMClient
from ..pipeline.indexing import IndexingPipeline
from ..pipeline.rag import RAGPipeline, RAGResult
from ..reranking.cross_encoder import CrossEncoderReranker
from ..retrieval.hybrid import HybridRetriever
from ..retrieval.result import RetrievalResult
from .middleware import (
    PrometheusMiddleware,
    generation_latency_ms,
    query_counter,
    reranking_latency_ms,
    retrieval_latency_ms,
)

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Application state
# ------------------------------------------------------------------

class AppState:
    pipeline: Optional[RAGPipeline] = None
    hybrid_retriever: Optional[HybridRetriever] = None
    indexing_stats: Optional[Dict] = None


_state = AppState()


# ------------------------------------------------------------------
# Lifespan
# ------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all pipeline components at startup."""
    logger.info("Loading DocRank pipeline components…")
    try:
        indexer = IndexingPipeline()
        bm25, dense = indexer.load_indexes()

        hybrid = HybridRetriever(
            bm25=bm25,
            dense=dense,
            bm25_weight=settings.hybrid_bm25_weight,
            dense_weight=settings.hybrid_dense_weight,
        )
        _state.hybrid_retriever = hybrid

        reranker = CrossEncoderReranker(
            model_name=settings.cross_encoder_model
        )
        llm = LLMClient(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            model=settings.llm_model,
        )
        ctx_builder = ContextBuilder(max_tokens=settings.max_context_tokens)

        _state.pipeline = RAGPipeline(
            hybrid_retriever=hybrid,
            reranker=reranker,
            llm=llm,
            context_builder=ctx_builder,
        )
        _state.indexing_stats = {
            "n_chunks_bm25": bm25.num_chunks,
            "n_chunks_dense": dense.num_chunks,
            "embedding_model": dense.model_name,
            "cross_encoder_model": settings.cross_encoder_model,
            "llm_model": settings.llm_model,
        }
        logger.info("Pipeline loaded successfully.")
    except FileNotFoundError as exc:
        logger.warning(
            "Index files not found — pipeline will be unavailable until "
            "you run `make index`. Error: %s",
            exc,
        )
    yield
    logger.info("Shutting down DocRank API.")


# ------------------------------------------------------------------
# App
# ------------------------------------------------------------------

app = FastAPI(
    title="DocRank",
    description=(
        "Production RAG platform with hybrid search (BM25 + FAISS), "
        "cross-encoder reranking, and LLM generation."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(PrometheusMiddleware)


# ------------------------------------------------------------------
# Request / Response models
# ------------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Natural-language question.")
    k: int = Field(default=5, ge=1, le=50, description="Number of context chunks.")
    use_reranker: bool = Field(default=True, description="Enable cross-encoder reranking.")


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[Dict]
    cited_chunks: List[str]
    latency: Dict[str, float]
    tokens_used: int


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    k: int = Field(default=10, ge=1, le=200)
    mode: Literal["hybrid", "bm25", "dense"] = Field(default="hybrid")


class SearchResult(BaseModel):
    chunk_id: str
    doc_id: str
    title: str
    text: str
    score: float
    rank: int


class SearchResponse(BaseModel):
    query: str
    mode: str
    results: List[SearchResult]
    latency_ms: float


class EvaluateRequest(BaseModel):
    questions: List[str] = Field(..., min_length=1)
    references: Optional[List[str]] = None


# ------------------------------------------------------------------
# Endpoints
# ------------------------------------------------------------------

@app.get("/health")
async def health() -> Dict:
    """Health check endpoint."""
    pipeline_ready = _state.pipeline is not None
    return {
        "status": "healthy" if pipeline_ready else "degraded",
        "pipeline_ready": pipeline_ready,
        "version": "1.0.0",
    }


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)


@app.get("/index/stats")
async def index_stats() -> Dict:
    """Return index metadata."""
    if _state.indexing_stats is None:
        raise HTTPException(status_code=503, detail="Index not loaded.")
    return _state.indexing_stats


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest) -> QueryResponse:
    """Run the full RAG pipeline: retrieve, rerank, generate."""
    if _state.pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Pipeline not loaded. Run `make index` first.",
        )

    query_counter.labels(mode="hybrid").inc()

    k_rerank = settings.top_k_rerank if req.use_reranker else 0
    result: RAGResult = _state.pipeline.query(
        question=req.question,
        k_retrieve=settings.top_k_retrieval,
        k_rerank=max(k_rerank, req.k),
        k_context=req.k,
    )

    # Record stage latencies.
    retrieval_latency_ms.observe(result.latency.retrieval_ms)
    reranking_latency_ms.observe(result.latency.reranking_ms)
    generation_latency_ms.observe(result.latency.generation_ms)

    return QueryResponse(
        question=result.question,
        answer=result.answer,
        sources=[
            {
                "chunk_id": r.chunk_id,
                "doc_id": r.doc_id,
                "title": r.title,
                "text": r.text[:500],
                "score": r.score,
                "rank": r.rank,
            }
            for r in result.sources
        ],
        cited_chunks=result.cited_chunks,
        latency={
            "retrieval_ms": result.latency.retrieval_ms,
            "reranking_ms": result.latency.reranking_ms,
            "context_ms": result.latency.context_ms,
            "generation_ms": result.latency.generation_ms,
            "total_ms": result.latency.total_ms,
        },
        tokens_used=(
            result.generation_metadata.tokens_used
            if result.generation_metadata
            else 0
        ),
    )


@app.post("/search", response_model=SearchResponse)
async def search_endpoint(req: SearchRequest) -> SearchResponse:
    """Retrieval-only endpoint — no LLM generation."""
    if _state.hybrid_retriever is None:
        raise HTTPException(
            status_code=503,
            detail="Retriever not loaded. Run `make index` first.",
        )

    import time

    query_counter.labels(mode=req.mode).inc()
    start = time.perf_counter()

    if req.mode == "bm25":
        results = _state.hybrid_retriever.search_bm25_only(req.query, k=req.k)
    elif req.mode == "dense":
        results = _state.hybrid_retriever.search_dense_only(req.query, k=req.k)
    else:
        results = _state.hybrid_retriever.search(req.query, k=req.k)

    latency_ms = (time.perf_counter() - start) * 1000.0
    retrieval_latency_ms.observe(latency_ms)

    return SearchResponse(
        query=req.query,
        mode=req.mode,
        results=[
            SearchResult(
                chunk_id=r.chunk_id,
                doc_id=r.doc_id,
                title=r.title,
                text=r.text[:500],
                score=r.score,
                rank=r.rank,
            )
            for r in results
        ],
        latency_ms=latency_ms,
    )


@app.post("/evaluate")
async def evaluate_endpoint(req: EvaluateRequest) -> Dict:
    """Run LLM-as-judge evaluation over a batch of questions."""
    if _state.pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Pipeline not loaded. Run `make index` first.",
        )

    if len(req.questions) > 50:
        raise HTTPException(
            status_code=400,
            detail="Maximum 50 questions per evaluation request.",
        )

    from ..evaluation.llm_judge import LLMJudge

    rag_results = _state.pipeline.batch_query(req.questions)
    answers = [r.answer for r in rag_results]
    contexts = [
        _state.pipeline.context_builder.build_context(r.sources)
        for r in rag_results
    ]

    judge = LLMJudge(llm_client=_state.pipeline.llm)
    report = judge.evaluate_batch(
        queries=req.questions,
        answers=answers,
        contexts=contexts,
        references=req.references,
    )

    return report.to_dict()
