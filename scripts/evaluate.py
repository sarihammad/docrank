#!/usr/bin/env python3
"""CLI script for running the DocRank evaluation suite.

Evaluates retrieval quality (NDCG@K, Recall@K, MRR, MAP) on the SciFact
test queries and optionally runs LLM-as-judge evaluation on generated answers.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --with-llm-judge --output results.json
    python scripts/evaluate.py --k-values 5 10 20 100
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings
from src.evaluation.retrieval_metrics import evaluate_retrieval
from src.ingestion.loader import CorpusLoader
from src.pipeline.indexing import IndexingPipeline
from src.reranking.cross_encoder import CrossEncoderReranker
from src.retrieval.hybrid import HybridRetriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate DocRank retrieval and generation quality.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--index-dir",
        default="indexes",
        help="Directory containing the BM25 and FAISS indexes.",
    )
    parser.add_argument(
        "--k-values",
        nargs="+",
        type=int,
        default=[5, 10, 20, 100],
        help="Cutoff values for Recall@K, NDCG@K, Precision@K.",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Limit evaluation to the first N queries (useful for quick testing).",
    )
    parser.add_argument(
        "--with-llm-judge",
        action="store_true",
        default=False,
        help="Run LLM-as-judge evaluation in addition to retrieval metrics.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to save evaluation results JSON.",
    )
    return parser.parse_args()


def _format_metric_table(metrics: dict, k_values: list) -> str:
    """Format evaluation metrics as a text table."""
    lines = []
    lines.append(f"{'Metric':<20}  {'Value':>8}")
    lines.append("-" * 32)

    row_order = (
        ["MRR", "MAP"]
        + [f"Recall@{k}" for k in k_values]
        + [f"NDCG@{k}" for k in k_values]
        + [f"Precision@{k}" for k in k_values]
    )
    for name in row_order:
        if name in metrics:
            lines.append(f"{name:<20}  {metrics[name]:>8.4f}")

    return "\n".join(lines)


def _run_retrieval_evaluation(
    hybrid: HybridRetriever,
    queries,
    qrels: dict,
    k_values: list,
    max_queries: int | None,
) -> tuple[dict, dict, dict, dict]:
    """Run retrieval across all ablation modes and return metric dicts."""
    eval_queries = queries[:max_queries] if max_queries else queries

    results_hybrid = {}
    results_bm25 = {}
    results_dense = {}
    results_reranked = {}

    reranker = CrossEncoderReranker(model_name=settings.cross_encoder_model)
    max_k = max(k_values)

    for i, q in enumerate(eval_queries):
        if q.query_id not in qrels:
            continue
        logger.debug("Query %d/%d: %s", i + 1, len(eval_queries), q.text[:60])

        # Hybrid retrieval.
        hybrid_res = hybrid.search(q.text, k=max_k)
        results_hybrid[q.query_id] = [r.doc_id for r in hybrid_res]

        # BM25 only.
        bm25_res = hybrid.search_bm25_only(q.text, k=max_k)
        results_bm25[q.query_id] = [r.doc_id for r in bm25_res]

        # Dense only.
        dense_res = hybrid.search_dense_only(q.text, k=max_k)
        results_dense[q.query_id] = [r.doc_id for r in dense_res]

        # Hybrid + reranker (use top-20 from hybrid, rerank, then evaluate).
        top20 = hybrid_res[:20]
        if top20:
            reranked = reranker.rerank(q.text, top20, top_k=20)
            results_reranked[q.query_id] = [r.doc_id for r in reranked]
        else:
            results_reranked[q.query_id] = []

        if (i + 1) % 50 == 0:
            logger.info("Evaluated %d/%d queries…", i + 1, len(eval_queries))

    metrics_hybrid = evaluate_retrieval(results_hybrid, qrels, k_values=k_values)
    metrics_bm25 = evaluate_retrieval(results_bm25, qrels, k_values=k_values)
    metrics_dense = evaluate_retrieval(results_dense, qrels, k_values=k_values)
    metrics_reranked = evaluate_retrieval(results_reranked, qrels, k_values=k_values)

    return metrics_hybrid, metrics_bm25, metrics_dense, metrics_reranked


def _run_llm_judge(
    hybrid: HybridRetriever,
    queries,
    qrels: dict,
    max_queries: int | None,
) -> dict:
    """Run RAG generation and LLM-as-judge evaluation."""
    from src.evaluation.llm_judge import LLMJudge
    from src.generation.context_builder import ContextBuilder
    from src.generation.llm import LLMClient
    from src.pipeline.rag import RAGPipeline
    from src.reranking.cross_encoder import CrossEncoderReranker

    reranker = CrossEncoderReranker(model_name=settings.cross_encoder_model)
    llm = LLMClient(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        model=settings.llm_model,
    )
    ctx_builder = ContextBuilder(max_tokens=settings.max_context_tokens)
    pipeline = RAGPipeline(
        hybrid_retriever=hybrid,
        reranker=reranker,
        llm=llm,
        context_builder=ctx_builder,
    )

    eval_queries = [q for q in queries if q.query_id in qrels]
    if max_queries:
        eval_queries = eval_queries[:max_queries]

    rag_results = []
    for q in eval_queries:
        result = pipeline.query(
            q.text,
            k_retrieve=settings.top_k_retrieval,
            k_rerank=settings.top_k_rerank,
            k_context=settings.top_k_context,
        )
        rag_results.append(result)

    judge = LLMJudge(llm_client=llm)
    report = judge.evaluate_batch(
        queries=[q.text for q in eval_queries],
        answers=[r.answer for r in rag_results],
        contexts=[ctx_builder.build_context(r.sources) for r in rag_results],
    )
    return report.to_dict()


def main() -> None:
    args = parse_args()

    index_dir = Path(args.index_dir)
    bm25_path = str(index_dir / "bm25.pkl")
    faiss_path = str(index_dir / "faiss.index")

    # ------------------------------------------------------------------
    # Load indexes
    # ------------------------------------------------------------------
    logger.info("Loading indexes from %s…", index_dir)
    indexer = IndexingPipeline(
        bm25_index_path=bm25_path,
        faiss_index_path=faiss_path,
    )
    try:
        bm25, dense = indexer.load_indexes()
    except FileNotFoundError as exc:
        logger.error(
            "Index files not found. Run `make index` or "
            "`python scripts/index_corpus.py --source scifact` first.\n%s",
            exc,
        )
        sys.exit(1)

    hybrid = HybridRetriever(
        bm25=bm25,
        dense=dense,
        bm25_weight=settings.hybrid_bm25_weight,
        dense_weight=settings.hybrid_dense_weight,
    )

    # ------------------------------------------------------------------
    # Load SciFact queries and qrels
    # ------------------------------------------------------------------
    logger.info("Loading SciFact queries and relevance judgments…")
    loader = CorpusLoader()
    _, queries, qrels = loader.load_scifact()
    logger.info("Loaded %d queries and %d qrel entries.", len(queries), len(qrels))

    if not qrels:
        logger.error("No relevance judgments found. Cannot run evaluation.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Retrieval evaluation
    # ------------------------------------------------------------------
    logger.info("Running retrieval evaluation (this may take a few minutes)…")
    (
        metrics_hybrid,
        metrics_bm25,
        metrics_dense,
        metrics_reranked,
    ) = _run_retrieval_evaluation(
        hybrid, queries, qrels, args.k_values, args.max_queries
    )

    # ------------------------------------------------------------------
    # Print results
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  DocRank Evaluation Results")
    print("=" * 60)

    for label, metrics in [
        ("Hybrid (BM25 + Dense, RRF)", metrics_hybrid),
        ("BM25 Only", metrics_bm25),
        ("Dense Only", metrics_dense),
        ("Hybrid + Cross-Encoder Reranker", metrics_reranked),
    ]:
        print(f"\n  {label}")
        print("-" * 60)
        print(_format_metric_table(metrics, args.k_values))

    print("\n" + "=" * 60)

    # ------------------------------------------------------------------
    # Optional LLM judge evaluation
    # ------------------------------------------------------------------
    llm_judge_report = None
    if args.with_llm_judge:
        logger.info("Running LLM-as-judge evaluation…")
        llm_judge_report = _run_llm_judge(
            hybrid, queries, qrels, args.max_queries
        )
        print("\n  LLM-as-Judge Results")
        print("-" * 60)
        print(f"  Mean Faithfulness   : {llm_judge_report['mean_faithfulness']:.4f}")
        print(f"  Mean Hallucination  : {llm_judge_report['mean_hallucination']:.4f}")
        if llm_judge_report.get("mean_correctness", 0) > 0:
            print(f"  Mean Correctness    : {llm_judge_report['mean_correctness']:.4f}")
        print("=" * 60)

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    output_data = {
        "retrieval": {
            "hybrid": metrics_hybrid,
            "bm25_only": metrics_bm25,
            "dense_only": metrics_dense,
            "hybrid_reranked": metrics_reranked,
        },
        "llm_judge": llm_judge_report,
        "config": {
            "k_values": args.k_values,
            "n_queries_evaluated": sum(
                1 for q in queries if q.query_id in qrels
            ) if not args.max_queries else args.max_queries,
            "embedding_model": settings.embedding_model,
            "cross_encoder_model": settings.cross_encoder_model,
        },
    }

    output_path = args.output or "evaluation_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)
    logger.info("Results saved to %s.", output_path)


if __name__ == "__main__":
    main()
