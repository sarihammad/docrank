"""Standard information retrieval evaluation metrics.

All functions operate on ranked lists of document IDs (strings) and
ground-truth relevance sets/dictionaries.  Metrics follow the standard
TREC/BEIR definitions.
"""

import math
from typing import Dict, List, Set


def recall_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """Fraction of relevant documents found in the top-k results.

    Args:
        retrieved: Ordered list of retrieved document IDs.
        relevant: Set of document IDs considered relevant.
        k: Cutoff rank.

    Returns:
        Recall@k in [0, 1].  Returns 0 if relevant is empty.
    """
    if not relevant:
        return 0.0
    top_k = set(retrieved[:k])
    return len(top_k & relevant) / len(relevant)


def precision_at_k(retrieved: List[str], relevant: Set[str], k: int) -> float:
    """Fraction of top-k results that are relevant.

    Args:
        retrieved: Ordered list of retrieved document IDs.
        relevant: Set of relevant document IDs.
        k: Cutoff rank.

    Returns:
        Precision@k in [0, 1].
    """
    if k == 0:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for d in top_k if d in relevant)
    return hits / k


def ndcg_at_k(
    retrieved: List[str],
    relevant_scores: Dict[str, int],
    k: int,
) -> float:
    """Normalised Discounted Cumulative Gain at rank k.

    Args:
        retrieved: Ordered list of retrieved document IDs.
        relevant_scores: Mapping from document ID to integer relevance grade
                         (e.g., {doc_id: 2} for highly relevant, 1 for relevant).
        k: Cutoff rank.

    Returns:
        NDCG@k in [0, 1].
    """
    if not relevant_scores or k == 0:
        return 0.0

    def dcg(ranked_ids: List[str], scores: Dict[str, int], cutoff: int) -> float:
        result = 0.0
        for i, doc_id in enumerate(ranked_ids[:cutoff], start=1):
            rel = scores.get(doc_id, 0)
            if rel > 0:
                result += (2 ** rel - 1) / math.log2(i + 1)
        return result

    actual_dcg = dcg(retrieved, relevant_scores, k)
    # Ideal ranking: sort relevant docs by descending grade.
    ideal_ranked = sorted(relevant_scores.keys(), key=lambda d: relevant_scores[d], reverse=True)
    ideal_dcg = dcg(ideal_ranked, relevant_scores, k)

    if ideal_dcg == 0:
        return 0.0
    return actual_dcg / ideal_dcg


def mean_reciprocal_rank(retrieved: List[str], relevant: Set[str]) -> float:
    """Reciprocal rank of the first relevant document.

    Args:
        retrieved: Ordered list of retrieved document IDs.
        relevant: Set of relevant document IDs.

    Returns:
        MRR score in (0, 1].  Returns 0 if no relevant doc is retrieved.
    """
    for i, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            return 1.0 / i
    return 0.0


def average_precision(retrieved: List[str], relevant: Set[str]) -> float:
    """Average Precision (AP) — area under the interpolated precision-recall curve.

    Args:
        retrieved: Ordered list of retrieved document IDs.
        relevant: Set of relevant document IDs.

    Returns:
        AP score in [0, 1].  Returns 0 if relevant is empty.
    """
    if not relevant:
        return 0.0

    hits = 0
    cumulative_precision = 0.0
    for i, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant:
            hits += 1
            cumulative_precision += hits / i

    if hits == 0:
        return 0.0
    return cumulative_precision / len(relevant)


def evaluate_retrieval(
    results_by_query: Dict[str, List[str]],
    qrels: Dict[str, Dict[str, int]],
    k_values: List[int] | None = None,
) -> Dict[str, float]:
    """Compute mean retrieval metrics across all queries.

    Args:
        results_by_query: Mapping from query_id to ordered list of retrieved
                          document IDs.
        qrels: Relevance judgments {query_id: {doc_id: relevance_score}}.
        k_values: List of cutoff values for Recall@k, P@k, NDCG@k.

    Returns:
        Dictionary mapping metric names to their mean values across queries.
        Example keys: "Recall@5", "NDCG@10", "MRR", "MAP".
    """
    if k_values is None:
        k_values = [5, 10, 20, 100]

    metrics: Dict[str, List[float]] = {f"Recall@{k}": [] for k in k_values}
    metrics.update({f"Precision@{k}": [] for k in k_values})
    metrics.update({f"NDCG@{k}": [] for k in k_values})
    metrics["MRR"] = []
    metrics["MAP"] = []

    for query_id, retrieved in results_by_query.items():
        if query_id not in qrels:
            continue

        relevant_scores = qrels[query_id]
        relevant_set = set(relevant_scores.keys())

        for k in k_values:
            metrics[f"Recall@{k}"].append(recall_at_k(retrieved, relevant_set, k))
            metrics[f"Precision@{k}"].append(precision_at_k(retrieved, relevant_set, k))
            metrics[f"NDCG@{k}"].append(ndcg_at_k(retrieved, relevant_scores, k))

        metrics["MRR"].append(mean_reciprocal_rank(retrieved, relevant_set))
        metrics["MAP"].append(average_precision(retrieved, relevant_set))

    # Return means.
    return {
        name: (sum(values) / len(values) if values else 0.0)
        for name, values in metrics.items()
    }
