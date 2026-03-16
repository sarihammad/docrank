"""Reference-based and reference-free generation quality metrics.

These heuristic metrics complement LLM-as-judge evaluation and provide
fast, API-free signals for continuous monitoring.
"""

import re
from typing import List, Set


def _tokenize(text: str) -> List[str]:
    """Simple whitespace + punctuation tokenizer."""
    return re.findall(r"[a-z0-9]+", text.lower())


def token_overlap_f1(prediction: str, reference: str) -> float:
    """Token-level F1 between a predicted answer and a reference answer.

    Ignores word order; measures bag-of-words overlap.  Equivalent to the
    metric used in SQuAD evaluation.

    Args:
        prediction: Model-generated answer.
        reference: Ground-truth reference answer.

    Returns:
        F1 score in [0, 1].
    """
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)

    if not pred_tokens or not ref_tokens:
        return float(pred_tokens == ref_tokens)

    pred_set: List[str] = list(pred_tokens)
    ref_set: List[str] = list(ref_tokens)

    # Count token-level matches (with multiplicity).
    pred_counts: dict = {}
    for t in pred_set:
        pred_counts[t] = pred_counts.get(t, 0) + 1

    ref_counts: dict = {}
    for t in ref_set:
        ref_counts[t] = ref_counts.get(t, 0) + 1

    common_tokens = 0
    for token, count in pred_counts.items():
        common_tokens += min(count, ref_counts.get(token, 0))

    if common_tokens == 0:
        return 0.0

    precision = common_tokens / len(pred_tokens)
    recall = common_tokens / len(ref_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def faithfulness_score(answer: str, context: str) -> float:
    """Estimate how well the answer is grounded in the provided context.

    For each sentence in the answer, we compute the token overlap with
    the entire context.  A sentence is considered faithful if it has at
    least 20% token overlap.  The overall score is the fraction of
    faithful sentences.

    This is a lightweight heuristic.  For production use, supplement with
    LLMJudge.score_faithfulness().

    Args:
        answer: The generated answer to evaluate.
        context: The context string passed to the LLM.

    Returns:
        Faithfulness score in [0, 1].
    """
    sentences = _split_sentences(answer)
    if not sentences:
        return 0.0

    context_tokens: Set[str] = set(_tokenize(context))
    if not context_tokens:
        return 0.0

    faithful_count = 0
    for sentence in sentences:
        sentence_tokens = set(_tokenize(sentence))
        if not sentence_tokens:
            faithful_count += 1
            continue
        overlap = sentence_tokens & context_tokens
        overlap_ratio = len(overlap) / len(sentence_tokens)
        if overlap_ratio >= 0.20:
            faithful_count += 1

    return faithful_count / len(sentences)


def context_utilization(answer: str, chunks: List[str]) -> float:
    """Fraction of context chunks that contributed tokens to the answer.

    A chunk is considered "utilised" if at least 10% of its non-trivial
    tokens appear in the answer.

    Args:
        answer: The generated answer.
        chunks: List of context chunk texts passed to the LLM.

    Returns:
        Utilization score in [0, 1].  Returns 0 if chunks is empty.
    """
    if not chunks:
        return 0.0

    answer_tokens: Set[str] = set(_tokenize(answer))
    utilised = 0

    for chunk in chunks:
        chunk_tokens = set(_tokenize(chunk))
        if not chunk_tokens:
            continue
        overlap = answer_tokens & chunk_tokens
        if len(overlap) / len(chunk_tokens) >= 0.10:
            utilised += 1

    return utilised / len(chunks)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _split_sentences(text: str) -> List[str]:
    """Split text into sentences using punctuation heuristics."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]
