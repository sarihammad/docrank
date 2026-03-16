"""LLM-as-judge evaluation for answer correctness, faithfulness, and hallucination."""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class JudgementResult:
    """Result of a single LLM judge evaluation.

    Attributes:
        score: Numeric score (scale depends on task: 1-5 for correctness,
               0-1 for faithfulness/hallucination).
        reasoning: Judge's chain-of-thought explanation.
        raw_response: Unprocessed LLM output for debugging.
        label: Optional categorical label (e.g., "faithful" / "unfaithful").
    """

    score: float
    reasoning: str
    raw_response: str
    label: Optional[str] = None


@dataclass
class EvaluationReport:
    """Aggregated evaluation results across a batch of queries.

    Attributes:
        correctness_scores: Per-query correctness scores (1-5 scale).
        faithfulness_scores: Per-query faithfulness scores (0-1).
        hallucination_scores: Per-query hallucination scores (0-1).
        mean_correctness: Mean correctness across all queries.
        mean_faithfulness: Mean faithfulness across all queries.
        mean_hallucination: Mean hallucination rate across all queries.
        per_query: Detailed per-query results.
    """

    correctness_scores: List[float] = field(default_factory=list)
    faithfulness_scores: List[float] = field(default_factory=list)
    hallucination_scores: List[float] = field(default_factory=list)
    mean_correctness: float = 0.0
    mean_faithfulness: float = 0.0
    mean_hallucination: float = 0.0
    per_query: List[Dict] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "mean_correctness": self.mean_correctness,
            "mean_faithfulness": self.mean_faithfulness,
            "mean_hallucination": self.mean_hallucination,
            "correctness_scores": self.correctness_scores,
            "faithfulness_scores": self.faithfulness_scores,
            "hallucination_scores": self.hallucination_scores,
            "per_query": self.per_query,
        }


class LLMJudge:
    """Uses an LLM to evaluate RAG output quality.

    Implements three evaluation tasks:
      - Answer correctness (1-5 scale vs. reference)
      - Faithfulness (is the answer supported by context?)
      - Hallucination detection (0-1 score)
    """

    def __init__(self, llm_client=None, model: Optional[str] = None) -> None:
        """
        Args:
            llm_client: An LLMClient instance.  If None, creates a default one.
            model: Override the LLM model used for judging.
        """
        if llm_client is None:
            from ..generation.llm import LLMClient
            from ..config import settings

            llm_client = LLMClient(
                api_key=settings.openai_api_key,
                base_url=settings.openai_base_url,
                model=model or settings.llm_judge_model,
            )
        self.llm = llm_client

    # ------------------------------------------------------------------
    # Individual scoring methods
    # ------------------------------------------------------------------

    def score_answer_correctness(
        self,
        question: str,
        answer: str,
        reference: str,
    ) -> JudgementResult:
        """Rate answer correctness on a 1-5 scale against a reference answer.

        Args:
            question: The original question.
            answer: The predicted answer to evaluate.
            reference: The ground-truth reference answer.

        Returns:
            JudgementResult with score in [1, 5] and reasoning.
        """
        prompt = f"""Rate the correctness of the predicted answer compared to the reference answer on a scale of 1-5.

Scoring guide:
1 = Completely wrong or contradicts the reference
2 = Mostly wrong, minor overlap with reference
3 = Partially correct, captures some key information
4 = Mostly correct, minor omissions or inaccuracies
5 = Fully correct, equivalent to or better than the reference

Question: {question}
Reference Answer: {reference}
Predicted Answer: {answer}

Output valid JSON only, no prose outside the JSON object:
{{"score": <integer 1-5>, "reasoning": "<brief explanation>"}}"""

        messages = [
            {
                "role": "system",
                "content": "You are an expert evaluator. Output only valid JSON.",
            },
            {"role": "user", "content": prompt},
        ]

        raw = self.llm.generate(messages, temperature=0.0, max_tokens=256)
        return self._parse_score_response(raw, scale_max=5.0)

    def score_faithfulness(
        self,
        question: str,
        answer: str,
        context: str,
    ) -> JudgementResult:
        """Evaluate whether the answer is fully supported by the context.

        Args:
            question: The original question.
            answer: The generated answer.
            context: The context passages provided to the LLM.

        Returns:
            JudgementResult with score 0 (unfaithful) or 1 (faithful).
        """
        prompt = f"""Determine whether the answer is fully supported by the provided context.

An answer is faithful if every factual claim it makes can be verified from the context.
An answer is unfaithful if it introduces information not present in the context.

Question: {question}
Context: {context[:2000]}
Answer: {answer}

Output valid JSON only:
{{"faithful": <true or false>, "score": <0 for unfaithful, 1 for faithful>, "reasoning": "<explanation>"}}"""

        messages = [
            {
                "role": "system",
                "content": "You are an expert evaluator. Output only valid JSON.",
            },
            {"role": "user", "content": prompt},
        ]

        raw = self.llm.generate(messages, temperature=0.0, max_tokens=256)
        result = self._parse_score_response(raw, scale_max=1.0)
        # Set human-readable label.
        result.label = "faithful" if result.score >= 0.5 else "unfaithful"
        return result

    def score_hallucination(
        self,
        answer: str,
        context: str,
    ) -> JudgementResult:
        """Estimate the hallucination rate of the answer.

        Args:
            answer: The generated answer.
            context: The context passages provided to the LLM.

        Returns:
            JudgementResult with score in [0, 1] where 1 = fully hallucinated.
        """
        prompt = f"""Assess the hallucination level in the answer below, given the context.

A hallucination score of 0 means every claim is grounded in the context.
A hallucination score of 1 means the answer is entirely fabricated.
Use a continuous scale between 0 and 1.

Context: {context[:2000]}
Answer: {answer}

Output valid JSON only:
{{"hallucination_score": <float 0.0-1.0>, "reasoning": "<explanation>"}}"""

        messages = [
            {
                "role": "system",
                "content": "You are an expert evaluator. Output only valid JSON.",
            },
            {"role": "user", "content": prompt},
        ]

        raw = self.llm.generate(messages, temperature=0.0, max_tokens=256)

        # Parse hallucination-specific JSON key.
        result = self._parse_generic_response(raw, score_key="hallucination_score")
        return result

    # ------------------------------------------------------------------
    # Batch evaluation
    # ------------------------------------------------------------------

    def evaluate_batch(
        self,
        queries: List[str],
        answers: List[str],
        contexts: List[str],
        references: Optional[List[str]] = None,
    ) -> EvaluationReport:
        """Run all judge evaluations over a batch of query-answer pairs.

        Args:
            queries: Original questions.
            answers: Generated answers.
            contexts: Context strings passed to the LLM for each query.
            references: Optional ground-truth reference answers.

        Returns:
            EvaluationReport with per-query and aggregate scores.
        """
        n = len(queries)
        report = EvaluationReport()

        for i in range(n):
            query = queries[i]
            answer = answers[i]
            context = contexts[i] if i < len(contexts) else ""
            reference = references[i] if references and i < len(references) else None

            entry: Dict = {"query": query, "answer": answer}

            # Faithfulness
            try:
                faith_result = self.score_faithfulness(query, answer, context)
                report.faithfulness_scores.append(faith_result.score)
                entry["faithfulness"] = {
                    "score": faith_result.score,
                    "label": faith_result.label,
                    "reasoning": faith_result.reasoning,
                }
            except Exception as exc:
                logger.warning("Faithfulness scoring failed for query %d: %s", i, exc)
                report.faithfulness_scores.append(0.0)
                entry["faithfulness"] = {"score": 0.0, "error": str(exc)}

            # Hallucination
            try:
                hall_result = self.score_hallucination(answer, context)
                report.hallucination_scores.append(hall_result.score)
                entry["hallucination"] = {
                    "score": hall_result.score,
                    "reasoning": hall_result.reasoning,
                }
            except Exception as exc:
                logger.warning("Hallucination scoring failed for query %d: %s", i, exc)
                report.hallucination_scores.append(0.0)
                entry["hallucination"] = {"score": 0.0, "error": str(exc)}

            # Correctness (only if reference available)
            if reference:
                try:
                    corr_result = self.score_answer_correctness(
                        query, answer, reference
                    )
                    report.correctness_scores.append(corr_result.score)
                    entry["correctness"] = {
                        "score": corr_result.score,
                        "reasoning": corr_result.reasoning,
                    }
                except Exception as exc:
                    logger.warning("Correctness scoring failed for query %d: %s", i, exc)
                    report.correctness_scores.append(0.0)
                    entry["correctness"] = {"score": 0.0, "error": str(exc)}

            report.per_query.append(entry)

        def _mean(lst: List[float]) -> float:
            return sum(lst) / len(lst) if lst else 0.0

        report.mean_faithfulness = _mean(report.faithfulness_scores)
        report.mean_hallucination = _mean(report.hallucination_scores)
        report.mean_correctness = _mean(report.correctness_scores)

        return report

    # ------------------------------------------------------------------
    # JSON parsing helpers
    # ------------------------------------------------------------------

    def _parse_score_response(
        self, raw: str, scale_max: float = 5.0
    ) -> JudgementResult:
        """Parse a JSON response that contains a "score" field."""
        try:
            data = _extract_json(raw)
            score = float(data.get("score", 0))
            # Normalise hallucination / faithfulness scores to [0,1].
            if scale_max == 1.0:
                score = max(0.0, min(1.0, score))
            else:
                score = max(1.0, min(scale_max, score))
            reasoning = data.get("reasoning", "")
            return JudgementResult(score=score, reasoning=reasoning, raw_response=raw)
        except Exception as exc:
            logger.warning("Could not parse judge response: %s\nRaw: %s", exc, raw)
            return JudgementResult(
                score=0.0,
                reasoning="Parsing failed.",
                raw_response=raw,
            )

    def _parse_generic_response(
        self, raw: str, score_key: str
    ) -> JudgementResult:
        """Parse a JSON response using a custom score key."""
        try:
            data = _extract_json(raw)
            score = float(data.get(score_key, data.get("score", 0)))
            score = max(0.0, min(1.0, score))
            reasoning = data.get("reasoning", "")
            return JudgementResult(score=score, reasoning=reasoning, raw_response=raw)
        except Exception as exc:
            logger.warning("Could not parse judge response: %s\nRaw: %s", exc, raw)
            return JudgementResult(
                score=0.0,
                reasoning="Parsing failed.",
                raw_response=raw,
            )


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _extract_json(text: str) -> dict:
    """Extract a JSON object from an LLM response that may contain prose."""
    # Try direct parse first.
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # Try to find a JSON object within the text.
    match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Last resort: try extracting key-value pairs manually.
    result = {}
    score_match = re.search(r'"?score"?\s*:\s*([0-9.]+)', text)
    if score_match:
        result["score"] = float(score_match.group(1))

    hallucination_match = re.search(
        r'"?hallucination_score"?\s*:\s*([0-9.]+)', text
    )
    if hallucination_match:
        result["hallucination_score"] = float(hallucination_match.group(1))

    reasoning_match = re.search(r'"?reasoning"?\s*:\s*"([^"]+)"', text)
    if reasoning_match:
        result["reasoning"] = reasoning_match.group(1)

    return result
