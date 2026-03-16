"""Context window construction and prompt formatting for RAG generation."""

import re
from typing import Dict, List

from ..retrieval.result import RetrievalResult

# Attempt to use tiktoken for accurate GPT token counts.
# Fall back to a character-based estimate if tiktoken is unavailable.
try:
    import tiktoken  # type: ignore

    _ENCODING = tiktoken.get_encoding("cl100k_base")

    def _count_tokens(text: str) -> int:
        return len(_ENCODING.encode(text))

except Exception:
    def _count_tokens(text: str) -> int:  # type: ignore[misc]
        # Conservative estimate: 4 characters per token.
        return max(1, len(text) // 4)


_SYSTEM_PROMPT = (
    "You are a precise question-answering system. "
    "Answer based only on the provided context. "
    "If the answer is not in the context, say "
    "'I don't have enough information to answer this.'"
)


class ContextBuilder:
    """Constructs the context string and prompt messages for LLM generation.

    Greedy token-budget selection ensures the most relevant chunks are
    prioritised while staying within the model's context window.
    """

    def __init__(self, max_tokens: int = 3000) -> None:
        """
        Args:
            max_tokens: Maximum number of tokens allowed in the context
                        string passed to the LLM.
        """
        self.max_tokens = max_tokens

    # ------------------------------------------------------------------
    # Context construction
    # ------------------------------------------------------------------

    def build_context(
        self,
        chunks: List[RetrievalResult],
        max_tokens: int | None = None,
    ) -> str:
        """Build a context string from retrieval results within a token budget.

        Chunks are assumed to arrive in order of descending relevance.
        We greedily add chunks until the token budget is exhausted.

        Args:
            chunks: Ranked retrieval results (most relevant first).
            max_tokens: Override the instance-level token budget.

        Returns:
            Formatted context string ready to be injected into the prompt.
        """
        budget = max_tokens if max_tokens is not None else self.max_tokens
        context_parts: List[str] = []
        tokens_used = 0

        for i, chunk in enumerate(chunks, start=1):
            title = chunk.title or f"Document {chunk.doc_id}"
            formatted = f"[Source {i}: {title}]\n{chunk.text}\n\n"
            chunk_tokens = _count_tokens(formatted)

            if tokens_used + chunk_tokens > budget:
                # Try to squeeze in a truncated version if we have budget left.
                remaining = budget - tokens_used
                if remaining > 50:
                    # Truncate at word boundary.
                    words = formatted.split()
                    truncated = ""
                    for word in words:
                        candidate = truncated + (" " if truncated else "") + word
                        if _count_tokens(candidate) > remaining:
                            break
                        truncated = candidate
                    if truncated.strip():
                        context_parts.append(truncated.strip())
                break

            context_parts.append(formatted)
            tokens_used += chunk_tokens

        return "".join(context_parts).rstrip()

    # ------------------------------------------------------------------
    # Prompt formatting
    # ------------------------------------------------------------------

    def build_prompt(
        self,
        query: str,
        context: str,
    ) -> List[Dict[str, str]]:
        """Construct the chat message list for the LLM.

        Args:
            query: The user's original question.
            context: The context string produced by build_context().

        Returns:
            List of message dicts compatible with the OpenAI chat API.
        """
        user_content = (
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            "Answer:"
        )
        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    # ------------------------------------------------------------------
    # Citation extraction
    # ------------------------------------------------------------------

    def extract_citations(
        self,
        response: str,
        chunks: List[RetrievalResult],
    ) -> List[str]:
        """Identify which source chunks contributed to the LLM response.

        Uses token-level overlap: a chunk is considered cited if it shares
        a sufficient number of non-trivial tokens with the response text.

        Args:
            response: The generated answer text.
            chunks: The context chunks passed to the LLM.

        Returns:
            List of chunk_ids for chunks that appear to be cited.
        """
        response_tokens = set(_tokenize_for_overlap(response))
        cited: List[str] = []

        for chunk in chunks:
            chunk_tokens = set(_tokenize_for_overlap(chunk.text))
            if not chunk_tokens:
                continue
            overlap = response_tokens & chunk_tokens
            overlap_ratio = len(overlap) / len(chunk_tokens)
            if overlap_ratio >= 0.15:
                cited.append(chunk.chunk_id)

        return cited


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "is", "are", "was", "were", "it", "this", "that",
    "be", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "i", "we", "you", "he", "she", "they",
}


def _tokenize_for_overlap(text: str) -> List[str]:
    """Lowercase, split on non-alphanumeric, remove stopwords."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 2]
