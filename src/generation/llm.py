"""OpenAI-compatible LLM client with retry logic and latency tracking."""

import logging
import time
from dataclasses import dataclass
from typing import Dict, List

logger = logging.getLogger(__name__)

_MOCK_RESPONSE = (
    "LLM not configured — set OPENAI_API_KEY in your environment or .env "
    "file to enable answer generation.  The retrieval stack is fully "
    "operational: see the `sources` field for relevant passages."
)


@dataclass
class GenerationResult:
    """Output of a single LLM generation call.

    Attributes:
        text: Generated answer text.
        tokens_used: Total tokens consumed (prompt + completion).
        latency_ms: Wall-clock generation time in milliseconds.
        model: Model name used for this generation.
    """

    text: str
    tokens_used: int
    latency_ms: float
    model: str


class LLMClient:
    """Thin wrapper around the OpenAI chat completions API.

    Supports any OpenAI-compatible endpoint (Ollama, vLLM, Azure OpenAI,
    Together AI, etc.) via the configurable base_url parameter.

    If OPENAI_API_KEY is not set, the client returns an informative stub
    response so the retrieval pipeline remains usable in demo environments.
    """

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o-mini",
        max_retries: int = 3,
    ) -> None:
        """
        Args:
            api_key: OpenAI (or compatible) API key.
            base_url: API base URL.
            model: Default model to use for generation.
            max_retries: Maximum number of retry attempts on transient errors.
        """
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.max_retries = max_retries
        self._client = None

    # ------------------------------------------------------------------
    # Client initialisation (lazy)
    # ------------------------------------------------------------------

    @property
    def client(self):
        if self._client is None:
            if not self.api_key:
                return None
            try:
                from openai import OpenAI  # type: ignore

                self._client = OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                )
            except ImportError as exc:
                raise ImportError(
                    "The `openai` library is required. "
                    "Install it with: pip install openai"
                ) from exc
        return self._client

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 512,
    ) -> str:
        """Generate a response from the LLM.

        Args:
            messages: Chat messages in OpenAI format.
            temperature: Sampling temperature (0 = deterministic).
            max_tokens: Maximum tokens to generate.

        Returns:
            Generated response text.
        """
        return self.generate_with_metadata(
            messages, temperature=temperature, max_tokens=max_tokens
        ).text

    def generate_with_metadata(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 512,
    ) -> GenerationResult:
        """Generate a response and return rich metadata.

        Args:
            messages: Chat messages in OpenAI format.
            temperature: Sampling temperature.
            max_tokens: Maximum completion tokens.

        Returns:
            GenerationResult with text, token counts, and latency.
        """
        if not self.api_key or self.client is None:
            return GenerationResult(
                text=_MOCK_RESPONSE,
                tokens_used=0,
                latency_ms=0.0,
                model="none",
            )

        last_error: Exception | None = None
        delay = 1.0

        for attempt in range(1, self.max_retries + 1):
            try:
                start = time.perf_counter()
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                latency_ms = (time.perf_counter() - start) * 1000.0

                text = response.choices[0].message.content or ""
                tokens_used = (
                    response.usage.total_tokens if response.usage else 0
                )
                return GenerationResult(
                    text=text.strip(),
                    tokens_used=tokens_used,
                    latency_ms=latency_ms,
                    model=self.model,
                )

            except Exception as exc:
                last_error = exc
                # Retry on rate-limit or transient server errors.
                err_str = str(exc).lower()
                retryable = any(
                    kw in err_str
                    for kw in ("rate", "timeout", "connection", "503", "502", "529")
                )
                if not retryable or attempt == self.max_retries:
                    logger.error(
                        "LLM generation failed (attempt %d/%d): %s",
                        attempt,
                        self.max_retries,
                        exc,
                    )
                    break
                logger.warning(
                    "LLM generation attempt %d/%d failed (%s). Retrying in %.1fs…",
                    attempt,
                    self.max_retries,
                    exc,
                    delay,
                )
                time.sleep(delay)
                delay = min(delay * 2, 30.0)

        raise RuntimeError(
            f"LLM generation failed after {self.max_retries} attempts."
        ) from last_error
