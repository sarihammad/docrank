"""Prometheus metrics middleware for DocRank API."""

import time
from typing import Callable

from fastapi import Request, Response
from prometheus_client import Counter, Histogram

# ------------------------------------------------------------------
# Metric definitions
# ------------------------------------------------------------------

query_latency_seconds = Histogram(
    "docrank_query_latency_seconds",
    "End-to-end query latency in seconds, by endpoint.",
    labelnames=["endpoint"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
)

retrieval_latency_ms = Histogram(
    "docrank_retrieval_latency_ms",
    "Retrieval stage latency in milliseconds.",
    buckets=[1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500],
)

reranking_latency_ms = Histogram(
    "docrank_reranking_latency_ms",
    "Cross-encoder reranking latency in milliseconds.",
    buckets=[1, 5, 10, 25, 50, 100, 250, 500, 1000, 2500],
)

generation_latency_ms = Histogram(
    "docrank_generation_latency_ms",
    "LLM generation latency in milliseconds.",
    buckets=[100, 250, 500, 1000, 2000, 5000, 10000, 30000],
)

cache_hit_counter = Counter(
    "docrank_cache_hits_total",
    "Number of cache hits across all endpoints.",
)

query_counter = Counter(
    "docrank_queries_total",
    "Total number of queries processed, by mode.",
    labelnames=["mode"],
)


# ------------------------------------------------------------------
# ASGI middleware
# ------------------------------------------------------------------

class PrometheusMiddleware:
    """ASGI middleware that records per-request latency metrics."""

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(
        self, scope, receive, send
    ) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "/")
        start = time.perf_counter()

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                elapsed = time.perf_counter() - start
                query_latency_seconds.labels(endpoint=path).observe(elapsed)
            await send(message)

        await self.app(scope, receive, send_wrapper)
