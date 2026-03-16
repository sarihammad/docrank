"""Document indexing pipeline: chunk → BM25 index + FAISS index."""

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from ..config import settings
from ..ingestion.chunker import RecursiveChunker
from ..ingestion.document import Document
from ..retrieval.bm25_retriever import BM25Retriever
from ..retrieval.dense_retriever import DenseRetriever

logger = logging.getLogger(__name__)


@dataclass
class IndexingStats:
    """Statistics produced by a completed indexing run.

    Attributes:
        n_docs: Number of input documents processed.
        n_chunks: Total number of chunks created.
        bm25_index_path: Path where the BM25 index was saved.
        faiss_index_path: Path where the FAISS index was saved.
        elapsed_seconds: Total wall-clock time for the indexing run.
    """

    n_docs: int
    n_chunks: int
    bm25_index_path: str
    faiss_index_path: str
    elapsed_seconds: float

    def __str__(self) -> str:
        return (
            f"IndexingStats("
            f"docs={self.n_docs}, "
            f"chunks={self.n_chunks}, "
            f"time={self.elapsed_seconds:.1f}s, "
            f"bm25={self.bm25_index_path}, "
            f"faiss={self.faiss_index_path})"
        )


class IndexingPipeline:
    """Orchestrates the full document ingestion and indexing workflow.

    Steps:
      1. Chunk documents using RecursiveChunker.
      2. Build BM25 index (rank_bm25).
      3. Build dense FAISS index (sentence-transformers + FAISS).
      4. Persist both indexes to disk.
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        embedding_model: str | None = None,
        bm25_index_path: str | None = None,
        faiss_index_path: str | None = None,
    ) -> None:
        """
        Args:
            chunk_size: Override default chunk size from config.
            chunk_overlap: Override default overlap from config.
            embedding_model: Override embedding model from config.
            bm25_index_path: Override BM25 index output path.
            faiss_index_path: Override FAISS index output path.
        """
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        self.embedding_model = embedding_model or settings.embedding_model
        self.bm25_index_path = bm25_index_path or settings.bm25_index_path
        self.faiss_index_path = faiss_index_path or settings.faiss_index_path

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self, documents: List[Document]) -> IndexingStats:
        """Execute the full indexing pipeline.

        Args:
            documents: Raw documents to index.

        Returns:
            IndexingStats with counts, paths, and timing.

        Raises:
            ValueError: If documents is empty.
        """
        if not documents:
            raise ValueError("Cannot index empty document list.")

        start = time.perf_counter()

        # Step 1: Chunking.
        logger.info("Chunking %d documents…", len(documents))
        chunker = RecursiveChunker(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        chunks = chunker.chunk_documents(documents)
        logger.info("Created %d chunks.", len(chunks))

        # Step 2: BM25 index.
        logger.info("Building BM25 index…")
        bm25 = BM25Retriever()
        bm25.build_index(chunks)
        bm25.save(self.bm25_index_path)

        # Step 3: Dense FAISS index.
        logger.info("Building dense FAISS index…")
        dense = DenseRetriever(model_name=self.embedding_model)
        dense.build_index(chunks)
        meta_path = self.faiss_index_path + ".meta.json"
        dense.save(self.faiss_index_path, meta_path)

        elapsed = time.perf_counter() - start

        stats = IndexingStats(
            n_docs=len(documents),
            n_chunks=len(chunks),
            bm25_index_path=str(Path(self.bm25_index_path).resolve()),
            faiss_index_path=str(Path(self.faiss_index_path).resolve()),
            elapsed_seconds=elapsed,
        )
        logger.info("Indexing complete: %s", stats)
        return stats

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_indexes(self) -> Tuple[BM25Retriever, DenseRetriever]:
        """Load previously built indexes from disk.

        Returns:
            Tuple of (BM25Retriever, DenseRetriever) ready for search.

        Raises:
            FileNotFoundError: If index files are not found.
        """
        logger.info("Loading BM25 index from %s…", self.bm25_index_path)
        bm25 = BM25Retriever()
        bm25.load(self.bm25_index_path)

        logger.info("Loading dense index from %s…", self.faiss_index_path)
        dense = DenseRetriever(model_name=self.embedding_model)
        meta_path = self.faiss_index_path + ".meta.json"
        dense.load(self.faiss_index_path, meta_path)

        return bm25, dense
