#!/usr/bin/env python3
"""CLI script for indexing a document corpus into DocRank.

Usage examples:
    python scripts/index_corpus.py --source scifact
    python scripts/index_corpus.py --source directory --data-path /path/to/docs
    python scripts/index_corpus.py --source jsonl --data-path corpus.jsonl \\
        --output-dir ./my_indexes
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# Ensure src is on the Python path when running as a standalone script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings
from src.ingestion.loader import CorpusLoader
from src.pipeline.indexing import IndexingPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Index a document corpus for DocRank retrieval.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source",
        choices=["scifact", "directory", "jsonl"],
        required=True,
        help="Corpus source type.",
    )
    parser.add_argument(
        "--data-path",
        default=None,
        help="Path to the data directory or JSONL file (required for "
             "directory and jsonl sources).",
    )
    parser.add_argument(
        "--output-dir",
        default="indexes",
        help="Directory where index files will be saved.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=settings.chunk_size,
        help="Maximum characters per text chunk.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=settings.chunk_overlap,
        help="Character overlap between consecutive chunks.",
    )
    parser.add_argument(
        "--embedding-model",
        default=settings.embedding_model,
        help="Sentence-transformers model name for dense encoding.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bm25_path = str(output_dir / "bm25.pkl")
    faiss_path = str(output_dir / "faiss.index")

    # ------------------------------------------------------------------
    # Load corpus
    # ------------------------------------------------------------------
    loader = CorpusLoader()

    if args.source == "scifact":
        logger.info("Loading SciFact corpus from HuggingFace…")
        documents, queries, qrels = loader.load_scifact()
        logger.info(
            "Loaded %d documents, %d queries, %d qrel entries.",
            len(documents),
            len(queries),
            len(qrels),
        )

    elif args.source == "directory":
        if not args.data_path:
            logger.error("--data-path is required for source=directory")
            sys.exit(1)
        logger.info("Loading documents from directory: %s", args.data_path)
        documents = loader.load_from_directory(args.data_path)

    elif args.source == "jsonl":
        if not args.data_path:
            logger.error("--data-path is required for source=jsonl")
            sys.exit(1)
        logger.info("Loading documents from JSONL: %s", args.data_path)
        documents = loader.load_from_jsonl(args.data_path)

    if not documents:
        logger.error("No documents loaded. Check --data-path and --source.")
        sys.exit(1)

    logger.info("Loaded %d documents total.", len(documents))

    # ------------------------------------------------------------------
    # Run indexing pipeline
    # ------------------------------------------------------------------
    pipeline = IndexingPipeline(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        embedding_model=args.embedding_model,
        bm25_index_path=bm25_path,
        faiss_index_path=faiss_path,
    )

    logger.info("Starting indexing pipeline…")
    stats = pipeline.run(documents)

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  DocRank Indexing Complete")
    print("=" * 60)
    print(f"  Documents indexed   : {stats.n_docs:,}")
    print(f"  Chunks created      : {stats.n_chunks:,}")
    print(f"  BM25 index          : {stats.bm25_index_path}")
    print(f"  FAISS index         : {stats.faiss_index_path}")
    print(f"  Elapsed             : {stats.elapsed_seconds:.1f}s")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  make serve          # Start the API server")
    print("  make evaluate       # Run evaluation on SciFact test queries")
    print()


if __name__ == "__main__":
    main()
