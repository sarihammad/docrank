"""Corpus loading utilities: BEIR/HuggingFace, directory, and JSONL sources."""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .document import Document

logger = logging.getLogger(__name__)

# Type alias: qrels map query_id -> {doc_id -> relevance_score}
QRelDict = Dict[str, Dict[str, int]]


class Query:
    """Represents a search query with an optional answer reference."""

    __slots__ = ("query_id", "text", "metadata")

    def __init__(
        self,
        query_id: str,
        text: str,
        metadata: Dict[str, Any] | None = None,
    ) -> None:
        self.query_id = query_id
        self.text = text
        self.metadata = metadata or {}

    def __repr__(self) -> str:
        return f"Query(query_id={self.query_id!r}, text={self.text[:60]!r})"


class CorpusLoader:
    """Unified loader for multiple corpus sources.

    Supports:
      - SciFact via HuggingFace datasets (allenai/scifact)
      - Arbitrary directory of .txt and .md files
      - JSONL files with {"id", "title", "text"} records
    """

    # ------------------------------------------------------------------
    # SciFact (via HuggingFace datasets)
    # ------------------------------------------------------------------

    def load_scifact(
        self,
    ) -> Tuple[List[Document], List[Query], QRelDict]:
        """Load the SciFact dataset from HuggingFace.

        Returns:
            documents: ~5K scientific corpus documents.
            queries: Test / validation queries.
            qrels: Relevance judgments {query_id: {doc_id: score}}.

        Raises:
            ImportError: If the `datasets` library is not installed.
            RuntimeError: If the dataset cannot be fetched.
        """
        try:
            from datasets import load_dataset  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "The `datasets` library is required to load SciFact. "
                "Install it with: pip install datasets"
            ) from exc

        logger.info("Loading SciFact corpus from HuggingFace…")
        corpus_ds = load_dataset("allenai/scifact", "corpus", trust_remote_code=True)
        queries_ds = load_dataset("allenai/scifact", "queries", trust_remote_code=True)

        documents: List[Document] = []
        # SciFact corpus has a single split called "train" for the full corpus.
        for row in corpus_ds["train"]:
            doc_id = str(row["doc_id"])
            title = row.get("title", "") or ""
            # SciFact corpus stores abstract as a list of sentences.
            abstract = row.get("abstract", []) or []
            if isinstance(abstract, list):
                text = " ".join(abstract)
            else:
                text = str(abstract)
            documents.append(
                Document(
                    doc_id=doc_id,
                    text=f"{title}. {text}".strip(". ") if title else text,
                    title=title,
                    metadata={"source": "scifact"},
                )
            )

        logger.info("Loaded %d corpus documents.", len(documents))

        queries: List[Query] = []
        qrels: QRelDict = {}

        # SciFact has train and validation splits for queries.
        for split in ("train", "validation", "test"):
            if split not in queries_ds:
                continue
            for row in queries_ds[split]:
                query_id = str(row["id"])
                text = row.get("question", "") or row.get("claim", "") or ""
                if not text:
                    continue
                queries.append(
                    Query(
                        query_id=query_id,
                        text=text,
                        metadata={"split": split},
                    )
                )
                # Build qrels from evidence field if present.
                evidence = row.get("evidence", {}) or {}
                for doc_id_str, evidence_list in evidence.items():
                    if not evidence_list:
                        continue
                    label = evidence_list[0].get("label", "SUPPORT") if isinstance(evidence_list[0], dict) else "SUPPORT"
                    score = 2 if label == "SUPPORT" else 1
                    if query_id not in qrels:
                        qrels[query_id] = {}
                    qrels[query_id][str(doc_id_str)] = score

        logger.info(
            "Loaded %d queries and %d qrels entries.", len(queries), len(qrels)
        )
        return documents, queries, qrels

    # ------------------------------------------------------------------
    # Directory source
    # ------------------------------------------------------------------

    def load_from_directory(self, path: str) -> List[Document]:
        """Load all .txt and .md files from a directory as Documents.

        Args:
            path: Path to a directory containing text files.

        Returns:
            List of Documents, one per file.
        """
        root = Path(path)
        if not root.is_dir():
            raise FileNotFoundError(f"Directory not found: {path}")

        documents: List[Document] = []
        for file_path in sorted(root.rglob("*")):
            if file_path.suffix.lower() not in {".txt", ".md"}:
                continue
            try:
                text = file_path.read_text(encoding="utf-8", errors="replace")
                doc_id = str(file_path.relative_to(root)).replace(
                    os.sep, "__"
                )
                title = file_path.stem.replace("_", " ").replace("-", " ").title()
                documents.append(
                    Document(
                        doc_id=doc_id,
                        text=text,
                        title=title,
                        metadata={
                            "source": "directory",
                            "file_path": str(file_path),
                        },
                    )
                )
            except Exception as exc:
                logger.warning("Failed to read %s: %s", file_path, exc)

        logger.info(
            "Loaded %d documents from directory %s.", len(documents), path
        )
        return documents

    # ------------------------------------------------------------------
    # JSONL source
    # ------------------------------------------------------------------

    def load_from_jsonl(self, path: str) -> List[Document]:
        """Load documents from a JSONL file.

        Each line must be a JSON object with at minimum an "id" field and a
        "text" field.  An optional "title" field is also supported.

        Args:
            path: Path to a JSONL file.

        Returns:
            List of Documents.

        Raises:
            FileNotFoundError: If the file does not exist.
        """
        file_path = Path(path)
        if not file_path.is_file():
            raise FileNotFoundError(f"JSONL file not found: {path}")

        documents: List[Document] = []
        with file_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.warning(
                        "Skipping invalid JSON on line %d: %s", line_no, exc
                    )
                    continue

                doc_id = str(
                    record.get("id", record.get("doc_id", f"doc_{line_no}"))
                )
                text = record.get("text", record.get("content", ""))
                title = record.get("title", "")
                metadata = {
                    k: v
                    for k, v in record.items()
                    if k not in {"id", "doc_id", "text", "content", "title"}
                }
                metadata["source"] = "jsonl"

                if not text:
                    logger.warning("Empty text on line %d, skipping.", line_no)
                    continue

                documents.append(
                    Document(
                        doc_id=doc_id,
                        text=text,
                        title=title,
                        metadata=metadata,
                    )
                )

        logger.info("Loaded %d documents from %s.", len(documents), path)
        return documents
