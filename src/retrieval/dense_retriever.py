"""Dense bi-encoder retriever backed by sentence-transformers and FAISS."""

import json
import logging
from pathlib import Path
from typing import List

import faiss  # type: ignore
import numpy as np
from sentence_transformers import SentenceTransformer  # type: ignore
from tqdm import tqdm  # type: ignore

from ..ingestion.document import Chunk
from .result import RetrievalResult

logger = logging.getLogger(__name__)


class DenseRetriever:
    """Bi-encoder retriever that encodes passages with a SentenceTransformer
    and stores embeddings in a FAISS IndexFlatIP (inner-product) index.

    Vectors are L2-normalised before indexing so that inner-product search
    is equivalent to cosine similarity.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        """
        Args:
            model_name: HuggingFace / sentence-transformers model identifier.
        """
        self.model_name = model_name
        self._model: SentenceTransformer | None = None
        self._index: faiss.IndexFlatIP | None = None
        self._chunks: List[Chunk] = []
        self._dimension: int = 0

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            logger.info("Loading embedding model: %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)
        return self._model

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def build_index(self, chunks: List[Chunk], batch_size: int = 64) -> None:
        """Encode all chunks and build a FAISS index.

        Args:
            chunks: Text chunks to index.
            batch_size: Number of passages to encode per forward pass.
        """
        if not chunks:
            raise ValueError("Cannot build dense index from empty chunk list.")

        logger.info("Encoding %d chunks with %s…", len(chunks), self.model_name)
        texts = [c.text for c in chunks]
        embeddings = self.encode_batch(texts, batch_size=batch_size)

        self._dimension = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(self._dimension)
        self._index.add(embeddings)
        self._chunks = list(chunks)

        logger.info(
            "FAISS index built: %d vectors of dimension %d.",
            self._index.ntotal,
            self._dimension,
        )

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode_batch(
        self, texts: List[str], batch_size: int = 64, show_progress: bool = True
    ) -> np.ndarray:
        """Encode a list of texts and return L2-normalised embeddings.

        Args:
            texts: Input strings.
            batch_size: Encoding batch size.
            show_progress: Whether to display a tqdm progress bar.

        Returns:
            Float32 array of shape (len(texts), embedding_dim), L2-normalised.
        """
        all_embeddings: List[np.ndarray] = []
        batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]

        iterator = tqdm(batches, desc="Encoding", unit="batch") if show_progress else batches
        for batch in iterator:
            batch_emb = self.model.encode(
                batch,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            all_embeddings.append(batch_emb.astype(np.float32))

        return np.vstack(all_embeddings)

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query string.

        Returns:
            Float32 array of shape (1, embedding_dim), L2-normalised.
        """
        emb = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return emb.astype(np.float32)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, k: int) -> List[RetrievalResult]:
        """Retrieve the top-k chunks by cosine similarity.

        Args:
            query: Natural-language query string.
            k: Number of results to return.

        Returns:
            List of RetrievalResult sorted by descending similarity score.

        Raises:
            RuntimeError: If the index has not been built yet.
        """
        if self._index is None or not self._chunks:
            raise RuntimeError(
                "Dense index is not built. Call build_index() first."
            )

        k = min(k, len(self._chunks))
        query_emb = self.encode_query(query)
        distances, indices = self._index.search(query_emb, k)

        results: List[RetrievalResult] = []
        for rank, (idx, score) in enumerate(
            zip(indices[0], distances[0])
        ):
            if idx < 0:
                continue
            chunk = self._chunks[int(idx)]
            results.append(
                RetrievalResult(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    score=float(score),
                    text=chunk.text,
                    title=chunk.title,
                    rank=rank,
                    metadata=dict(chunk.metadata),
                )
            )
        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, index_path: str, meta_path: str | None = None) -> None:
        """Save the FAISS index and chunk metadata to disk.

        Args:
            index_path: Path for the FAISS binary file (.index).
            meta_path: Path for the JSON metadata file. Defaults to
                       index_path + ".meta.json".
        """
        if self._index is None:
            raise RuntimeError("Nothing to save — index has not been built.")

        index_path_obj = Path(index_path)
        index_path_obj.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(index_path_obj))

        if meta_path is None:
            meta_path = str(index_path_obj) + ".meta.json"

        meta = {
            "model_name": self.model_name,
            "dimension": self._dimension,
            "chunks": [
                {
                    "chunk_id": c.chunk_id,
                    "doc_id": c.doc_id,
                    "text": c.text,
                    "title": c.title,
                    "start_char": c.start_char,
                    "end_char": c.end_char,
                    "metadata": c.metadata,
                }
                for c in self._chunks
            ],
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)

        logger.info(
            "Dense index saved: %s (%d vectors).", index_path, self._index.ntotal
        )

    def load(self, index_path: str, meta_path: str | None = None) -> None:
        """Load a previously saved FAISS index from disk.

        Args:
            index_path: Path to the FAISS binary file.
            meta_path: Path to the JSON metadata file. Defaults to
                       index_path + ".meta.json".

        Raises:
            FileNotFoundError: If either file does not exist.
        """
        if not Path(index_path).is_file():
            raise FileNotFoundError(f"FAISS index file not found: {index_path}")

        if meta_path is None:
            meta_path = str(index_path) + ".meta.json"

        if not Path(meta_path).is_file():
            raise FileNotFoundError(f"Dense index metadata not found: {meta_path}")

        self._index = faiss.read_index(index_path)

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        self.model_name = meta["model_name"]
        self._dimension = meta["dimension"]
        self._chunks = [
            Chunk(
                chunk_id=c["chunk_id"],
                doc_id=c["doc_id"],
                text=c["text"],
                title=c.get("title", ""),
                start_char=c.get("start_char", 0),
                end_char=c.get("end_char", 0),
                metadata=c.get("metadata", {}),
            )
            for c in meta["chunks"]
        ]
        logger.info(
            "Dense index loaded from %s (%d vectors).",
            index_path,
            self._index.ntotal,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_built(self) -> bool:
        return self._index is not None

    @property
    def num_chunks(self) -> int:
        return len(self._chunks)
