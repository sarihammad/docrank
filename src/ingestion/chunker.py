"""Recursive text chunker with overlap and character position tracking."""

import re
from typing import List

from .document import Chunk, Document


class RecursiveChunker:
    """Splits documents into overlapping chunks using a hierarchy of separators.

    Chunking strategy (in order of preference):
      1. Double newlines (paragraph boundaries)
      2. Single newlines
      3. Sentence-ending punctuation followed by space
      4. Spaces (word boundaries)
      5. Characters (last resort)

    The overlap window ensures that sentences spanning chunk boundaries are
    captured by at least one chunk, preventing context fragmentation.
    """

    # Separator hierarchy: try coarser splits first.
    SEPARATORS = [
        r"\n\n",        # paragraph
        r"\n",          # line break
        r"(?<=[.!?])\s+",  # sentence boundary
        r" ",           # word boundary
        r"",            # character (fallback)
    ]

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64) -> None:
        """
        Args:
            chunk_size: Maximum number of characters per chunk.
            chunk_overlap: Number of characters to overlap between consecutive
                chunks so context is not lost at boundaries.
        """
        if chunk_overlap >= chunk_size:
            raise ValueError(
                f"chunk_overlap ({chunk_overlap}) must be less than "
                f"chunk_size ({chunk_size})"
            )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk_documents(self, documents: List[Document]) -> List[Chunk]:
        """Chunk a list of documents into overlapping text chunks.

        Args:
            documents: Source documents.

        Returns:
            Flat list of Chunk objects with accurate character positions.
        """
        all_chunks: List[Chunk] = []
        for doc in documents:
            all_chunks.extend(self._chunk_document(doc))
        return all_chunks

    def chunk_document(self, document: Document) -> List[Chunk]:
        """Chunk a single document.

        Args:
            document: Source document.

        Returns:
            List of Chunk objects for this document.
        """
        return self._chunk_document(document)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _chunk_document(self, doc: Document) -> List[Chunk]:
        """Split a single document into chunks, tracking character offsets."""
        text = doc.text
        if not text.strip():
            return []

        raw_spans = self._split_with_offsets(text, 0)
        chunks: List[Chunk] = []
        i = 0

        while i < len(raw_spans):
            # Greedily accumulate spans into a single chunk.
            chunk_spans = []
            current_length = 0

            j = i
            while j < len(raw_spans):
                span_text, span_start, span_end = raw_spans[j]
                span_length = span_end - span_start
                if current_length + span_length > self.chunk_size and chunk_spans:
                    break
                chunk_spans.append((span_text, span_start, span_end))
                current_length += span_length
                j += 1

            if not chunk_spans:
                # Single span exceeds chunk_size — emit it whole (hard truncation
                # is worse than a slightly oversized chunk in practice).
                chunk_spans = [raw_spans[i]]
                j = i + 1

            chunk_text = "".join(s[0] for s in chunk_spans).strip()
            if chunk_text:
                chunk_start = chunk_spans[0][1]
                chunk_end = chunk_spans[-1][2]
                chunk_id = f"{doc.doc_id}__chunk_{len(chunks)}"
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        doc_id=doc.doc_id,
                        text=chunk_text,
                        start_char=chunk_start,
                        end_char=chunk_end,
                        title=doc.title,
                        metadata=dict(doc.metadata),
                    )
                )

            # Advance, stepping back by the overlap amount.
            if j >= len(raw_spans):
                break

            # Find the overlap start: walk back from j until we have
            # accumulated chunk_overlap characters worth of spans.
            overlap_chars = 0
            overlap_start = j - 1
            while overlap_start > i and overlap_chars < self.chunk_overlap:
                _, os, oe = raw_spans[overlap_start]
                overlap_chars += oe - os
                overlap_start -= 1

            i = max(overlap_start, i + 1)

        return chunks

    def _split_with_offsets(
        self, text: str, base_offset: int
    ) -> List[tuple]:
        """Recursively split text into atomic spans that fit within chunk_size.

        Returns a list of (span_text, abs_start, abs_end) tuples.
        """
        if len(text) <= self.chunk_size:
            if text:
                return [(text, base_offset, base_offset + len(text))]
            return []

        for sep_pattern in self.SEPARATORS:
            if sep_pattern == "":
                # Character-level fallback: hard split.
                spans = []
                for start in range(0, len(text), self.chunk_size):
                    end = min(start + self.chunk_size, len(text))
                    spans.append(
                        (
                            text[start:end],
                            base_offset + start,
                            base_offset + end,
                        )
                    )
                return spans

            # Split using the separator pattern.
            parts = re.split(sep_pattern, text)
            if len(parts) <= 1:
                continue

            # Reconstruct offsets by scanning through the original text.
            spans = []
            cursor = 0
            for part in parts:
                if not part:
                    continue
                part_start = text.find(part, cursor)
                if part_start == -1:
                    part_start = cursor
                part_end = part_start + len(part)
                cursor = part_end

                # Recursively split parts that are still too large.
                if len(part) > self.chunk_size:
                    sub_spans = self._split_with_offsets(
                        part, base_offset + part_start
                    )
                    spans.extend(sub_spans)
                else:
                    spans.append(
                        (
                            part,
                            base_offset + part_start,
                            base_offset + part_end,
                        )
                    )

            if spans:
                return spans

        return [(text, base_offset, base_offset + len(text))]
