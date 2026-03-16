"""Core document and chunk dataclasses."""

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class Document:
    """Represents a raw document before chunking.

    Attributes:
        doc_id: Unique identifier for this document.
        text: Full text content of the document.
        title: Optional document title.
        metadata: Arbitrary metadata dictionary (e.g., source, date, URL).
    """

    doc_id: str
    text: str
    title: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        preview = self.text[:80].replace("\n", " ")
        return (
            f"Document(doc_id={self.doc_id!r}, title={self.title!r}, "
            f"text={preview!r}...)"
        )

    def word_count(self) -> int:
        """Return approximate word count."""
        return len(self.text.split())


@dataclass
class Chunk:
    """Represents a text chunk derived from a Document.

    Chunk IDs follow the convention: "{doc_id}__chunk_{i}" so that the
    parent document can always be recovered from the chunk ID alone.

    Attributes:
        chunk_id: Unique identifier for this chunk.
        doc_id: ID of the parent Document.
        text: Text content of this chunk.
        start_char: Character offset of chunk start within the parent document.
        end_char: Character offset of chunk end within the parent document.
        title: Inherited document title (useful for citation formatting).
        metadata: Arbitrary metadata, inherits from parent Document.
    """

    chunk_id: str
    doc_id: str
    text: str
    start_char: int
    end_char: int
    title: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        preview = self.text[:80].replace("\n", " ")
        return (
            f"Chunk(chunk_id={self.chunk_id!r}, doc_id={self.doc_id!r}, "
            f"chars=[{self.start_char}:{self.end_char}], "
            f"text={preview!r}...)"
        )

    def char_length(self) -> int:
        """Return character length of this chunk."""
        return self.end_char - self.start_char

    def word_count(self) -> int:
        """Return approximate word count of this chunk."""
        return len(self.text.split())
