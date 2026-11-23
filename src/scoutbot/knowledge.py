"""Retrieval over the policy knowledge base: chunk documents, embed them, store vectors in
SQLite, and return the passages closest in meaning to a question. No LLM calls.

The store depends only on the ``Embedder`` function type; fastembed is imported solely
inside ``fastembed_embedder()``, so tests can use a fake embedder with no model download.
"""
import sqlite3
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple, Protocol

import numpy as np

from . import config

if TYPE_CHECKING:
    # typing.Self needs Python 3.11; the project supports 3.10.
    from typing_extensions import Self

#: Maps a batch of texts to one float32 vector per text, in order.
Embedder = Callable[[Sequence[str]], list[np.ndarray]]

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"  # 384 dimensions, ONNX, no PyTorch.
DEFAULT_CHUNK_SIZE = 60  # Words per chunk.
DEFAULT_CHUNK_OVERLAP = 15  # Words shared by consecutive chunks.


class Chunk(NamedTuple):
    """A window of a document, tagged with its source file and position for citing."""

    source: str
    ordinal: int
    text: str


class Hit(NamedTuple):
    """A search result: a stored chunk plus its cosine score against the query."""

    source: str
    ordinal: int
    text: str
    score: float


class KnowledgeStore(Protocol):
    """What the graph needs from a store: ranked passages for a question."""

    def search(self, query: str, k: int = 3) -> list[Hit]: ...


def cosine_scores(matrix: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Score each row of ``matrix`` against ``query`` by cosine similarity (direction only).

    Args:
        matrix: Array of shape ``(n, d)``; ``n`` may be 0.
        query: Array of shape ``(d,)``.

    Returns:
        Float32 array of shape ``(n,)`` in ``[-1, 1]``; zero vectors score 0.0, never nan.

    Raises:
        ValueError: If the vector dimensions differ.
    """
    matrix = np.asarray(matrix, dtype=np.float32)
    query = np.asarray(query, dtype=np.float32).ravel()
    if matrix.size == 0:
        return np.zeros(0, dtype=np.float32)
    matrix = np.atleast_2d(matrix)
    if matrix.shape[1] != query.shape[0]:
        raise ValueError(
            f"Dimension mismatch: rows have {matrix.shape[1]}, query has {query.shape[0]}."
        )

    denominator = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query)
    dots = matrix @ query
    # A zero vector has no direction: report 0.0 instead of dividing by zero.
    safe = np.where(denominator == 0, 1.0, denominator)
    return np.where(denominator == 0, 0.0, dots / safe).astype(np.float32)


def fastembed_embedder(model_name: str = DEFAULT_EMBEDDING_MODEL) -> Embedder:
    """Create an embedder backed by fastembed; the model (~130MB) downloads on first use.

    Args:
        model_name: fastembed model id.

    Returns:
        A function ``texts -> list of float32 vectors``.
    """
    from fastembed import TextEmbedding  # Imported here so the module loads without it.

    model = TextEmbedding(model_name=model_name)

    def embed(texts: Sequence[str]) -> list[np.ndarray]:
        return [np.asarray(vec, dtype=np.float32) for vec in model.embed(list(texts))]

    return embed


def chunk_document(
    text: str, size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP
) -> list[str]:
    """Split ``text`` into overlapping word windows, stepping by ``size - overlap``.

    Args:
        text: Document text; whitespace is normalised to single spaces.
        size: Words per window (> 0).
        overlap: Words shared by consecutive windows (0 <= overlap < size).

    Returns:
        Chunk texts in order; the short final window is kept. Empty text gives ``[]``.

    Raises:
        ValueError: If ``size`` or ``overlap`` is out of range.
    """
    if size <= 0:
        raise ValueError(f"size must be positive, got {size}.")
    if not 0 <= overlap < size:
        raise ValueError(f"overlap must be in [0, size), got overlap={overlap}, size={size}.")

    words = text.split()
    step = size - overlap
    chunks = []
    start = 0
    while start < len(words):
        chunks.append(" ".join(words[start : start + size]))
        if start + size >= len(words):
            break  # This window reached the end; another would only repeat the tail.
        start += step
    return chunks


def chunk_corpus(
    docs: dict[str, str], size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP
) -> list[Chunk]:
    """Chunk every document and tag each chunk with its source and ordinal.

    Args:
        docs: ``{source name: document text}``.
        size: Words per chunk.
        overlap: Words shared by consecutive chunks.

    Returns:
        Chunks grouped by document, ordinals starting at 0.
    """
    return [
        Chunk(source, ordinal, chunk_text)
        for source, text in docs.items()
        for ordinal, chunk_text in enumerate(chunk_document(text, size, overlap))
    ]


def load_corpus(directory: str | Path | None = None) -> dict[str, str]:
    """Read every ``*.md`` file in a folder.

    Args:
        directory: Folder to read; defaults to ``config.KNOWLEDGE_BASE_DIR``.

    Returns:
        ``{file name: text}``, sorted by file name.
    """
    folder = Path(directory or config.KNOWLEDGE_BASE_DIR)
    return {path.name: path.read_text(encoding="utf-8") for path in sorted(folder.glob("*.md"))}


_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    id      INTEGER PRIMARY KEY,
    source  TEXT    NOT NULL,
    ordinal INTEGER NOT NULL,
    text    TEXT    NOT NULL,
    vector  BLOB    NOT NULL,  -- float32 bytes
    UNIQUE (source, ordinal)
)
"""


class KnowledgeBase:
    """Vector store on plain SQLite with brute-force cosine search.

    The same embedder is used for chunks and queries. Use ``":memory:"`` for a throwaway
    store or a file path to persist it; a reopened file searches without re-indexing.
    """

    def __init__(self, embedder: Embedder, path: str | Path = ":memory:") -> None:
        """Open or create the store.

        Args:
            embedder: Function that embeds both chunks and queries.
            path: SQLite file, or ``":memory:"``.
        """
        self._embed = embedder
        self.path = str(path)
        self._conn = sqlite3.connect(self.path)
        self._conn.execute(_SCHEMA)
        self._conn.commit()

    def __enter__(self) -> "Self":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def __len__(self) -> int:
        """Return the number of stored chunks."""
        return self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    def close(self) -> None:
        """Close the SQLite connection."""
        self._conn.close()

    def add(self, chunks: Sequence[Chunk]) -> int:
        """Embed chunks in one batch and store them; same ``(source, ordinal)`` is replaced.

        Args:
            chunks: Chunks to index.

        Returns:
            Number of chunks written.

        Raises:
            ValueError: If the embedder returns a different number of vectors than texts.
        """
        if not chunks:
            return 0
        vectors = self._embed([chunk.text for chunk in chunks])
        if len(vectors) != len(chunks):
            raise ValueError(f"Embedder returned {len(vectors)} vectors for {len(chunks)} texts.")
        rows = [
            (c.source, c.ordinal, c.text, np.asarray(v, dtype=np.float32).tobytes())
            for c, v in zip(chunks, vectors)
        ]
        with self._conn:  # One transaction: all chunks or none.
            self._conn.executemany(
                "INSERT OR REPLACE INTO chunks (source, ordinal, text, vector) VALUES (?, ?, ?, ?)",
                rows,
            )
        return len(rows)

    def build(
        self,
        docs: dict[str, str],
        size: int = DEFAULT_CHUNK_SIZE,
        overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> int:
        """Replace the store's contents with a fresh index of ``docs``.

        Args:
            docs: ``{source name: document text}``.
            size: Words per chunk.
            overlap: Words shared by consecutive chunks.

        Returns:
            Number of chunks indexed.
        """
        chunks = chunk_corpus(docs, size, overlap)
        with self._conn:
            self._conn.execute("DELETE FROM chunks")
        return self.add(chunks)

    def search(self, query: str, k: int = 3) -> list[Hit]:
        """Find the stored chunks closest in meaning to ``query``.

        Args:
            query: Question text.
            k: Maximum number of results.

        Returns:
            Up to ``k`` hits ``(source, ordinal, text, score)``, best first; ``[]`` if empty.
        """
        if k <= 0:
            return []
        rows = self._conn.execute(
            "SELECT source, ordinal, text, vector FROM chunks ORDER BY id"
        ).fetchall()
        if not rows:
            return []

        matrix = np.stack([np.frombuffer(row[3], dtype=np.float32) for row in rows])
        scores = cosine_scores(matrix, self._embed([query])[0])
        top = np.argsort(-scores, kind="stable")[:k]  # Stable: ties keep insertion order.
        return [Hit(rows[i][0], rows[i][1], rows[i][2], float(scores[i])) for i in top]
