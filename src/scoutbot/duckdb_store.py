"""Knowledge base on DuckDB: vectors as ``FLOAT[dim]``, ranked in SQL by cosine similarity.

Indexing is incremental: a document is embedded again only when its sha256 changes, so
reopening the store does not re-embed the corpus. Changing the model or chunking settings
triggers a full rebuild.
"""
import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import duckdb
import numpy as np

from .knowledge import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_EMBEDDING_MODEL,
    Embedder,
    Hit,
    chunk_document,
)

if TYPE_CHECKING:
    from typing_extensions import Self


class IndexStats(NamedTuple):
    """Result of :meth:`DuckDBKnowledgeBase.index`, counted in documents."""

    embedded: int
    unchanged: int
    removed: int


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _as_text(vector: np.ndarray) -> str:
    # Binding a vector as text and casting in SQL is ~70x faster than binding a Python list.
    return str(np.asarray(vector, dtype=np.float32).tolist())


class DuckDBKnowledgeBase:
    """Vector store on DuckDB; same ``search`` interface as ``knowledge.KnowledgeBase``."""

    def __init__(
        self,
        embedder: Embedder,
        path: str | Path = ":memory:",
        model_name: str = DEFAULT_EMBEDDING_MODEL,
    ) -> None:
        """Open or create the store.

        Args:
            embedder: Function that embeds both chunks and queries.
            path: DuckDB file, or ``":memory:"``.
            model_name: Name of the model behind ``embedder``; stored next to the vectors.
        """
        self._embed = embedder
        self.model_name = model_name
        self._conn = duckdb.connect(str(path))
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS meta (key VARCHAR PRIMARY KEY, value VARCHAR)"
        )
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS documents (source VARCHAR PRIMARY KEY, sha256 VARCHAR)"
        )

    def __enter__(self) -> "Self":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def __len__(self) -> int:
        """Return the number of stored chunks."""
        if not self._has_chunks():
            return 0
        return self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    def close(self) -> None:
        """Close the DuckDB connection."""
        self._conn.close()

    def _meta(self) -> dict[str, str]:
        return dict(self._conn.execute("SELECT key, value FROM meta").fetchall())

    def _has_chunks(self) -> bool:
        return bool(
            self._conn.execute(
                "SELECT 1 FROM duckdb_tables() WHERE table_name = 'chunks'"
            ).fetchone()
        )

    def index(
        self,
        docs: dict[str, str],
        size: int = DEFAULT_CHUNK_SIZE,
        overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> IndexStats:
        """Sync the store with ``docs``, embedding only new or changed documents.

        Args:
            docs: ``{source name: document text}``; sources not in it are removed.
            size: Words per chunk.
            overlap: Words shared by consecutive chunks.

        Returns:
            How many documents were embedded, left unchanged, and removed.
        """
        settings = {"model": self.model_name, "size": str(size), "overlap": str(overlap)}
        stored = self._meta()
        # Vectors from another model or chunking are not comparable: rebuild from scratch.
        rebuild = any(stored.get(key) != value for key, value in settings.items())
        known = {} if rebuild else dict(
            self._conn.execute("SELECT source, sha256 FROM documents").fetchall()
        )
        hashes = {source: _sha256(text) for source, text in docs.items()}
        changed = [source for source in docs if known.get(source) != hashes[source]]
        removed = [source for source in known if source not in docs]

        rows = [
            (source, ordinal, text)
            for source in changed
            for ordinal, text in enumerate(chunk_document(docs[source], size, overlap))
        ]
        # Embed before the transaction: it is the slow step.
        vectors = self._embed([text for _, _, text in rows]) if rows else []
        dim = str(len(vectors[0])) if vectors else stored.get("dim")

        self._conn.begin()
        try:
            if rebuild:
                self._conn.execute("DROP TABLE IF EXISTS chunks")
                self._conn.execute("DELETE FROM documents")
                self._conn.execute("DELETE FROM meta")
            if vectors and not self._has_chunks():
                self._conn.execute(
                    "CREATE TABLE chunks (source VARCHAR, ordinal INTEGER, text VARCHAR, "
                    f"embedding FLOAT[{dim}])"
                )
            stale = changed + removed
            if stale:
                if self._has_chunks():
                    self._conn.execute("DELETE FROM chunks WHERE list_contains(?, source)", [stale])
                self._conn.execute("DELETE FROM documents WHERE list_contains(?, source)", [stale])
            # DuckDB's executemany rejects an empty list, hence the guards.
            if rows:
                self._conn.executemany(
                    f"INSERT INTO chunks VALUES (?, ?, ?, ?::VARCHAR::FLOAT[{dim}])",
                    [(*row, _as_text(vector)) for row, vector in zip(rows, vectors)],
                )
            if changed:
                self._conn.executemany(
                    "INSERT INTO documents VALUES (?, ?)", [(s, hashes[s]) for s in changed]
                )
            self._conn.executemany(
                "INSERT OR REPLACE INTO meta VALUES (?, ?)", [*settings.items(), ("dim", dim)]
            )
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        return IndexStats(len(changed), len(docs) - len(changed), len(removed))

    def search(self, query: str, k: int = 3) -> list[Hit]:
        """Find the stored chunks closest in meaning to ``query``, ranked inside DuckDB.

        Args:
            query: Question text.
            k: Maximum number of results.

        Returns:
            Up to ``k`` hits ``(source, ordinal, text, score)``, best first; ``[]`` if empty.

        Raises:
            ValueError: If the store was built with a different embedding model.
        """
        if k <= 0 or not len(self):
            return []
        meta = self._meta()
        if meta["model"] != self.model_name:
            raise ValueError(
                f"Store was built with {meta['model']!r}, not {self.model_name!r}; re-index it."
            )
        query_vector = _as_text(self._embed([query])[0])
        rows = self._conn.execute(
            f"""
            SELECT source, ordinal, text,
                   array_cosine_similarity(embedding, ?::VARCHAR::FLOAT[{meta['dim']}]) AS score
            FROM chunks
            ORDER BY score DESC, source, ordinal
            LIMIT ?
            """,
            [query_vector, k],
        ).fetchall()
        return [Hit(source, ordinal, text, float(score)) for source, ordinal, text, score in rows]
