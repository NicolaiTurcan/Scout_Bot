"""Explore what DuckDB offers for retrieval, on the real corpus and embedder.

1. Vector column types: the same vectors as BLOB, FLOAT[] (list) and FLOAT[384] (array).
2. An HNSW index (``vss`` extension) on the FLOAT[384] column.
3. Text search: BM25 (``fts`` extension), fuzzy matching, and a hybrid BM25 + vector ranking.

Extensions are downloaded once on first run.

Usage:
    uv run python explore_duckdb.py
"""
import time

import duckdb

from scoutbot.knowledge import chunk_corpus, fastembed_embedder, load_corpus

QUERIES = [
    "Can business customers get an invoice without VAT?",  # Exact keyword: VAT.
    "My headphones broke by themselves eight months in. Will you sort it out at no cost?",
    "What is the latest time to order for next-day express delivery?",
]


def first_line(exc: Exception) -> str:
    return str(exc).splitlines()[0]


def vector_types(conn: duckdb.DuckDBPyConnection, vectors: list, query: list[float]) -> None:
    """Try the cosine functions on each column type and print what works.

    Args:
        conn: Connection with a ``chunks`` table holding FLOAT[384] ``embedding``.
        vectors: The chunk vectors, in ``id`` order.
        query: Query vector.
    """
    print("=== 1. Vector column types")
    # FLOAT[384] cannot be cast to BLOB in SQL, so the bytes come from numpy.
    conn.execute("CREATE TABLE as_blob (id INTEGER, v BLOB)")
    conn.executemany(
        "INSERT INTO as_blob VALUES (?, ?)", [(i, v.tobytes()) for i, v in enumerate(vectors)]
    )
    conn.execute("CREATE TABLE as_list AS SELECT id, embedding::FLOAT[] AS v FROM chunks")
    conn.execute("CREATE TABLE as_array AS SELECT id, embedding AS v FROM chunks")
    as_text = str(query)
    # (column type, table, expression, query parameter)
    attempts = [
        ("BLOB", "as_blob", "array_cosine_similarity(v, ?::VARCHAR::FLOAT[384])", as_text),
        ("FLOAT[]", "as_list", "list_cosine_similarity(v, ?::VARCHAR::FLOAT[])", as_text),
        ("FLOAT[]", "as_list", "array_cosine_similarity(v, ?::VARCHAR::FLOAT[384])", as_text),
        ("FLOAT[384]", "as_array", "array_cosine_similarity(v, ?::VARCHAR::FLOAT[384])", as_text),
        # Same query, vector bound as a Python list instead of text: much slower.
        ("FLOAT[384]", "as_array", "array_cosine_similarity(v, ?::FLOAT[384])", query),
    ]
    for type_name, table, expression, param in attempts:
        sql = f"SELECT id, {expression} AS s FROM {table} ORDER BY s DESC LIMIT 1"
        try:
            conn.execute(sql, [param])  # Warm-up.
            start = time.perf_counter()
            row = conn.execute(sql, [param]).fetchone()
            ms = (time.perf_counter() - start) * 1000
            print(f"  {type_name:<10} {expression:<52} ok: top id {row[0]}, {ms:.1f} ms")
        except duckdb.Error as exc:
            print(f"  {type_name:<10} {expression:<52} ERROR: {first_line(exc)[:60]}")


def hnsw_index(conn: duckdb.DuckDBPyConnection, query: list[float]) -> None:
    """Build an HNSW index and show that the planner uses it for a top-k query.

    Args:
        conn: Connection with the ``as_array`` table.
        query: Query vector.
    """
    print("\n=== 2. HNSW index (vss extension)")
    conn.execute("INSTALL vss; LOAD vss")
    conn.execute("CREATE INDEX hnsw ON as_array USING HNSW (v) WITH (metric = 'cosine')")
    plan = conn.execute(
        "EXPLAIN SELECT id FROM as_array "
        "ORDER BY array_cosine_distance(v, ?::FLOAT[384]) LIMIT 3",
        [query],
    ).fetchall()[0][1]
    print(f"  plan uses HNSW_INDEX_SCAN: {'HNSW_INDEX_SCAN' in plan}")


def text_search(conn: duckdb.DuckDBPyConnection, embed) -> None:
    """Compare BM25, vector and hybrid (reciprocal rank fusion) top hits, plus fuzzy match.

    Args:
        conn: Connection with the ``chunks`` table.
        embed: Embedder for the queries.
    """
    print("\n=== 3. Text search (fts extension)")
    conn.execute("INSTALL fts; LOAD fts")
    conn.execute("PRAGMA create_fts_index('chunks', 'id', 'text')")
    for query in QUERIES:
        bm25 = [row[0] for row in conn.execute(
            "SELECT id FROM (SELECT id, fts_main_chunks.match_bm25(id, ?) AS s FROM chunks) "
            "WHERE s IS NOT NULL ORDER BY s DESC LIMIT 10",
            [query],
        ).fetchall()]
        vector = [row[0] for row in conn.execute(
            "SELECT id FROM chunks "
            "ORDER BY array_cosine_similarity(embedding, ?::FLOAT[384]) DESC LIMIT 10",
            [embed([query])[0].tolist()],
        ).fetchall()]
        # Reciprocal rank fusion: reward chunks ranked high by either method.
        fused = {}
        for ranking in (bm25, vector):
            for rank, chunk_id in enumerate(ranking):
                fused[chunk_id] = fused.get(chunk_id, 0) + 1 / (60 + rank)
        hybrid = sorted(fused, key=fused.get, reverse=True)

        print(f"  Q: {query}")
        for name, ranking in (("bm25", bm25), ("vector", vector), ("hybrid", hybrid)):
            print(f"    {name:<7} {label(conn, ranking[0]) if ranking else '(no match)'}")

    print("\n  Fuzzy match for the typo 'refnud' (jaro_winkler_similarity):")
    words = conn.execute(
        "SELECT DISTINCT w, jaro_winkler_similarity(w, 'refnud') AS s "
        "FROM (SELECT unnest(string_split(lower(text), ' ')) AS w FROM chunks) "
        "ORDER BY s DESC LIMIT 3"
    ).fetchall()
    print("   ", ", ".join(f"{w} ({s:.2f})" for w, s in words))


def label(conn: duckdb.DuckDBPyConnection, chunk_id: int) -> str:
    source, ordinal, text = conn.execute(
        "SELECT source, ordinal, text FROM chunks WHERE id = ?", [chunk_id]
    ).fetchone()
    return f"{source}#{ordinal}: {text[:70]}..."


def main() -> None:
    """Index the corpus into an in-memory DuckDB and run the three explorations."""
    embed = fastembed_embedder()
    chunks = chunk_corpus(load_corpus())
    vectors = embed([chunk.text for chunk in chunks])

    conn = duckdb.connect()
    conn.execute(
        "CREATE TABLE chunks (id INTEGER, source VARCHAR, ordinal INTEGER, text VARCHAR, "
        "embedding FLOAT[384])"
    )
    rows = list(zip(chunks, vectors))
    conn.executemany(
        "INSERT INTO chunks VALUES (?, ?, ?, ?, ?::VARCHAR::FLOAT[384])",
        [(i, c.source, c.ordinal, c.text, str(v.tolist())) for i, (c, v) in enumerate(rows)],
    )

    query = embed([QUERIES[1]])[0].tolist()
    vector_types(conn, vectors, query)
    hnsw_index(conn, query)
    text_search(conn, embed)


if __name__ == "__main__":
    main()
