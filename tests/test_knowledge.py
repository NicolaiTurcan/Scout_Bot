"""Offline tests for retrieval, using a keyword embedder instead of a model.

The one real-model test runs only with ``SCOUTBOT_REAL_EMBEDDINGS=1``.
"""
import os

import numpy as np
import pytest
from fakes import keyword_embedder

from scoutbot.knowledge import (
    Chunk,
    KnowledgeBase,
    chunk_corpus,
    chunk_document,
    cosine_scores,
)

DOCS = {
    "refunds.md": "Every refund is issued within five days. A refund needs an order number.",
    "shipping.md": "Standard shipping takes three days. Express shipping is faster.",
    "warranty.md": "The warranty covers defects for two years.",
}


# --- chunk_document / chunk_corpus ------------------------------------------------------------


def test_chunk_document_overlapping_windows_and_short_tail():
    text = " ".join(f"w{i}" for i in range(10))

    chunks = chunk_document(text, size=4, overlap=1)

    # Step is 3: windows start at 0, 3, 6, and the last one runs off the end.
    assert chunks == ["w0 w1 w2 w3", "w3 w4 w5 w6", "w6 w7 w8 w9"]


def test_chunk_document_keeps_short_final_window():
    text = " ".join(f"w{i}" for i in range(7))

    chunks = chunk_document(text, size=4, overlap=1)

    assert chunks == ["w0 w1 w2 w3", "w3 w4 w5 w6"]
    chunks = chunk_document(text, size=5, overlap=2)
    assert chunks[-1] == "w3 w4 w5 w6"  # Shorter than size, kept rather than dropped.


def test_chunk_document_short_text_is_a_single_chunk():
    assert chunk_document("just three words", size=10, overlap=2) == ["just three words"]


def test_chunk_document_empty_text_has_no_chunks():
    assert chunk_document("   \n ", size=10, overlap=2) == []


@pytest.mark.parametrize("overlap", [5, 6])
def test_chunk_document_rejects_overlap_not_smaller_than_size(overlap):
    with pytest.raises(ValueError):
        chunk_document("some text here", size=5, overlap=overlap)


def test_chunk_document_rejects_non_positive_size():
    with pytest.raises(ValueError):
        chunk_document("some text", size=0, overlap=0)


def test_chunk_corpus_tags_source_and_ordinal():
    docs = {"a.md": "one two three four five", "b.md": "six seven"}

    chunks = chunk_corpus(docs, size=3, overlap=1)

    assert chunks == [
        Chunk("a.md", 0, "one two three"),
        Chunk("a.md", 1, "three four five"),
        Chunk("b.md", 0, "six seven"),
    ]


# --- cosine_scores ----------------------------------------------------------------------------


def test_cosine_scores_ranks_by_direction():
    matrix = np.array([[1, 0, 0], [0, 1, 0], [1, 1, 0], [-1, 0, 0]], dtype=np.float32)

    scores = cosine_scores(matrix, np.array([1, 0, 0]))

    np.testing.assert_allclose(scores, [1.0, 0.0, np.sqrt(0.5), -1.0], atol=1e-6)


def test_cosine_scores_ignores_magnitude():
    matrix = np.array([[1, 0, 0], [100, 0, 0]], dtype=np.float32)

    scores = cosine_scores(matrix, np.array([3, 0, 0]))

    np.testing.assert_allclose(scores, [1.0, 1.0], atol=1e-6)


def test_cosine_scores_zero_vectors_score_zero_not_nan():
    matrix = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)

    row_zero = cosine_scores(matrix, np.array([1, 0, 0]))
    query_zero = cosine_scores(matrix, np.array([0, 0, 0]))

    assert not np.isnan(row_zero).any() and not np.isnan(query_zero).any()
    np.testing.assert_allclose(row_zero, [0.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(query_zero, [0.0, 0.0], atol=1e-6)


def test_cosine_scores_empty_matrix():
    assert cosine_scores(np.zeros((0, 3)), np.array([1, 0, 0])).shape == (0,)


def test_cosine_scores_rejects_dimension_mismatch():
    with pytest.raises(ValueError):
        cosine_scores(np.ones((2, 3)), np.ones(4))


# --- KnowledgeBase ----------------------------------------------------------------------------


def test_build_indexes_every_chunk():
    kb = KnowledgeBase(keyword_embedder)

    count = kb.build(DOCS, size=5, overlap=1)

    assert count == len(chunk_corpus(DOCS, size=5, overlap=1))
    assert len(kb) == count


def test_build_replaces_previous_index():
    kb = KnowledgeBase(keyword_embedder)
    kb.build(DOCS, size=3, overlap=1)

    kb.build({"only.md": "warranty text"}, size=3, overlap=1)

    assert len(kb) == 1


def test_vectors_round_trip_as_float32_blobs():
    kb = KnowledgeBase(keyword_embedder)
    kb.build({"a.md": "refund refund days"}, size=10, overlap=0)

    blob = kb._conn.execute("SELECT vector FROM chunks").fetchone()[0]

    assert isinstance(blob, bytes)
    np.testing.assert_array_equal(
        np.frombuffer(blob, dtype=np.float32), keyword_embedder(["refund refund days"])[0]
    )


def test_search_returns_best_chunk_first_with_score():
    kb = KnowledgeBase(keyword_embedder)
    kb.build(DOCS, size=50, overlap=0)  # One chunk per document.

    hits = kb.search("is my warranty still valid?", k=2)

    assert len(hits) == 2
    source, ordinal, text, score = hits[0]  # Unpacks as a plain tuple.
    assert (source, ordinal) == ("warranty.md", 0)
    assert "warranty" in text
    assert score == pytest.approx(1.0)
    assert hits[0].score >= hits[1].score


def test_search_ranks_by_meaning_of_the_embedder():
    kb = KnowledgeBase(keyword_embedder)
    kb.build(DOCS, size=50, overlap=0)

    hits = kb.search("how many days for a refund?", k=3)

    assert hits[0].source == "refunds.md"
    assert [hit.score for hit in hits] == sorted((hit.score for hit in hits), reverse=True)


def test_search_k_caps_results():
    kb = KnowledgeBase(keyword_embedder)
    kb.build(DOCS, size=50, overlap=0)

    assert len(kb.search("refund", k=1)) == 1
    assert len(kb.search("refund", k=10)) == len(DOCS)
    assert kb.search("refund", k=0) == []


def test_search_uses_the_same_embedder_for_the_query():
    calls = []

    def recording_embedder(texts):
        calls.append(list(texts))
        return keyword_embedder(texts)

    kb = KnowledgeBase(recording_embedder)
    kb.build(DOCS, size=50, overlap=0)
    kb.search("refund", k=1)

    assert calls[-1] == ["refund"]


def test_search_on_empty_store_returns_empty_list():
    kb = KnowledgeBase(keyword_embedder)

    assert kb.search("anything", k=3) == []


def test_file_backed_store_reopens_without_reindexing(tmp_path):
    db_path = tmp_path / "knowledge.db"
    with KnowledgeBase(keyword_embedder, db_path) as kb:
        kb.build(DOCS, size=50, overlap=0)

    calls = []

    def query_only_embedder(texts):
        calls.append(list(texts))
        return keyword_embedder(texts)

    with KnowledgeBase(query_only_embedder, db_path) as reopened:
        assert len(reopened) == len(DOCS)
        hits = reopened.search("shipping", k=1)

    assert hits[0].source == "shipping.md"
    assert calls == [["shipping"]]  # Only the query was embedded; nothing was re-indexed.


# --- Real model (opt-in) ----------------------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("SCOUTBOT_REAL_EMBEDDINGS") != "1",
    reason="Downloads a ~130MB model; set SCOUTBOT_REAL_EMBEDDINGS=1 to run.",
)
def test_real_fastembed_matches_meaning_not_words():
    from scoutbot.knowledge import fastembed_embedder

    embed = fastembed_embedder()
    query, refund, warranty = embed(
        [
            "how long do refunds take?",
            "refunds are issued within five business days",
            "the warranty covers manufacturing defects",
        ]
    )

    assert query.shape == refund.shape == warranty.shape == (384,)
    assert query.dtype == np.float32
    related, unrelated = cosine_scores(np.stack([refund, warranty]), query)
    assert related > unrelated + 0.1
