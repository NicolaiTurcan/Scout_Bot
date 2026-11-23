"""Offline tests for the DuckDB knowledge base, using the keyword embedder."""
import pytest
from fakes import keyword_embedder

from scoutbot.duckdb_store import DuckDBKnowledgeBase, IndexStats

DOCS = {
    "refunds.md": "Every refund is issued within five days. A refund needs an order number.",
    "shipping.md": "Standard shipping takes three days. Express shipping is faster.",
    "warranty.md": "The warranty covers defects for two years.",
}


def counting_embedder(calls):
    """Wrap the keyword embedder and record every batch it embeds.

    Args:
        calls: List that receives each batch of texts.

    Returns:
        An embedder function.
    """

    def embed(texts):
        calls.append(list(texts))
        return keyword_embedder(texts)

    return embed


def test_index_stores_every_chunk_as_fixed_size_array():
    kb = DuckDBKnowledgeBase(keyword_embedder)

    stats = kb.index(DOCS, size=5, overlap=1)

    assert stats == IndexStats(embedded=3, unchanged=0, removed=0)
    assert len(kb) == 7  # refunds: 3 windows, shipping: 3, warranty: 1.
    column_type = kb._conn.execute(
        "SELECT data_type FROM duckdb_columns() WHERE column_name = 'embedding'"
    ).fetchone()[0]
    assert column_type == "FLOAT[5]"


def test_search_returns_best_chunk_first_with_score():
    kb = DuckDBKnowledgeBase(keyword_embedder)
    kb.index(DOCS, size=50, overlap=0)

    hits = kb.search("how many days for a refund?", k=2)

    assert [hit.source for hit in hits] == ["refunds.md", "shipping.md"]
    source, ordinal, text, score = hits[0]
    assert (source, ordinal) == ("refunds.md", 0)
    assert "refund" in text
    assert score > hits[1].score


def test_search_k_caps_results():
    kb = DuckDBKnowledgeBase(keyword_embedder)
    kb.index(DOCS, size=50, overlap=0)

    assert len(kb.search("refund", k=1)) == 1
    assert len(kb.search("refund", k=10)) == len(DOCS)
    assert kb.search("refund", k=0) == []


def test_search_on_empty_store_returns_empty_list():
    assert DuckDBKnowledgeBase(keyword_embedder).search("anything") == []


def test_unchanged_documents_are_not_embedded_again(tmp_path):
    path = tmp_path / "kb.duckdb"
    with DuckDBKnowledgeBase(keyword_embedder, path) as kb:
        kb.index(DOCS, size=50, overlap=0)

    calls = []
    with DuckDBKnowledgeBase(counting_embedder(calls), path) as reopened:
        stats = reopened.index(DOCS, size=50, overlap=0)
        hits = reopened.search("shipping", k=1)

    assert stats == IndexStats(embedded=0, unchanged=3, removed=0)
    assert hits[0].source == "shipping.md"
    assert calls == [["shipping"]]  # Only the query was embedded.


def test_changed_and_removed_documents_are_synced():
    calls = []
    kb = DuckDBKnowledgeBase(counting_embedder(calls))
    kb.index(DOCS, size=50, overlap=0)
    calls.clear()

    docs = {**DOCS, "refunds.md": "Refunds take ten days."}
    del docs["warranty.md"]
    stats = kb.index(docs, size=50, overlap=0)

    assert stats == IndexStats(embedded=1, unchanged=1, removed=1)
    assert calls == [["Refunds take ten days."]]
    assert {hit.source for hit in kb.search("refund", k=10)} == {"refunds.md", "shipping.md"}


def test_new_chunking_settings_rebuild_everything():
    kb = DuckDBKnowledgeBase(keyword_embedder)
    kb.index(DOCS, size=50, overlap=0)

    stats = kb.index(DOCS, size=5, overlap=1)

    assert stats.embedded == 3
    assert len(kb) == 7


def test_store_refuses_queries_from_another_model(tmp_path):
    path = tmp_path / "kb.duckdb"
    with DuckDBKnowledgeBase(keyword_embedder, path, model_name="model-a") as kb:
        kb.index(DOCS)

    with DuckDBKnowledgeBase(keyword_embedder, path, model_name="model-b") as other:
        with pytest.raises(ValueError, match="model-a"):
            other.search("refund")
        assert other.index(DOCS).embedded == 3  # Re-indexing with the new model fixes it.
        assert other.search("refund")
