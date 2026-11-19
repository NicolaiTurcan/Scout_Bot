"""Tests for configuration (``scoutbot.config``) and sample data loaders (``scoutbot.data``)."""
import json

import pytest

from scoutbot import config
from scoutbot.data import find_ticket, load_customers, load_tickets


def test_project_root_holds_the_sample_data():
    assert config.CUSTOMERS_PATH.is_file()
    assert config.TICKETS_PATH.is_file()
    assert config.APP_LOG_PATH.is_file()


def test_get_model_defaults_when_env_is_unset(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)
    assert config.get_model() == config.DEFAULT_MODEL


def test_get_model_reads_env_lazily(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-from-env")
    assert config.get_model() == "claude-from-env"


def test_load_customers_is_keyed_by_id():
    customers = load_customers()
    assert customers["1001"]["plan"] == "pro"


def test_load_tickets_returns_sample_batch():
    tickets = load_tickets()
    assert [t["id"] for t in tickets][:3] == ["T-1001", "T-1002", "T-1003"]
    assert all({"id", "customer_id", "text"} <= set(t) for t in tickets)


def test_find_ticket_by_id():
    assert find_ticket("T-1003")["customer_id"] == "1003"


def test_find_ticket_unknown_id_raises_key_error():
    with pytest.raises(KeyError, match="T-404"):
        find_ticket("T-404")


def test_loaders_accept_an_explicit_path(tmp_path):
    path = tmp_path / "tickets.json"
    path.write_text(json.dumps([{"id": "X-1", "customer_id": "c", "text": "t"}]), encoding="utf-8")

    assert find_ticket("X-1", path=path)["text"] == "t"
