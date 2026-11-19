"""Tests for client and server tool definitions (``scoutbot.tools``)."""
import pytest
from pydantic import ValidationError

from scoutbot.tools import (
    CLIENT_TOOLS,
    SERVER_TOOLS,
    STATUS_PAGE_DOMAIN,
    TRIAGE_TOOL,
    WEB_FETCH_TOOL,
    escalate_to_human,
    look_up_customer,
    record_triage,
)


def test_look_up_customer_known_id():
    result = look_up_customer("1001")
    assert result["found"] is True
    assert result["plan"] == "pro"


def test_look_up_customer_unknown_id_is_graceful():
    assert look_up_customer("does-not-exist") == {"found": False, "customer_id": "does-not-exist"}


def test_escalate_to_human_approved():
    result = escalate_to_human("needs a refund decision", approve_fn=lambda reason: True)
    assert result["approved"] is True


def test_escalate_to_human_rejected():
    result = escalate_to_human("needs a refund decision", approve_fn=lambda reason: False)
    assert result["approved"] is False


def test_escalate_to_human_passes_reason_to_approve_fn():
    seen = []
    escalate_to_human("a specific reason", approve_fn=lambda reason: seen.append(reason) or True)
    assert seen == ["a specific reason"]


@pytest.mark.parametrize("answer, approved", [("y", True), ("Yes ", True), ("n", False), ("", False)])
def test_escalate_to_human_default_prompt_reads_stdin(monkeypatch, answer, approved):
    monkeypatch.setattr("builtins.input", lambda _prompt: answer)

    assert escalate_to_human("reason")["approved"] is approved


def test_record_triage_validates_and_returns_plain_values():
    result = record_triage({"category": "technical", "priority": "medium", "suggested_owner": "eng"})

    assert result == {"category": "technical", "priority": "medium", "suggested_owner": "eng"}


def test_record_triage_rejects_bad_input():
    with pytest.raises(ValidationError):
        record_triage({"category": "technical", "priority": "urgent!!", "suggested_owner": "eng"})


def test_triage_tool_schema_matches_the_pydantic_model():
    schema = TRIAGE_TOOL["input_schema"]

    assert set(schema["required"]) == {"category", "priority", "suggested_owner"}


def test_client_tool_names_are_unique():
    names = [tool["name"] for tool in CLIENT_TOOLS + SERVER_TOOLS]
    assert len(names) == len(set(names))


def test_web_fetch_is_restricted_to_the_status_page():
    assert WEB_FETCH_TOOL["allowed_domains"] == [STATUS_PAGE_DOMAIN]
