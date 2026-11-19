"""Unit tests for the agent loop's building blocks, without running the loop itself."""
from types import SimpleNamespace

from fakes import bash_result_block, text_block, web_fetch_result_block
from scoutbot.agent.dispatch import execute_client_tool
from scoutbot.agent.server_results import (
    STDOUT_PREVIEW_CHARS,
    WEB_FETCH_PREVIEW_CHARS,
    find_server_tool_result,
    summarize_server_tool_result,
)


def _result_block(content):
    return SimpleNamespace(type="some_tool_result", tool_use_id="srv_1", content=content)


# --- dispatch ---------------------------------------------------------------------------


def test_dispatch_look_up_customer():
    assert execute_client_tool("look_up_customer", {"customer_id": "1002"})["plan"] == "free"


def test_dispatch_escalation_forwards_approve_fn():
    seen = []
    result = execute_client_tool(
        "escalate_to_human",
        {"reason": "refund"},
        approve_fn=lambda reason: seen.append(reason) or True,
    )

    assert result["approved"] is True
    assert seen == ["refund"]


def test_dispatch_triage_returns_validated_dict():
    result = execute_client_tool(
        "triage_ticket", {"category": "billing", "priority": "low", "suggested_owner": "x"}
    )

    assert result == {"category": "billing", "priority": "low", "suggested_owner": "x"}


def test_dispatch_unknown_tool_returns_error():
    assert execute_client_tool("nope", {}) == {"error": "Unknown tool: nope"}


# --- server results ---------------------------------------------------------------------


def test_find_server_tool_result_matches_by_id():
    wanted = bash_result_block("srv_2", stdout="ok")
    blocks = [text_block("hi"), bash_result_block("srv_1", stdout="other"), wanted]

    assert find_server_tool_result(blocks, "srv_2") is wanted
    assert find_server_tool_result(blocks, "missing") is None


def test_summarize_missing_result_block():
    assert summarize_server_tool_result(None) == {
        "note": "no matching result block (unexpected)"
    }


def test_summarize_web_fetch_truncates_preview():
    block = web_fetch_result_block("srv_1", "https://status.stripe.com", "x" * 1000)

    summary = summarize_server_tool_result(block)

    assert summary["url"] == "https://status.stripe.com"
    assert len(summary["content_preview"]) == WEB_FETCH_PREVIEW_CHARS


def test_summarize_bash_result_truncates_stdout():
    block = bash_result_block("srv_1", stdout="y" * 1000, return_code=2, stderr="boom")

    summary = summarize_server_tool_result(block)

    assert len(summary["stdout"]) == STDOUT_PREVIEW_CHARS
    assert summary["stderr"] == "boom"
    assert summary["return_code"] == 2


def test_summarize_text_editor_operation():
    block = _result_block(SimpleNamespace(type="text_editor_code_execution_view_result"))

    assert summarize_server_tool_result(block) == {"file_op": "view"}


def test_summarize_error_result():
    block = _result_block(
        SimpleNamespace(type="web_fetch_tool_result_error", error_code="url_not_allowed")
    )

    assert summarize_server_tool_result(block) == {"error": "url_not_allowed"}


def test_summarize_unknown_result_falls_back_to_raw():
    block = _result_block("something unexpected")

    assert summarize_server_tool_result(block) == {"raw": "something unexpected"}
