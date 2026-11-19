"""Builders for fake Anthropic Messages API responses.

Real responses are SDK objects; the code under test only reads attributes from them, so
``SimpleNamespace`` instances with the same attribute names are enough.
"""
from types import SimpleNamespace

HIGH_TRIAGE = {"category": "billing", "priority": "high", "suggested_owner": "billing-team"}
LOW_TRIAGE = {"category": "account", "priority": "low", "suggested_owner": "support-team"}


def response(*blocks):
    """A Message response whose ``content`` is the given blocks."""
    return SimpleNamespace(content=list(blocks))


def text_block(text):
    """A ``text`` content block."""
    return SimpleNamespace(type="text", text=text)


def tool_use_block(name, input_, tool_use_id="toolu_1"):
    """A client ``tool_use`` content block. ``.input`` is already a parsed dict."""
    return SimpleNamespace(type="tool_use", name=name, input=input_, id=tool_use_id)


def triage_response(input_dict):
    """A response carrying exactly one ``triage_ticket`` tool call."""
    return response(tool_use_block("triage_ticket", input_dict))


def server_tool_use_block(name, input_, tool_use_id="srvtoolu_1"):
    """A ``server_tool_use`` content block (a tool Anthropic already ran)."""
    return SimpleNamespace(type="server_tool_use", name=name, input=input_, id=tool_use_id)


def web_fetch_result_block(tool_use_id, url, content_text):
    """The result block paired with a ``web_fetch`` server tool call."""
    return SimpleNamespace(
        type="web_fetch_tool_result",
        tool_use_id=tool_use_id,
        content=SimpleNamespace(
            type="web_fetch_result",
            url=url,
            content=SimpleNamespace(source=SimpleNamespace(data=content_text)),
        ),
    )


def bash_result_block(tool_use_id, stdout, return_code=0, stderr=""):
    """The result block paired with a ``bash_code_execution`` server tool call."""
    return SimpleNamespace(
        type="bash_code_execution_tool_result",
        tool_use_id=tool_use_id,
        content=SimpleNamespace(
            type="bash_code_execution_result",
            stdout=stdout,
            stderr=stderr,
            return_code=return_code,
        ),
    )
