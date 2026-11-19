"""Execution of *client* tool calls requested by the model.

Only client tools reach this module. Server tools (``web_fetch``, ``code_execution``) have
already been run by Anthropic by the time the response arrives; see
:mod:`scoutbot.agent.server_results` for those.
"""
from typing import Callable, Optional

from ..tools import escalate_to_human, look_up_customer, record_triage
from ..tools.escalation import ApproveFn


def _dispatch_table(approve_fn: Optional[ApproveFn]) -> dict[str, Callable[[dict], dict]]:
    """Map each client tool name to a callable taking the tool's input dict."""
    return {
        "look_up_customer": lambda args: look_up_customer(args["customer_id"]),
        "escalate_to_human": lambda args: escalate_to_human(args["reason"], approve_fn=approve_fn),
        "triage_ticket": record_triage,
    }


def execute_client_tool(name: str, args: dict, approve_fn: Optional[ApproveFn] = None) -> dict:
    """Run one client tool call and return its observation.

    Args:
        name: Tool name from the model's ``tool_use`` block.
        args: The tool's ``input`` dict.
        approve_fn: Approval callback forwarded to ``escalate_to_human``.

    Returns:
        The tool's result dict, or ``{"error": "Unknown tool: <name>"}`` for an unknown tool
        so the model can see and react to the mistake.
    """
    handler = _dispatch_table(approve_fn).get(name)
    if handler is None:
        return {"error": f"Unknown tool: {name}"}
    return handler(args)
