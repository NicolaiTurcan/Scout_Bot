"""Client tool: human-gated escalation.

A plain confirm prompt stands in for a real approval workflow (human-in-the-loop gate).
The escalation never takes effect without an explicit approval.
"""
from typing import Callable, Optional

#: Signature of an approval callback: receives the reason, returns ``True`` to approve.
ApproveFn = Callable[[str], bool]

ESCALATE_TOOL = {
    "name": "escalate_to_human",
    "description": (
        "Hand this ticket off to a human agent. Use this when the ticket needs a real person "
        "- e.g. a refund decision, or anything you're not confident resolving yourself. "
        "Requires human approval before it takes effect."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "reason": {"type": "string", "description": "Why this ticket needs a human."},
        },
        "required": ["reason"],
    },
}


def prompt_for_approval(reason: str) -> bool:
    """Ask the operator on stdin whether to approve an escalation.

    Args:
        reason: The model's stated reason for escalating.

    Returns:
        ``True`` if the answer starts with ``y`` (case-insensitive), else ``False``.
    """
    answer = input(f"[HUMAN APPROVAL NEEDED] Escalate ticket? Reason: {reason!r} (y/n): ")
    return answer.strip().lower().startswith("y")


def escalate_to_human(reason: str, approve_fn: Optional[ApproveFn] = None) -> dict:
    """Escalate a ticket to the human support queue, but only if approved.

    Args:
        reason: Why the ticket needs a human; passed verbatim to ``approve_fn``.
        approve_fn: Approval callback. Defaults to :func:`prompt_for_approval` (an
            interactive y/n prompt); tests and non-interactive callers pass their own.

    Returns:
        A dict with ``approved`` (bool) and a human-readable ``note``.
    """
    approve = approve_fn or prompt_for_approval
    if not approve(reason):
        return {"approved": False, "note": "Escalation rejected by human reviewer."}
    return {"approved": True, "note": "Escalated to the human support queue."}
