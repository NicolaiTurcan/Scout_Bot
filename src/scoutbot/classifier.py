"""One-shot ticket triage: one forced tool call, one schema, no regex.

The public function is :func:`triage`. This module is deliberately *not* named
``triage.py``: ``scoutbot/__init__.py`` re-exports the ``triage`` function, and a package
attribute can't be both the submodule and the function. The function would shadow the
module and break ``mock.patch("scoutbot.triage....")``-style patching.

The model is reached through :func:`scoutbot.llm.get_client`; tests replace that single
boundary to run offline (see ``tests/conftest.py``).
"""
from . import config, llm
from .schema import TicketTriage
from .tools.triage import TRIAGE_TOOL

TRIAGE_SYSTEM_PROMPT = "You triage support tickets by calling triage_ticket exactly once."


class TriageError(RuntimeError):
    """Raised when the model's reply can't be turned into a valid TicketTriage, even after a
    retry. Failing loudly and specifically beats a silent, wrong guess."""


def _request_triage(ticket_text: str) -> TicketTriage:
    """Make one triage request and validate the model's tool call.

    Args:
        ticket_text: The full text the model should classify.

    Returns:
        The validated triage.

    Raises:
        TriageError: If the model did not call ``triage_ticket``.
        pydantic.ValidationError: If the tool input does not match the schema.
    """
    response = llm.get_client().messages.create(
        model=config.get_model(),
        max_tokens=config.MAX_TOKENS,
        system=TRIAGE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": ticket_text}],
        tools=[TRIAGE_TOOL],
        tool_choice={"type": "tool", "name": TRIAGE_TOOL["name"]},
    )
    tool_use_blocks = [block for block in response.content if block.type == "tool_use"]
    if not tool_use_blocks:
        raise TriageError("Model did not call triage_ticket.")
    # Anthropic returns `.input` already parsed into a dict (unlike OpenAI's function-call
    # arguments, which are a JSON string), so no json.loads is needed here.
    return TicketTriage.model_validate(tool_use_blocks[0].input)


def triage(ticket_text: str) -> TicketTriage:
    """Classify one support ticket.

    Retry policy: retry once on any failure (malformed input, schema mismatch, no tool call),
    then escalate by raising instead of silently guessing. A production version would likely
    narrow the caught exceptions; they are kept broad here so the policy stays easy to read.

    Args:
        ticket_text: The ticket text, optionally enriched with context such as the customer
            record.

    Returns:
        The validated :class:`~scoutbot.schema.TicketTriage`.

    Raises:
        TriageError: If both attempts fail. The original exception is chained as
            ``__cause__``.
    """
    try:
        return _request_triage(ticket_text)
    except Exception:
        try:
            return _request_triage(ticket_text)
        except Exception as exc:
            raise TriageError(f"triage() failed twice for ticket: {ticket_text!r}") from exc
