"""Client tool: record a triage decision.

The tool's input schema is generated from :class:`scoutbot.schema.TicketTriage`, so the
model is asked for exactly the shape the rest of the code validates against. It is used both
by the one-shot classifier (as a forced tool call) and by the agent loop (as one option).
"""
from ..schema import TicketTriage

TRIAGE_TOOL = {
    "name": "triage_ticket",
    "description": "Record the triage decision for a single support ticket.",
    "input_schema": TicketTriage.model_json_schema(),
}


def record_triage(args: dict) -> dict:
    """Validate a ``triage_ticket`` tool call and return it as a plain dict.

    Args:
        args: The tool call's ``input`` as sent by the model.

    Returns:
        The validated triage as a dict.

    Raises:
        pydantic.ValidationError: If ``args`` does not match :class:`TicketTriage`.
    """
    return TicketTriage.model_validate(args).model_dump()
