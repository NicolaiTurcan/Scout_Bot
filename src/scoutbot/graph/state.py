"""The LangGraph state schema for one ticket run."""
import operator
from typing import Annotated, Optional, TypedDict


class _OptionalTicketFields(TypedDict, total=False):
    """Fields written only when the graph has a knowledge base."""

    policies: list[dict]  # Retrieved passages: source, ordinal, text, score.


class TicketState(_OptionalTicketFields):
    """Everything one ticket run accumulates, declared once, in one place.

    Plain fields are *replaced* by each node update. ``notes`` uses ``operator.add`` as its
    reducer, so every node's notes are *appended* to the list.
    """

    ticket_id: str
    ticket_text: str
    customer_id: str
    customer: dict
    triage: dict
    outcome: str
    notes: Annotated[list[str], operator.add]


def initial_state(
    ticket_text: str, customer_id: str, ticket_id: Optional[str] = None
) -> TicketState:
    """Build the starting state for one run.

    Args:
        ticket_text: The ticket body.
        customer_id: The customer's account id.
        ticket_id: Optional ticket id; stored as ``""`` when omitted.

    Returns:
        A fully populated :class:`TicketState` with empty outputs.
    """
    return {
        "ticket_id": ticket_id or "",
        "ticket_text": ticket_text,
        "customer_id": customer_id,
        "customer": {},
        "triage": {},
        "outcome": "",
        "notes": [],
    }
