"""ScoutBot's LangGraph workflow with streaming custom events, interrupts, and checkpointing.

Extends W8D2's routed graph with:
- Injected `StreamWriter` (with a no-op default) so nodes emit progress events while running
  and remain directly callable outside a graph.
- `interrupt()` inside `escalate_ticket_node` so high-priority tickets genuinely pause and
  wait for a human decision via `Command(resume=...)`.
"""
import json
import operator
import sqlite3
from typing import Annotated, Optional, TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import StreamWriter, interrupt

from .core import triage
from .tools import look_up_customer


class TicketState(TypedDict):
    """Everything one ticket run accumulates, declared once, in one place."""

    ticket_id: str
    ticket_text: str
    customer_id: str
    customer: dict
    triage: dict
    outcome: str
    notes: Annotated[list[str], operator.add]


def _noop_writer(_: dict) -> None:
    """Default no-op writer so node functions can still be called directly in unit tests."""
    return None


def _ticket_with_customer(state: TicketState) -> str:
    """Fold the looked-up record into the text `triage()` sees."""
    return (
        f"{state['ticket_text']}\n\n"
        f"Customer record: {json.dumps(state['customer'], sort_keys=True)}"
    )


def look_up_customer_node(
    state: TicketState, writer: StreamWriter = _noop_writer
) -> dict:
    """First node: read the customer record and write it into state."""
    writer(
        {
            "node": "look_up_customer",
            "status": "looking up customer",
            "customer_id": state["customer_id"],
        }
    )
    record = look_up_customer(state["customer_id"])
    outcome = "found" if record.get("found") else "not found"
    return {
        "customer": record,
        "notes": [f"looked up {state['customer_id']}: {outcome}"],
    }


def triage_ticket_node(
    state: TicketState, writer: StreamWriter = _noop_writer
) -> dict:
    """Second node: triage the ticket using what the first node wrote."""
    writer(
        {
            "node": "triage_ticket",
            "status": "triaging ticket with model",
            "ticket_id": state["ticket_id"],
        }
    )
    result = triage(_ticket_with_customer(state))
    return {
        "triage": result.model_dump(mode="json"),
        "notes": [f"triaged as {result.category.value}/{result.priority.value}"],
    }


def route_after_triage(state: TicketState) -> str:
    """Pure router: send high-priority tickets to a human, handle everything else."""
    if state.get("triage", {}).get("priority") == "high":
        return "escalate_ticket"
    return "auto_resolve"


def escalate_ticket_node(
    state: TicketState, writer: StreamWriter = _noop_writer
) -> dict:
    """Terminal node for urgent tickets: pause via interrupt() until a human responds."""
    # Printed at the very top of the node so resume runs clearly show this node re-executes.
    print(f"[escalate_ticket] entering node for ticket {state['ticket_id']!r}")
    writer(
        {
            "node": "escalate_ticket",
            "status": "requesting human decision",
            "ticket_id": state["ticket_id"],
        }
    )

    human_decision = interrupt(
        {
            "ticket_id": state["ticket_id"],
            "customer_id": state["customer_id"],
            "ticket_text": state["ticket_text"],
            "customer": state["customer"],
            "triage": state["triage"],
            "question": "Please review this high-priority ticket and provide a resolution decision.",
        }
    )

    print(f"[escalate_ticket] interrupt() returned: {human_decision!r}")
    return {
        "outcome": f"escalated_resolved: {human_decision}",
        "notes": [f"escalated and resolved by human: {human_decision}"],
    }


def auto_resolve_node(
    state: TicketState, writer: StreamWriter = _noop_writer
) -> dict:
    """Terminal node for non-urgent tickets: record that ScoutBot handled it."""
    writer(
        {
            "node": "auto_resolve",
            "status": "auto-resolving ticket",
            "ticket_id": state["ticket_id"],
        }
    )
    return {
        "outcome": "handled",
        "notes": ["resolved automatically by ScoutBot"],
    }


def make_sqlite_saver(db_path: str = "scoutbot_checkpoints.db") -> SqliteSaver:
    """Create a file-backed SqliteSaver checkpointer safe for multi-step graph runs."""
    conn = sqlite3.connect(db_path, check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()
    return saver


def build_ticket_graph(checkpointer=None):
    """Wire nodes and conditional edges, then compile with an optional checkpointer."""
    builder = StateGraph(TicketState)
    builder.add_node("look_up_customer", look_up_customer_node)
    builder.add_node("triage_ticket", triage_ticket_node)
    builder.add_node("escalate_ticket", escalate_ticket_node)
    builder.add_node("auto_resolve", auto_resolve_node)

    builder.add_edge(START, "look_up_customer")
    builder.add_edge("look_up_customer", "triage_ticket")
    builder.add_conditional_edges(
        "triage_ticket",
        route_after_triage,
        {
            "escalate_ticket": "escalate_ticket",
            "auto_resolve": "auto_resolve",
        },
    )
    builder.add_edge("escalate_ticket", END)
    builder.add_edge("auto_resolve", END)
    return builder.compile(checkpointer=checkpointer)


def initial_state(
    ticket_text: str, customer_id: str, ticket_id: Optional[str] = None
) -> TicketState:
    """The starting state for one run."""
    return {
        "ticket_id": ticket_id or "",
        "ticket_text": ticket_text,
        "customer_id": customer_id,
        "customer": {},
        "triage": {},
        "outcome": "",
        "notes": [],
    }


def run_ticket_graph(
    ticket_text: str,
    customer_id: str,
    ticket_id: Optional[str] = None,
    checkpointer=None,
    config: Optional[dict] = None,
) -> TicketState:
    """Run one ticket through the graph and return the resulting state (or interrupt state)."""
    graph = build_ticket_graph(checkpointer=checkpointer)
    return graph.invoke(initial_state(ticket_text, customer_id, ticket_id), config=config)