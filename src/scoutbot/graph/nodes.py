"""Node functions and the conditional router for the ticket graph.

Each node receives the current :class:`~scoutbot.graph.state.TicketState` and returns only
the fields it changed. Nodes take an injected ``StreamWriter`` with a no-op default, so they
emit ``"custom"`` progress events while running inside a graph and remain directly callable
in unit tests.

High-priority tickets pause inside :func:`escalate_ticket_node` via ``interrupt()`` until a
human answers with ``Command(resume=...)``.

Every node is wrapped with :func:`scoutbot.observability.traced_node`, which records timing,
ids, the state update and interrupts on an OpenTelemetry span. Nodes therefore contain no
``print()`` or logging calls.
"""
import json
from collections.abc import Callable

from langgraph.types import StreamWriter, interrupt

from ..classifier import triage
from ..knowledge import KnowledgeStore
from ..observability import traced_node
from ..schema import Priority
from ..tools import look_up_customer
from .state import TicketState

# Node names, shared by the nodes (in progress events), the router and the graph builder.
LOOK_UP_CUSTOMER = "look_up_customer"
RETRIEVE_POLICIES = "retrieve_policies"
TRIAGE_TICKET = "triage_ticket"
ESCALATE_TICKET = "escalate_ticket"
AUTO_RESOLVE = "auto_resolve"

ESCALATION_QUESTION = (
    "Please review this high-priority ticket and provide a resolution decision."
)


def noop_writer(_: dict) -> None:
    """Default ``StreamWriter`` that discards events, for calling nodes outside a graph."""
    return None  # noqa: RET501


#: Passages scoring below this are dropped. Tuned on sample_tickets.json with bge-small:
#: unrelated tickets (CSV export) peak at ~0.66, related passages score 0.70+.
MIN_POLICY_SCORE = 0.7


def _ticket_with_customer(state: TicketState) -> str:
    """Fold the customer record and any retrieved policies into the text :func:`triage` sees.

    Args:
        state: State containing ``ticket_text``, ``customer`` and optionally ``policies``.

    Returns:
        The ticket text, the customer record as sorted JSON, then cited policy passages.
    """
    text = (
        f"{state['ticket_text']}\n\n"
        f"Customer record: {json.dumps(state['customer'], sort_keys=True)}"
    )
    if state.get("policies"):
        passages = "\n".join(
            f"[{p['source']}#{p['ordinal']}] {p['text']}" for p in state["policies"]
        )
        text += f"\n\nRelevant policy passages:\n{passages}"
    return text


@traced_node(LOOK_UP_CUSTOMER)
def look_up_customer_node(state: TicketState, writer: StreamWriter = noop_writer) -> dict:
    """First node: read the customer record and write it into state.

    Args:
        state: Current state; ``customer_id`` is read.
        writer: Stream writer for custom progress events.

    Returns:
        Update with ``customer`` and one ``notes`` entry.
    """
    writer(
        {
            "node": LOOK_UP_CUSTOMER,
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


def make_retrieve_policies_node(
    knowledge_base: KnowledgeStore, k: int = 3, min_score: float = MIN_POLICY_SCORE
) -> Callable[..., dict]:
    """Build a node that searches the knowledge base with the ticket text.

    Args:
        knowledge_base: Store to search.
        k: Maximum passages to retrieve.
        min_score: Passages scoring below this are dropped.

    Returns:
        A node function returning ``policies`` and one ``notes`` entry.
    """

    @traced_node(RETRIEVE_POLICIES)
    def retrieve_policies_node(state: TicketState, writer: StreamWriter = noop_writer) -> dict:
        writer(
            {
                "node": RETRIEVE_POLICIES,
                "status": "searching knowledge base",
                "ticket_id": state["ticket_id"],
            }
        )
        hits = [h for h in knowledge_base.search(state["ticket_text"], k=k) if h.score >= min_score]
        cited = ", ".join(f"{h.source}#{h.ordinal} ({h.score:.2f})" for h in hits) or "none"
        return {
            "policies": [hit._asdict() for hit in hits],
            "notes": [f"retrieved policies: {cited}"],
        }

    return retrieve_policies_node


@traced_node(TRIAGE_TICKET)
def triage_ticket_node(state: TicketState, writer: StreamWriter = noop_writer) -> dict:
    """Second node: triage the ticket using the customer record from the first node.

    Args:
        state: Current state; ``ticket_text`` and ``customer`` are read.
        writer: Stream writer for custom progress events.

    Returns:
        Update with ``triage`` (as a JSON-compatible dict) and one ``notes`` entry.

    Raises:
        scoutbot.classifier.TriageError: If the model fails to triage twice.
    """
    writer(
        {
            "node": TRIAGE_TICKET,
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
    """Pure router: send high-priority tickets to a human, handle everything else.

    Args:
        state: State after triage.

    Returns:
        :data:`ESCALATE_TICKET` for high priority, otherwise :data:`AUTO_RESOLVE`.
    """
    if state.get("triage", {}).get("priority") == Priority.high.value:
        return ESCALATE_TICKET
    return AUTO_RESOLVE


@traced_node(ESCALATE_TICKET)
def escalate_ticket_node(state: TicketState, writer: StreamWriter = noop_writer) -> dict:
    """Terminal node for urgent tickets: pause via ``interrupt()`` until a human responds.

    On resume, LangGraph re-runs this node from the top and ``interrupt()`` immediately
    returns the value passed to ``Command(resume=...)``.

    Args:
        state: State after triage.
        writer: Stream writer for custom progress events.

    Returns:
        Update with ``outcome`` and one ``notes`` entry describing the human decision.
    """
    writer(
        {
            "node": ESCALATE_TICKET,
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
            "policies": state.get("policies", []),
            "question": ESCALATION_QUESTION,
        }
    )
    return {
        "outcome": f"escalated_resolved: {human_decision}",
        "notes": [f"escalated and resolved by human: {human_decision}"],
    }


@traced_node(AUTO_RESOLVE)
def auto_resolve_node(state: TicketState, writer: StreamWriter = noop_writer) -> dict:
    """Terminal node for non-urgent tickets: record that ScoutBot handled it.

    Args:
        state: State after triage.
        writer: Stream writer for custom progress events.

    Returns:
        Update with ``outcome="handled"`` and one ``notes`` entry.
    """
    writer(
        {
            "node": AUTO_RESOLVE,
            "status": "auto-resolving ticket",
            "ticket_id": state["ticket_id"],
        }
    )
    return {
        "outcome": "handled",
        "notes": ["resolved automatically by ScoutBot"],
    }
