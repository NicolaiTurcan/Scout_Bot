"""Graph wiring plus small helpers for running and inspecting ticket threads.

Topology::

    START -> look_up_customer -> triage_ticket -+-> escalate_ticket -> END   (priority high)
                                                +-> auto_resolve    -> END   (otherwise)
"""
from typing import Any, Optional

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from .nodes import (
    AUTO_RESOLVE,
    ESCALATE_TICKET,
    LOOK_UP_CUSTOMER,
    TRIAGE_TICKET,
    auto_resolve_node,
    escalate_ticket_node,
    look_up_customer_node,
    route_after_triage,
    triage_ticket_node,
)
from .state import TicketState, initial_state

#: Stream modes used by the demo scripts: node state updates plus custom progress events.
STREAM_MODES = ["updates", "custom"]


def build_ticket_graph(checkpointer: Optional[BaseCheckpointSaver] = None) -> CompiledStateGraph:
    """Wire nodes and conditional edges, then compile the graph.

    Args:
        checkpointer: Optional checkpointer. Required to resume after ``interrupt()`` or
            to inspect state with ``get_state()``.

    Returns:
        The compiled graph.
    """
    builder = StateGraph(TicketState)
    builder.add_node(LOOK_UP_CUSTOMER, look_up_customer_node)
    builder.add_node(TRIAGE_TICKET, triage_ticket_node)
    builder.add_node(ESCALATE_TICKET, escalate_ticket_node)
    builder.add_node(AUTO_RESOLVE, auto_resolve_node)

    builder.add_edge(START, LOOK_UP_CUSTOMER)
    builder.add_edge(LOOK_UP_CUSTOMER, TRIAGE_TICKET)
    builder.add_conditional_edges(
        TRIAGE_TICKET,
        route_after_triage,
        {ESCALATE_TICKET: ESCALATE_TICKET, AUTO_RESOLVE: AUTO_RESOLVE},
    )
    builder.add_edge(ESCALATE_TICKET, END)
    builder.add_edge(AUTO_RESOLVE, END)
    return builder.compile(checkpointer=checkpointer)


def run_ticket_graph(
    ticket_text: str,
    customer_id: str,
    ticket_id: Optional[str] = None,
    checkpointer: Optional[BaseCheckpointSaver] = None,
    config: Optional[dict] = None,
) -> TicketState:
    """Run one ticket through a freshly built graph.

    Args:
        ticket_text: The ticket body.
        customer_id: The customer's account id.
        ticket_id: Optional ticket id.
        checkpointer: Optional checkpointer passed to :func:`build_ticket_graph`.
        config: Optional LangGraph run config (e.g. from :func:`thread_config`).

    Returns:
        The resulting state. If the run paused, it contains an ``"__interrupt__"`` key.
    """
    graph = build_ticket_graph(checkpointer=checkpointer)
    return graph.invoke(initial_state(ticket_text, customer_id, ticket_id), config=config)


def thread_config(thread_id: str) -> dict:
    """Build a run config that pins a graph run to one checkpoint thread.

    Args:
        thread_id: Thread id; ScoutBot uses the ticket id.

    Returns:
        ``{"configurable": {"thread_id": thread_id}}``.
    """
    return {"configurable": {"thread_id": thread_id}}


def pending_interrupt(graph: CompiledStateGraph, config: dict) -> Optional[Any]:
    """Return the payload of the interrupt a thread is paused on, if any.

    Args:
        graph: A graph compiled with a checkpointer.
        config: Run config identifying the thread.

    Returns:
        The value passed to ``interrupt()``, or ``None`` if the thread is not paused.
    """
    snapshot = graph.get_state(config)
    for task in snapshot.tasks:
        if task.interrupts:
            return task.interrupts[0].value
    return None
