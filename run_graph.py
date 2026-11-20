"""Stream sample tickets through the ticket graph and show where high-priority ones pause.

Both ``"updates"`` and ``"custom"`` stream modes are printed, so each node's custom progress
event appears before that node's state update chunk. High-priority tickets stop at
``interrupt()`` and their payload is printed.

Usage:
    uv run python run_graph.py
"""
import json

from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver

from scoutbot import (
    STREAM_MODES,
    build_ticket_graph,
    initial_state,
    load_tickets,
    pending_interrupt,
    thread_config,
)
from scoutbot.observability import flush, setup_observability, traced_ticket_run


def main() -> None:
    """Run every sample ticket through the graph with an in-memory checkpointer."""
    load_dotenv()
    setup_observability()
    graph = build_ticket_graph(checkpointer=InMemorySaver())

    for ticket in load_tickets():
        print(f"=== {ticket['id']}: {ticket['text'][:60]}... ===")
        state = initial_state(ticket["text"], ticket["customer_id"], ticket["id"])
        config = thread_config(ticket["id"])

        # One root span per ticket: every graph, node and model-call span nests under it.
        with traced_ticket_run(config, run_mode="batch") as run:
            for mode, chunk in graph.stream(state, config=run.config, stream_mode=STREAM_MODES):
                print(f"  [{mode}] {json.dumps(chunk, default=str)}")
            run.record_result(graph.get_state(config))

        print(f"  checkpoint next nodes: {graph.get_state(config).next}")
        payload = pending_interrupt(graph, config)
        if payload is not None:
            print(f"  paused with interrupt payload: {payload}")
        print()

    flush()


if __name__ == "__main__":
    main()
