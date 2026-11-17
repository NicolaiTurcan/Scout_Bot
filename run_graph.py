"""Entry point for Week 8 Day 2: run the sample tickets through the routed graph with
checkpointing enabled, printing state updates after each node.

Usage:
    uv run python run_graph.py
"""
import json
from pathlib import Path

from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver

from scoutbot import build_ticket_graph, initial_state

load_dotenv()

TICKETS_PATH = Path(__file__).parent / "sample_tickets.json"
WATCHED_FIELDS = ("customer", "triage", "outcome", "notes")


def _print_fields(indent: str, data: dict, fields=WATCHED_FIELDS) -> None:
    for field in fields:
        if field in data:
            print(f"{indent}{field}: {json.dumps(data[field])}")


def main():
    checkpointer = InMemorySaver()
    graph = build_ticket_graph(checkpointer=checkpointer)
    tickets = json.loads(TICKETS_PATH.read_text())

    for ticket in tickets:
        print(f"=== {ticket['id']}: {ticket['text'][:60]}... ===")
        state = initial_state(ticket["text"], ticket["customer_id"], ticket["id"])
        config = {"configurable": {"thread_id": ticket["id"]}}

        first_snapshot = True
        for mode, chunk in graph.stream(
            state, config=config, stream_mode=["updates", "values"]
        ):
            if mode == "updates":
                for node_name, update in chunk.items():
                    print(f"  node {node_name!r} returned only:")
                    _print_fields("      ", update)
            else:
                print(f"  {'starting state:' if first_snapshot else 'state is now:'}")
                _print_fields("      ", chunk)
                first_snapshot = False

        snapshot = graph.get_state(config)
        print(f"  checkpoint next nodes: {snapshot.next}")
        print()


if __name__ == "__main__":
    main()