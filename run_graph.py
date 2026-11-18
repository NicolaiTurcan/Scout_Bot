"""Entry point for Week 8 Day 3: stream custom events alongside node updates and observe
high-priority tickets pausing on interrupt().

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


def main():
    checkpointer = InMemorySaver()
    graph = build_ticket_graph(checkpointer=checkpointer)
    tickets = json.loads(TICKETS_PATH.read_text())

    for ticket in tickets:
        print(f"=== {ticket['id']}: {ticket['text'][:60]}... ===")
        state = initial_state(ticket["text"], ticket["customer_id"], ticket["id"])
        config = {"configurable": {"thread_id": ticket["id"]}}

        # Stream both "updates" and "custom" so custom progress events appear before each
        # node's final state update chunk.
        for mode, chunk in graph.stream(
            state, config=config, stream_mode=["updates", "custom"]
        ):
            print(f"  [{mode}] {json.dumps(chunk, default=str)}")

        snapshot = graph.get_state(config)
        print(f"  checkpoint next nodes: {snapshot.next}")
        if snapshot.tasks and snapshot.tasks[0].interrupts:
            print(f"  paused with interrupt payload: {snapshot.tasks[0].interrupts[0].value}")
        print()


if __name__ == "__main__":
    main()