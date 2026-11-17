"""Demonstrate surviving a mid-run crash across two separate Python processes.

Usage:
    uv run python resume_demo.py start T-1001
    uv run python resume_demo.py resume T-1001
"""
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from scoutbot import build_ticket_graph, initial_state, make_sqlite_saver

load_dotenv()

DB_PATH = str(Path(__file__).parent / "scoutbot_checkpoints.db")
TICKETS_PATH = Path(__file__).parent / "sample_tickets.json"


def _load_ticket(ticket_id: str) -> dict:
    tickets = json.loads(TICKETS_PATH.read_text())
    for ticket in tickets:
        if ticket["id"] == ticket_id:
            return ticket
    raise SystemExit(f"Unknown ticket_id: {ticket_id}")


def start_and_crash(ticket_id: str) -> None:
    ticket = _load_ticket(ticket_id)
    saver = make_sqlite_saver(DB_PATH)
    graph = build_ticket_graph(checkpointer=saver)
    config = {"configurable": {"thread_id": ticket_id}}
    state = initial_state(ticket["text"], ticket["customer_id"], ticket["id"])

    print(f"[process 1] Starting {ticket_id} with durability='sync'...")
    for chunk in graph.stream(
        state, config=config, stream_mode="updates", durability="sync"
    ):
        for node_name, update in chunk.items():
            print(f"[process 1] Completed node {node_name!r}: {update}")
            if node_name == "triage_ticket":
                print("[process 1] Simulating abrupt crash right after triage_ticket!")
                os._exit(1)


def resume_from_checkpoint(ticket_id: str) -> None:
    saver = make_sqlite_saver(DB_PATH)
    graph = build_ticket_graph(checkpointer=saver)
    config = {"configurable": {"thread_id": ticket_id}}

    snapshot = graph.get_state(config)
    print(f"[process 2] Loaded checkpoint for {ticket_id}. Pending next: {snapshot.next}")
    if not snapshot.next:
        print("[process 2] Nothing left to run (already completed or no checkpoint found).")
        return

    print("[process 2] Resuming with graph.stream(None, config, durability='sync')...")
    for chunk in graph.stream(
        None, config=config, stream_mode="updates", durability="sync"
    ):
        for node_name, update in chunk.items():
            print(f"[process 2] Executed node {node_name!r}: {update}")

    final_state = graph.get_state(config).values
    print(f"[process 2] Finished! Final outcome={final_state['outcome']!r}, notes={final_state['notes']}")


def main() -> None:
    if len(sys.argv) != 3 or sys.argv[1] not in {"start", "resume"}:
        raise SystemExit("Usage: python resume_demo.py <start|resume> <ticket_id>")
    command, ticket_id = sys.argv[1], sys.argv[2]
    if command == "start":
        start_and_crash(ticket_id)
    else:
        resume_from_checkpoint(ticket_id)


if __name__ == "__main__":
    main()