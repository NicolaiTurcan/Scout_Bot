"""Demonstrate pausing on interrupt() in one process and resuming in a second process.

Usage:
    uv run python resume_demo.py start T-1001
    uv run python resume_demo.py resume T-1001 "Approved duplicate charge refund"
"""
import json
import sys
from pathlib import Path

from dotenv import load_dotenv
from langgraph.types import Command

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


def start_until_pause(ticket_id: str) -> None:
    ticket = _load_ticket(ticket_id)
    saver = make_sqlite_saver(DB_PATH)
    graph = build_ticket_graph(checkpointer=saver)
    config = {"configurable": {"thread_id": ticket_id}}
    state = initial_state(ticket["text"], ticket["customer_id"], ticket["id"])

    print(f"[process 1] Starting {ticket_id} with durability='sync'...")
    for mode, chunk in graph.stream(
        state, config=config, stream_mode=["updates", "custom"], durability="sync"
    ):
        print(f"[process 1] [{mode}] {chunk}")

    snapshot = graph.get_state(config)
    print(f"[process 1] Exiting. Pending next nodes: {snapshot.next}")
    if snapshot.tasks and snapshot.tasks[0].interrupts:
        print(f"[process 1] Interrupt payload: {snapshot.tasks[0].interrupts[0].value}")


def resume_from_interrupt(ticket_id: str, human_answer: str) -> None:
    saver = make_sqlite_saver(DB_PATH)
    graph = build_ticket_graph(checkpointer=saver)
    config = {"configurable": {"thread_id": ticket_id}}

    snapshot = graph.get_state(config)
    print(f"[process 2] Loaded checkpoint for {ticket_id}. Pending next: {snapshot.next}")
    if not snapshot.next:
        print("[process 2] Nothing left to resume (already completed or no checkpoint found).")
        return

    print(f"[process 2] Resuming with Command(resume={human_answer!r})...")
    for mode, chunk in graph.stream(
        Command(resume=human_answer),
        config=config,
        stream_mode=["updates", "custom"],
        durability="sync",
    ):
        print(f"[process 2] [{mode}] {chunk}")

    final_state = graph.get_state(config).values
    print(
        f"[process 2] Finished! Final outcome={final_state['outcome']!r}, "
        f"notes={final_state['notes']}"
    )


def main() -> None:
    if len(sys.argv) < 3 or sys.argv[1] not in {"start", "resume"}:
        raise SystemExit(
            "Usage:\n"
            "  python resume_demo.py start <ticket_id>\n"
            "  python resume_demo.py resume <ticket_id> <human_answer>"
        )

    command, ticket_id = sys.argv[1], sys.argv[2]
    if command == "start":
        start_until_pause(ticket_id)
    else:
        if len(sys.argv) < 4:
            raise SystemExit("Usage: python resume_demo.py resume <ticket_id> <human_answer>")
        human_answer = " ".join(sys.argv[3:])
        resume_from_interrupt(ticket_id, human_answer)


if __name__ == "__main__":
    main()