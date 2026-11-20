"""Pause on ``interrupt()`` in one process and resume it from a second process.

The checkpoint is stored in SQLite with ``durability="sync"``, so the second process can
pick up exactly where the first one stopped.

Usage:
    uv run python resume_demo.py start T-1001
    uv run python resume_demo.py resume T-1001 "Approved duplicate charge refund"
    uv run python resume_demo.py history T-1001
"""
import sys

from dotenv import load_dotenv
from langgraph.types import Command

from scoutbot import (
    STREAM_MODES,
    build_ticket_graph,
    checkpoint_history,
    find_ticket,
    initial_state,
    make_sqlite_saver,
    pending_interrupt,
    thread_config,
)
from scoutbot.config import DEFAULT_CHECKPOINT_PATH, get_trace_ui_url
from scoutbot.observability import flush, setup_observability, traced_ticket_run

USAGE = (
    "Usage:\n"
    "  python resume_demo.py start <ticket_id>\n"
    "  python resume_demo.py resume <ticket_id> <human_answer>\n"
    "  python resume_demo.py history <ticket_id>"
)


def _sqlite_graph():
    """Build the ticket graph backed by the shared SQLite checkpoint file."""
    return build_ticket_graph(checkpointer=make_sqlite_saver(DEFAULT_CHECKPOINT_PATH))


def start_until_pause(ticket_id: str) -> None:
    """Process 1: run a ticket until it finishes or pauses on ``interrupt()``.

    Args:
        ticket_id: Id of a ticket in ``sample_tickets.json``.

    Raises:
        SystemExit: If the ticket id is unknown.
    """
    try:
        ticket = find_ticket(ticket_id)
    except KeyError as exc:
        raise SystemExit(exc.args[0]) from exc

    graph = _sqlite_graph()
    config = thread_config(ticket_id)
    state = initial_state(ticket["text"], ticket["customer_id"], ticket["id"])

    print(f"[process 1] Starting {ticket_id} with durability='sync'...")
    with traced_ticket_run(config, run_mode="start") as run:
        for mode, chunk in graph.stream(
            state, config=run.config, stream_mode=STREAM_MODES, durability="sync"
        ):
            print(f"[process 1] [{mode}] {chunk}")
        run.record_result(graph.get_state(config))

    print(f"[process 1] Exiting. Pending next nodes: {graph.get_state(config).next}")
    payload = pending_interrupt(graph, config)
    if payload is not None:
        print(f"[process 1] Interrupt payload: {payload}")


def resume_from_interrupt(ticket_id: str, human_answer: str) -> None:
    """Process 2: load the checkpoint and resume the paused run with a human answer.

    Args:
        ticket_id: Thread id used in :func:`start_until_pause`.
        human_answer: Value returned by ``interrupt()`` inside ``escalate_ticket``.
    """
    graph = _sqlite_graph()
    config = thread_config(ticket_id)

    snapshot = graph.get_state(config)
    print(f"[process 2] Loaded checkpoint for {ticket_id}. Pending next: {snapshot.next}")
    if not snapshot.next:
        print("[process 2] Nothing left to resume (already completed or no checkpoint found).")
        return

    print(f"[process 2] Resuming with Command(resume={human_answer!r})...")
    with traced_ticket_run(config, run_mode="resume") as run:
        for mode, chunk in graph.stream(
            Command(resume=human_answer),
            config=run.config,
            stream_mode=STREAM_MODES,
            durability="sync",
        ):
            print(f"[process 2] [{mode}] {chunk}")
        run.record_result(graph.get_state(config))

    final_state = graph.get_state(config).values
    print(
        f"[process 2] Finished! Final outcome={final_state['outcome']!r}, "
        f"notes={final_state['notes']}"
    )


def show_history(ticket_id: str) -> None:
    """Print every checkpoint of a ticket's thread with the run and trace that wrote it.

    Args:
        ticket_id: Thread id used in :func:`start_until_pause`.
    """
    rows = checkpoint_history(_sqlite_graph(), thread_config(ticket_id))
    if not rows:
        print(f"No checkpoints found for {ticket_id}.")
        return

    print(f"Checkpoint history for {ticket_id} ({len(rows)} checkpoints, oldest first):")
    print(f"{'step':>4}  {'created_at':<32} {'source':<7} {'run_mode':<8} {'run_by':<12} next")
    trace_ids = []
    for row in rows:
        print(
            f"{row['step']!s:>4}  {row['created_at']:<32} {row['source']!s:<7} "
            f"{row['run_mode'] or '-':<8} {row['run_by'] or '-':<12} {row['next'] or row['outcome']}"
        )
        if row["trace_id"] and row["trace_id"] not in trace_ids:
            trace_ids.append(row["trace_id"])

    print("\nTraces that wrote these checkpoints:")
    for trace_id in trace_ids:
        print(f"  {get_trace_ui_url()}/trace/{trace_id}")


def main(argv: list[str]) -> None:
    """Parse the command line and run the requested phase.

    Args:
        argv: Arguments without the program name.

    Raises:
        SystemExit: On invalid usage.
    """
    if len(argv) < 2 or argv[0] not in {"start", "resume", "history"}:
        raise SystemExit(USAGE)

    command, ticket_id = argv[0], argv[1]
    if command == "start":
        start_until_pause(ticket_id)
        return
    if command == "history":
        show_history(ticket_id)
        return
    if len(argv) < 3:
        raise SystemExit("Usage: python resume_demo.py resume <ticket_id> <human_answer>")
    resume_from_interrupt(ticket_id, " ".join(argv[2:]))


if __name__ == "__main__":
    load_dotenv()
    setup_observability()
    try:
        main(sys.argv[1:])
    finally:
        # Short-lived process: export buffered spans before it exits.
        flush()
