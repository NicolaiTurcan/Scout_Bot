"""Run ScoutBot's ReAct agent loop over the sample ticket batch.

Prints the full Thought/Action/Observation trace for each ticket, then answers questions
straight from the saved telemetry log, without re-running anything.

Usage:
    uv run python main.py
"""
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

from scoutbot import AgentResult, load_tickets, read_log, run_agent_loop
from scoutbot.config import APP_LOG_PATH, DEFAULT_LOG_PATH

# T-2002 asks ScoutBot to compute something over the app log, so the model needs the log's
# actual content, not just a filename it cannot open. The text is inlined here rather than
# in the loop: the loop should not need to know which tickets come with an attachment.
TICKETS_WITH_LOG_ATTACHED = {"T-2002"}


def ticket_text(ticket: dict) -> str:
    """Return the text sent to the agent, with the app log inlined where needed.

    Args:
        ticket: A sample ticket with ``id`` and ``text``.

    Returns:
        The ticket text, followed by today's app log for tickets in
        :data:`TICKETS_WITH_LOG_ATTACHED`.
    """
    if ticket["id"] in TICKETS_WITH_LOG_ATTACHED:
        app_log = APP_LOG_PATH.read_text(encoding="utf-8")
        return f"{ticket['text']}\n\nHere is today's app log:\n{app_log}"
    return ticket["text"]


def print_trace(result: AgentResult) -> None:
    """Print one ticket's Thought/Action/Observation trace and outcome.

    Args:
        result: The agent loop result to print.
    """
    for i, step in enumerate(result.steps, start=1):
        print(f"  Step {i} - Thought: {step.thought}")
        if step.action:
            print(f"           Action: {step.action}({step.action_input})")
            print(f"           Observation: {step.observation}")
    print(f"  Final answer: {result.final_answer}")
    print(f"  Escalated (hit step cap): {result.escalated}")


def summarize_log(log_path: Path) -> None:
    """Answer a few questions straight from the saved JSONL log.

    No ticket is re-run: every number comes from records already on disk.

    Args:
        log_path: The telemetry log written during the batch run.
    """
    records = read_log(log_path)
    if not records:
        print("(no log records found)")
        return

    errored = [r for r in records if r["errored"]]
    slowest = max(records, key=lambda r: r["duration_ms"])
    tool_counts = Counter(r["tool"] or "(final answer, no tool)" for r in records)

    print(f"  Total steps logged: {len(records)}")
    print(f"  Steps that errored: {len(errored)}")
    print(
        f"  Slowest step: {slowest['ticket_id']} step {slowest['step']} "
        f"({slowest['tool']}, {slowest['duration_ms']} ms)"
    )
    print("  Tool usage:")
    for tool, count in tool_counts.most_common():
        print(f"    {tool}: {count}")


def main() -> None:
    """Run every sample ticket through the agent loop, then summarize the log."""
    load_dotenv()

    # Start from a fresh log so re-running the batch doesn't double-count old records.
    DEFAULT_LOG_PATH.unlink(missing_ok=True)

    # No approve_fn is passed, so escalate_to_human falls back to the interactive y/n
    # prompt and actually waits for the operator.
    for ticket in load_tickets():
        print(f"=== {ticket['id']}: {ticket['text'][:60]}... ===")
        result = run_agent_loop(
            ticket_text(ticket),
            ticket["customer_id"],
            ticket_id=ticket["id"],
            log_path=DEFAULT_LOG_PATH,
        )
        print_trace(result)
        print()

    print(f"=== Log saved to {DEFAULT_LOG_PATH.name} - answering from it, not re-running ===")
    summarize_log(DEFAULT_LOG_PATH)


if __name__ == "__main__":
    main()
