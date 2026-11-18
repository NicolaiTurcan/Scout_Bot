# ScoutBot

Support-ticket triage and escalation assistant built with Anthropic's SDK and LangGraph.

## Overview

- `src/scoutbot/graph.py` — LangGraph `StateGraph` (`TicketState`) with:
  - `look_up_customer → triage_ticket → (escalate_ticket | auto_resolve)`
  - Injected `StreamWriter` (`writer: StreamWriter = _noop_writer`) emitting custom progress events before each node finishes
  - Human-in-the-loop pause via `interrupt()` inside `escalate_ticket_node` for high-priority tickets, resumed with `Command(resume=...)`
  - Optional checkpointer (`InMemorySaver` or file-backed `SqliteSaver`)
- `src/scoutbot/core.py` / `schema.py` / `tools.py` / `server_tools.py` / `telemetry.py` — ticket triage logic, tool definitions, and JSONL step logger.
- `src/scoutbot/loop.py` — Week 7 bounded ReAct agent loop (unchanged).
- `run_graph.py` — runs sample tickets through the graph, streaming both `"updates"` and `"custom"` events and showing paused interrupt payloads.
- `resume_demo.py` — demonstrates pausing a high-priority ticket in one process (`SqliteSaver`) and resuming it with a human answer in a separate Python process.
- `main.py` — runs the ReAct loop (`loop.py`) over the sample tickets and summarizes telemetry logs.
- `mock_customers.json` / `sample_tickets.json` / `sample_app.log` — sample data.

### Resume behavior note

When a paused ticket is resumed with `Command(resume=...)`:
- `look_up_customer` and `triage_ticket` **do not re-run** (their outputs are loaded from the checkpoint; `0` model calls occur on resume).
- `escalate_ticket` **re-runs from the top of the node**, and `interrupt()` immediately returns the value passed to `Command(resume=...)`.

## Setup

```bash
cp .env.example .env   # set your ANTHROPIC_API_KEY
uv sync                # create .venv/ and install dependencies

# Run offline unit tests
uv run pytest

# Stream graph execution ("updates" + "custom" events) and pause on high-priority tickets
uv run python run_graph.py

# Pause in process 1 and resume in process 2 (SQLite checkpoint)
uv run python resume_demo.py start T-1001
uv run python resume_demo.py resume T-1001 "Approved duplicate charge refund"

# Run the Week 7 ReAct loop over all sample tickets
uv run python main.py