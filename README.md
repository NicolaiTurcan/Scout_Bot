# ScoutBot

ScoutBot is a support-ticket assistant for a fictional SaaS company, Anchorpoint. It reads an
incoming customer ticket, looks up the customer's account, classifies the ticket, and either
resolves it automatically or hands it to a human for a decision.

It is built on the [Anthropic Python SDK](https://github.com/anthropics/anthropic-sdk-python)
(every model call goes through the SDK directly) and
[LangGraph](https://github.com/langchain-ai/langgraph) (for the orchestrated workflow).

## What it does

ScoutBot offers two ways to process a ticket:

| | **Ticket workflow** (LangGraph) | **Agent loop** (ReAct) |
|---|---|---|
| Control flow | Fixed, explicit graph | The model decides the next step |
| Model calls | Exactly one (triage) | Up to `max_steps` turns (default 6) |
| Human in the loop | Pauses on `interrupt()` for high-priority tickets and resumes later, even from another process | `escalate_to_human` tool asks for y/n approval |
| Tools | Customer lookup + triage | Customer lookup, triage, escalation, `web_fetch`, `code_execution` |
| Entry point | `run_graph.py`, `resume_demo.py` | `main.py` |

### Ticket workflow

```
START → look_up_customer → triage_ticket ─┬─ priority = high → escalate_ticket → END
                                          └─ otherwise       → auto_resolve    → END
```

1. **look_up_customer** reads the account (plan, account age, recent refunds) from the mock
   customer database.
2. **triage_ticket** asks the model to classify the ticket together with the customer record.
   The result is validated against a Pydantic schema:
   `category` (billing / technical / account / other), `priority` (low / medium / high) and
   `suggested_owner`. An invalid reply is retried once and then raises `TriageError`, so
   ScoutBot never silently guesses.
3. **auto_resolve** marks low- and medium-priority tickets as handled.
4. **escalate_ticket** pauses the graph for high-priority tickets and waits for a human
   decision. With a checkpointer the paused run is saved, so the decision can arrive later
   from another process. On resume, the first two nodes are **not** re-run (no extra model
   calls); only `escalate_ticket` runs again, and `interrupt()` returns the human's answer.

While running, each node emits a progress event (`stream_mode="custom"`) before its state
update (`stream_mode="updates"`).

### Agent loop

A bounded ReAct loop: the model thinks, calls one tool, sees the result, and repeats until it
gives a final answer. If it hits the step cap, the ticket is escalated automatically.

- **Client tools** (run by ScoutBot): `look_up_customer`, `triage_ticket`,
  `escalate_to_human`. Escalation never takes effect without human approval.
- **Server tools** (run by Anthropic): `web_fetch`, restricted to `status.stripe.com`, for
  checking payment-provider outages, and `code_execution` for real computations over
  attached files, such as counting errors in a log.
- **Telemetry** (optional): every step can be appended to a JSONL log (ticket, step number,
  tool, duration, error flag) and analysed afterwards without re-running anything.

## Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- An Anthropic API key, needed only to run the demos. The tests run offline.

## Installation

```bash
git clone <repo-url>
cd scoutbot-baseline

uv sync                    # creates .venv/ and installs dependencies (incl. pytest)
cp .env.example .env       # Windows PowerShell: Copy-Item .env.example .env
```

Then edit `.env`:

| Variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | yes | Your Anthropic API key. |
| `ANTHROPIC_MODEL` | no | Model id. Defaults to `claude-haiku-4-5`. The agent loop needs a model that supports the `web_fetch` and `code_execution` server tools. |

Without uv: `python -m venv .venv`, activate it, then `pip install -e . pytest`, and run the
commands below without the `uv run` prefix.

## Usage

### 1. Stream the workflow over the sample tickets

```bash
uv run python run_graph.py
```

Every ticket in `sample_tickets.json` goes through the graph. For each ticket you will see:

- `[custom]` progress events and `[updates]` state changes, in execution order;
- `checkpoint next nodes` — `()` if the ticket finished, `('escalate_ticket',)` if it paused;
- for paused tickets, the interrupt payload: ticket, customer record, triage, and the
  question for the human.

### 2. Pause in one process, resume in another

```bash
# Process 1: run the ticket until it pauses for a human decision, then exit.
uv run python resume_demo.py start T-1001

# Process 2 (any time later): load the checkpoint and finish the run with a decision.
uv run python resume_demo.py resume T-1001 "Approved duplicate charge refund"
```

The checkpoint is stored in `scoutbot_checkpoints.db` in the project root. The second command
prints the final `outcome`, for example
`escalated_resolved: Approved duplicate charge refund`, and the accumulated `notes`. If the
model rated the ticket below high priority, it finishes in step one and there is nothing to
resume.

### 3. Run the ReAct agent loop

```bash
uv run python main.py
```

For each sample ticket this prints the full Thought → Action → Observation trace and the
final answer. When the model calls `escalate_to_human`, the script **waits for your input**:
type `y` to approve or `n` to reject. Ticket `T-2002` gets `sample_app.log` attached, so
the model can count errors with `code_execution`.

At the end, the script answers questions straight from the telemetry log
(`scoutbot_log.jsonl`, recreated on every run): total steps, errored steps, the slowest step,
and how often each tool was used.

### 4. Use it as a library

```python
from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from scoutbot import (
    build_ticket_graph,
    initial_state,
    pending_interrupt,
    run_agent_loop,
    thread_config,
    triage,
)

load_dotenv()

# One-shot classification.
result = triage("I was charged twice this month, please refund me.")
print(result.category.value, result.priority.value, result.suggested_owner)

# Workflow with a human decision.
graph = build_ticket_graph(checkpointer=InMemorySaver())
config = thread_config("T-42")
state = graph.invoke(initial_state("I was charged twice.", "1001", "T-42"), config=config)
if pending_interrupt(graph, config) is not None:
    state = graph.invoke(Command(resume="Refund approved"), config=config)
print(state["outcome"], state["notes"])

# Agent loop with non-interactive approval and telemetry.
agent_result = run_agent_loop(
    "Is Stripe having an outage right now?",
    customer_id="1001",
    ticket_id="T-43",
    approve_fn=lambda reason: True,
    log_path="scoutbot_log.jsonl",
)
print(agent_result.final_answer, agent_result.escalated)
```

## Sample data

| File | Contents |
|---|---|
| `mock_customers.json` | Three customers standing in for a real database: `1001` (pro, 2 recent refunds), `1002` (free, 12-day-old account), `1003` (pro, long-time customer). |
| `sample_tickets.json` | Five tickets: `T-1001` duplicate charge, `T-1002` stuck CSV export, `T-1003` email change, `T-2001` duplicate charge with a Stripe outage check, `T-2002` error count from today's app log. |
| `sample_app.log` | App log attached to `T-2002`. |

To try your own tickets, add entries with `id`, `customer_id` and `text` to
`sample_tickets.json`, or call the library functions directly.

## Tests

```bash
uv run pytest
```

All tests run offline, with no API key and no network. `tests/conftest.py` replaces the
Anthropic SDK boundary (`scoutbot.llm.get_client`) with a mock (the `messages_create`
fixture) and fails any test that would reach the real API. `tests/fakes.py` builds fake
response blocks.

## Project structure

```
├── main.py                # demo: ReAct agent loop + telemetry summary
├── run_graph.py           # demo: stream the workflow over all sample tickets
├── resume_demo.py         # demo: pause/resume across processes (SQLite)
├── src/scoutbot/
│   ├── config.py          # model name (read lazily from ANTHROPIC_MODEL), token budget, data paths
│   ├── llm.py             # get_client(): the single Anthropic SDK boundary
│   ├── data.py            # load_customers / load_tickets / find_ticket
│   ├── schema.py          # TicketTriage, Category, Priority
│   ├── classifier.py      # triage(): one forced tool call, retry once, then TriageError
│   ├── telemetry.py       # JSONL step logger: log_step / read_log
│   ├── tools/             # customer lookup, escalation, triage tool, server tools
│   ├── agent/             # ReAct loop: models, prompts, dispatch, server results, loop
│   └── graph/             # LangGraph workflow: state, nodes, builder, checkpoint
└── tests/                 # offline pytest suite
```
