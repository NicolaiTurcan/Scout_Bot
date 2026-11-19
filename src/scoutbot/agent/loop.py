"""Bounded ReAct agent loop on top of Anthropic tool calling.

Reason, act, observe, repeat, with a hard iteration cap and one human-in-the-loop gate.

Design choices:

- The model is reached through :func:`scoutbot.llm.get_client`; tests patch that single
  boundary to run offline.
- Only one *client* tool call is executed per turn. The model may request more; the first
  is executed and the next turn picks up from there, which keeps the trace easy to read.
- No retry on malformed output (unlike :func:`scoutbot.classifier.triage`): a bad tool call
  simply becomes an observation the model can react to, which is how ReAct recovers.
- Server tools (``web_fetch``, ``code_execution``) are run by Anthropic. The loop records
  them as steps but never executes them, and does not treat them as needing another turn.
- Telemetry is opt-in: pass ``log_path`` to append one JSONL record per step (see
  :mod:`scoutbot.telemetry`); the default ``None`` writes nothing.
"""
import json
import time
from typing import Any, Optional

from .. import config, llm
from ..telemetry import PathLike, log_step
from ..tools import CLIENT_TOOLS, SERVER_TOOLS
from ..tools.escalation import ApproveFn
from .dispatch import execute_client_tool
from .models import AgentResult, LoopStep
from .prompts import (
    SERVER_TOOL_THOUGHT,
    STEP_CAP_ANSWER,
    SYSTEM_PROMPT,
    ticket_message,
)
from .server_results import find_server_tool_result, summarize_server_tool_result

#: Default maximum number of model turns per ticket.
MAX_STEPS = 6

#: Every tool the agent may call, client and server.
ALL_TOOLS = [*CLIENT_TOOLS, *SERVER_TOOLS]


def _blocks_of_type(response: Any, block_type: str) -> list:
    """Return the response content blocks with the given ``type``."""
    return [block for block in response.content if block.type == block_type]


def _thought_text(response: Any) -> str:
    """Join all text blocks of a response into one stripped string."""
    return "".join(block.text for block in _blocks_of_type(response, "text")).strip()


def _server_tool_steps(response: Any) -> list[LoopStep]:
    """Build one step per server tool call already executed within ``response``.

    Args:
        response: A Messages API response.

    Returns:
        A step for each ``server_tool_use`` block, with a summarized observation.
    """
    steps = []
    for server_call in _blocks_of_type(response, "server_tool_use"):
        result_block = find_server_tool_result(response.content, server_call.id)
        steps.append(
            LoopStep(
                thought=SERVER_TOOL_THOUGHT,
                action=server_call.name,
                action_input=server_call.input,
                observation=summarize_server_tool_result(result_block),
            )
        )
    return steps


def _tool_result_message(tool_use_id: str, observation: dict) -> dict:
    """Build the user message that returns a client tool's observation to the model."""
    return {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": json.dumps(observation),
            }
        ],
    }


def _record_steps(
    steps: list[LoopStep],
    new_steps: list[LoopStep],
    ticket_id: Optional[str],
    duration_ms: float,
    log_path: Optional[PathLike],
) -> None:
    """Append ``new_steps`` to the trace and, if enabled, log each of them.

    All steps from one API response share the same turn-level ``duration_ms``: a client tool
    call and any server tool calls in the same response happened within one round trip, and
    the response exposes no finer-grained timing.

    Args:
        steps: The running trace; extended in place.
        new_steps: Steps produced by the current turn.
        ticket_id: Ticket id written to the log.
        duration_ms: Duration of the current turn.
        log_path: Log file, or ``None`` to skip logging.
    """
    first_index = len(steps) + 1
    steps.extend(new_steps)
    if not log_path:
        return
    for offset, step in enumerate(new_steps):
        log_step(ticket_id, first_index + offset, step, duration_ms, log_path)


def _elapsed_ms(start: float) -> float:
    """Milliseconds elapsed since a ``time.perf_counter()`` reading."""
    return (time.perf_counter() - start) * 1000


def run_agent_loop(
    ticket_text: str,
    customer_id: str,
    max_steps: int = MAX_STEPS,
    approve_fn: Optional[ApproveFn] = None,
    ticket_id: Optional[str] = None,
    log_path: Optional[PathLike] = None,
) -> AgentResult:
    """Run the ReAct loop on one ticket.

    The loop stops when the model answers without calling a client tool, or after
    ``max_steps`` turns, which is treated as an automatic escalation.

    Args:
        ticket_text: The ticket body (plus any inlined attachment).
        customer_id: The customer's account id.
        max_steps: Maximum number of model turns.
        approve_fn: Approval callback for ``escalate_to_human``. Defaults to an
            interactive y/n prompt.
        ticket_id: Ticket id recorded in the telemetry log.
        log_path: JSONL file to append one record per step to; ``None`` disables logging.

    Returns:
        The final answer, the full step trace, and whether the loop escalated.
    """
    messages: list[dict] = [{"role": "user", "content": ticket_message(ticket_text, customer_id)}]
    steps: list[LoopStep] = []

    for _ in range(max_steps):
        turn_start = time.perf_counter()
        response = llm.get_client().messages.create(
            model=config.get_model(),
            max_tokens=config.MAX_TOKENS,
            system=SYSTEM_PROMPT,
            messages=messages,
            tools=ALL_TOOLS,
        )
        thought = _thought_text(response)
        client_tool_uses = _blocks_of_type(response, "tool_use")

        # Server tools already ran: record each as its own step, never dispatch it.
        new_steps = _server_tool_steps(response)

        if not client_tool_uses:
            new_steps.append(LoopStep(thought=thought))
            _record_steps(steps, new_steps, ticket_id, _elapsed_ms(turn_start), log_path)
            return AgentResult(final_answer=thought, steps=steps)

        tool_use = client_tool_uses[0]
        observation = execute_client_tool(tool_use.name, tool_use.input, approve_fn)
        new_steps.append(
            LoopStep(
                thought=thought,
                action=tool_use.name,
                action_input=tool_use.input,
                observation=observation,
            )
        )
        _record_steps(steps, new_steps, ticket_id, _elapsed_ms(turn_start), log_path)
        messages.append({"role": "assistant", "content": response.content})
        messages.append(_tool_result_message(tool_use.id, observation))

    cap_step = LoopStep(thought=f"Hit the {max_steps}-step cap without a final answer.")
    _record_steps(steps, [cap_step], ticket_id, 0.0, log_path)
    return AgentResult(final_answer=STEP_CAP_ANSWER, steps=steps, escalated=True)
