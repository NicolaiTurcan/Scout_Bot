"""Structured JSONL logging for the agent loop.

No new dependency and no tracing library: just ``json`` and ``open()``. Each loop step is
appended as one JSON line to a plain file (JSON Lines), with the step's ticket id, index,
tool name, timing and an error flag.

This is deliberately just enough to make the loop's existing trace queryable after the
fact. See ``main.py``'s ``summarize_log()`` for an example of answering questions from the
saved file without re-running anything.
"""
import json
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

from . import config

if TYPE_CHECKING:  # Imported for type hints only, to avoid an import cycle with the loop.
    from .agent.models import LoopStep

PathLike = Union[str, Path]


def _resolve(log_path: Optional[PathLike]) -> Path:
    """Return ``log_path`` as a Path, falling back to the configured default."""
    return Path(log_path) if log_path else config.DEFAULT_LOG_PATH


def step_errored(step: "LoopStep") -> bool:
    """Decide whether a loop step should be flagged as errored.

    Args:
        step: The step to inspect.

    Returns:
        ``True`` if the observation is a client tool's ``{"error": ...}`` dict, or a server
        tool result with a non-zero ``return_code`` (see
        :func:`scoutbot.agent.server_results.summarize_server_tool_result`).
    """
    obs = step.observation
    if not isinstance(obs, dict):
        return False
    if "error" in obs:
        return True
    return obs.get("return_code") not in (None, 0)


def log_step(
    ticket_id: Optional[str],
    step_number: int,
    step: "LoopStep",
    duration_ms: float,
    log_path: Optional[PathLike] = None,
) -> None:
    """Append one JSON line describing a single loop step.

    The record has the same shape whether the step came from a client tool or a server
    tool: the logger does not branch on tool type.

    Args:
        ticket_id: Id of the ticket being processed (may be ``None``).
        step_number: 1-based index of the step within the ticket's run.
        step: The step to record.
        duration_ms: Wall-clock time of the API turn that produced the step.
        log_path: Target file. Defaults to :data:`scoutbot.config.DEFAULT_LOG_PATH`.
    """
    record = {
        "ticket_id": ticket_id,
        "step": step_number,
        "tool": step.action,
        "duration_ms": round(duration_ms, 1),
        "errored": step_errored(step),
    }
    with open(_resolve(log_path), "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def read_log(log_path: Optional[PathLike] = None) -> list:
    """Read every saved record back from the log file, oldest first.

    Args:
        log_path: Source file. Defaults to :data:`scoutbot.config.DEFAULT_LOG_PATH`.

    Returns:
        The list of record dicts, or ``[]`` if the file does not exist yet.
    """
    path = _resolve(log_path)
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
