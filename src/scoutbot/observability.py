"""OpenTelemetry tracing for ScoutBot (via TraceVerde): setup, node decorator, run spans.

Spans go to the console if ``OTEL_EXPORTER_OTLP_ENDPOINT`` is empty, otherwise over OTLP/HTTP.
"""
import functools
import getpass
import os
import socket
import threading
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, TypeVar

from langgraph.errors import GraphInterrupt
from langgraph.types import StateSnapshot
from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor
from opentelemetry.trace import Span, Status, StatusCode, Tracer

from . import config

NodeFunc = TypeVar("NodeFunc", bound=Callable[..., dict])

#: Service name shown in trace backends (Jaeger's "Service" drop-down).
SERVICE_NAME = "scoutbot"
#: Instrumentation scope for ScoutBot's own spans.
TRACER_NAME = "scoutbot"

# Defaults applied only when the variable is not set in the environment or in ``.env``.
_ENV_DEFAULTS = {
    "OTEL_SERVICE_NAME": SERVICE_NAME,
    # Instrument only what ScoutBot actually uses, to keep traces readable.
    "GENAI_ENABLED_INSTRUMENTORS": "anthropic,langgraph",
    # ScoutBot runs no local model, so GPU metrics would only add noise.
    "GENAI_ENABLE_GPU_METRICS": "false",
}

_initialized = False


def setup_observability() -> None:
    """Configure OpenTelemetry tracing for this process once; later calls do nothing."""
    global _initialized
    if _initialized:
        return
    for name, value in _ENV_DEFAULTS.items():
        os.environ.setdefault(name, value)

    # Imported lazily: importing ScoutBot (e.g. in unit tests) must not pull in TraceVerde.
    import genai_otel

    genai_otel.instrument()
    register_usage_collector(trace.get_tracer_provider())
    _initialized = True


def get_tracer() -> Tracer:
    """Return the tracer for ScoutBot's own spans.

    Returns:
        The ``scoutbot`` tracer (a no-op tracer until tracing is set up).
    """
    return trace.get_tracer(TRACER_NAME)


def flush() -> None:
    """Export every span still buffered in memory before the process exits."""
    provider = trace.get_tracer_provider()
    force_flush = getattr(provider, "force_flush", None)
    if force_flush is not None:
        force_flush()


# --- Node decorator ----------------------------------------------------------------------

#: Event added to a node span when the node pauses for a human (``interrupt()``).
PAUSED_EVENT = "scoutbot.paused_for_human"
#: Status description of a node span that paused for a human.
PAUSED_STATUS = "paused for human decision (interrupt)"


def _interrupt_event_attributes(exc: GraphInterrupt) -> dict:
    """Summarize an interrupt without customer content (payload keys and question only).

    Args:
        exc: The interrupt raised by ``interrupt()``.

    Returns:
        Span event attributes.
    """
    interrupts = exc.args[0] if exc.args else ()
    payloads = [getattr(item, "value", item) for item in interrupts]
    attributes: dict = {"interrupt.count": len(payloads)}
    first = payloads[0] if payloads else None
    if isinstance(first, dict):
        attributes["interrupt.payload_keys"] = sorted(first)
        if "question" in first:
            attributes["interrupt.question"] = str(first["question"])
    return attributes


def _record_update(span: Span, update: Any) -> None:
    """Write a node's state update summary (keys, outcome, notes, triage) onto its span.

    Args:
        span: The node span.
        update: The dict returned by the node.
    """
    if not isinstance(update, dict):
        return
    span.set_attribute("scoutbot.updated_keys", sorted(update))
    if update.get("outcome"):
        span.set_attribute("scoutbot.outcome", str(update["outcome"]))
    if update.get("notes"):
        span.set_attribute("scoutbot.notes", [str(note) for note in update["notes"]])
    if isinstance(update.get("triage"), dict):
        for key, value in update["triage"].items():
            span.set_attribute(f"scoutbot.triage.{key}", str(value))


def traced_node(node_name: str) -> Callable[[NodeFunc], NodeFunc]:
    """Wrap a node in a ``scoutbot.node.<name>`` span that marks ``interrupt()`` as an error.

    Args:
        node_name: Name the node is registered under in the graph.

    Returns:
        A decorator for node functions.
    """

    def decorator(func: NodeFunc) -> NodeFunc:
        # functools.wraps keeps the original signature, so LangGraph still injects `writer`.
        @functools.wraps(func)
        def wrapper(state: dict, *args: Any, **kwargs: Any) -> dict:
            with get_tracer().start_as_current_span(
                f"scoutbot.node.{node_name}",
                # Exceptions are handled below, so an interrupt is not logged as a crash.
                record_exception=False,
                set_status_on_exception=False,
            ) as span:
                span.set_attribute("scoutbot.node", node_name)
                span.set_attribute("scoutbot.ticket_id", str(state.get("ticket_id", "")))
                span.set_attribute("scoutbot.customer_id", str(state.get("customer_id", "")))
                try:
                    update = func(state, *args, **kwargs)
                except GraphInterrupt as exc:
                    # Re-raised unchanged, so LangGraph still pauses the run normally.
                    span.set_status(Status(StatusCode.ERROR, PAUSED_STATUS))
                    span.add_event(PAUSED_EVENT, _interrupt_event_attributes(exc))
                    raise
                except Exception as exc:
                    span.record_exception(exc)
                    span.set_status(Status(StatusCode.ERROR, f"{type(exc).__name__}: {exc}"))
                    raise
                _record_update(span, update)
                span.set_status(Status(StatusCode.OK))
                return update

        return wrapper  # type: ignore[return-value]

    return decorator


# --- Token usage per trace ---------------------------------------------------------------

#: Span attributes summed over every model call of one trace.
_USAGE_ATTRIBUTES = {
    "gen_ai.usage.input_tokens": "input_tokens",
    "gen_ai.usage.output_tokens": "output_tokens",
    "gen_ai.usage.cost.total": "cost_usd",
}


class TokenUsageCollector(SpanProcessor):
    """Sum token usage and cost of all model-call spans, per trace."""

    def __init__(self) -> None:
        self._totals: dict[int, dict[str, float]] = {}
        self._lock = threading.Lock()

    def on_end(self, span: ReadableSpan) -> None:
        """Add a finished span's usage attributes, if any, to its trace's totals.

        Args:
            span: The span that just ended.
        """
        attributes = span.attributes or {}
        found = {
            name: float(attributes[key])
            for key, name in _USAGE_ATTRIBUTES.items()
            if key in attributes
        }
        if not found:
            return
        with self._lock:
            totals = self._totals.setdefault(span.context.trace_id, {"model_calls": 0.0})
            totals["model_calls"] += 1
            for name, value in found.items():
                totals[name] = totals.get(name, 0.0) + value

    def pop_totals(self, trace_id: int) -> dict[str, float]:
        """Return and forget the usage totals of one trace.

        Args:
            trace_id: The trace to read.

        Returns:
            Totals by name, or ``{}`` if the trace made no model calls.
        """
        with self._lock:
            return self._totals.pop(trace_id, {})


#: The process-wide collector, registered on the tracer provider by setup_observability().
USAGE_COLLECTOR = TokenUsageCollector()


def register_usage_collector(provider: Any) -> None:
    """Attach :data:`USAGE_COLLECTOR` to a tracer provider that supports span processors.

    Args:
        provider: An OpenTelemetry SDK ``TracerProvider``.
    """
    add_span_processor = getattr(provider, "add_span_processor", None)
    if add_span_processor is not None:
        add_span_processor(USAGE_COLLECTOR)


# --- Root span for one ticket run --------------------------------------------------------

#: Name of the root span that groups one start or resume of a ticket's graph run.
TICKET_RUN_SPAN = "scoutbot.ticket_run"


def _current_user() -> str:
    """Return the OS user running this process.

    Returns:
        The user name, or ``"unknown"`` if it cannot be determined.
    """
    try:
        return getpass.getuser()
    except Exception:  # getuser() raises when no user name source is available.  # noqa: BLE001
        return "unknown"


def trace_id_hex(span: Span) -> str:
    """Format a span's trace id the way trace viewers such as Jaeger show it.

    Args:
        span: Any span of the trace.

    Returns:
        The trace id as 32 lowercase hex characters.
    """
    return format(span.get_span_context().trace_id, "032x")


@dataclass
class TicketRun:
    """One traced run of a ticket: its root span and the run config to pass to the graph."""

    span: Span
    config: dict

    def record_result(self, snapshot: StateSnapshot) -> None:
        """Write the run status (paused/completed), pending nodes and outcome onto the span.

        Args:
            snapshot: ``graph.get_state(run.config)`` taken after the run.
        """
        paused = bool(snapshot.next)
        self.span.set_attribute("scoutbot.run_status", "paused" if paused else "completed")
        if paused:
            self.span.set_attribute("scoutbot.waiting_for", list(snapshot.next))
        outcome: str | None = (snapshot.values or {}).get("outcome")
        if outcome:
            self.span.set_attribute("scoutbot.outcome", outcome)


@contextmanager
def traced_ticket_run(run_config: dict, run_mode: str) -> Generator[TicketRun, None, None]:
    """Open the root span of a ticket run and tag its checkpoints with the trace id.

    Args:
        run_config: Thread config from ``thread_config(ticket_id)``.
        run_mode: How the run was started: ``"start"``, ``"resume"`` or ``"batch"``.

    Yields:
        A :class:`TicketRun` whose ``config`` must be used for the graph run.
    """
    ticket_id = run_config["configurable"]["thread_id"]
    run_info = {
        "run_mode": run_mode,
        "run_by": _current_user(),
        "host": socket.gethostname(),
        "pid": os.getpid(),
    }
    attributes = {
        "scoutbot.ticket_id": ticket_id,
        "scoutbot.model": config.get_model(),
        **{f"scoutbot.{key}": value for key, value in run_info.items()},
        "enduser.id": run_info["run_by"],
        # Same key TraceVerde uses, so all spans of a ticket share one session id.
        "session.id": ticket_id,
    }
    with get_tracer().start_as_current_span(TICKET_RUN_SPAN, attributes=attributes) as span:
        # LangGraph copies config["metadata"] into every checkpoint the run writes.
        metadata = {**run_config.get("metadata", {}), **run_info, "trace_id": trace_id_hex(span)}
        try:
            yield TicketRun(span=span, config={**run_config, "metadata": metadata})
        finally:
            totals = USAGE_COLLECTOR.pop_totals(span.get_span_context().trace_id)
            for name, value in totals.items():
                key = f"scoutbot.usage.{name}"
                span.set_attribute(key, value if name == "cost_usd" else int(value))
