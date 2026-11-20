"""Offline tests for ScoutBot's OpenTelemetry node decorator.

Spans are captured in memory instead of being exported, so the tests need neither TraceVerde
nor a running trace backend.
"""
import inspect

import pytest
from fakes import HIGH_TRIAGE, LOW_TRIAGE, triage_response
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from scoutbot.graph import (
    build_ticket_graph,
    checkpoint_history,
    initial_state,
    look_up_customer_node,
    thread_config,
)
from scoutbot.observability import (
    PAUSED_EVENT,
    PAUSED_STATUS,
    TICKET_RUN_SPAN,
    get_tracer,
    register_usage_collector,
    trace_id_hex,
    traced_node,
    traced_ticket_run,
)

# OpenTelemetry allows the global tracer provider to be set only once per process, so one
# in-memory exporter is shared by every test and cleared before each of them.
_EXPORTER = InMemorySpanExporter()
_provider = TracerProvider()
_provider.add_span_processor(SimpleSpanProcessor(_EXPORTER))
register_usage_collector(_provider)
trace.set_tracer_provider(_provider)


@pytest.fixture
def spans():
    """Clear captured spans and return a function listing ScoutBot's finished spans by name."""
    _EXPORTER.clear()

    def finished() -> dict:
        return {
            span.name: span
            for span in _EXPORTER.get_finished_spans()
            if span.name.startswith("scoutbot.")
        }

    return finished


def test_decorator_keeps_the_signature_langgraph_inspects():
    params = inspect.signature(look_up_customer_node).parameters

    assert list(params) == ["state", "writer"]


def test_successful_node_span_has_ids_update_and_ok_status(spans):
    state = initial_state("Some ticket", customer_id="1001", ticket_id="T-9")

    look_up_customer_node(state)

    span = spans()["scoutbot.node.look_up_customer"]
    assert span.status.status_code is StatusCode.OK
    assert span.attributes["scoutbot.ticket_id"] == "T-9"
    assert span.attributes["scoutbot.customer_id"] == "1001"
    assert span.attributes["scoutbot.updated_keys"] == ("customer", "notes")
    assert span.attributes["scoutbot.notes"] == ("looked up 1001: found",)


def test_failing_node_records_exception_and_error_status(spans):
    @traced_node("broken")
    def broken_node(state):
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        broken_node({"ticket_id": "T-1"})

    span = spans()["scoutbot.node.broken"]
    assert span.status.status_code is StatusCode.ERROR
    assert "ValueError: boom" in span.status.description
    assert [event.name for event in span.events] == ["exception"]


def test_graph_run_creates_one_span_per_node(messages_create, spans):
    messages_create.return_value = triage_response(LOW_TRIAGE)
    graph = build_ticket_graph()

    graph.invoke(initial_state("Change my email", customer_id="1003", ticket_id="T-2"))

    recorded = spans()
    assert set(recorded) == {
        "scoutbot.node.look_up_customer",
        "scoutbot.node.triage_ticket",
        "scoutbot.node.auto_resolve",
    }
    assert recorded["scoutbot.node.triage_ticket"].attributes["scoutbot.triage.priority"] == "low"
    assert recorded["scoutbot.node.auto_resolve"].attributes["scoutbot.outcome"] == "handled"


def test_interrupt_marks_span_as_error_without_leaking_ticket_text(messages_create, spans):
    messages_create.return_value = triage_response(HIGH_TRIAGE)
    graph = build_ticket_graph(checkpointer=InMemorySaver())
    config = thread_config("T-3")

    graph.invoke(
        initial_state("Secret refund details", customer_id="1001", ticket_id="T-3"),
        config=config,
    )

    span = spans()["scoutbot.node.escalate_ticket"]
    assert span.status.status_code is StatusCode.ERROR
    assert span.status.description == PAUSED_STATUS
    (event,) = span.events
    assert event.name == PAUSED_EVENT
    assert "ticket_text" in event.attributes["interrupt.payload_keys"]
    assert "Secret refund details" not in str(dict(event.attributes))

    # On resume the node runs again, this time to completion with an OK span.
    _EXPORTER.clear()
    graph.invoke(Command(resume="refund approved"), config=config)

    resumed = spans()["scoutbot.node.escalate_ticket"]
    assert resumed.status.status_code is StatusCode.OK
    assert resumed.attributes["scoutbot.outcome"] == "escalated_resolved: refund approved"


# --- Root span per ticket run -------------------------------------------------------------


def _fake_model_call(input_tokens: int, output_tokens: int, cost: float) -> None:
    """Emit a span shaped like TraceVerde's ``anthropic.messages.create`` span.

    Args:
        input_tokens: Value for ``gen_ai.usage.input_tokens``.
        output_tokens: Value for ``gen_ai.usage.output_tokens``.
        cost: Value for ``gen_ai.usage.cost.total``.
    """
    with get_tracer().start_as_current_span("anthropic.messages.create") as span:
        span.set_attribute("gen_ai.usage.input_tokens", input_tokens)
        span.set_attribute("gen_ai.usage.output_tokens", output_tokens)
        span.set_attribute("gen_ai.usage.cost.total", cost)


def test_ticket_run_span_is_the_parent_of_every_node_span(messages_create, spans):
    messages_create.return_value = triage_response(LOW_TRIAGE)
    graph = build_ticket_graph(checkpointer=InMemorySaver())
    config = thread_config("T-4")

    with traced_ticket_run(config, run_mode="batch") as run:
        graph.invoke(initial_state("Change my email", "1003", "T-4"), config=run.config)
        run.record_result(graph.get_state(config))

    recorded = spans()
    root = recorded[TICKET_RUN_SPAN]
    assert root.parent is None
    assert root.attributes["scoutbot.ticket_id"] == "T-4"
    assert root.attributes["scoutbot.run_mode"] == "batch"
    assert root.attributes["enduser.id"] == root.attributes["scoutbot.run_by"]
    assert root.attributes["scoutbot.host"]
    assert root.attributes["scoutbot.run_status"] == "completed"
    assert root.attributes["scoutbot.outcome"] == "handled"
    trace_id = root.context.trace_id
    assert all(span.context.trace_id == trace_id for span in recorded.values())


def test_ticket_run_span_records_pause(messages_create, spans):
    messages_create.return_value = triage_response(HIGH_TRIAGE)
    graph = build_ticket_graph(checkpointer=InMemorySaver())
    config = thread_config("T-5")

    with traced_ticket_run(config, run_mode="start") as run:
        graph.invoke(initial_state("Refund now", "1001", "T-5"), config=run.config)
        run.record_result(graph.get_state(config))

    root = spans()[TICKET_RUN_SPAN]
    assert root.attributes["scoutbot.run_status"] == "paused"
    assert root.attributes["scoutbot.waiting_for"] == ("escalate_ticket",)


def test_ticket_run_span_sums_token_usage_of_its_own_trace_only(spans):
    _fake_model_call(999, 999, 9.0)  # Outside the run: belongs to another trace.

    with traced_ticket_run(thread_config("T-6"), run_mode="batch"):
        _fake_model_call(100, 10, 0.001)
        _fake_model_call(200, 20, 0.002)

    root = spans()[TICKET_RUN_SPAN]
    assert root.attributes["scoutbot.usage.model_calls"] == 2
    assert root.attributes["scoutbot.usage.input_tokens"] == 300
    assert root.attributes["scoutbot.usage.output_tokens"] == 30
    assert root.attributes["scoutbot.usage.cost_usd"] == pytest.approx(0.003)


# --- Checkpoint history -------------------------------------------------------------------


def test_checkpoints_carry_the_trace_id_of_the_run_that_wrote_them(messages_create, spans):
    messages_create.return_value = triage_response(HIGH_TRIAGE)
    graph = build_ticket_graph(checkpointer=InMemorySaver())
    config = thread_config("T-7")

    with traced_ticket_run(config, run_mode="start") as start:
        graph.invoke(initial_state("Refund now", "1001", "T-7"), config=start.config)
    with traced_ticket_run(config, run_mode="resume") as resume:
        graph.invoke(Command(resume="approved"), config=resume.config)

    history = checkpoint_history(graph, config)

    assert [row["step"] for row in history] == sorted(row["step"] for row in history)
    assert {row["run_mode"] for row in history} == {"start", "resume"}
    start_rows = [row for row in history if row["run_mode"] == "start"]
    resume_rows = [row for row in history if row["run_mode"] == "resume"]
    assert {row["trace_id"] for row in start_rows} == {trace_id_hex(start.span)}
    assert {row["trace_id"] for row in resume_rows} == {trace_id_hex(resume.span)}
    assert start_rows[-1]["next"] == ["escalate_ticket"]
    assert history[-1]["next"] == []
    assert history[-1]["outcome"] == "escalated_resolved: approved"


def test_checkpoint_history_is_empty_for_an_unknown_thread():
    graph = build_ticket_graph(checkpointer=InMemorySaver())

    assert checkpoint_history(graph, thread_config("nope")) == []
