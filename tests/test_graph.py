"""Offline tests for ScoutBot's routed, streaming, and interruptible LangGraph workflow."""
import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from fakes import HIGH_TRIAGE, LOW_TRIAGE, triage_response
from scoutbot.graph import (
    TicketState,
    auto_resolve_node,
    build_ticket_graph,
    initial_state,
    look_up_customer_node,
    make_sqlite_saver,
    pending_interrupt,
    route_after_triage,
    run_ticket_graph,
    thread_config,
)

# --- Pure functions (no model call) ---------------------------------------------------


def test_initial_state_has_empty_outputs():
    assert initial_state("text", "1001") == {
        "ticket_id": "",
        "ticket_text": "text",
        "customer_id": "1001",
        "customer": {},
        "triage": {},
        "outcome": "",
        "notes": [],
    }


def test_thread_config_shape():
    assert thread_config("T-1") == {"configurable": {"thread_id": "T-1"}}


def test_router_sends_high_priority_to_escalate():
    state = initial_state("Urgent refund!", customer_id="1001")
    state["triage"] = HIGH_TRIAGE

    assert route_after_triage(state) == "escalate_ticket"


def test_router_sends_non_high_priority_to_auto_resolve():
    state = initial_state("Change my email", customer_id="1003")
    state["triage"] = LOW_TRIAGE

    assert route_after_triage(state) == "auto_resolve"


def test_router_defaults_to_auto_resolve_without_triage():
    state = initial_state("No triage yet", customer_id="1003")

    assert route_after_triage(state) == "auto_resolve"


def test_node_returns_only_the_fields_it_changed():
    state: TicketState = initial_state("Some ticket", customer_id="1001")

    update = look_up_customer_node(state)

    assert set(update) == {"customer", "notes"}
    assert "ticket_text" not in update


def test_node_emits_a_custom_event_through_the_injected_writer():
    events = []

    auto_resolve_node(initial_state("t", "1003", "T-7"), writer=events.append)

    assert events == [{"node": "auto_resolve", "status": "auto-resolving ticket", "ticket_id": "T-7"}]


# --- Full graph runs (model patched) ---------------------------------------------------


def test_graph_routes_high_priority_to_escalate_pause_and_resume(messages_create):
    messages_create.return_value = triage_response(HIGH_TRIAGE)
    graph = build_ticket_graph(checkpointer=InMemorySaver())
    config = thread_config("T-3001")

    paused = graph.invoke(
        initial_state("I was charged twice.", customer_id="1001", ticket_id="T-3001"),
        config=config,
    )

    assert paused["customer"]["found"] is True
    assert paused["customer"]["plan"] == "pro"
    assert paused["triage"] == HIGH_TRIAGE
    assert paused["ticket_id"] == "T-3001"
    assert paused["customer_id"] == "1001"
    assert "__interrupt__" in paused
    payload = paused["__interrupt__"][0].value
    assert payload["ticket_id"] == "T-3001"
    assert payload["ticket_text"] == "I was charged twice."
    assert payload["triage"] == HIGH_TRIAGE
    assert pending_interrupt(graph, config) == payload
    assert graph.get_state(config).next == ("escalate_ticket",)
    assert messages_create.call_count == 1

    # Resuming passes the human answer back from interrupt() without re-running triage.
    resumed = graph.invoke(Command(resume="approved refund"), config=config)

    assert graph.get_state(config).next == ()
    assert pending_interrupt(graph, config) is None
    assert resumed["outcome"] == "escalated_resolved: approved refund"
    assert resumed["notes"][-1] == "escalated and resolved by human: approved refund"
    assert messages_create.call_count == 1


def test_graph_routes_low_priority_to_auto_resolve(messages_create):
    messages_create.return_value = triage_response(LOW_TRIAGE)

    final = run_ticket_graph("Update my email please.", customer_id="1003", ticket_id="T-3002")

    assert final["triage"] == LOW_TRIAGE
    assert final["outcome"] == "handled"
    assert final["notes"][-1] == "resolved automatically by ScoutBot"


def test_notes_accumulate_across_all_three_nodes(messages_create):
    """`notes` uses `operator.add`, so each visited node appends to the list."""
    messages_create.return_value = triage_response(HIGH_TRIAGE)
    graph = build_ticket_graph(checkpointer=InMemorySaver())
    config = thread_config("T-NOTES-1")

    graph.invoke(initial_state("I was charged twice.", customer_id="1001"), config=config)
    final = graph.invoke(Command(resume="approved"), config=config)

    assert final["notes"] == [
        "looked up 1001: found",
        "triaged as billing/high",
        "escalated and resolved by human: approved",
    ]


def test_plain_field_is_replaced_not_merged(messages_create):
    """`customer` has no reducer, so the node's record replaces the initial dict."""
    messages_create.return_value = triage_response(LOW_TRIAGE)
    start = initial_state("Update my email.", customer_id="1001")
    start["customer"] = {"stale": "should be gone"}

    final = build_ticket_graph().invoke(start)

    assert "stale" not in final["customer"]
    assert final["customer"]["customer_id"] == "1001"


def test_second_node_reads_what_the_first_node_wrote(messages_create):
    """The triage node must build its prompt from state, including the customer record."""
    messages_create.return_value = triage_response(LOW_TRIAGE)

    run_ticket_graph("I have a question.", customer_id="1001")

    prompt = messages_create.call_args.kwargs["messages"][0]["content"]
    assert "I have a question." in prompt
    assert "Customer record:" in prompt
    assert '"plan": "pro"' in prompt


def test_streaming_yields_one_update_per_node_in_order(messages_create):
    """`stream_mode="updates"` yields one chunk per visited node in execution order."""
    messages_create.return_value = triage_response(HIGH_TRIAGE)
    graph = build_ticket_graph(checkpointer=InMemorySaver())
    config = thread_config("T-STREAM-1")

    first_chunks = list(
        graph.stream(initial_state("I was charged twice.", "1001"), config=config, stream_mode="updates")
    )
    resumed_chunks = list(
        graph.stream(Command(resume="approved"), config=config, stream_mode="updates")
    )

    visited = [
        name
        for chunk in first_chunks + resumed_chunks
        for name in chunk
        if name != "__interrupt__"
    ]
    assert visited == ["look_up_customer", "triage_ticket", "escalate_ticket"]
    assert set(first_chunks[0]["look_up_customer"]) == {"customer", "notes"}
    assert set(first_chunks[1]["triage_ticket"]) == {"triage", "notes"}
    assert set(resumed_chunks[-1]["escalate_ticket"]) == {"outcome", "notes"}


def test_custom_event_arrives_before_node_updates_chunk(messages_create):
    """A node's custom event must arrive before that same node's updates chunk."""
    messages_create.return_value = triage_response(LOW_TRIAGE)
    graph = build_ticket_graph()

    events = list(
        graph.stream(
            initial_state("Update my email.", customer_id="1003", ticket_id="T-CUSTOM-1"),
            stream_mode=["updates", "custom"],
        )
    )

    triage_custom_idx = next(
        i
        for i, (mode, chunk) in enumerate(events)
        if mode == "custom" and chunk.get("node") == "triage_ticket"
    )
    triage_update_idx = next(
        i
        for i, (mode, chunk) in enumerate(events)
        if mode == "updates" and "triage_ticket" in chunk
    )
    assert triage_custom_idx < triage_update_idx


def test_interrupt_without_checkpointer_pauses_but_cannot_resume(messages_create):
    """Without a checkpointer, invoke() reaches __interrupt__, and the failure surfaces
    only when trying to resume."""
    messages_create.return_value = triage_response(HIGH_TRIAGE)
    graph = build_ticket_graph()
    config = thread_config("T-NO-CHECKPOINTER")

    paused = graph.invoke(initial_state("I was charged twice.", customer_id="1001"), config=config)
    assert "__interrupt__" in paused

    with pytest.raises((ValueError, RuntimeError)):
        graph.invoke(Command(resume="approved"), config=config)


def test_unknown_customer_still_reaches_the_triage_node(messages_create):
    messages_create.return_value = triage_response(LOW_TRIAGE)

    final = run_ticket_graph("Who am I?", customer_id="does-not-exist")

    assert final["customer"] == {"found": False, "customer_id": "does-not-exist"}
    assert final["notes"][0] == "looked up does-not-exist: not found"
    assert final["triage"] == LOW_TRIAGE


def test_sqlite_checkpoint_resumes_and_handles_interrupt_without_rerunning_triage(
    messages_create, tmp_path
):
    """Simulate a crash after triage_ticket, pause at interrupt(), then resume with Command."""
    messages_create.return_value = triage_response(HIGH_TRIAGE)
    db_path = tmp_path / "checkpoints.db"
    config = thread_config("T-CRASH-1")

    # Process 1: crashes right after triage_ticket completes.
    graph1 = build_ticket_graph(checkpointer=make_sqlite_saver(db_path))
    for chunk in graph1.stream(
        initial_state("Double charge", "1001", "T-CRASH-1"),
        config=config,
        stream_mode="updates",
        durability="sync",
    ):
        if "triage_ticket" in chunk:
            break

    assert messages_create.call_count == 1
    assert graph1.get_state(config).next == ("escalate_ticket",)

    # Process 2: recovers from the crash, runs escalate_ticket, and pauses at interrupt().
    graph2 = build_ticket_graph(checkpointer=make_sqlite_saver(db_path))
    paused = graph2.invoke(None, config=config, durability="sync")

    assert "__interrupt__" in paused
    assert graph2.get_state(config).next == ("escalate_ticket",)
    assert messages_create.call_count == 1

    # Process 3: a human provides a decision via Command(resume=...) and the graph completes.
    graph3 = build_ticket_graph(checkpointer=make_sqlite_saver(db_path))
    resumed = graph3.invoke(Command(resume="refund approved"), config=config, durability="sync")

    assert resumed["outcome"] == "escalated_resolved: refund approved"
    assert graph3.get_state(config).next == ()
    assert messages_create.call_count == 1
