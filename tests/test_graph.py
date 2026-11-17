"""Offline tests for ScoutBot's routed and checkpointed LangGraph workflow."""
from types import SimpleNamespace
from unittest.mock import patch

from scoutbot.graph import (
    TicketState,
    build_ticket_graph,
    initial_state,
    look_up_customer_node,
    make_sqlite_saver,
    route_after_triage,
    run_ticket_graph,
)

HIGH_TRIAGE = {"category": "billing", "priority": "high", "suggested_owner": "billing-team"}
LOW_TRIAGE = {"category": "account", "priority": "low", "suggested_owner": "support-team"}


def _fake_response(input_dict: dict = None):
    """Mimic an Anthropic Message response carrying one tool_use block."""
    block = SimpleNamespace(
        type="tool_use", name="triage_ticket", input=input_dict or HIGH_TRIAGE
    )
    return SimpleNamespace(content=[block])


def test_router_sends_high_priority_to_escalate_without_mocking():
    state = initial_state("Urgent refund!", customer_id="1001")
    state["triage"] = HIGH_TRIAGE

    assert route_after_triage(state) == "escalate_ticket"


def test_router_sends_non_high_priority_to_auto_resolve_without_mocking():
    state = initial_state("Change my email", customer_id="1003")
    state["triage"] = LOW_TRIAGE

    assert route_after_triage(state) == "auto_resolve"


@patch("scoutbot.core._client")
def test_graph_routes_high_priority_to_escalate_ticket(mock_client):
    mock_client.return_value.messages.create.return_value = _fake_response(HIGH_TRIAGE)

    final = run_ticket_graph("I was charged twice.", customer_id="1001", ticket_id="T-3001")

    assert final["customer"]["found"] is True
    assert final["customer"]["plan"] == "pro"
    assert final["triage"] == HIGH_TRIAGE
    assert final["outcome"] == "waiting_for_human"
    assert final["ticket_id"] == "T-3001"
    assert final["customer_id"] == "1001"


@patch("scoutbot.core._client")
def test_graph_routes_low_priority_to_auto_resolve(mock_client):
    mock_client.return_value.messages.create.return_value = _fake_response(LOW_TRIAGE)

    final = run_ticket_graph("Update my email please.", customer_id="1003", ticket_id="T-3002")

    assert final["triage"] == LOW_TRIAGE
    assert final["outcome"] == "handled"
    assert final["notes"][-1] == "resolved automatically by ScoutBot"


@patch("scoutbot.core._client")
def test_notes_accumulate_across_all_three_nodes(mock_client):
    """`notes` uses `operator.add`, so each visited node appends to the list."""
    mock_client.return_value.messages.create.return_value = _fake_response(HIGH_TRIAGE)

    final = run_ticket_graph("I was charged twice.", customer_id="1001")

    assert len(final["notes"]) == 3
    assert final["notes"][0] == "looked up 1001: found"
    assert final["notes"][1] == "triaged as billing/high"
    assert final["notes"][2] == "escalated: waiting for human review"


@patch("scoutbot.core._client")
def test_plain_field_is_replaced_not_merged(mock_client):
    """`customer` has no reducer, so the node's record replaces the initial dict."""
    mock_client.return_value.messages.create.return_value = _fake_response(HIGH_TRIAGE)
    start = initial_state("I was charged twice.", customer_id="1001")
    start["customer"] = {"stale": "should be gone"}

    final = build_ticket_graph().invoke(start)

    assert "stale" not in final["customer"]
    assert final["customer"]["customer_id"] == "1001"


@patch("scoutbot.core._client")
def test_second_node_reads_what_the_first_node_wrote(mock_client):
    """The triage node must build its prompt from state, including customer record."""
    mock_client.return_value.messages.create.return_value = _fake_response(HIGH_TRIAGE)

    run_ticket_graph("I was charged twice.", customer_id="1001")

    prompt = mock_client.return_value.messages.create.call_args.kwargs["messages"][0]["content"]
    assert "I was charged twice." in prompt
    assert "Customer record:" in prompt
    assert '"plan": "pro"' in prompt


@patch("scoutbot.core._client")
def test_streaming_yields_one_update_per_node_in_order(mock_client):
    """`stream_mode="updates"` yields one chunk per visited node in execution order."""
    mock_client.return_value.messages.create.return_value = _fake_response(HIGH_TRIAGE)
    graph = build_ticket_graph()

    chunks = list(
        graph.stream(initial_state("I was charged twice.", "1001"), stream_mode="updates")
    )

    assert [name for chunk in chunks for name in chunk] == [
        "look_up_customer",
        "triage_ticket",
        "escalate_ticket",
    ]
    assert set(chunks[0]["look_up_customer"]) == {"customer", "notes"}
    assert set(chunks[1]["triage_ticket"]) == {"triage", "notes"}
    assert set(chunks[2]["escalate_ticket"]) == {"outcome", "notes"}


@patch("scoutbot.core._client")
def test_unknown_customer_still_reaches_the_triage_node(mock_client):
    mock_client.return_value.messages.create.return_value = _fake_response(HIGH_TRIAGE)

    final = run_ticket_graph("Who am I?", customer_id="does-not-exist")

    assert final["customer"] == {"found": False, "customer_id": "does-not-exist"}
    assert final["notes"][0] == "looked up does-not-exist: not found"
    assert final["triage"] == HIGH_TRIAGE


def test_node_returns_only_the_fields_it_changed():
    state: TicketState = initial_state("Some ticket", customer_id="1001")

    update = look_up_customer_node(state)

    assert set(update) == {"customer", "notes"}
    assert "ticket_text" not in update


@patch("scoutbot.core._client")
def test_sqlite_checkpoint_resumes_without_rerunning_triage(mock_client, tmp_path):
    """Simulate a crash after triage_ticket and verify resume skips triage."""
    mock_client.return_value.messages.create.return_value = _fake_response(HIGH_TRIAGE)
    db_path = str(tmp_path / "checkpoints.db")
    config = {"configurable": {"thread_id": "T-CRASH-1"}}

    graph1 = build_ticket_graph(checkpointer=make_sqlite_saver(db_path))
    for chunk in graph1.stream(
        initial_state("Double charge", "1001", "T-CRASH-1"),
        config=config,
        stream_mode="updates",
        durability="sync",
    ):
        if "triage_ticket" in chunk:
            break

    assert mock_client.return_value.messages.create.call_count == 1
    assert graph1.get_state(config).next == ("escalate_ticket",)

    # Fresh graph + fresh SqliteSaver simulating a new Python process.
    graph2 = build_ticket_graph(checkpointer=make_sqlite_saver(db_path))
    resumed = graph2.invoke(None, config=config, durability="sync")

    assert resumed["outcome"] == "waiting_for_human"
    assert graph2.get_state(config).next == ()
    assert mock_client.return_value.messages.create.call_count == 1