"""Tests for the bounded ReAct agent loop (``scoutbot.agent.loop.run_agent_loop``)."""
import json

from fakes import (
    bash_result_block,
    response,
    server_tool_use_block,
    text_block,
    tool_use_block,
    web_fetch_result_block,
)
from scoutbot.agent import run_agent_loop
from scoutbot.telemetry import read_log

STRIPE_URL = "https://status.stripe.com"


def test_loop_looks_up_then_answers(messages_create):
    messages_create.side_effect = [
        response(
            text_block("I should check the account first."),
            tool_use_block("look_up_customer", {"customer_id": "1001"}),
        ),
        response(text_block("Final answer: pro customer, no need to escalate.")),
    ]

    result = run_agent_loop("Some billing ticket", customer_id="1001")

    assert result.final_answer == "Final answer: pro customer, no need to escalate."
    assert result.escalated is False
    assert len(result.steps) == 2
    assert result.steps[0].action == "look_up_customer"
    assert result.steps[0].observation == {
        "found": True,
        "customer_id": "1001",
        "plan": "pro",
        "account_age_days": 430,
        "past_refunds_90d": 2,
    }
    assert result.steps[1].action is None


def test_loop_sends_tool_result_back_to_the_model(messages_create):
    """The second turn must carry the assistant turn plus a matching tool_result."""
    first = response(
        text_block("Checking the account."),
        tool_use_block("look_up_customer", {"customer_id": "1001"}, tool_use_id="toolu_42"),
    )
    messages_create.side_effect = [first, response(text_block("Done."))]

    run_agent_loop("Some ticket", customer_id="1001")

    # The mock stores a reference to the (mutated) messages list, so inspect the final call.
    messages = messages_create.call_args_list[1].kwargs["messages"]
    assert messages[0] == {"role": "user", "content": "Ticket (customer_id=1001): Some ticket"}
    assert messages[1] == {"role": "assistant", "content": first.content}
    tool_result = messages[2]["content"][0]
    assert tool_result["type"] == "tool_result"
    assert tool_result["tool_use_id"] == "toolu_42"
    assert json.loads(tool_result["content"])["plan"] == "pro"


def test_loop_offers_client_and_server_tools(messages_create):
    messages_create.return_value = response(text_block("Done."))

    run_agent_loop("Some ticket", customer_id="1001")

    tool_names = [tool["name"] for tool in messages_create.call_args.kwargs["tools"]]
    assert tool_names == [
        "triage_ticket",
        "look_up_customer",
        "escalate_to_human",
        "web_fetch",
        "code_execution",
    ]


def test_loop_hits_step_cap_and_escalates(messages_create):
    # Always returns a tool call, never a final answer - the loop must stop on its own.
    messages_create.return_value = response(
        text_block("Still thinking..."),
        tool_use_block("look_up_customer", {"customer_id": "1001"}),
    )

    result = run_agent_loop("An unresolvable ticket", customer_id="1001", max_steps=3)

    assert result.escalated is True
    assert "step limit" in result.final_answer
    assert len(result.steps) == 4  # 3 tool-use steps + 1 cap-hit step
    assert messages_create.call_count == 3


def test_loop_escalation_requires_approval(messages_create):
    messages_create.side_effect = [
        response(
            text_block("This needs a human."),
            tool_use_block("escalate_to_human", {"reason": "needs a refund decision"}),
        ),
        response(text_block("Escalation was rejected; here's a draft reply instead.")),
    ]

    result = run_agent_loop(
        "A tricky refund request", customer_id="1001", approve_fn=lambda reason: False
    )

    assert result.steps[0].action == "escalate_to_human"
    assert result.steps[0].observation == {
        "approved": False,
        "note": "Escalation rejected by human reviewer.",
    }
    assert result.final_answer == "Escalation was rejected; here's a draft reply instead."


def test_loop_turns_an_unknown_tool_into_an_error_observation(messages_create):
    messages_create.side_effect = [
        response(text_block("Trying something."), tool_use_block("delete_account", {})),
        response(text_block("That tool doesn't exist; answering directly.")),
    ]

    result = run_agent_loop("Some ticket", customer_id="1001")

    assert result.steps[0].observation == {"error": "Unknown tool: delete_account"}
    assert result.escalated is False


def test_loop_only_executes_first_tool_use_per_turn(messages_create):
    messages_create.side_effect = [
        response(
            text_block("Doing two things at once."),
            tool_use_block("look_up_customer", {"customer_id": "1001"}, tool_use_id="toolu_a"),
            tool_use_block("escalate_to_human", {"reason": "just in case"}, tool_use_id="toolu_b"),
        ),
        response(text_block("Done.")),
    ]

    result = run_agent_loop("Ambiguous ticket", customer_id="1001")

    assert len(result.steps) == 2
    assert result.steps[0].action == "look_up_customer"


def test_loop_records_web_fetch_as_a_step_without_a_second_turn(messages_create):
    """A server tool's result is already part of the *same* response, so unlike a client
    tool the loop must not treat it as needing another turn."""
    tool_use_id = "srvtoolu_1"
    messages_create.return_value = response(
        text_block("Checking Stripe's status before refunding."),
        server_tool_use_block("web_fetch", {"url": STRIPE_URL}, tool_use_id),
        web_fetch_result_block(tool_use_id, STRIPE_URL, "All systems operational."),
        text_block("Stripe reports no outage, so this charge isn't on their end."),
    )

    result = run_agent_loop("Was I double-charged because of an outage?", customer_id="1001")

    assert messages_create.call_count == 1
    assert len(result.steps) == 2
    assert result.steps[0].action == "web_fetch"
    assert result.steps[0].observation["url"] == STRIPE_URL
    assert "All systems operational" in result.steps[0].observation["content_preview"]
    assert result.steps[1].action is None
    assert "no outage" in result.final_answer


def test_loop_records_code_execution_result(messages_create):
    tool_use_id = "srvtoolu_2"
    messages_create.return_value = response(
        text_block("Running a count against the log."),
        server_tool_use_block(
            "bash_code_execution", {"command": "grep ERROR log.txt | wc -l"}, tool_use_id
        ),
        bash_result_block(tool_use_id, stdout="5\n"),
        text_block("5 errors happened at or after 3pm."),
    )

    result = run_agent_loop("How many errors after 3pm?", customer_id="1002")

    assert result.steps[0].action == "bash_code_execution"
    assert result.steps[0].observation["stdout"] == "5\n"
    assert result.steps[0].observation["return_code"] == 0
    assert "5 errors" in result.final_answer


def test_loop_still_continues_when_a_client_tool_follows_a_server_tool(messages_create):
    """A server tool call doesn't stop a *later* client tool call in the same response from
    starting a normal next turn."""
    tool_use_id = "srvtoolu_3"
    messages_create.side_effect = [
        response(
            text_block("Checking Stripe first."),
            server_tool_use_block("web_fetch", {"url": STRIPE_URL}, tool_use_id),
            web_fetch_result_block(tool_use_id, STRIPE_URL, "All systems operational."),
            text_block("No outage - escalating for a human refund decision."),
            tool_use_block("escalate_to_human", {"reason": "duplicate charge, no known cause"}),
        ),
        response(text_block("Escalated.")),
    ]

    result = run_agent_loop(
        "I was double-charged, is it Stripe's fault?", customer_id="1001", approve_fn=lambda r: True
    )

    assert len(result.steps) == 3  # web_fetch step, escalate_to_human step, final answer step
    assert result.steps[0].action == "web_fetch"
    assert result.steps[1].action == "escalate_to_human"
    assert result.steps[1].observation["approved"] is True
    assert result.final_answer == "Escalated."


# --- Telemetry integration -------------------------------------------------------------


def test_loop_does_not_log_by_default(messages_create, tmp_path, monkeypatch):
    """Logging is opt-in (`log_path=None` by default): nothing may be written anywhere."""
    default_log = tmp_path / "default_log.jsonl"
    monkeypatch.setattr("scoutbot.config.DEFAULT_LOG_PATH", default_log)
    messages_create.side_effect = [
        response(
            text_block("Checking the account."),
            tool_use_block("look_up_customer", {"customer_id": "1001"}),
        ),
        response(text_block("All set.")),
    ]

    run_agent_loop("Some ticket", customer_id="1001")

    assert not default_log.exists()


def test_loop_logs_one_line_per_step_when_log_path_given(messages_create, tmp_path):
    log_path = tmp_path / "scoutbot_log.jsonl"
    messages_create.side_effect = [
        response(
            text_block("Checking the account."),
            tool_use_block("look_up_customer", {"customer_id": "1001"}),
        ),
        response(text_block("All set.")),
    ]

    result = run_agent_loop(
        "Some ticket", customer_id="1001", ticket_id="T-9001", log_path=log_path
    )

    records = read_log(log_path)
    assert len(records) == len(result.steps) == 2
    first = dict(records[0])
    duration = first.pop("duration_ms")  # Timing varies run to run; checked separately.
    assert first == {
        "ticket_id": "T-9001",
        "step": 1,
        "tool": "look_up_customer",
        "errored": False,
    }
    assert isinstance(duration, float)
    assert records[1]["tool"] is None
    assert records[1]["step"] == 2


def test_loop_logs_server_tool_steps_with_the_same_fields(messages_create, tmp_path):
    """A server tool step is logged exactly like a client tool step: same fields, no branch
    on tool type in the logger itself."""
    log_path = tmp_path / "scoutbot_log.jsonl"
    tool_use_id = "srvtoolu_9"
    messages_create.return_value = response(
        text_block("Checking Stripe's status before refunding."),
        server_tool_use_block("web_fetch", {"url": STRIPE_URL}, tool_use_id),
        web_fetch_result_block(tool_use_id, STRIPE_URL, "All systems operational."),
        text_block("No outage."),
    )

    run_agent_loop(
        "Was I double-charged?", customer_id="1001", ticket_id="T-9002", log_path=log_path
    )

    records = read_log(log_path)
    assert len(records) == 2
    assert records[0]["tool"] == "web_fetch"
    assert records[0]["ticket_id"] == "T-9002"
    assert records[0]["errored"] is False
    # Both steps came from the same API turn, so they share one turn-level duration.
    assert records[0]["duration_ms"] == records[1]["duration_ms"]


def test_loop_numbers_steps_consecutively_across_turns(messages_create, tmp_path):
    log_path = tmp_path / "scoutbot_log.jsonl"
    tool_use_id = "srvtoolu_5"
    messages_create.side_effect = [
        response(
            text_block("Checking Stripe first."),
            server_tool_use_block("web_fetch", {"url": STRIPE_URL}, tool_use_id),
            web_fetch_result_block(tool_use_id, STRIPE_URL, "All systems operational."),
            tool_use_block("look_up_customer", {"customer_id": "1001"}),
        ),
        response(text_block("Done.")),
    ]

    run_agent_loop("Ticket", customer_id="1001", ticket_id="T-9005", log_path=log_path)

    assert [r["step"] for r in read_log(log_path)] == [1, 2, 3]


def test_loop_marks_a_failed_code_execution_step_as_errored(messages_create, tmp_path):
    log_path = tmp_path / "scoutbot_log.jsonl"
    tool_use_id = "srvtoolu_10"
    messages_create.return_value = response(
        text_block("Running the count."),
        server_tool_use_block("bash_code_execution", {"command": "grep ERROR log.txt"}, tool_use_id),
        bash_result_block(tool_use_id, stdout="", return_code=1),
        text_block("The script failed."),
    )

    run_agent_loop("How many errors?", customer_id="1002", ticket_id="T-9003", log_path=log_path)

    assert read_log(log_path)[0]["errored"] is True


def test_loop_logs_the_cap_hit_step_too(messages_create, tmp_path):
    log_path = tmp_path / "scoutbot_log.jsonl"
    messages_create.return_value = response(
        text_block("Still thinking..."),
        tool_use_block("look_up_customer", {"customer_id": "1001"}),
    )

    run_agent_loop(
        "An unresolvable ticket", customer_id="1001", max_steps=2, ticket_id="T-9004", log_path=log_path
    )

    records = read_log(log_path)
    assert len(records) == 3  # 2 tool-use steps + 1 cap-hit step
    assert records[-1]["tool"] is None
    assert records[-1]["duration_ms"] == 0.0
