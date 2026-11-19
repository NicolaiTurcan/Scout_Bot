"""Tests for JSONL step logging (``scoutbot.telemetry``)."""
from scoutbot.agent import LoopStep
from scoutbot.telemetry import log_step, read_log, step_errored


def test_read_log_returns_empty_list_for_a_missing_file(tmp_path):
    assert read_log(tmp_path / "nope.jsonl") == []


def test_read_log_skips_blank_lines(tmp_path):
    log_path = tmp_path / "log.jsonl"
    log_path.write_text('{"step": 1}\n\n   \n{"step": 2}\n', encoding="utf-8")

    assert read_log(log_path) == [{"step": 1}, {"step": 2}]


def test_log_step_appends_one_json_line_per_call(tmp_path):
    log_path = tmp_path / "log.jsonl"
    step_one = LoopStep(thought="Looking it up.", action="look_up_customer", observation={"found": True})
    step_two = LoopStep(thought="All done.")

    log_step("T-1", 1, step_one, duration_ms=123.456, log_path=log_path)
    log_step("T-1", 2, step_two, duration_ms=45.0, log_path=log_path)

    assert read_log(log_path) == [
        {"ticket_id": "T-1", "step": 1, "tool": "look_up_customer", "duration_ms": 123.5, "errored": False},
        {"ticket_id": "T-1", "step": 2, "tool": None, "duration_ms": 45.0, "errored": False},
    ]


def test_log_step_uses_configured_default_path(tmp_path, monkeypatch):
    default_log = tmp_path / "default.jsonl"
    monkeypatch.setattr("scoutbot.config.DEFAULT_LOG_PATH", default_log)

    log_step("T-1", 1, LoopStep(thought="Done."), duration_ms=1.0)

    assert read_log() == read_log(default_log)
    assert len(read_log(default_log)) == 1


def test_log_step_appends_across_calls_rather_than_overwriting(tmp_path):
    log_path = tmp_path / "log.jsonl"
    step = LoopStep(thought="Done.")

    log_step("T-4", 1, step, duration_ms=1.0, log_path=log_path)
    log_step("T-5", 1, step, duration_ms=2.0, log_path=log_path)

    assert [r["ticket_id"] for r in read_log(log_path)] == ["T-4", "T-5"]


def test_step_errored_flags_a_client_tool_error_observation():
    step = LoopStep(thought="Trying.", action="look_up_customer", observation={"error": "not found"})

    assert step_errored(step) is True


def test_step_errored_flags_a_nonzero_return_code():
    step = LoopStep(
        thought="Running.", action="bash_code_execution", observation={"stdout": "", "return_code": 1}
    )

    assert step_errored(step) is True


def test_step_errored_is_false_for_success_and_missing_observation():
    assert step_errored(LoopStep(thought="ok", observation={"return_code": 0})) is False
    assert step_errored(LoopStep(thought="final answer")) is False
