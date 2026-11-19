"""Tests for the one-shot classifier (``scoutbot.classifier.triage``)."""
import pytest

from fakes import HIGH_TRIAGE, response, text_block, triage_response
from scoutbot.classifier import TriageError, triage


def test_triage_returns_validated_result(messages_create):
    messages_create.return_value = triage_response(HIGH_TRIAGE)

    result = triage("I was charged twice, please refund me.")

    assert result.category == "billing"
    assert result.priority == "high"
    assert result.suggested_owner == "billing-team"
    messages_create.assert_called_once()


def test_triage_forces_the_triage_tool(messages_create):
    messages_create.return_value = triage_response(HIGH_TRIAGE)

    triage("Some ticket")

    kwargs = messages_create.call_args.kwargs
    assert kwargs["tool_choice"] == {"type": "tool", "name": "triage_ticket"}
    assert [tool["name"] for tool in kwargs["tools"]] == ["triage_ticket"]
    assert kwargs["messages"] == [{"role": "user", "content": "Some ticket"}]


def test_triage_uses_model_from_environment_at_call_time(messages_create, monkeypatch):
    """The model is read lazily, so `load_dotenv()` after import still takes effect."""
    messages_create.return_value = triage_response(HIGH_TRIAGE)
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-test-model")

    triage("Some ticket")

    assert messages_create.call_args.kwargs["model"] == "claude-test-model"


def test_triage_retries_once_then_raises_on_repeated_bad_data(messages_create):
    messages_create.return_value = triage_response(
        {"category": "billing", "priority": "urgent!!", "suggested_owner": "billing-team"}
    )

    with pytest.raises(TriageError):
        triage("this will fail schema validation twice")

    assert messages_create.call_count == 2


def test_triage_succeeds_on_second_attempt(messages_create):
    bad = triage_response({"category": "billing", "priority": "urgent!!", "suggested_owner": "x"})
    good = triage_response(HIGH_TRIAGE)
    messages_create.side_effect = [bad, good]

    result = triage("flaky first reply, valid second reply")

    assert result.priority == "high"
    assert messages_create.call_count == 2


def test_triage_raises_when_model_never_calls_the_tool(messages_create):
    messages_create.return_value = response(text_block("I'd rather not classify this one."))

    with pytest.raises(TriageError) as exc_info:
        triage("model ignores the tool entirely")

    # The underlying failure is chained, so the cause is not lost.
    assert isinstance(exc_info.value.__cause__, TriageError)
