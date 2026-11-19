"""Tests for the triage structured-output contract (``scoutbot.schema``)."""
import pytest
from pydantic import ValidationError

from scoutbot.schema import Category, Priority, TicketTriage


def test_valid_triage_parses():
    result = TicketTriage(category="billing", priority="high", suggested_owner="billing-team")
    assert result.category is Category.billing
    assert result.priority is Priority.high
    assert result.suggested_owner == "billing-team"


def test_json_dump_uses_plain_strings():
    result = TicketTriage(category="billing", priority="high", suggested_owner="billing-team")
    assert result.model_dump(mode="json") == {
        "category": "billing",
        "priority": "high",
        "suggested_owner": "billing-team",
    }


def test_invalid_priority_is_rejected():
    with pytest.raises(ValidationError):
        TicketTriage(category="billing", priority="urgent!!", suggested_owner="billing-team")


def test_invalid_category_is_rejected():
    with pytest.raises(ValidationError):
        TicketTriage(category="not-a-real-category", priority="low", suggested_owner="someone")


def test_missing_field_is_rejected():
    with pytest.raises(ValidationError):
        TicketTriage(category="billing", priority="low")
