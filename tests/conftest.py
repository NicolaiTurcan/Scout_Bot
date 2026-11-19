"""Shared pytest fixtures.

Every test runs offline: the Anthropic SDK boundary (``scoutbot.llm.get_client``) is
patched, so no API key or network access is needed. Production code stays simple and
direct; only the tests reach in and replace that single boundary.
"""
from unittest.mock import patch

import pytest


@pytest.fixture
def messages_create():
    """Patch the SDK boundary and return the fake ``client.messages.create`` mock.

    Configure it with ``.return_value`` / ``.side_effect`` and inspect it with
    ``.call_count`` / ``.call_args``.
    """
    with patch("scoutbot.llm.get_client") as get_client:
        yield get_client.return_value.messages.create


@pytest.fixture(autouse=True)
def _no_real_client():
    """Fail loudly if any test reaches the real SDK without using ``messages_create``."""
    def _refuse():
        raise AssertionError("Test tried to create a real Anthropic client.")

    with patch("scoutbot.llm.get_client", side_effect=_refuse):
        yield
