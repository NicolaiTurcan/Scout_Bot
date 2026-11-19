"""The single boundary between ScoutBot and the Anthropic SDK.

Production code stays simple and direct: there is no injected client abstraction. Every
module that talks to the model calls ``llm.get_client()`` through the module attribute
(``from scoutbot import llm``; ``llm.get_client()``), never via
``from scoutbot.llm import get_client``. That way tests can replace the boundary in one
place with ``mock.patch("scoutbot.llm.get_client")`` and every caller sees the fake.
"""
from anthropic import Anthropic


def get_client() -> Anthropic:
    """Create an Anthropic client.

    The API key is read from the ``ANTHROPIC_API_KEY`` environment variable by the SDK
    itself; it must never be hardcoded.

    Returns:
        A ready-to-use :class:`anthropic.Anthropic` client.
    """
    return Anthropic()
