"""Loaders for ScoutBot's JSON sample data (customers and tickets).

Paths default to the files configured in :mod:`scoutbot.config`; every loader accepts an
explicit path so tests and scripts can point at other files.
"""
import json
from pathlib import Path
from typing import Optional, Union

from . import config

PathLike = Union[str, Path]


def _read_json(path: PathLike):
    """Read and parse a UTF-8 JSON file.

    Args:
        path: File to read.

    Returns:
        The parsed JSON value.
    """
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_customers(path: Optional[PathLike] = None) -> dict:
    """Load the mock customer database.

    Args:
        path: JSON file mapping ``customer_id`` to a customer record. Defaults to
            :data:`scoutbot.config.CUSTOMERS_PATH`.

    Returns:
        A dict keyed by ``customer_id``.
    """
    return _read_json(path or config.CUSTOMERS_PATH)


def load_tickets(path: Optional[PathLike] = None) -> list:
    """Load the sample ticket batch.

    Args:
        path: JSON file holding a list of tickets, each with ``id``, ``customer_id`` and
            ``text``. Defaults to :data:`scoutbot.config.TICKETS_PATH`.

    Returns:
        The list of ticket dicts, in file order.
    """
    return _read_json(path or config.TICKETS_PATH)


def find_ticket(ticket_id: str, path: Optional[PathLike] = None) -> dict:
    """Return a single sample ticket by id.

    Args:
        ticket_id: The ticket's ``id`` field, e.g. ``"T-1001"``.
        path: Optional tickets file, see :func:`load_tickets`.

    Returns:
        The matching ticket dict.

    Raises:
        KeyError: If no ticket has the given id.
    """
    for ticket in load_tickets(path):
        if ticket["id"] == ticket_id:
            return ticket
    raise KeyError(f"Unknown ticket_id: {ticket_id}")
