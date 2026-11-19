"""The structured-output contract for ScoutBot's triage step.

Enums (not free strings) are used for category and priority so that a malformed value fails
validation loudly instead of silently becoming whatever the model felt like writing.
"""
from enum import Enum

from pydantic import BaseModel, Field


class Category(str, Enum):
    """Which area of the product a ticket is about."""

    billing = "billing"
    technical = "technical"
    account = "account"
    other = "other"


class Priority(str, Enum):
    """How urgently a ticket needs attention. ``high`` routes to a human in the graph."""

    low = "low"
    medium = "medium"
    high = "high"


class TicketTriage(BaseModel):
    """The validated result ScoutBot must produce for every ticket."""

    category: Category
    priority: Priority
    suggested_owner: str = Field(
        description="Team or person who should handle this ticket, e.g. 'billing-team'."
    )
