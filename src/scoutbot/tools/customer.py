"""Client tool: read-only customer lookup.

A JSON file stands in for a real customer database. The tool schema is a plain dict (no
Pydantic model) because it only feeds an intermediate observation back into the agent loop;
nothing here validates a *final* structured answer.
"""
from ..data import load_customers

LOOK_UP_CUSTOMER_TOOL = {
    "name": "look_up_customer",
    "description": (
        "Look up an existing customer's account info by customer_id. Read-only - has no "
        "side effects. Returns plan, account age, and recent refund history."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "customer_id": {"type": "string", "description": "The customer's account ID."},
        },
        "required": ["customer_id"],
    },
}


def look_up_customer(customer_id: str) -> dict:
    """Look up a customer record by id.

    Args:
        customer_id: The customer's account id, e.g. ``"1001"``.

    Returns:
        ``{"found": True, **record}`` for a known customer, or
        ``{"found": False, "customer_id": customer_id}`` otherwise. Never raises for an
        unknown id, so the agent loop can show the model a graceful observation.
    """
    record = load_customers().get(customer_id)
    if record is None:
        return {"found": False, "customer_id": customer_id}
    return {"found": True, **record}
