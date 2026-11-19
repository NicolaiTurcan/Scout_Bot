"""Tool definitions and implementations available to ScoutBot.

- Client tools (executed by our code): :mod:`.customer`, :mod:`.escalation`, :mod:`.triage`.
- Server tools (executed by Anthropic): :mod:`.server`.
"""
from .customer import LOOK_UP_CUSTOMER_TOOL, look_up_customer
from .escalation import ESCALATE_TOOL, ApproveFn, escalate_to_human, prompt_for_approval
from .server import CODE_EXECUTION_TOOL, SERVER_TOOLS, STATUS_PAGE_DOMAIN, WEB_FETCH_TOOL
from .triage import TRIAGE_TOOL, record_triage

#: Tools whose calls are executed locally by :func:`scoutbot.agent.dispatch.execute_client_tool`.
CLIENT_TOOLS = [TRIAGE_TOOL, LOOK_UP_CUSTOMER_TOOL, ESCALATE_TOOL]

__all__ = [
    "CLIENT_TOOLS",
    "CODE_EXECUTION_TOOL",
    "ESCALATE_TOOL",
    "LOOK_UP_CUSTOMER_TOOL",
    "SERVER_TOOLS",
    "STATUS_PAGE_DOMAIN",
    "TRIAGE_TOOL",
    "WEB_FETCH_TOOL",
    "ApproveFn",
    "escalate_to_human",
    "look_up_customer",
    "prompt_for_approval",
    "record_triage",
]
