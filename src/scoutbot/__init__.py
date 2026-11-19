"""ScoutBot: support-ticket triage and escalation assistant.

Package layout:

- :mod:`scoutbot.config` - model name, token budget and data file paths.
- :mod:`scoutbot.llm` - the single Anthropic SDK boundary (patched in tests).
- :mod:`scoutbot.data` - loaders for the sample customers and tickets.
- :mod:`scoutbot.schema` - the ``TicketTriage`` structured-output contract.
- :mod:`scoutbot.classifier` - one-shot ``triage()`` with retry-once policy.
- :mod:`scoutbot.tools` - client and server tool definitions.
- :mod:`scoutbot.telemetry` - JSONL step logging.
- :mod:`scoutbot.agent` - bounded ReAct agent loop.
- :mod:`scoutbot.graph` - LangGraph workflow with interrupts and checkpoints.
"""
from .agent import AgentResult, LoopStep, run_agent_loop
from .classifier import TriageError, triage
from .config import DEFAULT_LOG_PATH, get_model
from .data import find_ticket, load_customers, load_tickets
from .graph import (
    STREAM_MODES,
    TicketState,
    auto_resolve_node,
    build_ticket_graph,
    escalate_ticket_node,
    initial_state,
    look_up_customer_node,
    make_sqlite_saver,
    pending_interrupt,
    route_after_triage,
    run_ticket_graph,
    thread_config,
    triage_ticket_node,
)
from .schema import Category, Priority, TicketTriage
from .telemetry import log_step, read_log
from .tools import (
    CODE_EXECUTION_TOOL,
    STATUS_PAGE_DOMAIN,
    WEB_FETCH_TOOL,
    escalate_to_human,
    look_up_customer,
)

__all__ = [
    "CODE_EXECUTION_TOOL",
    "DEFAULT_LOG_PATH",
    "STATUS_PAGE_DOMAIN",
    "STREAM_MODES",
    "WEB_FETCH_TOOL",
    "AgentResult",
    "Category",
    "LoopStep",
    "Priority",
    "TicketState",
    "TicketTriage",
    "TriageError",
    "auto_resolve_node",
    "build_ticket_graph",
    "escalate_ticket_node",
    "escalate_to_human",
    "find_ticket",
    "get_model",
    "initial_state",
    "load_customers",
    "load_tickets",
    "log_step",
    "look_up_customer",
    "look_up_customer_node",
    "make_sqlite_saver",
    "pending_interrupt",
    "read_log",
    "route_after_triage",
    "run_agent_loop",
    "run_ticket_graph",
    "thread_config",
    "triage",
    "triage_ticket_node",
]
