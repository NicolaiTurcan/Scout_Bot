"""Runtime configuration shared by every ScoutBot module.

Everything that used to be scattered across modules (model name, token budget, data file
locations) lives here, so there is exactly one place to change it.

Environment-driven values are read lazily, at call time rather than at import time. Entry
point scripts call ``load_dotenv()`` *after* ``import scoutbot``, so a module-level
``os.environ.get(...)`` would be evaluated too early and silently ignore ``.env``.
"""
import os
from pathlib import Path

#: Repository root (``src/scoutbot/config.py`` -> ``src`` -> repo root).
PROJECT_ROOT = Path(__file__).resolve().parents[2]

#: JSON file standing in for a real customer database.
CUSTOMERS_PATH = PROJECT_ROOT / "mock_customers.json"
#: Sample tickets used by the demo scripts.
TICKETS_PATH = PROJECT_ROOT / "sample_tickets.json"
#: Sample application log attached to tickets that ask for log analysis.
APP_LOG_PATH = PROJECT_ROOT / "sample_app.log"
#: Default JSONL file for per-step agent loop telemetry.
DEFAULT_LOG_PATH = PROJECT_ROOT / "scoutbot_log.jsonl"
#: Default SQLite file for LangGraph checkpoints.
DEFAULT_CHECKPOINT_PATH = PROJECT_ROOT / "scoutbot_checkpoints.db"
#: Folder of policy documents indexed by :mod:`scoutbot.knowledge`.
KNOWLEDGE_BASE_DIR = PROJECT_ROOT / "knowledge_base"
#: Default SQLite file for the retrieval store.
DEFAULT_KNOWLEDGE_DB_PATH = PROJECT_ROOT / "knowledge.db"
#: Default DuckDB file for the retrieval store used by the ticket graph.
DEFAULT_KNOWLEDGE_DUCKDB_PATH = PROJECT_ROOT / "knowledge.duckdb"

#: Model used when ``ANTHROPIC_MODEL`` is not set.
DEFAULT_MODEL = "claude-haiku-4-5"
#: Output token budget for every model call.
MAX_TOKENS = 1024


def get_model() -> str:
    """Return the Anthropic model id to use for the next request.

    Returns:
        The value of the ``ANTHROPIC_MODEL`` environment variable, or
        :data:`DEFAULT_MODEL` if it is unset.
    """
    return os.environ.get("ANTHROPIC_MODEL", DEFAULT_MODEL)


#: Jaeger UI used when ``TRACE_UI_URL`` is not set.
DEFAULT_TRACE_UI_URL = "http://127.0.0.1:16686"


def get_trace_ui_url() -> str:
    """Return the base URL of the trace viewer UI.

    Returns:
        ``TRACE_UI_URL`` from the environment, or :data:`DEFAULT_TRACE_UI_URL`.
    """
    return os.environ.get("TRACE_UI_URL", DEFAULT_TRACE_UI_URL).rstrip("/")
