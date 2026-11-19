"""Checkpointer factories for persisting graph runs across processes."""
import sqlite3
from pathlib import Path
from typing import Optional, Union

from langgraph.checkpoint.sqlite import SqliteSaver

from .. import config


def make_sqlite_saver(db_path: Optional[Union[str, Path]] = None) -> SqliteSaver:
    """Create a file-backed ``SqliteSaver`` checkpointer.

    The connection is opened with ``check_same_thread=False`` because LangGraph may touch
    the checkpointer from worker threads during a multi-step run.

    Args:
        db_path: SQLite file to use. Defaults to
            :data:`scoutbot.config.DEFAULT_CHECKPOINT_PATH`.

    Returns:
        A ready-to-use saver with its tables created.
    """
    conn = sqlite3.connect(str(db_path or config.DEFAULT_CHECKPOINT_PATH), check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()
    return saver
