"""Reading *server* tool results back out of a model response.

A server tool call (``server_tool_use``) and its result block are both just entries in the
same ``response.content`` list, paired by id. Nothing here executes anything: Anthropic has
already run the tool before this code sees the response. These helpers only turn the result
into a compact dict for the trace and the telemetry log.
"""
from typing import Any, Iterable, Optional

#: Max characters of fetched page content kept in a ``web_fetch`` summary.
WEB_FETCH_PREVIEW_CHARS = 300
#: Max characters of stdout kept in a ``bash_code_execution`` summary.
STDOUT_PREVIEW_CHARS = 500


def find_server_tool_result(content_blocks: Iterable[Any], tool_use_id: str) -> Optional[Any]:
    """Find the result block Anthropic paired with a given ``server_tool_use`` block.

    Args:
        content_blocks: The response's ``content`` list.
        tool_use_id: The ``id`` of the ``server_tool_use`` block.

    Returns:
        The matching result block, or ``None`` if there is none.
    """
    for block in content_blocks:
        if getattr(block, "tool_use_id", None) == tool_use_id:
            return block
    return None


def summarize_server_tool_result(result_block: Optional[Any]) -> dict:
    """Summarize what a server tool already did, for the trace and the log.

    Args:
        result_block: A result block from :func:`find_server_tool_result`, or ``None``.

    Returns:
        A small dict whose shape depends on the result type:

        - ``web_fetch_result``: ``{"url", "content_preview"}``
        - ``bash_code_execution_result``: ``{"stdout", "stderr", "return_code"}``
        - ``text_editor_code_execution_*``: ``{"file_op"}``
        - any ``*_error``: ``{"error"}``
        - anything else: ``{"raw"}``
    """
    if result_block is None:
        return {"note": "no matching result block (unexpected)"}
    content = result_block.content
    kind = getattr(content, "type", None) or ""

    if kind == "web_fetch_result":
        source = getattr(getattr(content, "content", None), "source", None)
        preview = (getattr(source, "data", "") or "")[:WEB_FETCH_PREVIEW_CHARS]
        return {"url": getattr(content, "url", None), "content_preview": preview}
    if kind == "bash_code_execution_result":
        return {
            "stdout": (getattr(content, "stdout", "") or "")[:STDOUT_PREVIEW_CHARS],
            "stderr": getattr(content, "stderr", "") or "",
            "return_code": getattr(content, "return_code", None),
        }
    if kind.startswith("text_editor_code_execution"):
        return {"file_op": kind.replace("text_editor_code_execution_", "").replace("_result", "")}
    if kind.endswith("_error"):
        return {"error": getattr(content, "error_code", "unknown_error")}
    return {"raw": str(content)}
