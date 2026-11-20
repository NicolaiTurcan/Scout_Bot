"""Read ScoutBot traces back from Jaeger (API) and summarize runs and node timings."""
import json
import statistics
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import Any

from . import config
from .observability import SERVICE_NAME, TICKET_RUN_SPAN

#: OTLP status codes as Jaeger's JSON encodes them.
#: Span names whose timings are summarized by :func:`node_stats`.
_STATUS_NAMES = {0: "UNSET", 1: "OK", 2: "ERROR"}
_TIMED_PREFIXES = ("scoutbot.node.", "anthropic.messages.create")


def _decode_value(value: dict) -> Any:
    """Convert one OTLP JSON ``AnyValue`` into a Python value.

    Args:
        value: E.g. ``{"intValue": "912"}`` or ``{"stringValue": "[\\"a\\"]"}``.

    Returns:
        The decoded int, float, bool, str or list.
    """
    if "intValue" in value:
        return int(value["intValue"])  # OTLP JSON sends 64-bit ints as strings.
    if "doubleValue" in value:
        return float(value["doubleValue"])
    if "boolValue" in value:
        return bool(value["boolValue"])
    if "arrayValue" in value:
        return [_decode_value(item) for item in value["arrayValue"].get("values", [])]
    text = value.get("stringValue", "")
    if text.startswith("["):  # Jaeger flattens array attributes into JSON strings.
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text
    return text


def _decode_attributes(attributes: list) -> dict:
    """Convert an OTLP JSON attribute list into a plain dict.

    Args:
        attributes: List of ``{"key": ..., "value": {...}}`` items.

    Returns:
        Attribute values by key.
    """
    return {item["key"]: _decode_value(item.get("value", {})) for item in attributes or []}


def parse_spans(payload: dict) -> list[dict]:
    """Flatten a Jaeger v3 ``/api/v3/traces`` response into one dict per span.

    Args:
        payload: The decoded JSON response.

    Returns:
        Spans with ids, name, start time, duration, status, attributes, events and host.
    """
    spans = []
    for resource_spans in payload.get("result", {}).get("resourceSpans", []):
        resource = _decode_attributes(resource_spans.get("resource", {}).get("attributes"))
        for scope_spans in resource_spans.get("scopeSpans", []):
            for span in scope_spans.get("spans", []):
                start_ns = int(span["startTimeUnixNano"])
                status = span.get("status", {})
                code = status.get("code", 0)
                spans.append(
                    {
                        "trace_id": span["traceId"],
                        "span_id": span["spanId"],
                        "parent_id": span.get("parentSpanId"),
                        "name": span["name"],
                        "start": datetime.fromtimestamp(start_ns / 1e9, tz=timezone.utc),
                        "duration_ms": (int(span["endTimeUnixNano"]) - start_ns) / 1e6,
                        "status": _STATUS_NAMES.get(code, str(code)),
                        "status_message": status.get("message", ""),
                        "attributes": _decode_attributes(span.get("attributes")),
                        "events": [event["name"] for event in span.get("events", [])],
                        "host": resource.get("host.name"),
                    }
                )
    return spans


def fetch_spans(
    hours: float = 24.0,
    ticket_id: str | None = None,
    limit: int = 100,
    base_url: str | None = None,
) -> list[dict]:
    """Download ScoutBot spans from Jaeger for a recent time window.

    Args:
        hours: How far back to look.
        ticket_id: Only traces of this ticket, if given.
        limit: Maximum number of traces.
        base_url: Jaeger UI/API base URL; defaults to :func:`scoutbot.config.get_trace_ui_url`.

    Returns:
        Parsed spans (see :func:`parse_spans`), or ``[]`` if nothing matched.
    """
    now = datetime.now(timezone.utc)
    params = {
        "query.service_name": SERVICE_NAME,
        "query.start_time_min": (now - timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "query.start_time_max": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "query.search_depth": str(limit),
    }
    if ticket_id:
        params["query.attributes"] = json.dumps({"scoutbot.ticket_id": ticket_id})
    url = f"{base_url or config.get_trace_ui_url()}/api/v3/traces?{urllib.parse.urlencode(params)}"
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            return parse_spans(json.load(response))
    except urllib.error.HTTPError as exc:
        if exc.code == 404:  # Jaeger answers 404 when no trace matches the query.
            return []
        raise


def run_rows(spans: list[dict]) -> list[dict]:
    """Build one summary row per ticket run (root span), newest first.

    Args:
        spans: Parsed spans.

    Returns:
        Rows with start, ticket, mode, status, duration, usage, host and trace id.
    """
    rows = []
    for span in spans:
        if span["name"] != TICKET_RUN_SPAN:
            continue
        attrs = span["attributes"]
        rows.append(
            {
                "start": span["start"],
                "ticket_id": attrs.get("scoutbot.ticket_id"),
                "run_mode": attrs.get("scoutbot.run_mode"),
                "run_status": attrs.get("scoutbot.run_status", span["status"]),
                "duration_ms": span["duration_ms"],
                "model_calls": attrs.get("scoutbot.usage.model_calls", 0),
                "input_tokens": attrs.get("scoutbot.usage.input_tokens", 0),
                "output_tokens": attrs.get("scoutbot.usage.output_tokens", 0),
                "cost_usd": attrs.get("scoutbot.usage.cost_usd", 0.0),
                "run_by": attrs.get("scoutbot.run_by"),
                "host": span["host"],
                "trace_id": span["trace_id"],
            }
        )
    return sorted(rows, key=lambda row: row["start"], reverse=True)


def node_stats(spans: list[dict]) -> list[dict]:
    """Benchmark node and model-call spans: count, mean, median, max duration and errors.

    Args:
        spans: Parsed spans.

    Returns:
        One row per span name, slowest mean first.
    """
    durations: dict[str, list[float]] = {}
    errors: dict[str, int] = {}
    for span in spans:
        if not span["name"].startswith(_TIMED_PREFIXES):
            continue
        durations.setdefault(span["name"], []).append(span["duration_ms"])
        errors[span["name"]] = errors.get(span["name"], 0) + (span["status"] == "ERROR")
    rows = [
        {
            "name": name,
            "count": len(values),
            "mean_ms": statistics.fmean(values),
            "median_ms": statistics.median(values),
            "max_ms": max(values),
            "errors": errors[name],
        }
        for name, values in durations.items()
    ]
    return sorted(rows, key=lambda row: row["mean_ms"], reverse=True)
