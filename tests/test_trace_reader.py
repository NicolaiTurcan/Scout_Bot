"""Offline tests for reading ScoutBot traces back from Jaeger's v3 API."""
import io
import json
import urllib.error
from unittest.mock import patch

import pytest

from scoutbot.trace_reader import fetch_spans, node_stats, parse_spans, run_rows

_MS = 1_000_000  # nanoseconds per millisecond


def _span(name, trace_id, start_ms, duration_ms, attributes=None, code=None, parent=None):
    """Build one span in Jaeger's OTLP JSON shape.

    Args:
        name: Span name.
        trace_id: Trace id (hex).
        start_ms: Start time in ms since the epoch.
        duration_ms: Duration in ms.
        attributes: Already-encoded OTLP attribute list.
        code: OTLP status code (1 = OK, 2 = ERROR), or ``None`` for unset.
        parent: Parent span id.

    Returns:
        The span dict.
    """
    span = {
        "traceId": trace_id,
        "spanId": f"{name}-{trace_id}",
        "name": name,
        "startTimeUnixNano": str(start_ms * _MS),
        "endTimeUnixNano": str((start_ms + duration_ms) * _MS),
        "attributes": attributes or [],
        "status": {"code": code} if code is not None else {},
    }
    if parent:
        span["parentSpanId"] = parent
    return span


def _payload(*spans):
    """Wrap spans in a ``/api/v3/traces`` response with one resource.

    Args:
        *spans: Spans from :func:`_span`.

    Returns:
        The response dict.
    """
    resource = {"attributes": [{"key": "host.name", "value": {"stringValue": "box-1"}}]}
    return {"result": {"resourceSpans": [{"resource": resource, "scopeSpans": [{"spans": list(spans)}]}]}}


def _str(key, value):
    return {"key": key, "value": {"stringValue": value}}


def _int(key, value):
    return {"key": key, "value": {"intValue": str(value)}}


PAYLOAD = _payload(
    _span(
        "scoutbot.ticket_run", "t1", 1_000, 1_200,
        [_str("scoutbot.ticket_id", "T-1"), _str("scoutbot.run_mode", "start"),
         _str("scoutbot.run_status", "paused"), _str("scoutbot.waiting_for", '["escalate_ticket"]'),
         _int("scoutbot.usage.input_tokens", 900), _int("scoutbot.usage.output_tokens", 80),
         {"key": "scoutbot.usage.cost_usd", "value": {"doubleValue": 0.0002}}],
    ),
    _span("anthropic.messages.create", "t1", 1_010, 1_000, code=1, parent="x"),
    _span("scoutbot.node.escalate_ticket", "t1", 2_100, 1, code=2, parent="x"),
    _span("scoutbot.ticket_run", "t2", 5_000, 10, [_str("scoutbot.ticket_id", "T-1"),
                                                  _str("scoutbot.run_mode", "resume")]),
    _span("scoutbot.node.escalate_ticket", "t2", 5_001, 3, code=1, parent="y"),
)


def test_parse_spans_decodes_values_status_and_resource():
    spans = {(s["trace_id"], s["name"]): s for s in parse_spans(PAYLOAD)}

    root = spans[("t1", "scoutbot.ticket_run")]
    assert root["duration_ms"] == 1_200
    assert root["status"] == "UNSET"
    assert root["host"] == "box-1"
    assert root["attributes"]["scoutbot.usage.input_tokens"] == 900
    assert root["attributes"]["scoutbot.waiting_for"] == ["escalate_ticket"]
    assert spans[("t1", "scoutbot.node.escalate_ticket")]["status"] == "ERROR"


def test_run_rows_are_newest_first_with_zero_usage_defaults():
    rows = run_rows(parse_spans(PAYLOAD))

    assert [(r["run_mode"], r["trace_id"]) for r in rows] == [("resume", "t2"), ("start", "t1")]
    assert rows[0]["input_tokens"] == 0
    assert rows[0]["run_status"] == "UNSET"
    assert rows[1]["cost_usd"] == pytest.approx(0.0002)


def test_node_stats_aggregates_durations_and_errors():
    stats = {row["name"]: row for row in node_stats(parse_spans(PAYLOAD))}

    assert set(stats) == {"anthropic.messages.create", "scoutbot.node.escalate_ticket"}
    escalate = stats["scoutbot.node.escalate_ticket"]
    assert (escalate["count"], escalate["mean_ms"], escalate["max_ms"]) == (2, 2.0, 3.0)
    assert escalate["errors"] == 1
    assert next(iter(stats)) == "anthropic.messages.create"  # slowest mean first


def test_fetch_spans_builds_query_and_parses_response():
    body = io.BytesIO(json.dumps(PAYLOAD).encode())
    with patch("urllib.request.urlopen", return_value=body) as urlopen:
        spans = fetch_spans(hours=2, ticket_id="T-1", base_url="http://jaeger:16686")

    url = urlopen.call_args.args[0]
    assert url.startswith("http://jaeger:16686/api/v3/traces?")
    assert "query.service_name=scoutbot" in url
    assert "scoutbot.ticket_id" in url
    assert len(spans) == 5


def test_fetch_spans_returns_empty_list_when_jaeger_finds_nothing():
    not_found = urllib.error.HTTPError("url", 404, "Not Found", {}, None)
    with patch("urllib.request.urlopen", side_effect=not_found):
        assert fetch_spans(base_url="http://jaeger:16686") == []
