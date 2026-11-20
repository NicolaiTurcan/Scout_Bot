"""Print ScoutBot runs and node timings read back from Jaeger.

Usage:
    uv run python read_traces.py                     # last 24 hours
    uv run python read_traces.py --hours 2
    uv run python read_traces.py --ticket T-2001
"""
import argparse
import urllib.error

from dotenv import load_dotenv

from scoutbot.config import get_trace_ui_url
from scoutbot.trace_reader import fetch_spans, node_stats, run_rows


def print_runs(rows: list[dict]) -> None:
    """Print one line per ticket run plus usage totals.

    Args:
        rows: Rows from :func:`scoutbot.trace_reader.run_rows`.
    """
    print(f"Ticket runs ({len(rows)}, newest first):")
    print(
        f"{'started (UTC)':<20} {'ticket':<7} {'mode':<7} {'status':<10} {'ms':>7} "
        f"{'calls':>5} {'in':>6} {'out':>5} {'cost $':>9}  {'run_by':<10} trace"
    )
    for row in rows:
        print(
            f"{row['start']:%Y-%m-%d %H:%M:%S}  {row['ticket_id']!s:<7} {row['run_mode']!s:<7} "
            f"{row['run_status']!s:<10} {row['duration_ms']:>7.0f} {row['model_calls']:>5} "
            f"{row['input_tokens']:>6} {row['output_tokens']:>5} {row['cost_usd']:>9.6f}  "
            f"{row['run_by']!s:<10} {row['trace_id']}"
        )
    tokens = sum(row["input_tokens"] + row["output_tokens"] for row in rows)
    cost = sum(row["cost_usd"] for row in rows)
    paused = sum(row["run_status"] == "paused" for row in rows)
    print(f"Total: {tokens} tokens, ${cost:.6f}, {paused} run(s) paused for a human.")


def print_node_stats(rows: list[dict]) -> None:
    """Print the per-span timing benchmark.

    Args:
        rows: Rows from :func:`scoutbot.trace_reader.node_stats`.
    """
    print("\nTimings by span (slowest mean first):")
    print(f"{'span':<36} {'count':>5} {'mean ms':>9} {'median ms':>10} {'max ms':>9} {'errors':>7}")
    for row in rows:
        print(
            f"{row['name']:<36} {row['count']:>5} {row['mean_ms']:>9.2f} "
            f"{row['median_ms']:>10.2f} {row['max_ms']:>9.2f} {row['errors']:>7}"
        )


def main() -> None:
    """Parse arguments, fetch spans from Jaeger and print both reports."""
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--hours", type=float, default=24.0, help="look-back window")
    parser.add_argument("--ticket", help="only traces of this ticket id")
    parser.add_argument("--limit", type=int, default=100, help="maximum number of traces")
    args = parser.parse_args()

    try:
        spans = fetch_spans(hours=args.hours, ticket_id=args.ticket, limit=args.limit)
    except urllib.error.URLError as exc:
        raise SystemExit(
            f"Cannot reach Jaeger at {get_trace_ui_url()} ({exc.reason}). "
            "Is it running? Try: docker compose up -d"
        ) from exc
    if not spans:
        print("No ScoutBot traces found in this time window.")
        return

    print_runs(run_rows(spans))
    print_node_stats(node_stats(spans))


if __name__ == "__main__":
    main()
