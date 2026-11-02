"""Run the parser against eval/queries.jsonl and report accuracy.

Usage:
    python eval/run_eval.py
    python eval/run_eval.py --delay 6 --limit 10

Metric helpers are pure functions so they can be tested without an LLM.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:  # allow `python eval/run_eval.py`
    sys.path.insert(0, str(PROJECT_ROOT))

from app.schemas import PropertyFilter  # noqa: E402
from app.services import describer as describer_service  # noqa: E402
from app.services.llm_parser import InvalidFilterError, parse_query  # noqa: E402

DEFAULT_QUERIES = PROJECT_ROOT / "eval" / "queries.jsonl"
DEFAULT_DESCRIPTIONS = PROJECT_ROOT / "eval" / "descriptions.jsonl"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "eval" / "results"
DEFAULT_MARKDOWN = PROJECT_ROOT / "docs" / "evaluation.md"
DEFAULT_DELAY_SECONDS = 4.0

FILTER_FIELDS = tuple(PropertyFilter.model_fields)


# --- metric helpers -------------------------------------------------------

def percentile(values: Iterable[float], p: float) -> float:
    """Linear-interpolated percentile (p in 0..100) of the given values."""
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]

    rank = (len(ordered) - 1) * (p / 100.0)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def normalise_filter(raw: dict[str, Any]) -> dict[str, Any]:
    """Drop unset values so an empty filter equals an absent one."""
    return {key: value for key, value in raw.items() if value is not None}


def is_exact_match(expected: dict[str, Any], actual: dict[str, Any]) -> bool:
    return normalise_filter(expected) == normalise_filter(actual)


def compare_fields(
    expected: dict[str, Any], actual: dict[str, Any]
) -> dict[str, bool]:
    """Compare only fields that either side actually mentions."""
    expected = normalise_filter(expected)
    actual = normalise_filter(actual)
    compared: dict[str, bool] = {}
    for field in FILTER_FIELDS:
        if field in expected or field in actual:
            compared[field] = expected.get(field) == actual.get(field)
    return compared


def _summarise(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate exact-match, field and latency metrics for a record group."""
    total = len(records)
    exact = sum(1 for record in records if record["exact_match"])
    schema_ok = sum(1 for record in records if record["schema_ok"])

    field_correct: dict[str, int] = {field: 0 for field in FILTER_FIELDS}
    field_total: dict[str, int] = {field: 0 for field in FILTER_FIELDS}
    for record in records:
        for field, correct in record["field_matches"].items():
            field_total[field] += 1
            field_correct[field] += int(correct)

    field_accuracy = {
        field: {
            "correct": field_correct[field],
            "total": field_total[field],
            "accuracy": round(field_correct[field] / field_total[field], 4),
        }
        for field in FILTER_FIELDS
        if field_total[field]
    }

    accuracies = [entry["accuracy"] for entry in field_accuracy.values()]
    latencies = [record["latency_ms"] for record in records]

    return {
        "total": total,
        "exact_match_count": exact,
        "exact_match": round(exact / total, 4) if total else 0.0,
        "schema_compliance": round(schema_ok / total, 4) if total else 0.0,
        "field_accuracy": field_accuracy,
        "mean_field_accuracy": (
            round(sum(accuracies) / len(accuracies), 4) if accuracies else 0.0
        ),
        "latency_ms": {
            "p50": round(percentile(latencies, 50), 2),
            "p95": round(percentile(latencies, 95), 2),
            "mean": round(sum(latencies) / total, 2) if total else 0.0,
        },
    }


def compute_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute overall metrics plus a breakdown per category."""
    enriched = []
    for record in records:
        enriched.append(
            {
                **record,
                # A rejected query is not a successful parse, even if the
                # fallback empty filter happens to equal the expectation.
                "exact_match": record["schema_ok"]
                and is_exact_match(record["expected"], record["actual"]),
                "field_matches": compare_fields(record["expected"], record["actual"]),
            }
        )

    categories = sorted({record["category"] for record in enriched})
    summary = _summarise(enriched)
    summary["by_category"] = {
        category: _summarise(
            [record for record in enriched if record["category"] == category]
        )
        for category in categories
    }
    summary["records"] = enriched
    return summary


# --- execution ------------------------------------------------------------

def load_queries(path: Path | str = DEFAULT_QUERIES) -> list[dict[str, Any]]:
    """Read a JSONL file, skipping blank lines."""
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def run_queries(
    queries: list[dict[str, Any]],
    parser: Callable[[str], PropertyFilter] = parse_query,
    delay: float = DEFAULT_DELAY_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
) -> list[dict[str, Any]]:
    """Parse every query with a delay between calls to respect rate limits."""
    records = []
    for index, item in enumerate(queries):
        if index and delay > 0:
            sleep(delay)

        started = time.perf_counter()
        schema_ok = True
        error = None
        try:
            parsed = parser(item["query"])
            actual = normalise_filter(parsed.model_dump(exclude_none=True))
        except InvalidFilterError as exc:
            actual = {}
            schema_ok = False
            error = str(exc)

        latency_ms = (time.perf_counter() - started) * 1000
        records.append(
            {
                "id": item["id"],
                "query": item["query"],
                "category": item["category"],
                "expected": normalise_filter(item["expected"]),
                "actual": actual,
                "latency_ms": latency_ms,
                "schema_ok": schema_ok,
                "error": error,
            }
        )
    return records


# --- descriptions (hallucination rate) ------------------------------------

def evaluate_descriptions(
    cases: list[dict[str, Any]],
    describe: Callable[[dict[str, Any]], Any] = describer_service.describe_listing,
    delay: float = DEFAULT_DELAY_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
) -> list[dict[str, Any]]:
    """Describe each sample listing and check the text against its row."""
    records = []
    for index, case in enumerate(cases):
        if index and delay > 0:
            sleep(delay)
        outcome = describe(case["listing"])
        records.append(
            {
                "id": case["id"],
                "passed": bool(outcome.check.passed),
                "fallback_used": bool(outcome.fallback_used),
                "errors": list(outcome.check.errors),
            }
        )
    return records


def description_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Hallucination rate over checked descriptions."""
    total = len(records)
    failed = sum(1 for record in records if not record["passed"])
    return {
        "total": total,
        "failed": failed,
        "hallucination_rate": round(failed / total, 4) if total else 0.0,
        "records": records,
    }


# --- reporting ------------------------------------------------------------

def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def render_markdown(
    metrics: dict[str, Any],
    generated_at: str,
    query_count: int,
) -> str:
    """Render the evaluation report as Markdown."""
    lines = [
        "# Evaluation",
        "",
        f"Generated: {generated_at}",
        f"Queries: {query_count}",
        "",
        "## Overall",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Exact match | {_pct(metrics['exact_match'])}"
        f" ({metrics['exact_match_count']}/{metrics['total']}) |",
        f"| Schema compliance | {_pct(metrics['schema_compliance'])} |",
        f"| Mean field accuracy | {_pct(metrics['mean_field_accuracy'])} |",
        f"| Latency p50 | {metrics['latency_ms']['p50']} ms |",
        f"| Latency p95 | {metrics['latency_ms']['p95']} ms |",
        "",
        "## Per field",
        "",
        "| Field | Correct | Total | Accuracy |",
        "| --- | --- | --- | --- |",
    ]
    for field, stats in metrics["field_accuracy"].items():
        lines.append(
            f"| {field} | {stats['correct']} | {stats['total']} |"
            f" {_pct(stats['accuracy'])} |"
        )

    lines += [
        "",
        "## Per category",
        "",
        "| Category | N | Exact match | Mean field accuracy | p50 ms | p95 ms |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for category, stats in metrics["by_category"].items():
        lines.append(
            f"| {category} | {stats['total']} | {_pct(stats['exact_match'])} |"
            f" {_pct(stats['mean_field_accuracy'])} | {stats['latency_ms']['p50']} |"
            f" {stats['latency_ms']['p95']} |"
        )

    descriptions = metrics.get("descriptions")
    if descriptions:
        lines += [
            "",
            "## Descriptions (hallucination check)",
            "",
            "| Metric | Value |",
            "| --- | --- |",
            f"| Samples | {descriptions['total']} |",
            f"| Failed checks | {descriptions['failed']} |",
            f"| Hallucination rate | {_pct(descriptions['hallucination_rate'])} |",
        ]
        failed_descriptions = [
            record for record in descriptions["records"] if not record["passed"]
        ]
        for record in failed_descriptions:
            lines.append(
                f"- Sample {record['id']}: {'; '.join(record['errors']) or 'empty output'}"
            )

    failures = [record for record in metrics["records"] if not record["exact_match"]]
    lines += ["", "## Failed queries", ""]
    if not failures:
        lines.append("None.")
    else:
        lines += [
            "| ID | Category | Query | Expected | Actual |",
            "| --- | --- | --- | --- | --- |",
        ]
        for record in failures:
            expected = json.dumps(record["expected"], sort_keys=True)
            actual = json.dumps(record["actual"], sort_keys=True)
            lines.append(
                f"| {record['id']} | {record['category']} | {record['query']} |"
                f" `{expected}` | `{actual}` |"
            )

    return "\n".join(lines) + "\n"


def write_outputs(
    metrics: dict[str, Any],
    generated_at: str,
    results_dir: Path | str = DEFAULT_RESULTS_DIR,
    markdown_path: Path | str = DEFAULT_MARKDOWN,
) -> tuple[Path, Path]:
    """Write the JSON results and the Markdown summary; return both paths."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = generated_at.replace(":", "").replace("-", "")
    json_path = results_dir / f"{stamp}.json"
    json_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")

    markdown_path = Path(markdown_path)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(
        render_markdown(metrics, generated_at, metrics["total"]), encoding="utf-8"
    )
    return json_path, markdown_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the query parser.")
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_DELAY_SECONDS,
        help="Seconds to wait between LLM calls",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only run N queries")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument(
        "--descriptions",
        type=Path,
        default=DEFAULT_DESCRIPTIONS,
        help="JSONL of sample listings for the hallucination check",
    )
    parser.add_argument(
        "--skip-descriptions",
        action="store_true",
        help="Skip the description / hallucination evaluation",
    )
    args = parser.parse_args(argv)

    queries = load_queries(args.queries)
    if args.limit:
        queries = queries[: args.limit]
    if not queries:
        print("error: no queries found", file=sys.stderr)
        return 1

    records = run_queries(queries, delay=args.delay)
    metrics = compute_metrics(records)

    if not args.skip_descriptions and Path(args.descriptions).exists():
        cases = load_queries(args.descriptions)
        metrics["descriptions"] = description_metrics(
            evaluate_descriptions(cases, delay=args.delay)
        )

    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    json_path, markdown_path = write_outputs(
        metrics, generated_at, args.results_dir, args.markdown
    )

    print(
        f"exact match {_pct(metrics['exact_match'])}"
        f" | schema {_pct(metrics['schema_compliance'])}"
        f" | mean field {_pct(metrics['mean_field_accuracy'])}"
        f" | p50 {metrics['latency_ms']['p50']} ms"
        f" | p95 {metrics['latency_ms']['p95']} ms"
    )
    print(f"wrote {json_path}")
    print(f"wrote {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
