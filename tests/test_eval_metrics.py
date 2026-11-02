"""Tests for the metric helpers in eval/run_eval.py (no LLM involved)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "eval" / "run_eval.py"
_spec = importlib.util.spec_from_file_location("run_eval", SCRIPT_PATH)
run_eval = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(run_eval)


def _record(rid, category, expected, actual, latency=100.0, schema_ok=True):
    return {
        "id": rid,
        "query": f"query {rid}",
        "category": category,
        "expected": expected,
        "actual": actual,
        "latency_ms": latency,
        "schema_ok": schema_ok,
        "error": None,
    }


RECORDS = [
    # perfect match
    _record(1, "single_field", {"town": "BEDOK"}, {"town": "BEDOK"}, 100.0),
    # one wrong field, one right field
    _record(
        2,
        "multi_field",
        {"town": "TAMPINES", "flat_type": "4 ROOM", "max_price": 600000},
        {"town": "TAMPINES", "flat_type": "4 ROOM", "max_price": 650000},
        200.0,
    ),
    # extra field invented by the model
    _record(3, "ambiguous", {}, {"town": "BEDOK"}, 300.0),
    # parser rejected the query
    _record(4, "adversarial", {}, {}, 400.0, schema_ok=False),
    # exact match on an empty filter
    _record(5, "ambiguous", {}, {}, 500.0),
]


def test_normalise_filter_drops_none():
    assert run_eval.normalise_filter({"town": "BEDOK", "max_price": None}) == {
        "town": "BEDOK"
    }


def test_exact_match_ignores_unset_values():
    assert run_eval.is_exact_match({"town": "BEDOK"}, {"town": "BEDOK"})
    assert run_eval.is_exact_match({}, {})
    assert not run_eval.is_exact_match({"town": "BEDOK"}, {})
    assert not run_eval.is_exact_match(
        {"town": "BEDOK"}, {"town": "BEDOK", "max_price": 500000}
    )


def test_compare_fields_only_counts_mentioned_fields():
    compared = run_eval.compare_fields(
        {"town": "BEDOK", "max_price": 500000},
        {"town": "BEDOK", "max_price": 450000, "flat_type": "3 ROOM"},
    )
    assert compared == {
        "town": True,
        "max_price": False,
        "flat_type": False,
    }


def test_percentile_handles_empty_single_and_interpolation():
    assert run_eval.percentile([], 50) == 0.0
    assert run_eval.percentile([42.0], 95) == 42.0
    assert run_eval.percentile([10, 20, 30, 40], 50) == 25.0
    assert run_eval.percentile([10, 20, 30, 40], 0) == 10.0
    assert run_eval.percentile([10, 20, 30, 40], 100) == 40.0
    # nearest-rank style check on 20 samples: p95 sits near the top
    assert run_eval.percentile(list(range(1, 21)), 95) == pytest.approx(19.05)


def test_compute_metrics_overall_numbers():
    metrics = run_eval.compute_metrics(RECORDS)

    assert metrics["total"] == 5
    assert metrics["exact_match_count"] == 2  # ids 1 and 5
    assert metrics["exact_match"] == 0.4
    assert metrics["schema_compliance"] == 0.8  # id 4 failed
    assert metrics["latency_ms"]["p50"] == 300.0
    assert metrics["latency_ms"]["mean"] == 300.0


def test_compute_metrics_field_accuracy():
    metrics = run_eval.compute_metrics(RECORDS)

    fields = metrics["field_accuracy"]
    assert fields["town"] == {"correct": 2, "total": 3, "accuracy": 0.6667}
    assert fields["max_price"] == {"correct": 0, "total": 1, "accuracy": 0.0}
    assert fields["flat_type"] == {"correct": 1, "total": 1, "accuracy": 1.0}
    assert metrics["mean_field_accuracy"] == pytest.approx(
        (0.6667 + 0.0 + 1.0) / 3, abs=1e-4
    )


def test_compute_metrics_per_category():
    metrics = run_eval.compute_metrics(RECORDS)

    ambiguous = metrics["by_category"]["ambiguous"]
    assert ambiguous["total"] == 2
    assert ambiguous["exact_match"] == 0.5  # id 5 matched, id 3 invented a town

    adversarial = metrics["by_category"]["adversarial"]
    assert adversarial["schema_compliance"] == 0.0

    single = metrics["by_category"]["single_field"]
    assert single["exact_match"] == 1.0
    assert single["latency_ms"]["p95"] == 100.0


def test_compute_metrics_handles_no_records():
    metrics = run_eval.compute_metrics([])

    assert metrics["total"] == 0
    assert metrics["exact_match"] == 0.0
    assert metrics["field_accuracy"] == {}
    assert metrics["latency_ms"] == {"p50": 0.0, "p95": 0.0, "mean": 0.0}
    assert metrics["by_category"] == {}


def test_render_markdown_lists_failures():
    metrics = run_eval.compute_metrics(RECORDS)

    markdown = run_eval.render_markdown(metrics, "2026-11-01T00:00:00+00:00", 5)

    assert "# Evaluation" in markdown
    assert "Exact match | 40.0% (2/5)" in markdown
    assert "## Per field" in markdown
    assert "| town | 2 | 3 | 66.7% |" in markdown
    assert "## Per category" in markdown
    assert "## Failed queries" in markdown
    assert "| 2 | multi_field | query 2 |" in markdown
    assert "650000" in markdown


def test_write_outputs_creates_json_and_markdown(tmp_path):
    metrics = run_eval.compute_metrics(RECORDS)

    json_path, md_path = run_eval.write_outputs(
        metrics,
        "2026-11-01T00:00:00+00:00",
        results_dir=tmp_path / "results",
        markdown_path=tmp_path / "docs" / "evaluation.md",
    )

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["total"] == 5
    assert json_path.name.endswith(".json")
    assert "# Evaluation" in md_path.read_text(encoding="utf-8")


def test_run_queries_throttles_and_records_errors():
    queries = [
        {"id": 1, "query": "a", "expected": {"town": "BEDOK"}, "category": "x"},
        {"id": 2, "query": "b", "expected": {}, "category": "y"},
    ]
    sleeps: list[float] = []

    class FakeFilter:
        def __init__(self, payload):
            self.payload = payload

        def model_dump(self, exclude_none=True):
            return self.payload

    def fake_parser(query):
        if query == "b":
            from app.services.llm_parser import InvalidFilterError

            raise InvalidFilterError("nope")
        return FakeFilter({"town": "BEDOK", "max_price": None})

    records = run_eval.run_queries(
        queries, parser=fake_parser, delay=2.0, sleep=sleeps.append
    )

    assert sleeps == [2.0]  # only between calls, not before the first
    assert records[0]["actual"] == {"town": "BEDOK"}
    assert records[0]["schema_ok"] is True
    assert records[1]["actual"] == {}
    assert records[1]["schema_ok"] is False
    assert records[1]["error"] == "nope"


def test_load_queries_skips_blank_lines(tmp_path):
    path = tmp_path / "q.jsonl"
    path.write_text(
        '{"id": 1, "query": "a", "expected": {}, "category": "x"}\n\n', encoding="utf-8"
    )

    assert len(run_eval.load_queries(path)) == 1
