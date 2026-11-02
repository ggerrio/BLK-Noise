"""API tests for app/main.py with the LLM and model mocked."""

from __future__ import annotations

import json
import sqlite3

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import main
from app.config import settings
from app.schemas import PropertyFilter
from app.services.llm_parser import InvalidFilterError, LLMTimeoutError

SCHEMA = """
CREATE TABLE listings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    month TEXT NOT NULL,
    town TEXT NOT NULL,
    flat_type TEXT NOT NULL,
    block TEXT,
    street_name TEXT,
    storey_min INTEGER,
    storey_max INTEGER,
    floor_area_sqm REAL,
    flat_model TEXT,
    lease_commence_year INTEGER,
    remaining_lease_years REAL,
    resale_price INTEGER NOT NULL,
    summary_text TEXT,
    embedding BLOB
);
CREATE TABLE audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_utc TEXT NOT NULL,
    request_id TEXT NOT NULL,
    masked_query TEXT NOT NULL,
    parsed_filter TEXT,
    result_count INTEGER,
    latency_ms INTEGER,
    llm_model TEXT,
    status TEXT NOT NULL,
    error_detail TEXT
);
"""


def _blob(values) -> bytes:
    return np.asarray(values, dtype=np.float32).tobytes()


def _seed(db_path):
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    conn.executemany(
        "INSERT INTO listings (month, town, flat_type, block, street_name,"
        " storey_min, storey_max, floor_area_sqm, flat_model,"
        " lease_commence_year, remaining_lease_years, resale_price,"
        " summary_text, embedding)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        [
            ("2026-01", "TAMPINES", "4 ROOM", "301", "TAMPINES ST 21",
             10, 12, 92.0, "MODEL A", 1990, 70.0, 580000,
             "4 ROOM in TAMPINES, 92 sqm", _blob((1.0, 0.0, 0.0))),
            ("2026-02", "TAMPINES", "5 ROOM", "302", "TAMPINES ST 22",
             13, 15, 110.0, "IMPROVED", 1995, 80.0, 680000,
             "5 ROOM in TAMPINES, 110 sqm", _blob((0.9, 0.1, 0.0))),
        ],
    )
    conn.commit()
    conn.close()


@pytest.fixture
def client(tmp_path, monkeypatch):
    db_path = tmp_path / "listings.db"
    _seed(db_path)
    monkeypatch.setattr(settings, "db_path", db_path)
    main.reset_rate_limits()
    yield TestClient(main.app)
    main.reset_rate_limits()


@pytest.fixture
def fake_llm(monkeypatch):
    """Stub the parser, ranker and explainer so no model or network is used."""

    def install(parsed=None, ranker_impl=None, explanation="Two listings match."):
        calls = {"parse": []}

        def fake_parse(query):
            calls["parse"].append(query)
            return parsed if parsed is not None else PropertyFilter()

        def fake_rank(query, rows, top_n=5):
            calls.setdefault("rank_query", query)
            if ranker_impl is not None:
                return ranker_impl(rows, top_n)
            return rows[:top_n]

        real_explain = main.explainer.explain

        def fake_explain(query, rows):
            # Empty results never reach the LLM, so keep the real message.
            return real_explain(query, rows) if not rows else explanation

        monkeypatch.setattr(main, "parse_query", fake_parse)
        monkeypatch.setattr(main.ranker, "rank", fake_rank)
        monkeypatch.setattr(main.explainer, "explain", fake_explain)
        return calls

    return install


def _audit_rows(db_path):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(row) for row in conn.execute("SELECT * FROM audit_log")]
    finally:
        conn.close()


# --- health ---------------------------------------------------------------

def test_health_ok(client):
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["listings"] == 2


def test_health_fails_without_database(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "db_path", tmp_path / "missing.db")
    response = TestClient(main.app).get("/health")

    assert response.status_code == 503


# --- search happy path ----------------------------------------------------

def test_search_returns_ranked_results(client, fake_llm, tmp_path, monkeypatch):
    fake_llm(parsed=PropertyFilter(town="TAMPINES"), explanation="Two TAMPINES flats.")

    response = client.post("/search", json={"query": "4 room in Tampines", "top_n": 2})

    assert response.status_code == 200
    body = response.json()
    assert body["parsed_filter"]["town"] == "TAMPINES"
    assert len(body["results"]) == 2
    assert body["explanation"] == "Two TAMPINES flats."
    assert body["request_id"]
    assert body["latency_ms"] >= 0
    assert "embedding" not in body["results"][0]

    audit = _audit_rows(settings.db_path)
    assert len(audit) == 1
    assert audit[0]["masked_query"] == "4 room in Tampines"
    assert audit[0]["result_count"] == 2
    assert audit[0]["status"] == "ok"
    assert audit[0]["request_id"] == body["request_id"]
    assert audit[0]["llm_model"] == settings.gemini_model
    assert json.loads(audit[0]["parsed_filter"])["town"] == "TAMPINES"
    assert audit[0]["ts_utc"]


def test_search_masks_pii_before_llm_and_audit(client, fake_llm, monkeypatch):
    calls = fake_llm()

    client.post(
        "/search",
        json={"query": "Call 9123 4567 about a 4 room flat, email me@x.com"},
    )

    parsed_input = calls["parse"][0]
    assert parsed_input == "Call [PHONE] about a 4 room flat, email [EMAIL]"

    audit = _audit_rows(settings.db_path)
    assert "[PHONE]" in audit[0]["masked_query"]
    assert "9123" not in audit[0]["masked_query"]
    assert "me@x.com" not in audit[0]["masked_query"]


def test_search_uses_top_n(client, fake_llm):
    fake_llm()

    response = client.post("/search", json={"query": "flats", "top_n": 1})

    assert response.status_code == 200
    assert len(response.json()["results"]) == 1


# --- empty results --------------------------------------------------------

def test_search_without_matches_returns_200_and_message(client, fake_llm):
    fake_llm(parsed=PropertyFilter(town="TAMPINES", max_price=100))

    response = client.post("/search", json={"query": "3 room in Bedok"})

    assert response.status_code == 200
    body = response.json()
    assert body["results"] == []
    assert "No listings matched" in body["explanation"]

    audit = _audit_rows(settings.db_path)
    assert audit[0]["status"] == "no_results"
    assert audit[0]["result_count"] == 0


# --- errors ---------------------------------------------------------------

def test_invalid_filter_returns_422(client, fake_llm, monkeypatch):
    def boom(_query):
        raise InvalidFilterError("Gemini returned an invalid filter after retry")

    monkeypatch.setattr(main, "parse_query", boom)

    response = client.post("/search", json={"query": "nonsense filters"})

    assert response.status_code == 422
    assert "could not understand" in response.json()["detail"].lower()
    audit = _audit_rows(settings.db_path)[0]
    assert audit["status"] == "invalid_filter"
    assert "invalid filter" in audit["error_detail"]


def test_llm_timeout_returns_503(client, monkeypatch):
    def boom(_query):
        raise LLMTimeoutError("timed out")

    monkeypatch.setattr(main, "parse_query", boom)

    response = client.post("/search", json={"query": "4 room Tampines"})

    assert response.status_code == 503
    assert "took too long" in response.json()["detail"]
    assert _audit_rows(settings.db_path)[0]["status"] == "llm_error"


def test_missing_database_returns_503(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "db_path", tmp_path / "nope.db")
    main.reset_rate_limits()
    monkeypatch.setattr(main, "parse_query", lambda q: PropertyFilter())

    response = TestClient(main.app).post("/search", json={"query": "flat"})

    assert response.status_code == 503
    assert "not ready" in response.json()["detail"]


# --- input limits and rate limiting ---------------------------------------

def test_query_longer_than_300_chars_is_rejected(client, fake_llm):
    fake_llm()

    response = client.post("/search", json={"query": "a" * 301})

    assert response.status_code == 422


def test_rate_limit_returns_429(client, fake_llm, monkeypatch):
    fake_llm()
    monkeypatch.setattr(main, "RATE_LIMIT_MAX_REQUESTS", 2)

    assert client.post("/search", json={"query": "one"}).status_code == 200
    assert client.post("/search", json={"query": "two"}).status_code == 200

    blocked = client.post("/search", json={"query": "three"})

    assert blocked.status_code == 429
    assert "Too many requests" in blocked.json()["detail"]
    assert blocked.headers["Retry-After"] == "60"
