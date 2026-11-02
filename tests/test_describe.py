"""Tests for POST /describe/{listing_id} and the describer fallback."""

from __future__ import annotations

import sqlite3
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app import main
from app.config import settings
from app.services import describer
from app.services.llm_parser import LLMTimeoutError
from app.services.var_check import check_description

SCHEMA = """
CREATE TABLE listings (
    id INTEGER PRIMARY KEY,
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
"""

ROW = {
    "id": 1,
    "month": "2026-01",
    "town": "TAMPINES",
    "flat_type": "4 ROOM",
    "block": "301",
    "street_name": "TAMPINES ST 21",
    "storey_min": 10,
    "storey_max": 12,
    "floor_area_sqm": 93.0,
    "flat_model": "MODEL A",
    "lease_commence_year": 1990,
    "remaining_lease_years": 61.5,
    "resale_price": 450000,
    "summary_text": "4 ROOM in TAMPINES, 93 sqm",
    "embedding": None,
}

GOOD_TEXT = (
    "This 4 ROOM flat in TAMPINES sits on storeys 10 to 12 with 93 sqm. "
    "It has 61.5 years of lease left and costs S$450,000."
)
BAD_TEXT = "This 4 ROOM flat in BEDOK costs S$900,000 with 120 sqm."


@pytest.fixture
def client(tmp_path, monkeypatch):
    db_path = tmp_path / "listings.db"
    conn = sqlite3.connect(db_path)
    conn.execute(SCHEMA)
    conn.execute(
        "INSERT INTO listings (id, month, town, flat_type, block, street_name,"
        " storey_min, storey_max, floor_area_sqm, flat_model, lease_commence_year,"
        " remaining_lease_years, resale_price, summary_text) VALUES"
        " (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        tuple(ROW[key] for key in (
            "id", "month", "town", "flat_type", "block", "street_name",
            "storey_min", "storey_max", "floor_area_sqm", "flat_model",
            "lease_commence_year", "remaining_lease_years", "resale_price",
            "summary_text",
        )),
    )
    conn.commit()
    conn.close()

    monkeypatch.setattr(settings, "db_path", db_path)
    main.reset_rate_limits()
    yield TestClient(main.app)
    main.reset_rate_limits()


def _outcome(text: str) -> describer.DescribeOutcome:
    check = check_description(text, ROW)
    return describer.DescribeOutcome(
        description=text, check=check, fallback_used=not check.passed
    )


def test_describe_returns_verified_text(client, monkeypatch):
    monkeypatch.setattr(
        main.describer, "describe_listing", lambda row: _outcome(GOOD_TEXT)
    )

    response = client.post("/describe/1")

    assert response.status_code == 200
    body = response.json()
    assert body["listing_id"] == 1
    assert body["description"] == GOOD_TEXT
    assert body["check"]["passed"] is True
    assert body["fallback_used"] is False
    assert body["request_id"]
    assert body["latency_ms"] >= 0


def test_describe_reports_failed_check(client, monkeypatch):
    monkeypatch.setattr(
        main.describer, "describe_listing", lambda row: _outcome(BAD_TEXT)
    )

    response = client.post("/describe/1")

    assert response.status_code == 200
    body = response.json()
    assert body["check"]["passed"] is False
    assert body["check"]["errors"]
    assert body["fallback_used"] is True


def test_describe_unknown_id_returns_404(client):
    assert client.post("/describe/999").status_code == 404


def test_describe_timeout_returns_503(client, monkeypatch):
    def boom(_row):
        raise LLMTimeoutError("slow")

    monkeypatch.setattr(main.describer, "describe_listing", boom)

    assert client.post("/describe/1").status_code == 503


# --- describer fallback ---------------------------------------------------

def test_describer_uses_fallback_when_claims_are_wrong(monkeypatch):
    class FakeClient:
        interactions = SimpleNamespace(
            create=lambda **kwargs: SimpleNamespace(output_text=BAD_TEXT)
        )

    outcome = describer.describe_listing(ROW, client=FakeClient())

    assert outcome.fallback_used is True
    assert outcome.check.passed is False
    assert outcome.description == describer.fallback_description(ROW)
    assert "BEDOK" not in outcome.description
    assert "900000" not in outcome.description


def test_describer_keeps_good_text(monkeypatch):
    class FakeClient:
        interactions = SimpleNamespace(
            create=lambda **kwargs: SimpleNamespace(output_text=GOOD_TEXT)
        )

    outcome = describer.describe_listing(ROW, client=FakeClient())

    assert outcome.fallback_used is False
    assert outcome.description == GOOD_TEXT


def test_describer_timeout_propagates():
    class TimeoutClient:
        def create(self, **kwargs):
            raise TimeoutError("deadline exceeded")

        interactions = SimpleNamespace(create=create)

    with pytest.raises(LLMTimeoutError):
        describer.describe_listing(ROW, client=TimeoutClient())
