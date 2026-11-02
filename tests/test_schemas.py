"""Tests for app/schemas.py validation behaviour."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from pydantic import ValidationError

from app import schemas

TOWNS = ("ANG MO KIO", "BEDOK", "TAMPINES", "TOA PAYOH", "WOODLANDS")


@pytest.fixture(autouse=True)
def town_db(tmp_path, monkeypatch) -> Path:
    """Point the schema layer at a temporary database with known towns."""
    db = tmp_path / "listings.db"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE listings (town TEXT, resale_price REAL)")
        conn.executemany(
            "INSERT INTO listings (town, resale_price) VALUES (?, ?)",
            [(town, 400000.0) for town in TOWNS],
        )

    schemas._read_towns.cache_clear()
    monkeypatch.setattr(schemas.settings, "db_path", db)
    yield db
    schemas._read_towns.cache_clear()


def test_valid_filter_normalises_and_keeps_values():
    f = schemas.PropertyFilter(
        town=" bedok ",
        flat_type="4 ROOM",
        min_price=300000,
        max_price=500000,
        min_area_sqm=90,
        storey_min=10,
        storey_max=12,
        min_remaining_lease_years=60,
        sort_by="price_asc",
    )

    assert f.town == "BEDOK"
    assert f.flat_type == "4 ROOM"
    assert f.min_price == 300000
    assert f.max_price == 500000
    assert f.min_area_sqm == 90
    assert (f.storey_min, f.storey_max) == (10, 12)
    assert f.min_remaining_lease_years == 60
    assert f.sort_by == "price_asc"


def test_min_price_greater_than_max_price_rejected():
    with pytest.raises(ValidationError, match="min_price"):
        schemas.PropertyFilter(min_price=600000, max_price=500000)


def test_storey_min_greater_than_storey_max_rejected():
    with pytest.raises(ValidationError, match="storey_min"):
        schemas.PropertyFilter(storey_min=12, storey_max=10)


def test_equal_bounds_are_allowed():
    f = schemas.PropertyFilter(
        min_price=400000, max_price=400000, storey_min=5, storey_max=5
    )
    assert f.max_price == 400000
    assert f.storey_max == 5


def test_town_typo_is_corrected():
    f = schemas.PropertyFilter(town="TAMPINESS")
    assert f.town == "TAMPINES"


def test_nonsense_town_rejected_with_clear_message():
    with pytest.raises(ValidationError) as excinfo:
        schemas.PropertyFilter(town="GIANT BANANA")

    message = str(excinfo.value)
    assert "Unknown town" in message
    assert "GIANT BANANA" in message
    assert "TAMPINES" in message  # lists the valid towns


@pytest.mark.parametrize(
    "payload",
    [
        {"min_price": -1},
        {"max_price": -500000},
        {"min_area_sqm": -10},
        {"min_remaining_lease_years": -0.5},
        {"storey_min": -2},
        {"storey_max": 0},
    ],
)
def test_negative_values_rejected(payload):
    with pytest.raises(ValidationError):
        schemas.PropertyFilter(**payload)


@pytest.mark.parametrize("flat_type", ["3-ROOM", "SHOPHOUSE", "room"])
def test_flat_type_outside_literal_is_rejected(flat_type):
    with pytest.raises(ValidationError):
        schemas.PropertyFilter(flat_type=flat_type)


@pytest.mark.parametrize(
    "flat_type",
    ["1 ROOM", "2 ROOM", "3 ROOM", "4 ROOM", "5 ROOM", "EXECUTIVE",
     "MULTI-GENERATION"],
)
def test_all_documented_flat_types_are_accepted(flat_type):
    assert schemas.PropertyFilter(flat_type=flat_type).flat_type == flat_type


@pytest.mark.parametrize("sort_by", ["price_asc", "price_desc", "area_desc",
                                     "lease_desc"])
def test_documented_sort_values_are_accepted(sort_by):
    assert schemas.PropertyFilter(sort_by=sort_by).sort_by == sort_by


def test_unknown_sort_value_is_rejected():
    with pytest.raises(ValidationError):
        schemas.PropertyFilter(sort_by="floor_area; DROP")


def test_unknown_field_rejected():
    with pytest.raises(ValidationError):
        schemas.PropertyFilter(town="BEDOK", postal_code="123456")


# --- SearchRequest --------------------------------------------------------

def test_search_request_defaults():
    req = schemas.SearchRequest(query="4 room flat in tampines under 500k")
    assert req.top_n == 5


def test_search_request_query_too_long_rejected():
    with pytest.raises(ValidationError):
        schemas.SearchRequest(query="a" * 301)


def test_search_request_query_max_length_accepted():
    req = schemas.SearchRequest(query="a" * 300)
    assert len(req.query) == 300


@pytest.mark.parametrize("top_n", [0, -3, 21])
def test_search_request_top_n_bounds(top_n):
    with pytest.raises(ValidationError):
        schemas.SearchRequest(query="3 room bedok", top_n=top_n)


def test_search_request_empty_query_rejected():
    with pytest.raises(ValidationError):
        schemas.SearchRequest(query="   ")


# --- SearchResponse -------------------------------------------------------

def test_search_response_builds_with_request_id():
    resp = schemas.SearchResponse(
        parsed_filter=schemas.PropertyFilter(town="BEDOK"),
        results=[{"town": "BEDOK", "resale_price": 400000}],
        explanation="Cheapest 3-room flats in Bedok.",
        latency_ms=42.5,
    )

    assert resp.request_id
    assert len(resp.request_id) == 32
    assert resp.latency_ms == 42.5
    assert resp.parsed_filter.town == "BEDOK"


def test_search_response_rejects_negative_latency():
    with pytest.raises(ValidationError):
        schemas.SearchResponse(
            parsed_filter=schemas.PropertyFilter(),
            latency_ms=-1,
        )


def test_town_validation_skipped_when_db_has_no_towns(tmp_path, monkeypatch):
    empty_db = tmp_path / "empty.db"
    with sqlite3.connect(empty_db) as conn:
        conn.execute("CREATE TABLE listings (town TEXT)")

    schemas._read_towns.cache_clear()
    monkeypatch.setattr(schemas.settings, "db_path", empty_db)

    f = schemas.PropertyFilter(town="nowhere")
    assert f.town == "NOWHERE"
