"""Tests for scripts/load_data.py parsing and loading behaviour."""

from __future__ import annotations

import importlib.util
import math
import sqlite3
from pathlib import Path

import pandas as pd
import pytest

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "load_data.py"
_spec = importlib.util.spec_from_file_location("load_data", SCRIPT_PATH)
load_data = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(load_data)


# --- parse_storey_range ---------------------------------------------------

@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("10 TO 12", (10, 12)),
        (" 01 to 03 ", (1, 3)),
        ("12 TO 10", (10, 12)),
        ("07 TO 07", (7, 7)),
        ("", (None, None)),
        ("N/A", (None, None)),
        (None, (None, None)),
        (float("nan"), (None, None)),
    ],
)
def test_parse_storey_range(raw, expected):
    assert load_data.parse_storey_range(raw) == expected


# --- parse_remaining_lease ------------------------------------------------

@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("61 years 06 months", 61.5),
        ("45 years", 45.0),
        ("07 months", 0.58),
        ("99 years 11 months", 99.92),
        ("61", 61.0),
        (62, 62.0),
        ("", None),
        ("unknown", None),
        (None, None),
        (float("nan"), None),
    ],
)
def test_parse_remaining_lease(raw, expected):
    assert load_data.parse_remaining_lease(raw) == expected


# --- normalize_text -------------------------------------------------------

@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("  ang mo kio ", "ANG MO KIO"),
        ("BEDOK", "BEDOK"),
        ("4  room  flat", "4 ROOM FLAT"),
        ("", ""),
        (None, ""),
        (float("nan"), ""),
    ],
)
def test_normalize_text(raw, expected):
    assert load_data.normalize_text(raw) == expected


# --- parse_month ----------------------------------------------------------

@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("2017-01", pd.Timestamp("2017-01-01")),
        ("2017-1", pd.Timestamp("2017-01-01")),
        ("2020-12", pd.Timestamp("2020-12-01")),
        ("2017-01-15", pd.Timestamp("2017-01-01")),
        ("2017-13", None),
        ("", None),
        (None, None),
    ],
)
def test_parse_month(raw, expected):
    assert load_data.parse_month(raw) == expected


# --- filter_recent_months -------------------------------------------------

def test_filter_recent_months_keeps_only_window():
    months = pd.date_range("2020-01-01", "2021-12-01", freq="MS")
    df = pd.DataFrame({"month": months.tolist() + [pd.Timestamp("2019-12-01")]})

    out = load_data.filter_recent_months(df, months=24)

    assert len(out) == 24
    assert out["month"].min() == pd.Timestamp("2020-01-01")
    assert out["month"].max() == pd.Timestamp("2021-12-01")


def test_filter_recent_months_empty_frame():
    df = pd.DataFrame({"month": pd.Series([], dtype="datetime64[ns]")})
    assert load_data.filter_recent_months(df, months=24).empty


def test_filter_recent_months_rejects_zero():
    with pytest.raises(ValueError):
        load_data.filter_recent_months(pd.DataFrame({"month": []}), months=0)


# --- build_summary_text ---------------------------------------------------

def test_build_summary_text_matches_embedding_format():
    summary = load_data.build_summary_text(
        {
            "town": "TAMPINES",
            "flat_type": "4 ROOM",
            "floor_area_sqm": 92.0,
            "storey_min": 10,
            "storey_max": 12,
            "remaining_lease_years": 70.0,
            "resale_price": 580000,
        }
    )
    assert summary == "4 ROOM in TAMPINES, 92 sqm, storey 10-12, 70 years lease left, S$580000"


# --- build_listings -------------------------------------------------------

RAW_ROWS = [
    {
        "month": "2021-12",
        "town": " ang mo kio ",
        "flat_type": "4 room",
        "block": "123",
        "street_name": "Ang Mo Kio Ave 1",
        "storey_range": "10 TO 12",
        "floor_area_sqm": "93",
        "flat_model": "Improved",
        "lease_commence_date": "1980",
        "remaining_lease": "61 years 06 months",
        "resale_price": "450000",
    },
    {
        "month": "2021-11",
        "town": "BEDOK",
        "flat_type": "3 ROOM",
        "block": "456",
        "street_name": "Bedok North Rd",
        "storey_range": "",
        "floor_area_sqm": "67",
        "flat_model": "New Generation",
        "lease_commence_date": "1985",
        "remaining_lease": "",
        "resale_price": "320000",
    },
    {
        "month": "not-a-month",
        "town": "TAMPINES",
        "flat_type": "5 ROOM",
        "block": "789",
        "street_name": "Tampines St 21",
        "storey_range": "04 TO 06",
        "floor_area_sqm": "110",
        "flat_model": "Improved",
        "lease_commence_date": "1990",
        "remaining_lease": "70 years",
        "resale_price": "600000",
    },
]


def _raw_frame() -> pd.DataFrame:
    return pd.DataFrame(RAW_ROWS)


def test_build_listings_transforms_and_drops_invalid_rows():
    out = load_data.build_listings(_raw_frame())

    assert list(out.columns) == list(load_data.LISTINGS_COLUMNS)
    assert len(out) == 2  # row with an unparseable month is dropped

    first = out.iloc[0]
    assert first["town"] == "ANG MO KIO"
    assert first["flat_type"] == "4 ROOM"
    assert (first["storey_min"], first["storey_max"]) == (10, 12)
    assert first["remaining_lease_years"] == 61.5
    assert first["lease_commence_year"] == 1980
    assert first["month"] == pd.Timestamp("2021-12-01")
    assert first["resale_price"] == 450000
    assert first["summary_text"].startswith("4 ROOM in ANG MO KIO, 93 sqm")

    second = out.iloc[1]
    assert math.isnan(second["storey_min"])
    assert math.isnan(second["remaining_lease_years"])


def test_build_listings_requires_core_columns():
    with pytest.raises(ValueError, match="town"):
        load_data.build_listings(pd.DataFrame({"month": ["2021-01"]}))


# --- write_listings / load: idempotency -----------------------------------

def test_write_listings_is_idempotent(tmp_path):
    db = tmp_path / "test.db"
    listings = load_data.build_listings(_raw_frame())

    assert load_data.write_listings(listings, db) == 2
    assert load_data.write_listings(listings, db) == 2

    with sqlite3.connect(db) as conn:
        (rows,) = conn.execute("SELECT COUNT(*) FROM listings").fetchone()
        (distinct,) = conn.execute(
            "SELECT COUNT(*) FROM (SELECT DISTINCT month, town, block, resale_price"
            " FROM listings)"
        ).fetchone()
        indexes = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index'"
            )
        }
    assert rows == 2
    assert distinct == 2
    assert {"idx_listings_town_type", "idx_listings_price"} <= indexes


def test_write_listings_uses_architecture_schema(tmp_path):
    db = tmp_path / "test.db"
    load_data.write_listings(load_data.build_listings(_raw_frame()), db)

    with sqlite3.connect(db) as conn:
        columns = [row[1] for row in conn.execute("PRAGMA table_info(listings)")]

    assert columns == [
        "id",
        "month",
        "town",
        "flat_type",
        "block",
        "street_name",
        "storey_min",
        "storey_max",
        "floor_area_sqm",
        "flat_model",
        "lease_commence_year",
        "remaining_lease_years",
        "resale_price",
        "summary_text",
    ]
    assert "embedding" not in columns


def test_load_reads_csv_and_applies_month_window(tmp_path):
    csv_path = tmp_path / "resale.csv"
    rows = []
    for index in range(30):
        month = (
            pd.Timestamp("2021-01-01") + pd.DateOffset(months=index)
        ).strftime("%Y-%m")
        rows.append({**RAW_ROWS[0], "month": month, "block": str(100 + index)})
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    db = tmp_path / "test.db"
    count = load_data.load(csv_path, db, months=24)

    assert count == 24
    with sqlite3.connect(db) as conn:
        (lowest,) = conn.execute("SELECT MIN(month) FROM listings").fetchone()
    assert lowest == "2021-07"
