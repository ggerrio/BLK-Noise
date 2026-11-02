"""Tests for app/services/retriever.py using an in-memory SQLite database."""

from __future__ import annotations

import sqlite3

import pytest

from app import schemas
from app.schemas import PropertyFilter
from app.services import retriever

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
    summary_text TEXT
);
"""

ROWS = [
    # month, town, flat_type, storey_min, storey_max, area, lease_years, price
    ("2026-01", "ANG MO KIO", "3 ROOM", 4, 6, 67.0, 45.0, 320000),
    ("2026-01", "ANG MO KIO", "4 ROOM", 10, 12, 93.0, 61.0, 450000),
    ("2026-02", "BEDOK", "3 ROOM", 1, 3, 60.0, 40.0, 280000),
    ("2026-02", "BEDOK", "5 ROOM", 7, 9, 110.0, 55.0, 620000),
    ("2026-03", "TAMPINES", "4 ROOM", 10, 12, 95.0, 70.5, 520000),
    ("2026-03", "TAMPINES", "5 ROOM", 13, 15, 120.0, 78.0, 750000),
    ("2026-04", "WOODLANDS", "2 ROOM", 4, 6, 45.0, 82.0, 210000),
    ("2026-04", "WOODLANDS", "4 ROOM", 7, 9, 92.0, 88.5, 400000),
    ("2026-05", "TOA PAYOH", "3 ROOM", 7, 9, 68.0, 35.0, 350000),
    ("2026-05", "TOA PAYOH", "4 ROOM", 10, 12, 90.0, 37.5, 500000),
]


@pytest.fixture(autouse=True)
def _isolate_town_validation(tmp_path, monkeypatch):
    """Keep PropertyFilter's town check away from any real database."""
    monkeypatch.setattr(schemas.settings, "db_path", tmp_path / "missing.db")
    schemas._read_towns.cache_clear()
    yield
    schemas._read_towns.cache_clear()


@pytest.fixture
def conn():
    connection = sqlite3.connect(":memory:")
    connection.execute(SCHEMA)
    connection.executemany(
        "INSERT INTO listings (month, town, flat_type, block, street_name,"
        " storey_min, storey_max, floor_area_sqm, flat_model,"
        " lease_commence_year, remaining_lease_years, resale_price, summary_text)"
        " VALUES (?, ?, ?, '1', 'TEST ST', ?, ?, ?, 'IMPROVED', 1980, ?, ?, NULL)",
        [
            (month, town, flat_type, lo, hi, area, lease, price)
            for month, town, flat_type, lo, hi, area, lease, price in ROWS
        ],
    )
    connection.commit()
    yield connection
    connection.close()


def _towns(rows):
    return [row["town"] for row in rows]


# --- one test per filter field -------------------------------------------

def test_town_filter(conn):
    rows = retriever.retrieve(PropertyFilter(town="tampines"), limit=10, conn=conn)
    assert _towns(rows) == ["TAMPINES", "TAMPINES"]


def test_flat_type_filter(conn):
    rows = retriever.retrieve(PropertyFilter(flat_type="3 ROOM"), limit=10, conn=conn)
    assert len(rows) == 3
    assert {row["flat_type"] for row in rows} == {"3 ROOM"}


def test_min_price_filter(conn):
    rows = retriever.retrieve(PropertyFilter(min_price=500000), limit=10, conn=conn)
    assert [row["resale_price"] for row in rows] == [500000, 520000, 620000, 750000]


def test_max_price_filter(conn):
    rows = retriever.retrieve(PropertyFilter(max_price=320000), limit=10, conn=conn)
    assert [row["resale_price"] for row in rows] == [210000, 280000, 320000]


def test_min_area_sqm_filter(conn):
    rows = retriever.retrieve(PropertyFilter(min_area_sqm=100), limit=10, conn=conn)
    assert [row["floor_area_sqm"] for row in rows] == [110.0, 120.0]


def test_storey_min_filter_matches_overlapping_bands(conn):
    rows = retriever.retrieve(PropertyFilter(storey_min=10), limit=10, conn=conn)
    assert len(rows) == 4
    assert all(row["storey_max"] >= 10 for row in rows)


def test_storey_max_filter_matches_overlapping_bands(conn):
    rows = retriever.retrieve(PropertyFilter(storey_max=6), limit=10, conn=conn)
    assert len(rows) == 3
    assert all(row["storey_min"] <= 6 for row in rows)


def test_min_remaining_lease_years_filter(conn):
    rows = retriever.retrieve(
        PropertyFilter(min_remaining_lease_years=70), limit=10, conn=conn
    )
    assert len(rows) == 4
    assert all(row["remaining_lease_years"] >= 70 for row in rows)


# --- combinations and edge cases -----------------------------------------

def test_combined_filters(conn):
    rows = retriever.retrieve(
        PropertyFilter(town="TAMPINES", max_price=700000, storey_min=10),
        limit=10,
        conn=conn,
    )
    assert len(rows) == 1
    assert rows[0]["resale_price"] == 520000


def test_combined_filter_with_four_fields(conn):
    rows = retriever.retrieve(
        PropertyFilter(
            town="WOODLANDS", flat_type="4 ROOM", min_area_sqm=90, min_price=300000
        ),
        limit=10,
        conn=conn,
    )
    assert len(rows) == 1
    assert rows[0]["resale_price"] == 400000


def test_no_matches_returns_empty_list(conn):
    rows = retriever.retrieve(
        PropertyFilter(town="BEDOK", min_price=900000), limit=10, conn=conn
    )
    assert rows == []


def test_empty_filter_returns_rows_up_to_limit(conn):
    assert len(retriever.retrieve(PropertyFilter(), limit=3, conn=conn)) == 3
    assert len(retriever.retrieve(PropertyFilter(), limit=50, conn=conn)) == 10


def test_limit_is_bound_and_validated(conn):
    assert len(retriever.retrieve(PropertyFilter(), limit=4, conn=conn)) == 4
    with pytest.raises(ValueError, match="limit"):
        retriever.retrieve(PropertyFilter(), limit=0, conn=conn)


# --- sorting --------------------------------------------------------------

def test_sort_from_filter(conn):
    rows = retriever.retrieve(
        PropertyFilter(sort_by="price_desc"), limit=3, conn=conn
    )
    assert [row["resale_price"] for row in rows] == [750000, 620000, 520000]


def test_sort_whitelist_area_and_lease(conn):
    areas = retriever.retrieve(PropertyFilter(), limit=2, sort_by="area_desc", conn=conn)
    assert [row["floor_area_sqm"] for row in areas] == [120.0, 110.0]

    leases = retriever.retrieve(
        PropertyFilter(), limit=2, sort_by="lease_desc", conn=conn
    )
    assert [row["remaining_lease_years"] for row in leases] == [88.5, 82.0]


def test_unknown_sort_is_rejected(conn):
    with pytest.raises(ValueError, match="Unsupported sort_by"):
        retriever.retrieve(PropertyFilter(), limit=5, sort_by="price; DROP", conn=conn)


# --- injection ------------------------------------------------------------

def test_dangerous_town_cannot_break_the_query(conn):
    evil = "'; DROP TABLE listings; --"
    flt = PropertyFilter.model_construct(town=evil)

    assert retriever.retrieve(flt, limit=10, conn=conn) == []

    (count,) = conn.execute("SELECT COUNT(*) FROM listings").fetchone()
    assert count == 10


def test_dangerous_flat_type_is_treated_as_data(conn):
    flt = PropertyFilter.model_construct(flat_type="3 ROOM' OR '1'='1")
    assert retriever.retrieve(flt, limit=10, conn=conn) == []


def test_sql_and_params_use_placeholders():
    sql, params = retriever.build_query(
        PropertyFilter.model_construct(town="BEDOK", max_price=500000), limit=7
    )
    assert "DROP" not in sql
    assert "BEDOK" not in sql  # values are never interpolated
    assert sql.count("?") == len(params) == 3
    assert params == ["BEDOK", 500000, 7]
