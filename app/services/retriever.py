"""Query the `listings` table for rows matching a PropertyFilter.

All values are passed as bound parameters. Only whitelisted SQL fragments
(column names and ORDER BY) are interpolated, never user input.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from app.config import settings
from app.schemas import PropertyFilter

# sort_by value -> ORDER BY. Keys mirror the SortBy literal in schemas.py.
SORT_OPTIONS: dict[str, str] = {
    "price_asc": "resale_price ASC, id ASC",
    "price_desc": "resale_price DESC, id ASC",
    "area_desc": "floor_area_sqm DESC, id ASC",
    "lease_desc": "remaining_lease_years DESC, id ASC",
}

DEFAULT_SORT = "price_asc"

SELECT_COLUMNS = (
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
)


def _text(value: str) -> str:
    """Normalise a text filter to the stored form (upper case, single spaces)."""
    return " ".join(str(value).split()).upper()


def resolve_sort(sort_by: str | None) -> str:
    """Validate a sort value against the whitelist.

    Args:
        sort_by: Requested ordering, or None for the default.

    Returns:
        The ORDER BY fragment for a whitelisted key.

    Raises:
        ValueError: When the key is not whitelisted.
    """
    key = sort_by or DEFAULT_SORT
    if key not in SORT_OPTIONS:
        allowed = ", ".join(sorted(SORT_OPTIONS))
        raise ValueError(f"Unsupported sort_by {key!r}. Allowed: {allowed}")
    return SORT_OPTIONS[key]


def build_query(
    flt: PropertyFilter,
    limit: int,
    sort_by: str | None = None,
) -> tuple[str, list[Any]]:
    """Build the SELECT statement and its bound parameters."""
    order_by = resolve_sort(sort_by or flt.sort_by)

    limit = int(limit)
    if limit < 1:
        raise ValueError("limit must be at least 1")

    where: list[str] = []
    params: list[Any] = []

    if flt.town:
        where.append("town = ?")
        params.append(_text(flt.town))

    if flt.flat_type:
        where.append("flat_type = ?")
        params.append(_text(flt.flat_type))

    if flt.min_price is not None:
        where.append("resale_price >= ?")
        params.append(flt.min_price)

    if flt.max_price is not None:
        where.append("resale_price <= ?")
        params.append(flt.max_price)

    if flt.min_area_sqm is not None:
        where.append("floor_area_sqm >= ?")
        params.append(flt.min_area_sqm)

    # Storeys are stored as a band, so a requested minimum floor matches any
    # band reaching that high and a requested maximum matches any band starting
    # at or below it.
    if flt.storey_min is not None:
        where.append("storey_max >= ?")
        params.append(flt.storey_min)

    if flt.storey_max is not None:
        where.append("storey_min <= ?")
        params.append(flt.storey_max)

    if flt.min_remaining_lease_years is not None:
        where.append("remaining_lease_years >= ?")
        params.append(flt.min_remaining_lease_years)

    sql = f"SELECT {', '.join(SELECT_COLUMNS)} FROM listings"  # noqa: S608 - literal columns
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += f" ORDER BY {order_by} LIMIT ?"
    params.append(limit)

    return sql, params


def retrieve(
    filter: PropertyFilter,  # noqa: A002 - name required by the public API
    limit: int,
    sort_by: str | None = None,
    conn: sqlite3.Connection | None = None,
) -> list[dict[str, Any]]:
    """Return matching listings as dicts, or an empty list when none match.

    Args:
        filter: Validated filters to apply. Its `sort_by` is used unless
            `sort_by` is passed explicitly.
        limit: Maximum number of rows.
        sort_by: Whitelisted ordering key overriding the filter's own.
        conn: Optional open connection (e.g. in-memory in tests). Otherwise the
            database at `settings.db_path` is opened read-only.

    Returns:
        Matching rows as dicts, empty when nothing matches.
    """
    sql, params = build_query(filter, limit, sort_by)

    own_connection = conn is None
    if own_connection:
        db_path = Path(settings.db_path)
        if not db_path.exists():
            raise FileNotFoundError(f"Database not found: {db_path}")
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)

    try:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(sql, params)
        return [dict(row) for row in cursor.fetchall()]
    finally:
        if own_connection:
            conn.close()
