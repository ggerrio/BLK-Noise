"""Load HDB resale flat prices from a raw CSV into the SQLite `listings` table.

Usage:
    python scripts/load_data.py data/raw/resale.csv
    python scripts/load_data.py data/raw/resale.csv --db data/listings.db --months 24

The schema follows ARCHITECTURE.md section 4 (without the `embedding` column,
which `scripts/build_embeddings.py` adds later). The loader is idempotent: the
`listings` table is rebuilt inside one transaction, so re-running never
duplicates rows.
"""

from __future__ import annotations

import argparse
import math
import re
import sqlite3
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "listings.db"
DEFAULT_MONTHS = 24

REQUIRED_COLUMNS = ("month", "town", "flat_type", "resale_price")

LISTINGS_COLUMNS = (
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

LISTINGS_SCHEMA = """
CREATE TABLE listings (
    id                    INTEGER PRIMARY KEY,
    month                 TEXT NOT NULL,
    town                  TEXT NOT NULL,
    flat_type             TEXT NOT NULL,
    block                 TEXT,
    street_name           TEXT,
    storey_min            INTEGER,
    storey_max            INTEGER,
    floor_area_sqm        REAL,
    flat_model            TEXT,
    lease_commence_year   INTEGER,
    remaining_lease_years REAL,
    resale_price          INTEGER NOT NULL,
    summary_text          TEXT
);
CREATE INDEX idx_listings_town_type ON listings (town, flat_type);
CREATE INDEX idx_listings_price ON listings (resale_price);
"""

_MONTH_RE = re.compile(r"^\s*(\d{4})[-/](\d{1,2})")
_DIGITS_RE = re.compile(r"\d+(?:\.\d+)?")
_LEASE_YEAR_RE = re.compile(r"(\d+)\s*years?", re.IGNORECASE)
_LEASE_MONTH_RE = re.compile(r"(\d+)\s*months?", re.IGNORECASE)


def _is_missing(value: object) -> bool:
    """Return True for None, NaN and other null-ish values."""
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def normalize_text(value: object) -> str:
    """Trim whitespace, collapse inner spaces and upper-case a text value."""
    if _is_missing(value):
        return ""
    return " ".join(str(value).split()).upper()


def parse_storey_range(value: object) -> tuple[int | None, int | None]:
    """Parse "10 TO 12" into (10, 12).

    Returns (None, None) when the value is empty or has no number, and
    normalises a reversed range. A single number yields (n, n).
    """
    if _is_missing(value):
        return None, None

    text = str(value).strip()
    if not text:
        return None, None

    numbers = [int(float(number)) for number in _DIGITS_RE.findall(text)]
    if not numbers:
        return None, None
    if len(numbers) == 1:
        return numbers[0], numbers[0]

    low, high = numbers[0], numbers[1]
    if low > high:
        low, high = high, low
    return low, high


def parse_remaining_lease(value: object) -> float | None:
    """Parse "61 years 06 months" into 61.5.

    Also accepts "45 years", "07 months", a bare number, and returns None for
    empty or unparseable input.
    """
    if _is_missing(value):
        return None

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)

    text = str(value).strip()
    if not text:
        return None

    years_match = _LEASE_YEAR_RE.search(text)
    months_match = _LEASE_MONTH_RE.search(text)
    if years_match or months_match:
        years = int(years_match.group(1)) if years_match else 0
        months = int(months_match.group(1)) if months_match else 0
        return round(years + months / 12, 2)

    try:
        return float(text)
    except ValueError:
        return None


def parse_month(value: object) -> pd.Timestamp | None:
    """Parse "2017-01" (or a full date) into the first day of that month."""
    if _is_missing(value):
        return None

    text = str(value).strip()
    if not text:
        return None

    match = _MONTH_RE.match(text)
    if match:
        year, month = int(match.group(1)), int(match.group(2))
        if 1 <= month <= 12:
            return pd.Timestamp(year=year, month=month, day=1)

    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return None
    return pd.Timestamp(year=parsed.year, month=parsed.month, day=1)


def filter_recent_months(
    df: pd.DataFrame,
    months: int = DEFAULT_MONTHS,
    reference: pd.Timestamp | str | None = None,
) -> pd.DataFrame:
    """Keep rows within the last `months` months of the data.

    The reference point defaults to the newest month in the data, so a
    historical CSV still yields a full window.
    """
    if months <= 0:
        raise ValueError("months must be a positive integer")

    result = df[df["month"].notna()].copy()
    if result.empty:
        return result

    ref = pd.Timestamp(reference) if reference is not None else result["month"].max()
    cutoff = ref - pd.DateOffset(months=months - 1)
    return result[result["month"] >= cutoff].copy()


def build_summary_text(row: dict[str, object]) -> str:
    """Render a row as the text embedded for semantic search.

    Example: "4 ROOM in TAMPINES, 92 sqm, storey 10-12, 70 years lease left,
    S$580000". Kept identical to the format used by build_embeddings.py.
    """

    def number(value: object) -> str:
        return str(int(value)) if float(value).is_integer() else f"{float(value):g}"

    def clean(value: object) -> object | None:
        return None if _is_missing(value) else value

    parts: list[str] = []

    flat_type = str(row.get("flat_type") or "").strip()
    town = str(row.get("town") or "").strip()
    if flat_type and town:
        parts.append(f"{flat_type} in {town}")
    elif flat_type or town:
        parts.append(flat_type or town)

    area = clean(row.get("floor_area_sqm"))
    if area is not None:
        parts.append(f"{number(area)} sqm")

    storey_min = clean(row.get("storey_min"))
    storey_max = clean(row.get("storey_max"))
    if storey_min is not None and storey_max is not None:
        parts.append(f"storey {int(storey_min)}-{int(storey_max)}")
    elif storey_min is not None or storey_max is not None:
        parts.append(f"storey {int(storey_min or storey_max)}")

    lease = clean(row.get("remaining_lease_years"))
    if lease is not None:
        parts.append(f"{number(lease)} years lease left")

    price = clean(row.get("resale_price"))
    if price is not None:
        parts.append(f"S${int(round(float(price)))}")

    return ", ".join(parts)


def _normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    """Lower-case headers and turn spaces into underscores."""
    return df.rename(columns=lambda c: re.sub(r"\s+", "_", str(c).strip().lower()))


def _optional(df: pd.DataFrame, column: str) -> pd.Series:
    if column in df.columns:
        return df[column]
    return pd.Series([None] * len(df), index=df.index, dtype=object)


def _to_float(value: object) -> float | None:
    if _is_missing(value):
        return None
    try:
        return float(str(value).replace(",", "").strip())
    except ValueError:
        return None


def _to_int(value: object) -> int | None:
    number = _to_float(value)
    return int(round(number)) if number is not None else None


def build_listings(df: pd.DataFrame) -> pd.DataFrame:
    """Transform a raw resale-price frame into the `listings` column layout."""
    raw = _normalize_headers(df)

    missing = [column for column in REQUIRED_COLUMNS if column not in raw.columns]
    if missing:
        raise ValueError("CSV is missing required column(s): " + ", ".join(missing))

    out = pd.DataFrame(index=raw.index)
    out["month"] = raw["month"].map(parse_month)
    out["town"] = raw["town"].map(normalize_text)
    out["flat_type"] = raw["flat_type"].map(normalize_text)
    out["block"] = _optional(raw, "block").map(normalize_text)
    out["street_name"] = _optional(raw, "street_name").map(normalize_text)

    storey = _optional(raw, "storey_range").map(parse_storey_range)
    out["storey_min"] = [pair[0] for pair in storey]
    out["storey_max"] = [pair[1] for pair in storey]

    out["floor_area_sqm"] = _optional(raw, "floor_area_sqm").map(_to_float)
    out["flat_model"] = _optional(raw, "flat_model").map(normalize_text)
    out["lease_commence_year"] = _optional(raw, "lease_commence_date").map(_to_int)
    out["remaining_lease_years"] = _optional(raw, "remaining_lease").map(
        parse_remaining_lease
    )
    out["resale_price"] = raw["resale_price"].map(_to_float)

    out = out.dropna(subset=["month", "resale_price"])
    out = out[(out["town"] != "") & (out["flat_type"] != "")]

    out["month"] = pd.to_datetime(out["month"])
    out["resale_price"] = out["resale_price"].astype("int64")
    out = out.loc[:, list(LISTINGS_COLUMNS[:-1])].reset_index(drop=True)

    out["summary_text"] = [
        build_summary_text(record) for record in out.to_dict(orient="records")
    ]
    return out.loc[:, list(LISTINGS_COLUMNS)]


def write_listings(df: pd.DataFrame, db_path: Path | str = DEFAULT_DB_PATH) -> int:
    """Recreate the `listings` table and insert `df`. Returns rows written."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    frame = df.copy()
    frame["month"] = pd.to_datetime(frame["month"]).dt.strftime("%Y-%m")
    records = frame.astype(object).where(pd.notna(frame), None).values.tolist()

    placeholders = ", ".join("?" * len(LISTINGS_COLUMNS))
    insert_sql = (
        f"INSERT INTO listings ({', '.join(LISTINGS_COLUMNS)}) "
        f"VALUES ({placeholders})"
    )

    conn = sqlite3.connect(db_path)
    try:
        conn.execute("BEGIN")
        conn.execute("DROP TABLE IF EXISTS listings")
        conn.executescript(LISTINGS_SCHEMA)
        conn.executemany(insert_sql, records)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    return len(records)


def load(
    csv_path: Path | str,
    db_path: Path | str = DEFAULT_DB_PATH,
    months: int = DEFAULT_MONTHS,
) -> int:
    """Read a CSV, keep the last `months` months, and load them into SQLite."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    raw = pd.read_csv(csv_path, dtype=str)
    listings = filter_recent_months(build_listings(raw), months)
    return write_listings(listings, db_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Load HDB resale prices into SQLite.")
    parser.add_argument("csv_path", type=Path, help="Path to the raw CSV file")
    parser.add_argument(
        "--db", type=Path, default=DEFAULT_DB_PATH, help="SQLite database path"
    )
    parser.add_argument(
        "--months",
        type=int,
        default=DEFAULT_MONTHS,
        help="Keep only this many recent months (default: 24)",
    )
    args = parser.parse_args(argv)

    try:
        count = load(args.csv_path, args.db, args.months)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"Loaded {count} rows into {args.db} (last {args.months} months)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
