"""Embed every listing and store the vector as a float32 BLOB.

Usage:
    python scripts/build_embeddings.py
    python scripts/build_embeddings.py --db data/listings.db --batch-size 64
    python scripts/build_embeddings.py --rebuild
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "listings.db"
MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_BATCH_SIZE = 32

SOURCE_COLUMNS = (
    "id",
    "town",
    "flat_type",
    "floor_area_sqm",
    "storey_min",
    "storey_max",
    "remaining_lease_years",
    "resale_price",
    "summary_text",
)


def _number(value: Any) -> str:
    """Format a number without a trailing '.0'."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return f"{value:g}"


def _get(row: Any, key: str) -> Any:
    """Read a column from a dict or sqlite3.Row, tolerating missing keys."""
    try:
        return row[key]
    except (KeyError, IndexError):
        return None


def build_summary(row: dict[str, Any] | sqlite3.Row) -> str:
    """Build the searchable text for one listing.

    Example: "4 ROOM in TAMPINES, 92 sqm, storey 10-12, 70 years lease left,
    S$580000". The loader already stores this text, so reuse it when present.
    """
    stored = _get(row, "summary_text")
    if isinstance(stored, str) and stored.strip():
        return stored.strip()

    parts: list[str] = []

    flat_type = (_get(row, "flat_type") or "").strip()
    town = (_get(row, "town") or "").strip()
    if flat_type and town:
        parts.append(f"{flat_type} in {town}")
    elif flat_type or town:
        parts.append(flat_type or town)

    area = _get(row, "floor_area_sqm")
    if area is not None:
        parts.append(f"{_number(float(area))} sqm")

    storey_min = _get(row, "storey_min")
    storey_max = _get(row, "storey_max")
    if storey_min is not None and storey_max is not None:
        parts.append(f"storey {int(storey_min)}-{int(storey_max)}")
    elif storey_min is not None or storey_max is not None:
        parts.append(f"storey {int(storey_min or storey_max)}")

    lease = _get(row, "remaining_lease_years")
    if lease is not None:
        parts.append(f"{_number(float(lease))} years lease left")

    price = _get(row, "resale_price")
    if price is not None:
        parts.append(f"S${int(round(float(price)))}")

    return ", ".join(parts)


def ensure_embedding_column(conn: sqlite3.Connection) -> None:
    """Add the `embedding` BLOB column when it does not exist yet."""
    columns = {
        row[1] for row in conn.execute("PRAGMA table_info(listings)").fetchall()
    }
    if "embedding" not in columns:
        conn.execute("ALTER TABLE listings ADD COLUMN embedding BLOB")
        conn.commit()


def fetch_rows(conn: sqlite3.Connection, rebuild: bool) -> list[sqlite3.Row]:
    """Return the rows that still need an embedding."""
    where = "" if rebuild else " WHERE embedding IS NULL"
    sql = f"SELECT {', '.join(SOURCE_COLUMNS)} FROM listings{where} ORDER BY id"
    return conn.execute(sql).fetchall()


def load_model(model_name: str = MODEL_NAME):
    """Load the sentence-transformers model (imported lazily)."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def _render_progress(
    done: int, total: int, started: float, width: int = 30, finish: bool = False
) -> None:
    fraction = done / total if total else 1.0
    filled = int(width * fraction)
    bar = "#" * filled + "-" * (width - filled)
    elapsed = time.time() - started
    eta = (elapsed / fraction - elapsed) if fraction > 0 else 0.0
    message = (
        f"\r[{bar}] {done}/{total} ({fraction * 100:5.1f}%) eta {eta:4.0f}s"
    )
    sys.stderr.write(message + ("\n" if finish else ""))
    sys.stderr.flush()


def _chunks(items: list[sqlite3.Row], size: int) -> Iterable[list[sqlite3.Row]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def to_blob(vector: np.ndarray) -> bytes:
    """Serialise a vector as a little-endian float32 BLOB."""
    return np.asarray(vector, dtype=np.float32).tobytes()


def build_embeddings(
    db_path: Path | str = DEFAULT_DB_PATH,
    model_name: str = MODEL_NAME,
    batch_size: int = DEFAULT_BATCH_SIZE,
    rebuild: bool = False,
    model: Any | None = None,
) -> int:
    """Embed listings lacking a vector and store them. Returns rows written."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        ensure_embedding_column(conn)
        rows = fetch_rows(conn, rebuild)

        total = len(rows)
        if total == 0:
            print("All listings already have embeddings (use --rebuild to redo).")
            return 0

        encoder = model if model is not None else load_model(model_name)
        started = time.time()
        written = 0
        _render_progress(0, total, started)

        for batch in _chunks(rows, batch_size):
            texts = [build_summary(row) for row in batch]
            vectors = encoder.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            vectors = np.asarray(vectors, dtype=np.float32)

            conn.executemany(
                "UPDATE listings SET embedding = ? WHERE id = ?",
                [(to_blob(vector), row["id"]) for vector, row in zip(vectors, batch)],
            )
            conn.commit()

            written += len(batch)
            if written < total:
                _render_progress(written, total, started)

        _render_progress(written, total, started, finish=True)
        return written
    finally:
        conn.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Embed listings and store them as float32 BLOBs."
    )
    parser.add_argument(
        "--db", type=Path, default=DEFAULT_DB_PATH, help="SQLite database path"
    )
    parser.add_argument("--model", default=MODEL_NAME, help="sentence-transformers model")
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="encode batch size"
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Recompute embeddings for rows that already have one",
    )
    args = parser.parse_args(argv)

    try:
        count = build_embeddings(
            db_path=args.db,
            model_name=args.model,
            batch_size=args.batch_size,
            rebuild=args.rebuild,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if count:
        print(f"Stored {count} embeddings in {args.db}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
