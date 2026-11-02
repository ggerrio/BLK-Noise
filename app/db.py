"""SQLite connection helpers and schema bootstrap."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from app.config import settings

LISTINGS_TABLE = "listings"
AUDIT_TABLE = "audit_log"

# Mirrors ARCHITECTURE.md section 4.
AUDIT_SCHEMA = """
CREATE TABLE IF NOT EXISTS audit_log (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_utc        TEXT NOT NULL,
    request_id    TEXT NOT NULL,
    masked_query  TEXT NOT NULL,
    parsed_filter TEXT,
    result_count  INTEGER,
    latency_ms    INTEGER,
    llm_model     TEXT,
    status        TEXT NOT NULL,
    error_detail  TEXT
);
CREATE INDEX IF NOT EXISTS idx_audit_log_ts ON audit_log (ts_utc);
"""


def resolve_path(db_path: Path | str | None = None) -> Path:
    """Return the database path used by the app."""
    return Path(db_path or settings.db_path)


def connect(
    db_path: Path | str | None = None, *, read_only: bool = False
) -> sqlite3.Connection:
    """Open a connection to the listings database.

    Read-only connections fail loudly when the file is missing; writable ones
    create the parent directory first.
    """
    path = resolve_path(db_path)

    if read_only:
        if not path.exists():
            raise FileNotFoundError(f"Database not found: {path}")
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path)

    conn.row_factory = sqlite3.Row
    return conn


def ensure_audit_table(conn: sqlite3.Connection) -> None:
    """Create the audit_log table when it does not exist."""
    conn.executescript(AUDIT_SCHEMA)
    conn.commit()


def init_db(db_path: Path | str | None = None) -> None:
    """Create the audit table for the configured database."""
    conn = connect(db_path)
    try:
        ensure_audit_table(conn)
    finally:
        conn.close()


def table_exists(conn: sqlite3.Connection, name: str) -> bool:
    """Return True when the named table exists."""
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (name,)
    ).fetchone()
    return row is not None


def count_listings(conn: sqlite3.Connection) -> int:
    """Return the number of rows in listings, or 0 when the table is absent."""
    if not table_exists(conn, LISTINGS_TABLE):
        return 0
    row = conn.execute("SELECT COUNT(*) FROM listings").fetchone()
    return int(row[0]) if row else 0
