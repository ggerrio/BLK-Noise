"""Persist a masked record of each search request."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any

from app import db
from app.schemas import PropertyFilter


def log_search(
    *,
    request_id: str,
    masked_query: str,
    parsed_filter: PropertyFilter | None = None,
    result_count: int = 0,
    latency_ms: float = 0.0,
    llm_model: str | None = None,
    status: str = "ok",
    error_detail: str | None = None,
    conn: sqlite3.Connection | None = None,
    db_path: Any | None = None,
) -> int:
    """Insert one audit row and return its id.

    Only the masked query is stored; callers must never pass raw user input.

    Args:
        request_id: Correlation id also returned to the client.
        masked_query: Query after PII masking.
        parsed_filter: Validated filter, serialised as JSON.
        result_count: Number of results returned.
        latency_ms: End-to-end latency in milliseconds.
        llm_model: Model used for this request.
        status: ok | invalid_filter | llm_error | no_results | error.
        error_detail: Short machine-readable failure reason, if any.
        conn: Optional existing connection.
        db_path: Optional database path override.

    Returns:
        The inserted row id.
    """
    own_connection = conn is None
    if own_connection:
        conn = db.connect(db_path)

    try:
        db.ensure_audit_table(conn)
        parsed_json = (
            json.dumps(parsed_filter.model_dump(mode="json"), sort_keys=True)
            if parsed_filter is not None
            else None
        )
        cursor = conn.execute(
            "INSERT INTO audit_log"
            " (ts_utc, request_id, masked_query, parsed_filter, result_count,"
            " latency_ms, llm_model, status, error_detail)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                request_id,
                masked_query,
                parsed_json,
                int(result_count),
                int(round(latency_ms)),
                llm_model,
                status,
                error_detail,
            ),
        )
        conn.commit()
        return int(cursor.lastrowid or 0)
    finally:
        if own_connection:
            conn.close()
