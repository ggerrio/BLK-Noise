"""FastAPI application: POST /search and GET /health.

Pipeline for /search (ARCHITECTURE.md 5.2):
mask PII -> parse -> retrieve -> rank -> explain -> audit log.
"""

from __future__ import annotations

import sqlite3
import time
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException, Request

from app import db
from app.config import settings
from app.schemas import DescriptionResponse, SearchRequest, SearchResponse
from app.services import audit, describer, explainer, ranker, retriever
from app.services.llm_parser import (
    InvalidFilterError,
    LLMTimeoutError,
    parse_query,
)
from app.services.pii import mask_pii

RATE_LIMIT_MAX_REQUESTS = 20
RATE_LIMIT_WINDOW_SECONDS = 60.0
RETRIEVE_LIMIT = 200

MESSAGE_INVALID_FILTER = (
    "Could not understand the location or filters. Try a Singapore town name "
    "and a budget, for example '4 room in Tampines under 500k'."
)
MESSAGE_LLM_TIMEOUT = (
    "The search assistant took too long to respond. Please try again in a moment."
)
MESSAGE_NO_DATA = "Search data is not ready yet. Please try again later."
MESSAGE_RATE_LIMITED = "Too many requests. Please wait a moment and try again."

app = FastAPI(title="BLK NOISE", version="0.1.0")

_rate_hits: dict[str, list[float]] = {}


def reset_rate_limits() -> None:
    """Clear the in-memory rate limit counters."""
    _rate_hits.clear()


def _check_rate_limit(request: Request) -> None:
    """Allow RATE_LIMIT_MAX_REQUESTS per client per sliding window."""
    client = request.client
    key = client.host if client and client.host else "unknown"
    now = time.monotonic()
    recent = [
        stamp
        for stamp in _rate_hits.get(key, [])
        if now - stamp < RATE_LIMIT_WINDOW_SECONDS
    ]

    if len(recent) >= RATE_LIMIT_MAX_REQUESTS:
        _rate_hits[key] = recent
        raise HTTPException(
            status_code=429,
            detail=MESSAGE_RATE_LIMITED,
            headers={"Retry-After": str(int(RATE_LIMIT_WINDOW_SECONDS))},
        )

    recent.append(now)
    _rate_hits[key] = recent


def _attach_embeddings(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fetch stored embeddings for the retrieved rows."""
    ids = [row["id"] for row in rows if row.get("id") is not None]
    if not ids:
        return rows

    placeholders = ", ".join("?" * len(ids))
    try:
        conn = db.connect(read_only=True)
    except (FileNotFoundError, sqlite3.Error):
        return rows

    try:
        found = {
            row["id"]: row["embedding"]
            for row in conn.execute(
                f"SELECT id, embedding FROM listings WHERE id IN ({placeholders})",
                ids,
            )
        }
    except sqlite3.Error:
        return rows
    finally:
        conn.close()

    return [{**row, "embedding": found.get(row["id"])} for row in rows]


def _public_row(row: dict[str, Any]) -> dict[str, Any]:
    """Strip non-serialisable internals before returning a result."""
    return {key: value for key, value in row.items() if key != "embedding"}


@app.get("/health")
def health() -> dict[str, Any]:
    """Report service and data readiness."""
    try:
        conn = db.connect(read_only=True)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=MESSAGE_NO_DATA) from exc

    try:
        listings = db.count_listings(conn)
    finally:
        conn.close()

    return {
        "status": "ok",
        "listings": listings,
        "database": str(db.resolve_path()),
    }


@app.post("/describe/{listing_id}", response_model=DescriptionResponse)
def describe(listing_id: int) -> DescriptionResponse:
    """Generate and verify a description for one listing."""
    started = time.perf_counter()
    request_id = uuid.uuid4().hex

    try:
        conn = db.connect(read_only=True)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=MESSAGE_NO_DATA) from exc

    try:
        row = conn.execute(
            "SELECT * FROM listings WHERE id = ?", (listing_id,)
        ).fetchone()
    finally:
        conn.close()

    if row is None:
        raise HTTPException(status_code=404, detail="No listing with that id.")

    try:
        outcome = describer.describe_listing(dict(row))
    except LLMTimeoutError as exc:
        raise HTTPException(status_code=503, detail=MESSAGE_LLM_TIMEOUT) from exc
    except Exception as exc:  # network, rate limit, provider errors
        raise HTTPException(
            status_code=503,
            detail="Could not write a description right now. Try again shortly.",
        ) from exc

    return DescriptionResponse(
        listing_id=listing_id,
        description=outcome.description,
        check=outcome.check,
        fallback_used=outcome.fallback_used,
        request_id=request_id,
        latency_ms=round((time.perf_counter() - started) * 1000, 2),
    )


@app.post("/search", response_model=SearchResponse)
def search(payload: SearchRequest, request: Request) -> SearchResponse:
    """Run the full search pipeline for one natural-language query."""
    _check_rate_limit(request)

    started = time.perf_counter()
    request_id = uuid.uuid4().hex
    status = "error"
    error_detail: str | None = None
    parsed = None
    result_count = 0

    masked_query = mask_pii(payload.query)

    try:
        parsed = parse_query(masked_query)
        rows = retriever.retrieve(parsed, limit=RETRIEVE_LIMIT)
        ranked = ranker.rank(masked_query, _attach_embeddings(rows), top_n=payload.top_n)
        if not ranked and rows:  # no embeddings yet: return unranked results
            ranked = rows[: payload.top_n]

        results = [_public_row(row) for row in ranked]
        result_count = len(results)
        explanation = explainer.explain(masked_query, results)
        status = "ok" if results else "no_results"

        return SearchResponse(
            parsed_filter=parsed,
            results=results,
            explanation=explanation,
            request_id=request_id,
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
        )
    except LLMTimeoutError as exc:
        status, error_detail = "llm_error", str(exc)
        raise HTTPException(status_code=503, detail=MESSAGE_LLM_TIMEOUT) from exc
    except InvalidFilterError as exc:
        status, error_detail = "invalid_filter", str(exc)
        raise HTTPException(status_code=422, detail=MESSAGE_INVALID_FILTER) from exc
    except FileNotFoundError as exc:
        status, error_detail = "error", str(exc)
        raise HTTPException(status_code=503, detail=MESSAGE_NO_DATA) from exc
    finally:
        latency_ms = (time.perf_counter() - started) * 1000
        try:
            audit.log_search(
                request_id=request_id,
                masked_query=masked_query,
                parsed_filter=parsed,
                result_count=result_count,
                latency_ms=latency_ms,
                llm_model=settings.gemini_model,
                status=status,
                error_detail=error_detail,
            )
        except sqlite3.Error:
            pass  # auditing must never break a search
