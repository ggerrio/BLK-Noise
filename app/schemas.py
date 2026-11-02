"""Pydantic models for search filters and API payloads.

`PropertyFilter` follows ARCHITECTURE.md section 5.1 exactly.
"""

from __future__ import annotations

import difflib
import sqlite3
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.config import settings

# Minimum similarity for a town name to be accepted as a typo of a real town.
TOWN_FUZZY_THRESHOLD = 0.75

FlatType = Literal[
    "1 ROOM",
    "2 ROOM",
    "3 ROOM",
    "4 ROOM",
    "5 ROOM",
    "EXECUTIVE",
    "MULTI-GENERATION",
]

SortBy = Literal["price_asc", "price_desc", "area_desc", "lease_desc"]


@lru_cache(maxsize=8)
def _read_towns(db_path: str) -> frozenset[str]:
    """Read the distinct towns stored in the listings table.

    Returns an empty set when the database or table does not exist yet.
    """
    path = Path(db_path)
    if not path.exists():
        return frozenset()

    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
            rows = conn.execute("SELECT DISTINCT town FROM listings").fetchall()
    except sqlite3.Error:
        return frozenset()

    return frozenset(town for (town,) in rows if town)


def get_valid_towns(db_path: Path | None = None) -> frozenset[str]:
    """Return the canonical town names available in the database."""
    return _read_towns(str(db_path or settings.db_path))


def resolve_town(value: str, db_path: Path | None = None) -> str:
    """Normalise a town name and snap typos to the closest known town.

    Args:
        value: Raw town text from the query.
        db_path: Optional database to read valid towns from.

    Returns:
        The canonical town name, or the normalised input when no towns exist.

    Raises:
        ValueError: When the name is too far from every known town.
    """
    normalised = " ".join(str(value).split()).upper()
    towns = get_valid_towns(db_path)

    if not normalised or not towns:
        return normalised

    scored = sorted(
        (
            (difflib.SequenceMatcher(None, normalised, town).ratio(), town)
            for town in towns
        ),
        reverse=True,
    )
    best_score, best_town = scored[0]

    if best_score == 1.0 or best_score >= TOWN_FUZZY_THRESHOLD:
        return best_town

    known = ", ".join(sorted(towns))
    if best_score >= 0.5:
        raise ValueError(
            f"Unknown town {value!r}. Did you mean {best_town!r}? "
            f"Valid towns: {known}."
        )
    raise ValueError(f"Unknown town {value!r}. Valid towns: {known}.")


class PropertyFilter(BaseModel):
    """Structured filters extracted from a natural-language query."""

    model_config = ConfigDict(extra="forbid")

    town: str | None = None
    flat_type: FlatType | None = None
    min_price: int | None = Field(default=None, ge=0)
    max_price: int | None = Field(default=None, ge=0)
    min_area_sqm: float | None = Field(default=None, ge=0)
    storey_min: int | None = Field(default=None, ge=1)
    storey_max: int | None = Field(default=None, ge=1)
    min_remaining_lease_years: float | None = Field(default=None, ge=0)
    sort_by: SortBy | None = None

    @field_validator("town")
    @classmethod
    def _canonical_town(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return resolve_town(value)

    @model_validator(mode="after")
    def _check_ranges(self) -> PropertyFilter:
        if (
            self.min_price is not None
            and self.max_price is not None
            and self.min_price > self.max_price
        ):
            raise ValueError(
                f"min_price ({self.min_price}) must not exceed max_price "
                f"({self.max_price})"
            )

        if (
            self.storey_min is not None
            and self.storey_max is not None
            and self.storey_min > self.storey_max
        ):
            raise ValueError(
                f"storey_min ({self.storey_min}) must not exceed storey_max "
                f"({self.storey_max})"
            )

        return self


class SearchRequest(BaseModel):
    """Incoming natural-language search query."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    query: str = Field(min_length=1, max_length=300)
    top_n: int = Field(default=5, ge=1, le=20)


class SearchResponse(BaseModel):
    """Search result payload returned to the client."""

    parsed_filter: PropertyFilter
    results: list[dict[str, Any]] = Field(default_factory=list)
    explanation: str = ""
    request_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    latency_ms: float = Field(ge=0)


class Claim(BaseModel):
    """One factual claim extracted from generated text."""

    kind: str
    text: str
    value: str
    expected: str | None = None
    ok: bool


class VarCheckResult(BaseModel):
    """Outcome of checking a description against its source row."""

    passed: bool
    claims: list[Claim] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class DescriptionResponse(BaseModel):
    """Generated, checked description for one listing."""

    listing_id: int
    description: str
    check: VarCheckResult
    fallback_used: bool = False
    request_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    latency_ms: float = Field(ge=0)
