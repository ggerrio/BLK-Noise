"""Hallucination check for generated listing text.

Every number and named attribute in the text (price, area, storey, remaining
lease, town, flat type) must match the source row. Any mismatch fails the text.
"""

from __future__ import annotations

import re
from typing import Any, Mapping

from app.schemas import Claim, VarCheckResult

# Towns used by HDB resale data.
KNOWN_TOWNS: tuple[str, ...] = (
    "ANG MO KIO",
    "BEDOK",
    "BISHAN",
    "BUKIT BATOK",
    "BUKIT MERAH",
    "BUKIT PANJANG",
    "BUKIT TIMAH",
    "CENTRAL AREA",
    "CHOA CHU KANG",
    "CLEMENTI",
    "GEYLANG",
    "HOUGANG",
    "JURONG EAST",
    "JURONG WEST",
    "KALLANG/WHAMPOA",
    "MARINE PARADE",
    "PASIR RIS",
    "PUNGGOL",
    "QUEENSTOWN",
    "SEMBAWANG",
    "SENGKANG",
    "SERANGOON",
    "TAMPINES",
    "TOA PAYOH",
    "WOODLANDS",
    "YISHUN",
)

MONEY_RE = re.compile(r"(?:S\$|\$)\s?([0-9][0-9,]*)")
AREA_RE = re.compile(
    r"([0-9]+(?:\.[0-9]+)?)\s*(?:sqm|sq\.?\s?m\b|square\s+met(?:re|er)s?)",
    re.IGNORECASE,
)
LEASE_RE = re.compile(
    r"([0-9]+(?:\.[0-9]+)?)[\s-]*(?:years?|yrs?)[^.\n]{0,20}?lease"
    r"|lease[^.\n]{0,20}?([0-9]+(?:\.[0-9]+)?)[\s-]*(?:years?|yrs?)",
    re.IGNORECASE,
)
STOREY_RANGE_RE = re.compile(
    r"(?:storeys?|floors?|levels?)\s*([0-9]+)(?:st|nd|rd|th)?"
    r"\s*(?:to|and|-|\u2013)\s*([0-9]+)(?:st|nd|rd|th)?",
    re.IGNORECASE,
)
STOREY_SINGLE_RE = re.compile(
    r"\b([0-9]+)(?:st|nd|rd|th)\s+(?:floor|storey|level)\b"
    r"|\b(?:floor|storey|level)\s+([0-9]+)\b",
    re.IGNORECASE,
)
FLAT_TYPE_RE = re.compile(
    r"\b([1-5])[\s-]?rooms?\b|\bexecutive\b|\bmulti[\s-]generation\b",
    re.IGNORECASE,
)

_TOLERANCE = 0.5


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _storey_band(row: Mapping[str, Any]) -> tuple[int, int] | None:
    low = row.get("storey_min")
    high = row.get("storey_max")
    if low is None or high is None:
        return None
    return int(low), int(high)


def _range_overlaps(span: tuple[int, int], taken: list[tuple[int, int]]) -> bool:
    start, end = span
    return any(start < other_end and other_start < end for other_start, other_end in taken)


def check_description(text: str, row: Mapping[str, Any]) -> VarCheckResult:
    """Verify that every claim in `text` matches the source row.

    Args:
        text: Generated description.
        row: The listing row the text was written from.

    Returns:
        A VarCheckResult with one Claim per extracted value. `passed` is False
        when any claim disagrees with the row.
    """
    claims: list[Claim] = []
    errors: list[str] = []

    def record(kind: str, matched: str, value: Any, expected: Any, ok: bool) -> None:
        claims.append(
            Claim(
                kind=kind,
                text=matched,
                value=str(value),
                expected=None if expected is None else str(expected),
                ok=ok,
            )
        )
        if not ok:
            errors.append(f"{kind}: {matched!r} does not match the row ({expected!r})")

    price = _as_float(row.get("resale_price"))
    for match in MONEY_RE.finditer(text):
        amount = int(match.group(1).replace(",", ""))
        ok = price is not None and abs(amount - price) < 1
        record("price", match.group(0), amount, row.get("resale_price"), ok)

    area = _as_float(row.get("floor_area_sqm"))
    for match in AREA_RE.finditer(text):
        value = float(match.group(1))
        ok = area is not None and abs(value - area) <= _TOLERANCE
        record("area_sqm", match.group(0), value, row.get("floor_area_sqm"), ok)

    band = _storey_band(row)
    used_spans: list[tuple[int, int]] = []
    for match in STOREY_RANGE_RE.finditer(text):
        low, high = int(match.group(1)), int(match.group(2))
        used_spans.append((match.start(), match.end()))
        ok = band is not None and band[0] <= low and high <= band[1]
        record("storey", match.group(0), f"{low}-{high}", band, ok)

    for match in STOREY_SINGLE_RE.finditer(text):
        if _range_overlaps((match.start(), match.end()), used_spans):
            continue
        floor = int(match.group(1) or match.group(2))
        ok = band is not None and band[0] <= floor <= band[1]
        record("storey", match.group(0), floor, band, ok)

    lease = _as_float(row.get("remaining_lease_years"))
    for match in LEASE_RE.finditer(text):
        raw = match.group(1) or match.group(2)
        value = float(raw)
        ok = lease is not None and abs(value - lease) <= _TOLERANCE
        record(
            "remaining_lease_years",
            match.group(0),
            value,
            row.get("remaining_lease_years"),
            ok,
        )

    row_town = str(row.get("town") or "").upper()
    for town in KNOWN_TOWNS:
        match = re.search(rf"\b{re.escape(town)}\b", text.upper())
        if match:
            record("town", town, town, row_town or None, town == row_town)

    row_flat_type = str(row.get("flat_type") or "").upper()
    for match in FLAT_TYPE_RE.finditer(text):
        token = match.group(0).upper()
        if token[0].isdigit():
            normalised = f"{token[0]} ROOM"
        elif "MULTI" in token:
            normalised = "MULTI-GENERATION"
        else:
            normalised = "EXECUTIVE"
        record("flat_type", match.group(0), normalised, row_flat_type or None,
               normalised == row_flat_type)

    return VarCheckResult(passed=not errors, claims=claims, errors=errors)


def hallucination_rate(results: list[VarCheckResult]) -> float:
    """Fraction of checked texts that failed, 0.0 when nothing was checked."""
    if not results:
        return 0.0
    return round(sum(1 for result in results if not result.passed) / len(results), 4)
