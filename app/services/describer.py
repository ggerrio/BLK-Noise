"""Generate a short description for one listing and verify it.

The model sees a single row and writes 2-3 sentences. Every claim is then
checked by `var_check`; a failed description is replaced by a deterministic one
built from the same row, and the failure is reported to the caller.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from app.config import settings
from app.schemas import VarCheckResult
from app.services.llm_parser import LLMTimeoutError, _get_client, _is_timeout
from app.services.var_check import check_description

DESCRIBE_TIMEOUT_SECONDS = 20.0

DESCRIBE_SYSTEM_PROMPT = """You write 2-3 short sentences describing one HDB
resale flat to a buyer.

Rules:
1. Use ONLY the values in the provided row: town, flat type, block, street,
   floor area, storey range, remaining lease and resale price.
2. Never invent amenities, schools, MRT distances, renovation, view, or any
   other detail that is not in the row.
3. No investment or financial advice, no predictions about price.
4. Plain sentences, no bullet points, no markdown, no SQL.
"""


@dataclass
class DescribeOutcome:
    """Result of describing one listing."""

    description: str
    check: VarCheckResult
    fallback_used: bool


def _field(row: Mapping[str, Any], key: str) -> Any:
    return row.get(key)


def fallback_description(row: Mapping[str, Any]) -> str:
    """Deterministic description built only from the row's own values."""
    town = _field(row, "town") or "Singapore"
    flat_type = _field(row, "flat_type") or "HDB"
    block = _field(row, "block")
    street = _field(row, "street_name")
    area = _field(row, "floor_area_sqm")
    storey_min = _field(row, "storey_min")
    storey_max = _field(row, "storey_max")
    lease = _field(row, "remaining_lease_years")
    price = _field(row, "resale_price")

    location = f"Blk {block} {street}".strip() if block and street else town
    sentence = f"{flat_type} flat in {town} at {location}."
    details = []
    if area is not None:
        details.append(f"{float(area):g} sqm")
    if storey_min is not None and storey_max is not None:
        details.append(f"storeys {int(storey_min)}-{int(storey_max)}")
    if lease is not None:
        details.append(f"{float(lease):g} years of lease left")
    if details:
        sentence += " It has " + ", ".join(details) + "."
    if price is not None:
        sentence += f" The resale price is S${int(round(float(price)))}."
    return sentence


def _format_row(row: Mapping[str, Any]) -> str:
    keys = (
        "id",
        "town",
        "flat_type",
        "block",
        "street_name",
        "storey_min",
        "storey_max",
        "floor_area_sqm",
        "remaining_lease_years",
        "resale_price",
    )
    return "\n".join(f"{key}: {row[key]}" for key in keys if row.get(key) is not None)


def _request_description(row: Mapping[str, Any], client: Any | None = None) -> str:
    """Call Gemini for one row; raises LLMTimeoutError on timeouts."""
    client = client if client is not None else _get_client()
    try:
        interaction = client.interactions.create(
            model=settings.gemini_model,
            input="Listing row:\n" + _format_row(row) + "\n\nDescribe this flat.",
            system_instruction=DESCRIBE_SYSTEM_PROMPT,
            timeout=DESCRIBE_TIMEOUT_SECONDS,
        )
    except Exception as exc:
        if _is_timeout(exc):
            raise LLMTimeoutError("Describer request timed out") from exc
        raise

    return (getattr(interaction, "output_text", None) or "").strip()


def describe_listing(
    row: Mapping[str, Any],
    client: Any | None = None,
) -> DescribeOutcome:
    """Describe one listing and return a verified description.

    Args:
        row: The listing row to describe.
        client: Optional pre-built Gemini client (used by tests).

    Returns:
        A DescribeOutcome whose `description` is always safe to show: when the
        model output fails the check, a deterministic description is returned
        instead and `fallback_used` is True.
    """
    text = _request_description(row, client)
    check = check_description(text, row)

    if not check.passed or not text:
        return DescribeOutcome(
            description=fallback_description(row),
            check=check,
            fallback_used=True,
        )

    return DescribeOutcome(description=text, check=check, fallback_used=False)
