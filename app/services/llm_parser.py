"""Turn a masked natural-language query into a validated PropertyFilter.

The model may only fill in the declared function arguments; no SQL is ever
generated or executed here. The declaration mirrors ARCHITECTURE.md 5.1.
"""

from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from typing import Any

from google import genai
from pydantic import ValidationError

from app.config import settings
from app.prompts import FUNCTION_NAME, SYSTEM_PROMPT
from app.schemas import PropertyFilter

REQUEST_TIMEOUT_SECONDS = 20.0
MAX_API_ATTEMPTS = 2
BACKOFF_BASE_SECONDS = 0.5
MAX_VALIDATION_ATTEMPTS = 2  # original call + one correction retry
CACHE_MAX_SIZE = 256


class InvalidFilterError(RuntimeError):
    """Raised when the LLM cannot produce a valid PropertyFilter."""


class LLMTimeoutError(InvalidFilterError):
    """Raised when the LLM call timed out or was unreachable."""


def _is_timeout(exc: BaseException) -> bool:
    """Detect timeout-ish failures without depending on a specific transport."""
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    return "timeout" in name or "timeout" in text or "deadline" in text


# --- function declaration -------------------------------------------------

PROPERTY_FILTER_TOOL: dict[str, Any] = {
    "type": "function",
    "name": FUNCTION_NAME,
    "description": (
        "Return the structured filters implied by the user's HDB resale flat "
        "search. Omit any field the user did not mention."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "town": {
                "type": "string",
                "description": "Singapore HDB town, e.g. TAMPINES, BEDOK, ANG MO KIO.",
            },
            "flat_type": {
                "type": "string",
                "enum": [
                    "1 ROOM",
                    "2 ROOM",
                    "3 ROOM",
                    "4 ROOM",
                    "5 ROOM",
                    "EXECUTIVE",
                    "MULTI-GENERATION",
                ],
                "description": "Flat type, exactly as written in the enum.",
            },
            "min_price": {
                "type": "integer",
                "description": "Minimum resale price in SGD. Non-negative.",
            },
            "max_price": {
                "type": "integer",
                "description": "Maximum resale price in SGD. Non-negative.",
            },
            "min_area_sqm": {
                "type": "number",
                "description": "Minimum floor area in square metres. Non-negative.",
            },
            "storey_min": {
                "type": "integer",
                "description": "Lowest storey to include, 1 or higher.",
            },
            "storey_max": {
                "type": "integer",
                "description": "Highest storey to include, 1 or higher.",
            },
            "min_remaining_lease_years": {
                "type": "number",
                "description": "Minimum remaining lease in years. Non-negative.",
            },
            "sort_by": {
                "type": "string",
                "enum": ["price_asc", "price_desc", "area_desc", "lease_desc"],
                "description": "Ordering requested by the user, if any.",
            },
        },
    },
}

# The declaration must stay in sync with the model.
assert set(PROPERTY_FILTER_TOOL["parameters"]["properties"]) == set(
    PropertyFilter.model_fields
)


# --- cache ----------------------------------------------------------------

_CACHE: "OrderedDict[str, PropertyFilter]" = OrderedDict()


def _cache_key(query: str) -> str:
    payload = f"{settings.gemini_model}:{query}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def clear_cache() -> None:
    """Drop all cached filters."""
    _CACHE.clear()


def _cache_get(key: str) -> PropertyFilter | None:
    cached = _CACHE.get(key)
    if cached is None:
        return None
    _CACHE.move_to_end(key)
    return cached


def _cache_put(key: str, value: PropertyFilter) -> None:
    _CACHE[key] = value
    _CACHE.move_to_end(key)
    while len(_CACHE) > CACHE_MAX_SIZE:
        _CACHE.popitem(last=False)


# --- Gemini plumbing ------------------------------------------------------

def _get_client() -> genai.Client:
    """Create a Gemini client; isolated so tests can patch it."""
    return genai.Client(api_key=settings.require_gemini_api_key())


def _extract_arguments(interaction: Any) -> dict[str, Any]:
    """Pull the arguments of our function call out of an interaction."""
    for step in getattr(interaction, "steps", None) or []:
        if getattr(step, "type", None) != "function_call":
            continue
        if getattr(step, "name", None) != FUNCTION_NAME:
            continue
        arguments = getattr(step, "arguments", None) or {}
        if not isinstance(arguments, dict):
            raise InvalidFilterError("Function arguments were not an object")
        return arguments

    raise InvalidFilterError(f"Gemini did not call {FUNCTION_NAME} for this query")


def _request_function_call(user_input: str) -> dict[str, Any]:
    """Call Gemini with retry/backoff on transport errors."""
    client = _get_client()
    last_error: Exception | None = None

    for attempt in range(MAX_API_ATTEMPTS):
        try:
            interaction = client.interactions.create(
                model=settings.gemini_model,
                input=user_input,
                system_instruction=SYSTEM_PROMPT,
                tools=[PROPERTY_FILTER_TOOL],
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            return _extract_arguments(interaction)
        except InvalidFilterError:
            raise
        except Exception as exc:  # network, timeout, rate limit, 5xx ...
            last_error = exc
            if attempt + 1 < MAX_API_ATTEMPTS:
                time.sleep(BACKOFF_BASE_SECONDS * (2**attempt))

    if last_error is not None and _is_timeout(last_error):
        raise LLMTimeoutError("Gemini request timed out, please retry shortly") from last_error

    raise InvalidFilterError(f"Gemini request failed: {last_error}") from last_error


def _format_validation_error(exc: ValidationError) -> str:
    lines = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"]) or "(root)"
        lines.append(f"- {location}: {error['msg']}")
    return "\n".join(lines)


def _build_retry_input(query: str, error: str) -> str:
    return (
        f"{query}\n\n"
        "---\n"
        "Your previous function call was rejected by schema validation:\n"
        f"{error}\n"
        f"Call {FUNCTION_NAME} again with corrected arguments. "
        "Use only the declared fields and never return SQL."
    )


# --- public API -----------------------------------------------------------

def parse_query(masked_query: str) -> PropertyFilter:
    """Parse a PII-masked query into a validated PropertyFilter.

    Args:
        masked_query: User query with phone numbers and emails already masked.

    Returns:
        A validated filter. Unmappable queries yield an empty filter.

    Raises:
        InvalidFilterError: When no valid filter could be produced after one
            corrective retry.
        LLMTimeoutError: When the model call timed out.
    """
    if not masked_query or not masked_query.strip():
        return PropertyFilter()

    key = _cache_key(masked_query)
    cached = _cache_get(key)
    if cached is not None:
        return cached.model_copy(deep=True)

    user_input = masked_query
    last_error = "the function call was missing"

    for _attempt in range(MAX_VALIDATION_ATTEMPTS):
        try:
            arguments = _request_function_call(user_input)
            result = PropertyFilter(**arguments)
        except ValidationError as exc:
            last_error = _format_validation_error(exc)
            user_input = _build_retry_input(masked_query, last_error)
            continue
        except LLMTimeoutError:
            raise
        except InvalidFilterError as exc:
            last_error = str(exc)
            user_input = _build_retry_input(masked_query, last_error)
            continue

        _cache_put(key, result)
        return result

    raise InvalidFilterError(f"Gemini returned an invalid filter after retry: {last_error}")
