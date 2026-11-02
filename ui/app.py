"""BLK NOISE Streamlit UI.

Talks to the FastAPI service over HTTP; the API base URL comes from API_URL.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import requests
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:  # allow `streamlit run ui/app.py`
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import settings  # noqa: E402

REQUEST_TIMEOUT_SECONDS = 45
TOP_N = 10

EXAMPLES: tuple[str, ...] = (
    "4 room in Tampines under 600k",
    "high floor 3 room in Bedok with long lease",
    "5 room in Sengkang, at least 100 sqm, under 700k",
    "4 room in Woodlands, cheapest first",
)

def _set_query(text: str) -> None:
    """Put an example query into the input box."""
    st.session_state["query"] = text


def _error_detail(response: requests.Response) -> str | None:
    """Pull the API's friendly detail out of an error response."""
    try:
        detail = response.json().get("detail")
    except (ValueError, AttributeError):
        return None
    return detail if isinstance(detail, str) else None


def call_search(query: str, api_url: str) -> dict[str, Any] | None:
    """POST /search and return the payload, or show a friendly error."""
    url = f"{api_url.rstrip('/')}/search"
    try:
        response = requests.post(
            url,
            json={"query": query, "top_n": TOP_N},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException:
        st.error(
            "Can't reach the search service. Check that the API is running "
            f"at {api_url}, then try again."
        )
        return None

    if response.status_code == 200:
        return response.json()

    detail = _error_detail(response)
    if response.status_code == 422:
        st.error(
            detail
            or "Could not understand the location or filters. Try a Singapore "
            "town name and a budget, for example '4 room in Tampines under 500k'."
        )
    elif response.status_code == 429:
        st.error(detail or "Too many searches in a row. Wait a minute, then retry.")
    elif response.status_code == 503:
        st.error(detail or "The search service is not ready. Try again shortly.")
    else:
        st.error(detail or f"Search failed ({response.status_code}). Try again.")
    return None


def _table_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert raw result rows into display-friendly rows."""
    rows = []
    for row in results:
        storey = ""
        if row.get("storey_min") is not None and row.get("storey_max") is not None:
            storey = f"{row['storey_min']}-{row['storey_max']}"
        rows.append(
            {
                "Town": row.get("town", ""),
                "Flat type": row.get("flat_type", ""),
                "Block": row.get("block", ""),
                "Street": row.get("street_name", ""),
                "Sqm": row.get("floor_area_sqm"),
                "Storey": storey,
                "Lease (yrs)": row.get("remaining_lease_years"),
                "Price (S$)": row.get("resale_price"),
            }
        )
    return rows


def render_result(payload: dict[str, Any]) -> None:
    """Render the parsed filter, the table and the AI explanation."""
    left, right = st.columns([1, 2])

    with left:
        st.subheader("Parsed filter")
        parsed = payload.get("parsed_filter") or {}
        st.code(json.dumps(parsed, indent=2, sort_keys=True), language="json")
        st.caption(
            f"Request {payload.get('request_id', '')[:8]} · "
            f"{payload.get('latency_ms', 0):.0f} ms"
        )

    with right:
        st.subheader("Listings")
        results = payload.get("results") or []
        if results:
            st.dataframe(_table_rows(results), hide_index=True)
        else:
            st.info(
                "No listings matched. Widen the budget or try a nearby town."
            )

        st.markdown("**AI-generated summary**")
        st.write(payload.get("explanation", ""))


def main() -> None:
    """Render the page."""
    st.set_page_config(page_title="BLK NOISE", layout="wide")

    api_url = settings.api_url
    st.session_state.setdefault("query", "")

    st.title("BLK NOISE")
    st.caption("Type it messy. Get a flat.")
    st.write(
        "Ask in plain language. We turn it into filters, pull matching HDB "
        "resale flats, and rank them against your words."
    )

    st.text_input(
        "What are you looking for?",
        key="query",
        placeholder="e.g. 4 room in Tampines under 600k, high floor",
    )

    search_col, _spacer = st.columns([1, 3])
    with search_col:
        search_clicked = st.button("Search", type="primary")

    if search_clicked:
        query = st.session_state.get("query", "").strip()
        if not query:
            st.warning("Type a query first, or pick one of the examples.")
        else:
            payload = call_search(query, api_url)
            if payload is not None:
                st.session_state["last_payload"] = payload

    st.caption("Try one of these:")
    example_columns = st.columns(len(EXAMPLES))
    for column, example in zip(example_columns, EXAMPLES):
        with column:
            st.button(
                example,
                key=f"example-{example}",
                on_click=_set_query,
                args=(example,),
            )

    payload = st.session_state.get("last_payload")
    if payload:
        render_result(payload)

    st.divider()
    st.caption(
        "Phone numbers and emails are masked before your query reaches the model."
    )


main()
