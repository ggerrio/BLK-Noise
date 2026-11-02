"""Tests for app/services/explainer.py with the LLM mocked."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.services import explainer
from app.services.llm_parser import LLMTimeoutError

ROWS = [
    {
        "town": "TAMPINES",
        "flat_type": "4 ROOM",
        "floor_area_sqm": 92.0,
        "storey_min": 10,
        "storey_max": 12,
        "remaining_lease_years": 70.0,
        "resale_price": 580000.0,
    },
    {
        "town": "TAMPINES",
        "flat_type": "5 ROOM",
        "floor_area_sqm": 110.0,
        "storey_min": 13,
        "storey_max": 15,
        "remaining_lease_years": 80.0,
        "resale_price": 680000.0,
    },
]


class FakeClient:
    def __init__(self, text=None, error=None):
        self.text = text
        self.error = error
        self.calls: list[dict] = []
        self.interactions = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(output_text=self.text)


def _install(monkeypatch, client):
    monkeypatch.setattr(explainer, "_get_client", lambda: client)
    return client


def test_explain_returns_grounded_text(monkeypatch):
    text = "Two TAMPINES flats match: 92 sqm at S$580000 and 110 sqm at S$680000."
    client = _install(monkeypatch, FakeClient(text=text))

    result = explainer.explain("4 room in Tampines", ROWS)

    assert result == text
    assert client.calls[0]["system_instruction"] == explainer.EXPLAIN_SYSTEM_PROMPT
    assert "TAMPINES" in client.calls[0]["input"]
    assert client.calls[0]["timeout"] == explainer.EXPLAIN_TIMEOUT_SECONDS


def test_explain_rejects_invented_values(monkeypatch):
    _install(
        monkeypatch,
        FakeClient(text="A 4 ROOM flat in TAMPINES at S$123456789 - great value."),
    )

    result = explainer.explain("4 room in Tampines", ROWS)

    assert "123456789" not in result
    assert result == explainer._fallback_explanation(ROWS)
    assert "S$580000" in result


def test_explain_falls_back_on_empty_response(monkeypatch):
    _install(monkeypatch, FakeClient(text=""))

    result = explainer.explain("4 room", ROWS)

    assert result == explainer._fallback_explanation(ROWS)


def test_explain_raises_on_timeout(monkeypatch):
    _install(monkeypatch, FakeClient(error=TimeoutError("deadline exceeded")))

    with pytest.raises(LLMTimeoutError):
        explainer.explain("4 room", ROWS)


def test_explain_falls_back_on_other_errors(monkeypatch):
    _install(monkeypatch, FakeClient(error=RuntimeError("boom")))

    assert explainer.explain("4 room", ROWS) == explainer._fallback_explanation(ROWS)


def test_explain_without_rows_returns_clear_message(monkeypatch):
    client = _install(monkeypatch, FakeClient(text="unused"))

    result = explainer.explain("5 room in Bedok", [])

    assert "No listings matched" in result
    assert client.calls == []  # no LLM call when there is nothing to explain


def test_mentions_only_known_values():
    assert explainer.mentions_only_known_values(
        "92 sqm at S$580000 in TAMPINES", ROWS
    )
    assert not explainer.mentions_only_known_values("92 sqm at S$999999", ROWS)
    assert explainer.mentions_only_known_values("No numbers here", ROWS)
