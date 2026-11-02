"""Tests for app/services/llm_parser.py with the Gemini client mocked."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app import prompts
from app.services import llm_parser
from app.services.llm_parser import InvalidFilterError, parse_query


class FakeInteractions:
    """Stands in for client.interactions, replaying scripted responses."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class FakeClient:
    def __init__(self, responses):
        self.interactions = FakeInteractions(responses)


def _interaction(*, name: str = prompts.FUNCTION_NAME, arguments=None):
    """Build a fake interaction carrying one function call step."""
    step = SimpleNamespace(type="function_call", name=name, arguments=arguments or {})
    return SimpleNamespace(steps=[step])


def _no_call_interaction(text: str = "Sure, here is your answer."):
    return SimpleNamespace(
        steps=[SimpleNamespace(type="model_output", name=None, arguments=None)],
        output_text=text,
    )


@pytest.fixture(autouse=True)
def _clean_cache():
    llm_parser.clear_cache()
    yield
    llm_parser.clear_cache()


@pytest.fixture
def fake_client(monkeypatch):
    def install(responses):
        client = FakeClient(responses)
        monkeypatch.setattr(llm_parser, "_get_client", lambda: client)
        return client

    return install


# --- happy path -----------------------------------------------------------

def test_valid_response_returns_filter(fake_client):
    client = fake_client(
        [_interaction(arguments={"town": "TAMPINES", "max_price": 500000})]
    )

    result = parse_query("4 room under 500k in Tampines")

    assert result.town == "TAMPINES"
    assert result.max_price == 500000
    assert result.min_price is None
    assert len(client.interactions.calls) == 1


def test_request_uses_function_declaration_and_system_prompt(fake_client):
    client = fake_client([_interaction(arguments={"storey_min": 10})])

    parse_query("high floor please")

    call = client.interactions.calls[0]
    tools = call["tools"]
    assert tools[0]["type"] == "function"
    assert tools[0]["name"] == prompts.FUNCTION_NAME
    assert set(tools[0]["parameters"]["properties"]) == set(
        llm_parser.PropertyFilter.model_fields
    )
    assert call["system_instruction"] == prompts.SYSTEM_PROMPT
    assert "never write sql" in call["system_instruction"].lower()
    assert call["timeout"] == llm_parser.REQUEST_TIMEOUT_SECONDS
    assert call["model"]


def test_empty_query_returns_empty_filter_without_calling_llm(fake_client):
    client = fake_client([])

    result = parse_query("   ")

    assert result == llm_parser.PropertyFilter()
    assert client.interactions.calls == []


def test_cache_reuses_result(fake_client):
    client = fake_client([_interaction(arguments={"town": "BEDOK"})])

    first = parse_query("3 room in Bedok")
    second = parse_query("3 room in Bedok")

    assert first == second
    assert len(client.interactions.calls) == 1

    second.town = "TAMPINES"  # mutating the copy must not poison the cache
    assert parse_query("3 room in Bedok").town == "BEDOK"


# --- validation retry -----------------------------------------------------

def test_invalid_then_valid_retries_once(fake_client):
    client = fake_client(
        [
            _interaction(arguments={"min_price": 600000, "max_price": 500000}),
            _interaction(arguments={"min_price": 400000, "max_price": 500000}),
        ]
    )

    result = parse_query("between 600k and 500k")

    assert (result.min_price, result.max_price) == (400000, 500000)
    assert len(client.interactions.calls) == 2

    retry_input = client.interactions.calls[1]["input"]
    assert "min_price" in retry_input
    assert "max_price" in retry_input
    assert prompts.FUNCTION_NAME in retry_input


def test_invalid_twice_raises_invalid_filter_error(fake_client):
    client = fake_client(
        [
            _interaction(arguments={"storey_min": 12, "storey_max": 10}),
            _interaction(arguments={"storey_min": 15, "storey_max": 2}),
        ]
    )

    with pytest.raises(InvalidFilterError, match="invalid filter after retry"):
        parse_query("tall building")

    assert len(client.interactions.calls) == 2


def test_negative_values_are_retried_then_rejected(fake_client):
    client = fake_client(
        [
            _interaction(arguments={"min_price": -5}),
            _interaction(arguments={"max_price": -100}),
        ]
    )

    with pytest.raises(InvalidFilterError):
        parse_query("negative price")

    assert len(client.interactions.calls) == 2


def test_sql_output_is_rejected(fake_client):
    client = fake_client(
        [
            _interaction(arguments={"sql": "DROP TABLE listings"}),
            _interaction(arguments={"sql": "SELECT * FROM listings"}),
        ]
    )

    with pytest.raises(InvalidFilterError):
        parse_query("drop the table")

    assert len(client.interactions.calls) == 2


# --- failures -------------------------------------------------------------

def test_missing_function_call_retried_then_raises(fake_client):
    client = fake_client([_no_call_interaction(), _no_call_interaction()])

    with pytest.raises(InvalidFilterError, match="invalid filter after retry"):
        parse_query("4 room flat")

    assert len(client.interactions.calls) == 2


def test_transport_error_is_retried_with_backoff(fake_client, monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr(llm_parser.time, "sleep", sleeps.append)

    client = fake_client(
        [TimeoutError("deadline exceeded"), _interaction(arguments={"town": "BEDOK"})]
    )

    result = parse_query("bedok flat")

    assert result.town == "BEDOK"
    assert sleeps == [llm_parser.BACKOFF_BASE_SECONDS]
    assert len(client.interactions.calls) == 2


def test_persistent_transport_error_raises(fake_client, monkeypatch):
    monkeypatch.setattr(llm_parser.time, "sleep", lambda _s: None)
    client = fake_client([RuntimeError("boom"), RuntimeError("boom again")])

    with pytest.raises(InvalidFilterError, match="Gemini request failed"):
        parse_query("anything")


def test_timeout_raises_llm_timeout_error(fake_client, monkeypatch):
    monkeypatch.setattr(llm_parser.time, "sleep", lambda _s: None)
    fake_client([TimeoutError("deadline exceeded"), TimeoutError("still slow")])

    with pytest.raises(llm_parser.LLMTimeoutError, match="timed out"):
        parse_query("anything")
