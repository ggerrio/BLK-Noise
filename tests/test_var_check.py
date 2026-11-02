"""Tests for app/services/var_check.py."""

from __future__ import annotations

import pytest

from app.services.var_check import check_description, hallucination_rate

ROW = {
    "id": 1,
    "town": "TAMPINES",
    "flat_type": "4 ROOM",
    "block": "301",
    "street_name": "TAMPINES ST 21",
    "storey_min": 10,
    "storey_max": 12,
    "floor_area_sqm": 93.0,
    "remaining_lease_years": 61.5,
    "resale_price": 450000,
}

GOOD_TEXT = (
    "This 4 ROOM flat in TAMPINES sits on storeys 10 to 12 with 93 sqm. "
    "It has 61.5 years of lease left and costs S$450,000."
)


def test_correct_text_passes_and_lists_claims():
    result = check_description(GOOD_TEXT, ROW)

    assert result.passed
    assert result.errors == []
    kinds = {claim.kind for claim in result.claims}
    assert kinds == {
        "town",
        "flat_type",
        "storey",
        "area_sqm",
        "remaining_lease_years",
        "price",
    }
    assert all(claim.ok for claim in result.claims)


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("A 4 ROOM flat in TAMPINES for S$500,000.", "price"),
        ("A 4 ROOM flat in BEDOK, 93 sqm for S$450,000.", "town"),
        ("A 4 ROOM flat in TAMPINES on storeys 20 to 22.", "storey"),
        ("A 4 ROOM flat in TAMPINES with 120 sqm.", "area_sqm"),
        ("A 4 ROOM flat in TAMPINES with 45 years of lease left.", "remaining_lease_years"),
        ("A 5 ROOM flat in TAMPINES for S$450,000.", "flat_type"),
    ],
)
def test_false_claim_fails_the_check(text, kind):
    result = check_description(text, ROW)

    assert not result.passed
    failed = [claim for claim in result.claims if not claim.ok]
    assert failed
    assert {claim.kind for claim in failed} == {kind}
    assert any(kind in error for error in result.errors)


def test_single_storey_outside_band_fails():
    result = check_description("A quiet 4th floor unit in TAMPINES.", ROW)

    assert not result.passed
    assert [claim.kind for claim in result.claims if not claim.ok] == ["storey"]


def test_single_storey_inside_band_passes():
    result = check_description("A bright 11th floor unit in TAMPINES.", ROW)

    assert result.passed


def test_text_without_claims_is_vacuous_pass():
    result = check_description("A comfortable flat worth a look.", ROW)

    assert result.passed
    assert result.claims == []


def test_missing_row_value_fails_matching_claim():
    row = {**ROW, "remaining_lease_years": None}
    result = check_description("It has 61.5 years of lease left.", row)

    assert not result.passed


def test_hallucination_rate():
    good = check_description(GOOD_TEXT, ROW)
    bad = check_description("A 4 ROOM flat in BEDOK for S$900,000.", ROW)

    assert hallucination_rate([good, good]) == 0.0
    assert hallucination_rate([good, bad]) == 0.5
    assert hallucination_rate([bad, bad]) == 1.0
    assert hallucination_rate([]) == 0.0
