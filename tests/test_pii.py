"""Tests for app/services/pii.py."""

from __future__ import annotations

import pytest

from app.services.pii import mask_pii


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # 1. plain text without PII is untouched
        ("Any 4 room flat in Bedok?", "Any 4 room flat in Bedok?"),
        # 2. prices and other short numbers must never be masked
        ("The flat costs 600000 dollars", "The flat costs 600000 dollars"),
        # 3. several non-PII numbers in one sentence
        (
            "Built in 1985, 93 sqm, postal 600123, price 450000",
            "Built in 1985, 93 sqm, postal 600123, price 450000",
        ),
        # 4. email addresses
        (
            "Email me at john.doe@example.com.sg please",
            "Email me at [EMAIL] please",
        ),
        # 5. Indonesian number with +62 and spaces/dashes
        (
            "Call me at +62 812-3456-7890",
            "Call me at [PHONE]",
        ),
        # 6. Indonesian local mobile in dashed groups
        (
            "HP saya 0812-3456-7890",
            "HP saya [PHONE]",
        ),
        # 7. Indonesian number with 62 prefix only
        (
            "WA 6281234567890",
            "WA [PHONE]",
        ),
        # 8. Singapore number with +65
        (
            "WhatsApp +65 9123 4567",
            "WhatsApp [PHONE]",
        ),
        # 9. Singapore local mobile without country code
        (
            "sms 9123-4567 done",
            "sms [PHONE] done",
        ),
        # 10. price next to a phone number: only the phone is masked
        (
            "Price 600000, contact 81234567 now",
            "Price 600000, contact [PHONE] now",
        ),
        # 11. email and phone together
        (
            "Reach a_b@sub.domain.co or 021-555-1234",
            "Reach [EMAIL] or [PHONE]",
        ),
        # 12. empty and whitespace-only input
        ("", ""),
        ("   ", "   "),
    ],
)
def test_mask_pii(text, expected):
    assert mask_pii(text) == expected


def test_mask_pii_is_idempotent():
    once = mask_pii("Call +65 9123 4567 or mail me@x.com")
    assert mask_pii(once) == once


def test_mask_pii_leaves_other_text_intact():
    text = "3 room flat in Toa Payoh, 67 sqm, lease 61 years"
    assert mask_pii(text) == text
