"""Mask personally identifiable information in free text.

Handles Indonesian and Singapore phone numbers (with or without country code,
spaces, dots and dashes) plus email addresses. Bare numbers such as prices
("600000") are deliberately left untouched: every pattern requires a real
phone shape (country code, mobile prefix or grouped digit blocks).
"""

from __future__ import annotations

import re

PHONE_TOKEN = "[PHONE]"
EMAIL_TOKEN = "[EMAIL]"

EMAIL_PATTERN = re.compile(
    r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)+"
)

_SEP = r"[\s.\-]?"

# Order matters: the most specific (country code) forms run first so a number
# is not partially consumed by the more permissive local patterns.
PHONE_PATTERNS: tuple[re.Pattern[str], ...] = (
    # +62 812 3456 7890 / 62-812-3456-7890 (Indonesian mobile)
    re.compile(rf"(?<![\d+])\+?62{_SEP}0?8[1-9](?:{_SEP}\d){{6,10}}(?!\d)"),
    # +62 21 555 1234 (Indonesian landline)
    re.compile(rf"(?<![\d+])\+?62{_SEP}2[1-9]\d{{1,2}}{_SEP}\d{{3,4}}{_SEP}\d{{3,4}}(?!\d)"),
    # +65 9123 4567 / 65-9123-4567 (Singapore)
    re.compile(rf"(?<![\d+])\+?65{_SEP}[689]\d{{3}}{_SEP}\d{{4}}(?!\d)"),
    # 0812 3456 7890 / 0812-3456-7890 (local Indonesian mobile)
    re.compile(rf"(?<!\d)0?8[1-9](?:{_SEP}\d){{6,10}}(?!\d)"),
    # (021) 555 1234 / 021-5551234 (local Indonesian landline)
    re.compile(rf"(?<!\d)\(?0[2-9]\d{{1,2}}\)?{_SEP}\d{{3,4}}{_SEP}\d{{3,4}}(?!\d)"),
    # 9123 4567 / 9123-4567 / 61234567 (local Singapore)
    re.compile(rf"(?<!\d)[689]\d{{3}}{_SEP}\d{{4}}(?!\d)"),
)


def mask_pii(text: str) -> str:
    """Replace emails and phone numbers in `text` with placeholder tokens."""
    if not text:
        return text

    masked = EMAIL_PATTERN.sub(EMAIL_TOKEN, text)
    for pattern in PHONE_PATTERNS:
        masked = pattern.sub(PHONE_TOKEN, masked)
    return masked
