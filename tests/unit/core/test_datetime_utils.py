from datetime import date
import pytest
from core.datetime_utils import parse_thai_or_iso_date, format_iso_date


def test_parse_iso_date():
    assert parse_thai_or_iso_date("2026-10-25") == date(2026, 10, 25)
    assert parse_thai_or_iso_date("2026/10/25") == date(2026, 10, 25)
    assert parse_thai_or_iso_date("2026-10-25T14:30:00Z") == date(2026, 10, 25)


def test_parse_thai_buddhist_era_slash_format():
    # 2569 BE should be converted to 2026 CE (2569 - 543)
    assert parse_thai_or_iso_date("25/10/2569") == date(2026, 10, 25)
    assert parse_thai_or_iso_date("01/01/2567") == date(2024, 1, 1)


def test_parse_thai_textual_format():
    # Thai month abbr with 4-digit BE
    assert parse_thai_or_iso_date("25 ต.ค. 2567") == date(2024, 10, 25)
    # Thai full month with 4-digit BE
    assert parse_thai_or_iso_date("15 มกราคม 2568") == date(2025, 1, 15)
    # Thai month with 2-digit BE (67 -> 2567 -> 2024)
    assert parse_thai_or_iso_date("10 เม.ย. 67") == date(2024, 4, 10)


def test_parse_english_textual_format():
    assert parse_thai_or_iso_date("25 Oct 2026") == date(2026, 10, 25)
    assert parse_thai_or_iso_date("October 25, 2026") == date(2026, 10, 25)


def test_parse_compact_numeric():
    assert parse_thai_or_iso_date("20261025") == date(2026, 10, 25)
    assert parse_thai_or_iso_date("25691025") == date(2026, 10, 25)


def test_parse_invalid_or_none():
    assert parse_thai_or_iso_date(None) is None
    assert parse_thai_or_iso_date("") is None
    assert parse_thai_or_iso_date("N/A") is None
    assert parse_thai_or_iso_date("invalid_date_string") is None


def test_format_iso_date():
    assert format_iso_date(date(2026, 10, 25)) == "2026-10-25"
    assert format_iso_date(None) == ""
