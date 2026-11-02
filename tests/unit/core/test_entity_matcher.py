"""Unit tests for core/news_digest/entity_matcher.py.

Verifies:
- Explicit cashtags ($NVDA, $ON)
- Ambiguous stopword suppression (e.g. 'ON Monday', 'IT department' ignored unless cashtagged)
- Thai-adjacent ticker boundaries (e.g. 'หุ้นDELTA', 'ทยอยซื้อPTT')
- Thai company alias resolution (e.g. 'หุ้นเดลต้า' -> DELTA, 'ปตท.' -> PTT)
"""
from core.news_digest.entity_matcher import (
    contains_ticker_boundary,
    find_matching_tickers,
    is_ambiguous_ticker,
)


def test_is_ambiguous_ticker():
    assert is_ambiguous_ticker("ON") is True
    assert is_ambiguous_ticker("IT") is True
    assert is_ambiguous_ticker("ALL") is True
    assert is_ambiguous_ticker("A") is True
    assert is_ambiguous_ticker("US") is True
    assert is_ambiguous_ticker("DELTA") is False
    assert is_ambiguous_ticker("PTT") is False
    assert is_ambiguous_ticker("NVDA") is False


def test_contains_ticker_boundary():
    # Standard English boundaries
    assert contains_ticker_boundary("Review DELTA today", "DELTA") is True
    assert contains_ticker_boundary("DELTA.BK performance", "DELTA") is True
    assert contains_ticker_boundary("NON-DELTA assets", "DELTA") is True
    assert contains_ticker_boundary("SUBDELTA assets", "DELTA") is False

    # Thai adjacent boundaries (where standard \b fails because Thai chars are treated as \w)
    assert contains_ticker_boundary("หุ้นDELTAปรับตัวขึ้นแรง", "DELTA") is True
    assert contains_ticker_boundary("แนะนำซื้อPTTเมื่อย่อตัว", "PTT") is True
    assert contains_ticker_boundary("กลุ่มADVANCและTRUE", "ADVANC") is True
    assert contains_ticker_boundary("กลุ่มADVANCและTRUE", "TRUE") is True


def test_find_matching_tickers_cashtag():
    # Disambiguates short tickers like ON with $
    text = "Earnings report: $ON posted strong results while IT sector dropped."
    target_symbols = {"ON", "IT", "AAPL"}
    matched = find_matching_tickers(text, target_symbols=target_symbols)
    # $ON is matched via cashtag, 'IT' is ignored as ambiguous stopword
    assert "ON" in matched
    assert "IT" not in matched


def test_find_matching_tickers_ambiguous_suppression():
    text = "On Monday, all teams attended the meeting in the IT department."
    target_symbols = {"ON", "ALL", "IN", "IT", "CPALL"}
    matched = find_matching_tickers(text, target_symbols=target_symbols)
    # All are ambiguous words and should not match without cashtag
    assert matched == set()


def test_find_matching_tickers_thai_aliases():
    text = "นักลงทุนจับตาหุ้นเดลต้า และ ปตท. หลังประกาศผลประกอบการไตรมาสล่าสุด"
    target_symbols = {"DELTA", "PTT", "CPALL"}
    matched = find_matching_tickers(text, target_symbols=target_symbols)
    assert "DELTA" in matched
    assert "PTT" in matched
    assert "CPALL" not in matched


def test_find_matching_tickers_explicit_override():
    # If a ticker is explicitly given (e.g. from an RSS feed's category tag), it is included
    text = "General market update without mentioning any ticker."
    matched = find_matching_tickers(
        text,
        target_symbols={"DELTA", "ON"},
        explicit_tickers=["DELTA.BK", "$ON"],
    )
    assert matched == {"DELTA", "ON"}
