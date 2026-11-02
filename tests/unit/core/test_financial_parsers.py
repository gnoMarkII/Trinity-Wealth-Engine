"""Unit tests for core/financial_parsers.py.

Verifies:
- Standard integers and floats pass through intact
- Unit anchoring (x, เท่า, %, bps, THB, USD, บาท)
- Label anchoring (PE, P/E, EPS, Yield, Target)
- Disambiguation against calendar years (FY2026, 2025, 2567)
- Arrow indicator handling (▲, ▼)
- Comma thousand separators (1,588.05)
- Null and invalid string handling
"""
from decimal import Decimal

from core.financial_parsers import (
    extract_financial_decimal,
    extract_json_block,
    extract_valuation_metric,
    extract_labeled_decimal,
)
from core.datetime_utils import extract_date_by_labels
from core.text_utils import extract_labeled_value


def test_numeric_pass_through():
    assert extract_valuation_metric(15.5) == 15.5
    assert extract_valuation_metric(42) == 42.0
    assert extract_valuation_metric(0) == 0.0


def test_null_and_empty():
    assert extract_valuation_metric(None) is None
    assert extract_valuation_metric("") is None
    assert extract_valuation_metric("   ") is None
    assert extract_valuation_metric("-") is None
    assert extract_valuation_metric("N/A") is None
    assert extract_valuation_metric("null") is None
    assert extract_valuation_metric("invalid string") is None


def test_comma_and_indicators():
    assert extract_valuation_metric("1,588.05") == 1588.05
    assert extract_valuation_metric("▲0.50%") == 0.5
    assert extract_valuation_metric("▼-1.25%") == -1.25
    assert extract_valuation_metric("▼1.25%") == -1.25
    assert extract_valuation_metric("32.4700 THB") == 32.47


def test_unit_and_label_anchored():
    # Unit anchored
    assert extract_valuation_metric("Target multiple 18.5x") == 18.5
    assert extract_valuation_metric("P/E 12.4เท่า") == 12.4
    assert extract_valuation_metric("P/E 12.4เท่าของกำไร") == 12.4
    assert extract_valuation_metric("Spread widened by 45 bps") == 45.0
    assert extract_valuation_metric("เป้าหมาย 45.50 บาท") == 45.50
    assert extract_valuation_metric("เป้าหมาย 45.50บาทต่อหุ้น") == 45.50

    # Label anchored
    assert extract_valuation_metric("Forward P/E: 14.8") == 14.8
    assert extract_valuation_metric("Current EPS = 3.25") == 3.25
    assert extract_valuation_metric("Target: 75.00") == 75.0


def test_year_token_disambiguation():
    # In naive regex, FY2026 matched 2026 instead of 15.5
    assert extract_valuation_metric("FY2026 Target PE is 15.5x") == 15.5
    assert extract_valuation_metric("FY2025 EPS: 4.5") == 4.5
    assert extract_valuation_metric("2025 Yield: 4.2%") == 4.2
    assert extract_valuation_metric("Target for 2026 is 55.0 บาท") == 55.0
    assert extract_valuation_metric("กำไรปี2026อยู่ที่ 15.5x") == 15.5

    # Standalone year without any other metric falls back safely
    assert extract_valuation_metric("2026") == 2026.0


def test_parse_first_json_extractor():
    from tools.market.adapters.promise_extractor_adapter import _extract_json_block

    # 1. Direct JSON
    assert _extract_json_block('{"key": "value"}') == {"key": "value"}

    # 2. Markdown fence with prefix and suffix text
    text_fence = (
        "Here is the synthesized output:\n"
        "```json\n"
        '{"status": "ok", "items": [1, 2, 3]}\n'
        "```\n"
        "Hope this helps!"
    )
    assert _extract_json_block(text_fence) == {"status": "ok", "items": [1, 2, 3]}

    # 3. Non-greedy balanced braces
    text_multi = 'Analysis: {"main": {"nested": true}} trailing text {"another": 99}'
    assert _extract_json_block(text_multi) == {"main": {"nested": True}}

    # 4. Invalid
    assert _extract_json_block("Plain text with no json") is None
    assert _extract_json_block(None) is None
    assert extract_json_block('{"direct": 123}') == {"direct": 123}


def test_extract_financial_decimal():
    # Pass-through & Numbers
    assert extract_financial_decimal(Decimal("100.50")) == Decimal("100.50")
    assert extract_financial_decimal(1500) == Decimal("1500")
    assert extract_financial_decimal(12.75) == Decimal("12.75")

    # Commas & Currencies
    assert extract_financial_decimal("1,588,200.50 THB") == Decimal("1588200.50")
    assert extract_financial_decimal("35.50 บาท") == Decimal("35.50")
    assert extract_financial_decimal("฿12,450.00") == Decimal("12450.00")

    # Accounting parentheses (Negative)
    assert extract_financial_decimal("(1,500.25)") == Decimal("-1500.25")
    assert extract_financial_decimal("(45.0)") == Decimal("-45.0")

    # Direction indicators
    assert extract_financial_decimal("▲ 0.75%") == Decimal("0.75")
    assert extract_financial_decimal("▼ 1.25%") == Decimal("-1.25")

    # Null / Invalid
    assert extract_financial_decimal(None) is None
    assert extract_financial_decimal("") is None
    assert extract_financial_decimal("-") is None
    assert extract_financial_decimal("N/A") is None


def test_extract_labeled_decimal():
    sample = "Total Buy: 15,250.75 THB \n Amount: 1,500.00"
    assert extract_labeled_decimal(sample, ["Total Buy", "ยอดรวม"]) == Decimal("15250.75")
    assert extract_labeled_decimal(sample, ["Amount", "จำนวนเงิน"]) == Decimal("1500.00")
    assert extract_labeled_decimal(sample, ["Nonexistent"], default=Decimal("0")) == Decimal("0")


def test_extract_date_by_labels():
    from datetime import date
    sample = "Effective Date: 15/06/2026 \n วันที่ออกเอกสาร: 12/06/2569"
    assert extract_date_by_labels(sample, ["Effective Date", "วันที่คำสั่งมีผล"]) == date(2026, 6, 15)
    assert extract_date_by_labels(sample, ["Date of Issue", "วันที่ออกเอกสาร"]) == date(2026, 6, 12)
    assert extract_date_by_labels(sample, ["Missing Date"]) is None


def test_extract_labeled_value():
    sample = "Tax Invoice No.: INV-2026-001 \n เลขที่อ้างอิง: B25092602371"
    assert extract_labeled_value(sample, ["Tax Invoice No.", "เลขที่ใบกำกับภาษี"]) == "INV-2026-001"
    assert extract_labeled_value(sample, ["Ref No", "เลขที่อ้างอิง"]) == "B25092602371"
    assert extract_labeled_value(sample, ["Missing"]) is None

