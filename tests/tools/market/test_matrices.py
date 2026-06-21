"""Test market_tools: ticker normalization, currency-aware formatting, TH/US market routing"""
import pytest


import tools.market.core as core
import tools.market.fundamentals as fundamentals
import tools.market.financials as financials
import tools.market.news as news
import tools.market.technical as technical
import tools.market.consensus as consensus
from types import SimpleNamespace
mt = SimpleNamespace(
    ingest_stock_fundamentals=fundamentals.ingest_stock_fundamentals,
    ingest_financial_health=fundamentals.ingest_financial_health,
    ingest_financial_trends=financials.ingest_financial_trends,
    ingest_stock_news=news.ingest_stock_news,
    ingest_stock_momentum=technical.ingest_stock_momentum,
    ingest_stock_consensus=consensus.ingest_stock_consensus,
)

ALL_STOCK_TOOLS = [
    "ingest_stock_fundamentals",
    "ingest_financial_health",
    "ingest_financial_trends",
    "ingest_stock_news",
    "ingest_stock_momentum",
    "ingest_stock_consensus",
]
MARKET_CASES = [
    ("TH", "PTT", "PTT.BK", "THB"),
    ("US", "AAPL", "AAPL", "USD"),
]
MONEY_BEARING_TOOLS = [t for t in ALL_STOCK_TOOLS if t != "ingest_stock_news"]



# --- Pure helpers ---




class TestBackCompatBareTickerCall:
    def test_us_default_when_market_omitted(self, mock_yf_ticker):
        """back-compat: tools เดิมที่ไม่ส่ง market ต้อง default 'US'"""
        result = fundamentals.ingest_stock_fundamentals.invoke({"ticker": "MSFT"})
        assert mock_yf_ticker["ticker"] == "MSFT"
        assert "USD" in result


@pytest.mark.parametrize("tool_name", ALL_STOCK_TOOLS)
@pytest.mark.parametrize("market,input_sym,expected_yf,expected_currency", MARKET_CASES)
class TestMarketCoverageMatrix:
    """Contract: ทุก stock tool × ทุก market ต้องทำงานสอดคล้องกัน
    - ticker ที่ส่งเข้า yfinance ต้องเป็น format ที่ถูกต้อง (TH = .BK, US = bare)
    - YAML frontmatter ต้องมี market field + market_{xx} tag
    - YAML ต้องระบุ ticker เป็น bare symbol (ไม่มี .BK)
    """

    def test_yf_ticker_normalized(
        self, tool_name, market, input_sym, expected_yf, expected_currency, mock_yf_ticker
    ):
        tool = getattr(mt, tool_name)
        tool.invoke({"ticker": input_sym, "market": market})
        assert mock_yf_ticker["ticker"] == expected_yf, (
            f"{tool_name}(market={market}) ส่ง '{mock_yf_ticker['ticker']}' "
            f"เข้า yfinance แทนที่จะเป็น '{expected_yf}'"
        )

    def test_yaml_has_market_field(
        self, tool_name, market, input_sym, expected_yf, expected_currency, mock_yf_ticker
    ):
        tool = getattr(mt, tool_name)
        result = tool.invoke({"ticker": input_sym, "market": market})
        assert f"market: {market}" in result, (
            f"{tool_name}({market}): YAML frontmatter ไม่มี 'market: {market}'"
        )

    def test_yaml_has_market_tag(
        self, tool_name, market, input_sym, expected_yf, expected_currency, mock_yf_ticker
    ):
        tool = getattr(mt, tool_name)
        result = tool.invoke({"ticker": input_sym, "market": market})
        market_tag = f"market_{market.lower()}"
        assert market_tag in result, (
            f"{tool_name}({market}): tags ไม่มี '{market_tag}'"
        )

    def test_yaml_uses_bare_ticker_not_bk(
        self, tool_name, market, input_sym, expected_yf, expected_currency, mock_yf_ticker
    ):
        """Display layer ต้องไม่หลุด .BK ออกมาให้ user เห็น"""
        tool = getattr(mt, tool_name)
        result = tool.invoke({"ticker": input_sym, "market": market})
        # ใน title ต้องเป็น bare symbol
        assert f"title: {input_sym} " in result
        # YAML tags ไม่ควรมี ".bk"
        if market == "TH":
            tag_line = next(line for line in result.split("\n") if line.startswith("tags:"))
            assert ".bk" not in tag_line.lower(), f"tag line leaked .BK: {tag_line}"


@pytest.mark.parametrize("tool_name", MONEY_BEARING_TOOLS)
@pytest.mark.parametrize("market,input_sym,expected_yf,expected_currency", MARKET_CASES)
class TestCurrencyLabelMatrix:
    """ทุก tool ที่มีตัวเลขเงินต้องแสดง currency label ตรงกับ market"""

    def test_currency_label_in_output(
        self, tool_name, market, input_sym, expected_yf, expected_currency, mock_yf_ticker
    ):
        tool = getattr(mt, tool_name)
        result = tool.invoke({"ticker": input_sym, "market": market})
        assert expected_currency in result, (
            f"{tool_name}({market}): output ไม่มี '{expected_currency}' label"
        )

    def test_wrong_currency_not_in_output(
        self, tool_name, market, input_sym, expected_yf, expected_currency, mock_yf_ticker
    ):
        """TH market output ห้ามมี 'USD' (และ vice versa) — กัน label โกหก"""
        tool = getattr(mt, tool_name)
        result = tool.invoke({"ticker": input_sym, "market": market})
        wrong_currency = "USD" if expected_currency == "THB" else "THB"
        assert wrong_currency not in result, (
            f"{tool_name}({market}): output มี '{wrong_currency}' ปนมา (ควรเป็น {expected_currency} ล้วน)"
        )


class TestUserInputTolerance:
    def test_lowercase_ticker_normalized(self, mock_yf_ticker):
        fundamentals.ingest_stock_fundamentals.invoke({"ticker": "ptt", "market": "TH"})
        assert mock_yf_ticker["ticker"] == "PTT.BK"

    def test_ticker_with_spaces_stripped(self, mock_yf_ticker):
        fundamentals.ingest_stock_fundamentals.invoke({"ticker": "  AAPL  ", "market": "US"})
        assert mock_yf_ticker["ticker"] == "AAPL"

    def test_user_supplied_bk_suffix_handled(self, mock_yf_ticker):
        """User เผลอใส่ PTT.BK เอง — ระบบไม่ควร double-suffix"""
        fundamentals.ingest_stock_fundamentals.invoke({"ticker": "PTT.BK", "market": "TH"})
        assert mock_yf_ticker["ticker"] == "PTT.BK"
        # ไม่ใช่ PTT.BK.BK

    def test_user_supplied_bk_stripped_for_display(self, mock_yf_ticker):
        """Display title ต้องโชว์ PTT ไม่ใช่ PTT.BK"""
        result = fundamentals.ingest_stock_fundamentals.invoke({"ticker": "PTT.BK", "market": "TH"})
        assert "title: PTT Fundamentals" in result
        assert "PTT.BK Fundamentals" not in result


@pytest.mark.parametrize("tool_name", ALL_STOCK_TOOLS)
@pytest.mark.parametrize("market,input_sym,expected_yf,expected_currency", MARKET_CASES)
class TestYamlFrontmatterConsistency:
    """ทุก tool output ต้องมี YAML structure ที่ครบ + parsable"""

    def test_yaml_opening_and_closing(
        self, tool_name, market, input_sym, expected_yf, expected_currency, mock_yf_ticker
    ):
        tool = getattr(mt, tool_name)
        result = tool.invoke({"ticker": input_sym, "market": market})
        lines = result.split("\n")
        assert lines[0] == "---", f"{tool_name}: ไม่มี YAML opening '---'"
        # หา closing '---' ในช่วง 20 บรรทัดแรก
        closing_idx = next(
            (i for i, line in enumerate(lines[1:20], start=1) if line == "---"), None
        )
        assert closing_idx is not None, f"{tool_name}: ไม่มี YAML closing '---'"

    def test_yaml_required_fields_present(
        self, tool_name, market, input_sym, expected_yf, expected_currency, mock_yf_ticker
    ):
        tool = getattr(mt, tool_name)
        result = tool.invoke({"ticker": input_sym, "market": market})
        # Required fields ทุก stock tool ต้องมี
        required = ["title:", "entity_type:", "market:", "date:", "last_updated:", "tags:"]
        for field in required:
            assert field in result, f"{tool_name}({market}): YAML ขาด field '{field}'"


class TestCrossMarketIsolation:
    def test_th_then_us_no_state_leak(self, mock_yf_ticker):
        """เรียก TH ก่อน → US ครั้งถัดไป — ticker ที่ส่งเข้า yfinance ต้องเป็นของ US ล้วน"""
        fundamentals.ingest_stock_fundamentals.invoke({"ticker": "PTT", "market": "TH"})
        assert mock_yf_ticker["ticker"] == "PTT.BK"

        fundamentals.ingest_stock_fundamentals.invoke({"ticker": "AAPL", "market": "US"})
        assert mock_yf_ticker["ticker"] == "AAPL"  # ไม่ใช่ AAPL.BK

    def test_us_then_th_no_state_leak(self, mock_yf_ticker):
        fundamentals.ingest_stock_fundamentals.invoke({"ticker": "AAPL", "market": "US"})
        assert mock_yf_ticker["ticker"] == "AAPL"

        fundamentals.ingest_stock_fundamentals.invoke({"ticker": "PTT", "market": "TH"})
        assert mock_yf_ticker["ticker"] == "PTT.BK"

