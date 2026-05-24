"""Test market_tools: ticker normalization, currency-aware formatting, TH/US market routing"""
import pytest

from tools import market_tools as mt


# --- Pure helpers ---


class TestNormalizeYfTicker:
    def test_us_unchanged(self):
        assert mt._normalize_yf_ticker("AAPL", "US") == "AAPL"

    def test_th_adds_bk_suffix(self):
        assert mt._normalize_yf_ticker("PTT", "TH") == "PTT.BK"

    def test_th_no_double_suffix(self):
        assert mt._normalize_yf_ticker("PTT.BK", "TH") == "PTT.BK"

    def test_strip_and_upper(self):
        assert mt._normalize_yf_ticker(" ptt ", "TH") == "PTT.BK"
        assert mt._normalize_yf_ticker(" aapl ", "US") == "AAPL"


class TestCurrencyFor:
    def test_th_returns_thb(self):
        assert mt._currency_for("TH") == "THB"

    def test_us_returns_usd(self):
        assert mt._currency_for("US") == "USD"


class TestFmtLargeWithCurrency:
    def test_default_currency_usd(self):
        assert mt._fmt_large(1_500_000_000) == "1.50B USD"

    def test_custom_currency_thb(self):
        assert mt._fmt_large(1_500_000_000, "THB") == "1.50B THB"

    def test_trillion(self):
        assert mt._fmt_large(2.5e12, "USD") == "2.50T USD"

    def test_million(self):
        assert mt._fmt_large(5_000_000, "THB") == "5.00M THB"

    def test_none_returns_na(self):
        assert mt._fmt_large(None, "THB") == "N/A"


class TestFmtFinWithCurrency:
    def test_positive_billions_thb(self):
        assert mt._fmt_fin(2_500_000_000, "THB") == "2.50B THB"

    def test_negative_billions_usd(self):
        assert mt._fmt_fin(-3_000_000_000, "USD") == "-3.00B USD"

    def test_default_back_compat(self):
        assert "USD" in mt._fmt_fin(1_000_000_000)


# --- Integration: tools call yfinance with normalized ticker + currency in output ---


@pytest.fixture
def mock_yf_ticker(monkeypatch):
    """Mock yf.Ticker — track ticker arg ที่ส่งเข้าไป + คืน fake info"""
    captured = {}

    fake_info = {
        "quoteType": "EQUITY",
        "shortName": "Mock Company",
        "sector": "Energy",
        "industry": "Oil & Gas",
        "marketCap": 1_500_000_000_000,
        "trailingPE": 12.5,
        "currentPrice": 40.0,
        "operatingCashflow": 100_000_000_000,
        "freeCashflow": 80_000_000_000,
        "totalCash": 200_000_000_000,
        "totalDebt": 150_000_000_000,
        "targetMeanPrice": 45.0,
        "targetLowPrice": 38.0,
        "targetHighPrice": 50.0,
        "fiftyDayAverage": 39.0,
        "twoHundredDayAverage": 37.0,
        "fiftyTwoWeekHigh": 45.0,
        "fiftyTwoWeekLow": 30.0,
    }

    class FakeTicker:
        def __init__(self, ticker):
            captured["ticker"] = ticker
            self.info = fake_info
            self.news = [{"content": {"title": "Test news", "provider": {"displayName": "Test"},
                                       "canonicalUrl": {"url": "http://test"}}}]

        @property
        def sustainability(self):
            return None

        @property
        def financials(self):
            import pandas as pd
            from datetime import datetime
            return pd.DataFrame(
                {datetime(2024, 12, 31): [100e9, 30e9]},
                index=["Total Revenue", "Net Income"],
            )

        @property
        def insider_transactions(self):
            return None

    monkeypatch.setattr(mt.yf, "Ticker", FakeTicker)
    # bypass retry wrapper
    monkeypatch.setattr(mt, "_with_retry", lambda fn: fn())
    return captured


class TestIngestFundamentalsThaiStock:
    def test_th_market_normalizes_ticker(self, mock_yf_ticker):
        result = mt.ingest_stock_fundamentals.invoke({"ticker": "PTT", "market": "TH"})
        assert mock_yf_ticker["ticker"] == "PTT.BK"
        # Display ใช้ bare ticker
        assert "PTT Fundamentals" in result
        assert "(PTT, TH)" in result
        # Currency suffix ต้องเป็น THB
        assert "THB" in result
        assert "1.50T THB" in result  # market cap formatted

    def test_us_market_default(self, mock_yf_ticker):
        result = mt.ingest_stock_fundamentals.invoke({"ticker": "AAPL"})
        assert mock_yf_ticker["ticker"] == "AAPL"
        assert "USD" in result
        assert "(AAPL, US)" in result


class TestIngestFinancialHealthThaiStock:
    def test_th_market_currency_in_output(self, mock_yf_ticker):
        result = mt.ingest_financial_health.invoke({"ticker": "PTT", "market": "TH"})
        assert mock_yf_ticker["ticker"] == "PTT.BK"
        # Cash flow ต้องแสดงเป็น THB
        assert "100.00B THB" in result  # operatingCashflow
        assert "80.00B THB" in result   # freeCashflow
        assert "USD" not in result.replace("USDT", "")  # ไม่มี USD label (ยกเว้น token USDT ถ้ามี)


class TestIngestFinancialTrendsThaiStock:
    def test_th_market_currency(self, mock_yf_ticker):
        result = mt.ingest_financial_trends.invoke({"ticker": "PTT", "market": "TH"})
        assert mock_yf_ticker["ticker"] == "PTT.BK"
        assert "B THB" in result  # revenue/income formatted in THB
        assert "market: TH" in result


class TestIngestStockNewsThaiStock:
    def test_th_market_ticker_normalization(self, mock_yf_ticker):
        result = mt.ingest_stock_news.invoke({"ticker": "PTT", "market": "TH"})
        assert mock_yf_ticker["ticker"] == "PTT.BK"
        assert "ข่าวล่าสุด: PTT (TH)" in result


class TestIngestStockMomentumThaiStock:
    def test_th_market_currency(self, mock_yf_ticker):
        result = mt.ingest_stock_momentum.invoke({"ticker": "PTT", "market": "TH"})
        assert mock_yf_ticker["ticker"] == "PTT.BK"
        assert "40.00 THB" in result  # current price
        assert "39.00 THB" in result  # MA50


class TestIngestStockConsensusThaiStock:
    def test_th_market_currency(self, mock_yf_ticker):
        result = mt.ingest_stock_consensus.invoke({"ticker": "PTT", "market": "TH"})
        assert mock_yf_ticker["ticker"] == "PTT.BK"
        assert "45.00 THB" in result  # target mean
        assert "market: TH" in result


class TestBackCompatBareTickerCall:
    def test_us_default_when_market_omitted(self, mock_yf_ticker):
        """back-compat: tools เดิมที่ไม่ส่ง market ต้อง default 'US'"""
        result = mt.ingest_stock_fundamentals.invoke({"ticker": "MSFT"})
        assert mock_yf_ticker["ticker"] == "MSFT"
        assert "USD" in result


# === Coverage Matrix: 6 tools × 2 markets ===

# ทุก ingest_stock_* tool ต้องรับ market='TH' หรือ market='US' และทำงานถูกทั้ง 2 กรณี
ALL_STOCK_TOOLS = [
    "ingest_stock_fundamentals",
    "ingest_stock_news",
    "ingest_stock_consensus",
    "ingest_financial_trends",
    "ingest_stock_momentum",
    "ingest_financial_health",
]

# ticker ที่ใช้ทดสอบจริงต่อ market — symbol bare, ไม่มี suffix
MARKET_CASES = [
    ("TH", "PTT", "PTT.BK", "THB"),
    ("US", "AAPL", "AAPL", "USD"),
]


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


# News tool ไม่มี money fields → แยก currency check ออกมาเฉพาะ tools ที่มี
MONEY_BEARING_TOOLS = [t for t in ALL_STOCK_TOOLS if t != "ingest_stock_news"]


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


# === User Input Tolerance ===


class TestUserInputTolerance:
    def test_lowercase_ticker_normalized(self, mock_yf_ticker):
        mt.ingest_stock_fundamentals.invoke({"ticker": "ptt", "market": "TH"})
        assert mock_yf_ticker["ticker"] == "PTT.BK"

    def test_ticker_with_spaces_stripped(self, mock_yf_ticker):
        mt.ingest_stock_fundamentals.invoke({"ticker": "  AAPL  ", "market": "US"})
        assert mock_yf_ticker["ticker"] == "AAPL"

    def test_user_supplied_bk_suffix_handled(self, mock_yf_ticker):
        """User เผลอใส่ PTT.BK เอง — ระบบไม่ควร double-suffix"""
        mt.ingest_stock_fundamentals.invoke({"ticker": "PTT.BK", "market": "TH"})
        assert mock_yf_ticker["ticker"] == "PTT.BK"
        # ไม่ใช่ PTT.BK.BK

    def test_user_supplied_bk_stripped_for_display(self, mock_yf_ticker):
        """Display title ต้องโชว์ PTT ไม่ใช่ PTT.BK"""
        result = mt.ingest_stock_fundamentals.invoke({"ticker": "PTT.BK", "market": "TH"})
        assert "title: PTT Fundamentals" in result
        assert "PTT.BK Fundamentals" not in result


# === YAML Frontmatter Consistency ===


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


# === Cross-market round-trip: เรียก TH + US สลับกัน ต้องไม่ leak state ===


class TestCrossMarketIsolation:
    def test_th_then_us_no_state_leak(self, mock_yf_ticker):
        """เรียก TH ก่อน → US ครั้งถัดไป — ticker ที่ส่งเข้า yfinance ต้องเป็นของ US ล้วน"""
        mt.ingest_stock_fundamentals.invoke({"ticker": "PTT", "market": "TH"})
        assert mock_yf_ticker["ticker"] == "PTT.BK"

        mt.ingest_stock_fundamentals.invoke({"ticker": "AAPL", "market": "US"})
        assert mock_yf_ticker["ticker"] == "AAPL"  # ไม่ใช่ AAPL.BK

    def test_us_then_th_no_state_leak(self, mock_yf_ticker):
        mt.ingest_stock_fundamentals.invoke({"ticker": "AAPL", "market": "US"})
        assert mock_yf_ticker["ticker"] == "AAPL"

        mt.ingest_stock_fundamentals.invoke({"ticker": "PTT", "market": "TH"})
        assert mock_yf_ticker["ticker"] == "PTT.BK"
