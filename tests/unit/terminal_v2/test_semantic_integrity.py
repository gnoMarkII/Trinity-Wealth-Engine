"""Unit tests verifying Semantic Integrity (No Cross-Asset Silent Substitutions).

Crucial business & architectural rules:
1. ^SET.BK cannot substitute for SET 4-investor-type flow.
2. GC=F (gold futures) cannot substitute for Thai retail physical gold (96.5%).
3. Stale cache is flagged with is_stale=True, or DataUnavailableError is raised.
"""
from unittest.mock import MagicMock, patch
import pytest

from tools.market.terminal_v2.adapters.goldtraders_adapter import GoldTradersAdapter
from tools.market.terminal_v2.adapters.settrade_adapter import SettradeAdapter
from tools.market.terminal_v2.application.cache import ThreadSafeTTLCache
from tools.market.terminal_v2.domain.errors import DataUnavailableError
from tools.market.terminal_v2.domain.models import ThaiFundFlowSnapshot, ThaiRetailGoldQuote


def test_settrade_never_falls_back_to_yfinance_or_index():
    """Verify that Settrade adapter fails with DataUnavailableError when no cache exists,

    rather than attempting to query yfinance or ^SET.BK.
    """
    cache = ThreadSafeTTLCache(default_ttl_seconds=60.0)
    adapter = SettradeAdapter(cache=cache)

    with patch("requests.get", side_effect=Exception("Connection refused to api.settrade.com")):
        with pytest.raises(DataUnavailableError) as exc_info:
            adapter.get_investor_type_flow("SET")

        assert "temporarily unavailable" in str(exc_info.value).lower()
        assert exc_info.value.capability == "investor_type_flow"
        assert exc_info.value.source == "Settrade"


def test_settrade_serves_stale_flagged_data_on_subsequent_failure():
    """Verify that when Settrade is down after a good fetch, it serves the cached data with is_stale=True."""
    cache = ThreadSafeTTLCache(default_ttl_seconds=60.0)
    adapter = SettradeAdapter(cache=cache)

    mock_json = {
        "asof_date": "26/09/2026",
        "total_value": 50000000000,
        "investors": [
            {"type": "Foreign", "type_name_en": "Foreign Investors", "buy_value": 25000000000, "sell_value": 24000000000, "net_value": 1000000000},
            {"type": "Institution", "type_name_en": "Local Institutions", "buy_value": 10000000000, "sell_value": 11000000000, "net_value": -1000000000},
        ],
    }

    mock_resp = MagicMock()
    mock_resp.json.return_value = mock_json
    mock_resp.raise_for_status = MagicMock()

    with patch("requests.get", return_value=mock_resp):
        first_call = adapter.get_investor_type_flow("SET")
        assert not first_call.is_stale
        assert len(first_call.investors) == 2
        assert first_call.investors[0].net_value == 1000000000

    # Invalidate expiration so cache reloads on next call
    cache._entries["settrade:flow:SET"].expires_at = 0.0

    # Now upstream fails (e.g. Settrade WAF blocks or 503)
    with patch("requests.get", side_effect=Exception("HTTP 503 Service Unavailable")):
        second_call = adapter.get_investor_type_flow("SET")
        # Must return the snapshot with is_stale=True
        assert second_call.is_stale is True
        assert second_call.stale_reason is not None
        assert "503" in second_call.stale_reason or "upstream error" in second_call.stale_reason.lower()
        # Data itself is preserved accurately
        assert len(second_call.investors) == 2
        assert second_call.total_value == 50000000000


def test_goldtraders_never_falls_back_to_gcf_gold_futures():
    """Verify that GoldTraders adapter fails with DataUnavailableError when GTA is down,

    rather than attempting to calculate from GC=F futures.
    """
    cache = ThreadSafeTTLCache(default_ttl_seconds=120.0)
    adapter = GoldTradersAdapter(cache=cache)

    with patch("requests.get", side_effect=Exception("Connection timed out to classic.goldtraders.or.th")):
        with pytest.raises(DataUnavailableError) as exc_info:
            adapter.get_retail_gold_quote()

        assert "temporarily unavailable" in str(exc_info.value).lower()
        assert exc_info.value.capability == "retail_gold_price"
        assert exc_info.value.source == "Gold Traders Association"


def test_goldtraders_parses_html_accurately():
    """Verify accurate extraction of retail gold quotes from GTA HTML markup."""
    sample_html = """
    <html>
        <body>
            <span id="DetailPlace_uc_goldprices1_lblBLBuy">43,200.00</span>
            <span id="DetailPlace_uc_goldprices1_lblBLSell">43,300.00</span>
            <span id="DetailPlace_uc_goldprices1_lblOMBuy">42,417.72</span>
            <span id="DetailPlace_uc_goldprices1_lblOMSell">43,800.00</span>
            <span id="DetailPlace_uc_goldprices1_lblAsTime">26/09/2569 เวลา 09:30 น. (ครั้งที่ 1)</span>
        </body>
    </html>
    """
    cache = ThreadSafeTTLCache(default_ttl_seconds=120.0)
    adapter = GoldTradersAdapter(cache=cache)

    mock_resp = MagicMock()
    mock_resp.text = sample_html
    mock_resp.raise_for_status = MagicMock()

    with patch("requests.get", return_value=mock_resp):
        quote = adapter.get_retail_gold_quote()
        assert quote.bar.buy == 43200.00
        assert quote.bar.sell == 43300.00
        assert quote.ornament.buy == 42417.72
        assert quote.ornament.sell == 43800.00
        assert quote.revision == 1
        assert "26/09/2569" in quote.announced_at
        assert quote.is_stale is False
