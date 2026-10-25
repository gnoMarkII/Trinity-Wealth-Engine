import json
import pytest
from decimal import Decimal
from unittest.mock import MagicMock, patch

from tools.portfolio.adapters.thai_fund.finnomena_adapter import FinnomenaFundAdapter


@pytest.fixture
def sample_catalog_json():
    return [
        {"id": "F00000SP500", "short_code": "SCBS&P500E", "name_th": "เอสซีบี หุ้นยูเอส"},
        {"id": "F000000BTP", "short_code": "BTP", "name_th": "บัวหลวงทศพล"},
        {"id": "F00000LHGRID", "short_code": "LHGRID", "name_th": "แอล เอช โกลบอล อินฟราสตรัคเจอร์"},
    ]


@pytest.fixture
def sp500_portfolio_json():
    return {
        "status": True,
        "data": {
            "as_of_date": "2026-03-31T00:00:00Z",
            "top_holdings": [
                {
                    "name": "iShares Core S&P 500 ETF",
                    "short_code": "",
                    "percent": 99.04,
                    "asset_type": "FUND",
                }
            ],
            "sector": [
                {"name": "เทคโนโลยี", "percent": 31.5},
                {"name": "บริการด้านการเงิน", "percent": 13.0},
            ],
            "asset_allocation": {
                "equity": 99.04,
                "cash": 0.96,
            },
        },
    }


@pytest.fixture
def btp_portfolio_json():
    return {
        "status": True,
        "data": {
            "as_of_date": "2026-03-31T00:00:00Z",
            "top_holdings": [
                {
                    "name": "หุ้นสามัญ บริษัทเดลต้า อีเลคโทรนิคส์ (ประเทศไทย) จำกัด (มหาชน)",
                    "short_code": "",
                    "percent": 12.5,
                    "asset_type": "EQUITY",
                },
                {
                    "name": "บริษัท ปตท. จำกัด (มหาชน)",
                    "short_code": "PTT",
                    "percent": 9.8,
                    "asset_type": "EQUITY",
                },
            ],
            "sector": [
                {"name": "เทคโนโลยี", "percent": 12.5},
                {"name": "พลังงาน", "percent": 9.8},
            ],
            "asset_allocation": {
                "equity": 95.0,
                "cash": 5.0,
            },
        },
    }


def test_fetch_constituents_feeder_fund_2tier_resolution(tmp_path, sample_catalog_json, sp500_portfolio_json):
    adapter = FinnomenaFundAdapter(cache_dir=str(tmp_path))

    def mock_get(url, **kwargs):
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        if "list" in url:
            resp.json.return_value = sample_catalog_json
        elif "portfolio" in url:
            resp.json.return_value = sp500_portfolio_json
        return resp

    with patch.object(adapter._session, "get", side_effect=mock_get):
        data = adapter.fetch_constituents("SCBS&P500E")

        assert data is not None
        assert data.symbol == "SCBS&P500E"
        assert data.master_fund_name == "iShares Core S&P 500 ETF"
        assert data.master_fund_ticker == "IVV"
        assert len(data.top_holdings) >= 5

        # Check that top holding MSFT is present and scaled by feeder weight
        tickers = {h.ticker: h for h in data.top_holdings}
        assert "MSFT" in tickers
        assert "AAPL" in tickers
        assert "NVDA" in tickers

        # Weight should be roughly 7.1% * 0.9904 = ~7.03%
        assert tickers["MSFT"].percent > Decimal("6.0")

        # Sector breakdown check
        sec_names = {s.sector_name_en for s in data.sector_breakdown}
        assert "Information Technology" in sec_names
        assert "Financials" in sec_names


def test_fetch_constituents_thai_stock_normalization(tmp_path, sample_catalog_json, btp_portfolio_json):
    adapter = FinnomenaFundAdapter(cache_dir=str(tmp_path))

    def mock_get(url, **kwargs):
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        if "list" in url:
            resp.json.return_value = sample_catalog_json
        elif "portfolio" in url:
            resp.json.return_value = btp_portfolio_json
        return resp

    with patch.object(adapter._session, "get", side_effect=mock_get):
        data = adapter.fetch_constituents("BTP")

        assert data is not None
        assert len(data.top_holdings) == 2

        # Check normalization of DELTA from Thai legal description
        delta_holding = data.top_holdings[0]
        assert delta_holding.ticker == "DELTA"
        assert delta_holding.percent == Decimal("12.5")

        ptt_holding = data.top_holdings[1]
        assert ptt_holding.ticker == "PTT"
        assert ptt_holding.percent == Decimal("9.8")


def test_fetch_constituents_caching(tmp_path, sample_catalog_json, btp_portfolio_json):
    adapter = FinnomenaFundAdapter(cache_dir=str(tmp_path))

    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = btp_portfolio_json

    adapter._catalog = {"BTP": "F000000BTP"}
    adapter._catalog_loaded_at = 9999999999.0

    with patch.object(adapter._session, "get", return_value=mock_resp) as mock_get:
        data1 = adapter.fetch_constituents("BTP")
        assert mock_get.call_count == 1

        # Second call should use cache, not HTTP
        data2 = adapter.fetch_constituents("BTP")
        assert mock_get.call_count == 1
        assert data2.symbol == data1.symbol
        assert len(data2.top_holdings) == len(data1.top_holdings)
