"""Contract tests for the equity financials application boundary."""
from dataclasses import dataclass

import pytest

from application.equity.service import EquityFinancialsApplicationService


@dataclass
class _Asset:
    provider_symbol: str
    market: str
    asset_class: str = "STOCK_US"


class _Resolver:
    def __init__(self, asset):
        self.asset = asset

    def resolve(self, ticker: str):
        return self.asset


class _Financials:
    def __init__(self):
        self.calls = []

    def get_financial_statements(self, **kwargs):
        self.calls.append(kwargs)
        return {"ticker": kwargs["ticker"], "market": kwargs["market"]}


def test_financials_use_case_owns_asset_and_market_resolution():
    provider = _Financials()
    service = EquityFinancialsApplicationService(
        resolver=_Resolver(_Asset(provider_symbol="PTT.BK", market="TH", asset_class="STOCK_TH")),
        financials=provider,
    )

    result = service.get_statements("ptt", market="th", force_refresh=True)

    assert result == {"ticker": "PTT", "market": "TH"}
    assert provider.calls == [
        {
            "ticker": "PTT",
            "market": "TH",
            "provider_symbol": "PTT.BK",
            "force_refresh": True,
        }
    ]


def test_financials_use_case_rejects_market_mismatch():
    service = EquityFinancialsApplicationService(
        resolver=_Resolver(_Asset(provider_symbol="AAPL", market="US")),
        financials=_Financials(),
    )

    with pytest.raises(ValueError, match="Market mismatch"):
        service.get_statements("AAPL", market="TH")


def test_financials_use_case_rejects_non_equity_asset():
    service = EquityFinancialsApplicationService(
        resolver=_Resolver(_Asset(provider_symbol="BTC-USD", market="US", asset_class="CRYPTO")),
        financials=_Financials(),
    )

    with pytest.raises(ValueError, match="not an equity"):
        service.get_statements("BTC-USD")
