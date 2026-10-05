"""Unit tests for Level 1 Crypto Macro Liquidity Adapters and Service."""
import pytest
from unittest.mock import MagicMock, patch

from tools.market.terminal_v2.adapters.crypto_benchmark_adapter import CryptoBenchmarkAdapter
from tools.market.terminal_v2.adapters.defillama_stablecoin_adapter import DefiLlamaStablecoinsAdapter
from tools.market.terminal_v2.application.terminal_data_service import TerminalDataService
from tools.market.terminal_v2.domain.errors import DataUnavailableError
from tools.market.terminal_v2.domain.models import (
    CryptoBenchmarkSnapshot,
    CryptoMacroLiquiditySnapshot,
    SpotEtfFlowSnapshot,
    StablecoinItem,
    StablecoinSupplySnapshot,
)


def test_defillama_adapter_offline_parsing():
    """Test parsing DeFiLlama peggedAssets JSON response into domain model."""
    mock_payload = {
        "peggedAssets": [
            {
                "id": "1",
                "name": "Tether",
                "symbol": "USDT",
                "pegType": "peggedUSD",
                "price": 1.0,
                "circulating": {"peggedUSD": 100_000_000_000.0},
                "circulatingPrevWeek": {"peggedUSD": 99_000_000_000.0},
                "circulatingPrevMonth": {"peggedUSD": 95_000_000_000.0},
            },
            {
                "id": "2",
                "name": "USD Coin",
                "symbol": "USDC",
                "pegType": "peggedUSD",
                "price": 1.0,
                "circulating": {"peggedUSD": 50_000_000_000.0},
                "circulatingPrevWeek": {"peggedUSD": 49_000_000_000.0},
                "circulatingPrevMonth": {"peggedUSD": 48_000_000_000.0},
            },
        ]
    }
    adapter = DefiLlamaStablecoinsAdapter()
    with patch("requests.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.json.return_value = mock_payload
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        snap = adapter._fetch_snapshot()
        assert snap.total_circulating_usd == 150_000_000_000.0
        assert snap.change_7d_pct is not None
        assert snap.change_7d_pct > 0.0
        assert snap.change_30d_pct is not None
        assert snap.change_30d_pct > 0.0
        assert len(snap.top_stablecoins) == 2
        assert snap.top_stablecoins[0].symbol == "USDT"
        assert snap.top_stablecoins[0].market_share_pct == 66.67
        assert snap.source == "DeFiLlama"


def test_defillama_feature_flag():
    """Test that disabling the adapter raises DataUnavailableError."""
    adapter = DefiLlamaStablecoinsAdapter(enabled=False)
    with pytest.raises(DataUnavailableError):
        adapter._fetch_snapshot()


def test_crypto_benchmark_calculation():
    """Test calculation of BTC/Gold ratio and price metrics."""
    adapter = CryptoBenchmarkAdapter()
    with patch("yfinance.Ticker") as mock_ticker:
        # Mock BTC ticker
        mock_btc = MagicMock()
        import pandas as pd
        mock_btc.history.return_value = pd.DataFrame(
            {"Close": [80000.0, 81000.0, 82000.0, 83000.0, 84000.0, 85000.0, 86000.0, 87000.0]}
        )
        # Mock Gold ticker
        mock_gold = MagicMock()
        mock_gold.fast_info.last_price = 4000.0

        def ticker_side_effect(symbol):
            if symbol == "BTC-USD":
                return mock_btc
            elif symbol == "GC=F":
                return mock_gold
            return MagicMock()

        mock_ticker.side_effect = ticker_side_effect

        snap = adapter._fetch_snapshot()
        assert snap.symbol == "BTC"
        assert snap.price_usd == 87000.0
        assert snap.gold_price_usd == 4000.0
        assert snap.btc_gold_ratio == round(87000.0 / 4000.0, 2)
        assert snap.change_24h_pct is not None


def test_terminal_data_service_liquidity_synthesis():
    """Test that TerminalDataService correctly synthesizes Level 1 liquidity regime."""
    mock_stables_port = MagicMock()
    mock_stables_port.get_stablecoin_supply.return_value = StablecoinSupplySnapshot(
        total_circulating_usd=200_000_000_000.0,
        change_7d_pct=1.2,
        change_30d_pct=2.5,
        top_stablecoins=(),
        as_of_date="2026-10-04",
    )

    mock_bench_port = MagicMock()
    mock_bench_port.get_crypto_benchmark.return_value = CryptoBenchmarkSnapshot(
        symbol="BTC",
        price_usd=85000.0,
        change_24h_pct=1.5,
        change_7d_pct=3.0,
        gold_price_usd=4000.0,
        btc_gold_ratio=21.25,
        as_of_date="2026-10-04",
    )

    mock_etf_port = MagicMock()
    mock_etf_port.get_spot_etf_flows.return_value = SpotEtfFlowSnapshot(
        asset="BTC",
        report_date="2026-10-04",
        daily_total_usd=150_000_000.0,
        cumulative_total_usd=30_000_000_000.0,
        issuers=(),
    )

    service = TerminalDataService(
        short_volume=MagicMock(),
        nasdaq_intelligence=MagicMock(),
        sec_financials=MagicMock(),
        sec_insider_trades=MagicMock(),
        ticker_news=MagicMock(),
        reference_rates=MagicMock(),
        ofr_stress=MagicMock(),
        global_policy_rates=MagicMock(),
        treasury_data=MagicMock(),
        auction_history=MagicMock(),
        thai_bond_market=MagicMock(),
        thai_public_debt=MagicMock(),
        thai_yield_curve=MagicMock(),
        options_chain=MagicMock(),
        commodity_vol=MagicMock(),
        metals_cot=MagicMock(),
        prediction_market=MagicMock(),
        thai_fund_allocation=MagicMock(),
        spot_etf_flows=mock_etf_port,
        stablecoin_supply=mock_stables_port,
        crypto_benchmark=mock_bench_port,
    )

    liq = service.get_crypto_macro_liquidity()
    assert isinstance(liq, CryptoMacroLiquiditySnapshot)
    assert liq.btc_price_usd == 85000.0
    assert liq.btc_gold_ratio == 21.25
    assert liq.stablecoin_total_usd == 200_000_000_000.0
    assert liq.liquidity_regime == "Expanding Liquidity"
    assert liq.etf_daily_net_inflow_usd == 150_000_000.0
