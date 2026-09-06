import pytest
from unittest.mock import MagicMock

from tools.portfolio.adapters.composite_price_adapter import CompositeMarketPriceAdapter
from tools.portfolio.ports.price_port import MarketPricePort
from tools.portfolio.ports.thai_fund_port import ThaiFundPricePort, FundNavData
from tools.portfolio.domain.models import PortfolioState, Holding


@pytest.fixture
def mock_equity_provider():
    provider = MagicMock(spec=MarketPricePort)
    provider.fetch_fx_rate.return_value = (35.0, "live")
    provider.fetch_price.side_effect = lambda sym, ccy: 150.0 if sym == "AAPL" else 30.0
    provider.fetch_fundamentals.return_value = {"AAPL": "ok"}
    return provider


@pytest.fixture
def mock_fund_provider():
    provider = MagicMock(spec=ThaiFundPricePort)
    provider.has_fund.side_effect = lambda sym: sym in ("PRINCIPAL VNEQ-A", "KT-ASIAG-A")
    provider.fetch_nav.side_effect = lambda sym: (
        FundNavData(symbol="PRINCIPAL VNEQ-A", nav=12.5, nav_date="2026-09-04", percent_change=1.0)
        if sym == "PRINCIPAL VNEQ-A"
        else None
    )
    return provider


def test_fetch_price_routes_to_fund_when_thai_fund(mock_equity_provider, mock_fund_provider):
    adapter = CompositeMarketPriceAdapter(
        equity_provider=mock_equity_provider,
        fund_provider=mock_fund_provider,
    )

    # Fund in THB
    nav = adapter.fetch_price("PRINCIPAL VNEQ-A", "THB")
    assert nav == 12.5
    mock_fund_provider.fetch_nav.assert_called_once_with("PRINCIPAL VNEQ-A")
    mock_equity_provider.fetch_price.assert_not_called()


def test_fetch_price_routes_to_equity_when_stock(mock_equity_provider, mock_fund_provider):
    adapter = CompositeMarketPriceAdapter(
        equity_provider=mock_equity_provider,
        fund_provider=mock_fund_provider,
    )

    # US Stock
    price = adapter.fetch_price("AAPL", "USD")
    assert price == 150.0
    mock_equity_provider.fetch_price.assert_called_once_with("AAPL", "USD")

    # Thai Stock
    mock_equity_provider.fetch_price.reset_mock()
    price_thb = adapter.fetch_price("PTT", "THB")
    assert price_thb == 30.0
    mock_equity_provider.fetch_price.assert_called_once_with("PTT", "THB")


def test_refresh_portfolio_prices_mixed_holdings(mock_equity_provider, mock_fund_provider):
    adapter = CompositeMarketPriceAdapter(
        equity_provider=mock_equity_provider,
        fund_provider=mock_fund_provider,
    )

    state = PortfolioState(
        last_updated="2026-09-06T00:00:00Z",
        holdings=[
            Holding(symbol="CASH-THB", asset_type="Cash", units=10000, avg_cost_thb=1.0),
            Holding(symbol="AAPL", asset_type="Stock", units=10, avg_cost_usd=140.0),
            Holding(symbol="PRINCIPAL VNEQ-A", asset_type="Fund", units=500, avg_cost_thb=10.0),
        ]
    )

    results = adapter.refresh_portfolio_prices(state)

    assert "USDTHB" in results
    assert results["AAPL"] == "150.00 USD"
    assert "12.5000 THB" in results["PRINCIPAL VNEQ-A"]
    assert "CASH-THB" not in results

    # Verify updated prices on holding objects
    aapl = next(h for h in state.holdings if h.symbol == "AAPL")
    assert aapl.current_price_usd == 150.0
    assert aapl.current_price_thb == 150.0 * 35.0

    vneq = next(h for h in state.holdings if h.symbol == "PRINCIPAL VNEQ-A")
    assert vneq.current_price_thb == 12.5
    assert vneq.current_price_usd == round(12.5 / 35.0, 4)


def test_fetch_fundamentals_skips_funds(mock_equity_provider, mock_fund_provider):
    adapter = CompositeMarketPriceAdapter(
        equity_provider=mock_equity_provider,
        fund_provider=mock_fund_provider,
    )

    state = PortfolioState(
        last_updated="2026-09-06T00:00:00Z",
        holdings=[
            Holding(symbol="AAPL", asset_type="Stock", units=10, avg_cost_usd=140.0),
            Holding(symbol="PRINCIPAL VNEQ-A", asset_type="Fund", units=500, avg_cost_thb=10.0),
        ]
    )

    results = adapter.fetch_fundamentals(state)
    assert results["PRINCIPAL VNEQ-A"] == "Fund (no stock ratios)"
    assert results["AAPL"] == "ok"


def test_trading_service_sync_market_prices_with_composite(mock_equity_provider, mock_fund_provider):
    from tools.portfolio.services.trading_service import PortfolioTradingService
    from tools.portfolio.ports.repository_port import PortfolioRepositoryPort

    adapter = CompositeMarketPriceAdapter(
        equity_provider=mock_equity_provider,
        fund_provider=mock_fund_provider,
    )

    state = PortfolioState(
        last_updated="2026-09-06T00:00:00Z",
        holdings=[
            Holding(symbol="CASH-THB", asset_type="Cash", units=10000, avg_cost_thb=1.0),
            Holding(symbol="AAPL", asset_type="Stock", units=10, avg_cost_usd=140.0),
            Holding(symbol="PRINCIPAL VNEQ-A", asset_type="Fund", units=500, avg_cost_thb=10.0),
        ]
    )

    mock_uow = MagicMock()
    mock_uow.load_state.return_value = state
    mock_uow.__enter__.return_value = mock_uow
    mock_uow.__exit__.return_value = None

    mock_repo = MagicMock(spec=PortfolioRepositoryPort)
    mock_repo.unit_of_work.return_value = mock_uow

    trading_svc = PortfolioTradingService(
        repo=mock_repo,
        price_provider=adapter,
        journal_provider=MagicMock(),
    )

    msg = trading_svc.sync_market_prices("default")
    assert "[SYNC] updated prices: refreshed 2/2" in msg
    assert "(" not in msg  # No failure parenthetical since 2/2 succeeded

