from datetime import date
from decimal import Decimal

from tools.portfolio.adapters.composite_price_adapter import CompositeMarketPriceAdapter
from tools.portfolio.domain.models import Holding, PortfolioState, _now_iso
from tools.portfolio.domain.pvd_models import PvdHoldingValuation


class EquityProvider:
    def __init__(self):
        self.fx_calls = 0
        self.price_calls = []
        self.fundamental_symbols = []

    def fetch_fx_rate(self, **_kwargs):
        self.fx_calls += 1
        return 36.5, "live"

    def fetch_price(self, symbol, currency):
        self.price_calls.append((symbol, currency))
        return 1

    def fetch_fundamentals(self, state, force=False):
        self.fundamental_symbols.extend(h.symbol for h in state.holdings)
        return {}


class FundProvider:
    def has_fund(self, symbol):
        raise AssertionError(f"PVD symbol reached fund resolver: {symbol}")

    def fetch_nav(self, symbol):
        raise AssertionError(f"PVD symbol reached NAV resolver: {symbol}")


def test_pvd_only_refresh_uses_statement_nav_without_market_or_fx_calls():
    equity, adapter = EquityProvider(), None
    adapter = CompositeMarketPriceAdapter(equity, FundProvider())
    holding = Holding(
        symbol="PVD_SINSA_FI", asset_type="PVD", units=1, bucket_id="retirement_pvd",
        pvd_valuation=PvdHoldingValuation(
            fund_code="BBLAM", sub_fund_code="SINSA_FI", units=Decimal("1"), statement_nav=Decimal("10"),
            statement_market_value=Decimal("10"), cost_basis=Decimal("9"), as_of_date=date(2026, 9, 30),
            statement_fingerprint="f" * 64, archive_id="a" * 64,
        ),
    )
    state = PortfolioState(last_updated=_now_iso(), holdings=[holding])
    info = adapter.refresh_portfolio_prices(state)
    adapter.fetch_fundamentals(state, force=True)
    assert info["PVD_SINSA_FI"].startswith("Statement NAV 10")
    assert equity.fx_calls == 0
    assert equity.price_calls == []
    assert equity.fundamental_symbols == []

