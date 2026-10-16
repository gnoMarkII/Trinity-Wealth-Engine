import concurrent.futures
from typing import Optional, Tuple, Literal, Dict

from core.logger import get_logger
from tools.portfolio.domain.models import PortfolioState, Holding
from tools.portfolio.ports.price_port import MarketPricePort
from tools.portfolio.ports.thai_fund_port import ThaiFundPricePort, FundNavData

log = get_logger(__name__)

_TIMEOUT_PER_ITEM = 8.0


class CompositeMarketPriceAdapter(MarketPricePort):
    """Composite market price adapter routing between equity market providers and Thai mutual fund providers.
    
    Adheres strictly to Hexagonal Architecture MarketPricePort boundary.
    - Stocks, ETFs, FX rates, and corporate fundamentals -> equity_provider (e.g. PriceYFinanceAdapter)
    - Thai Mutual Funds -> fund_provider (e.g. FinnomenaFundAdapter)
    """

    def __init__(
        self,
        equity_provider: MarketPricePort,
        fund_provider: ThaiFundPricePort,
    ):
        self.equity_provider = equity_provider
        self.fund_provider = fund_provider

    def _is_fund(self, symbol: str, asset_type: Optional[str] = None) -> bool:
        """Determine if a holding/symbol should be treated as a Thai mutual fund."""
        if asset_type == "PVD" or symbol.strip().upper().startswith("PVD_"):
            return False
        if asset_type == "Fund":
            return True
        return self.fund_provider.has_fund(symbol)

    def fetch_price(self, symbol: str, currency: Literal["THB", "USD"]) -> Optional[float]:
        """Fetch latest price / NAV for a single symbol."""
        if currency not in ("THB", "USD"):
            raise ValueError(f"currency ต้องเป็น 'THB' หรือ 'USD' (got '{currency}')")
        if symbol.strip().upper().startswith("PVD_"):
            return None

        if currency == "THB" and self._is_fund(symbol):
            nav_data = self.fund_provider.fetch_nav(symbol)
            if nav_data and nav_data.nav > 0:
                return nav_data.nav

        return self.equity_provider.fetch_price(symbol, currency)

    def fetch_fx_rate(
        self, date_str: Optional[str] = None, fallback_rate: Optional[float] = None
    ) -> Tuple[float, Literal["historical", "live", "fallback"]]:
        """Fetch FX rate for USDTHB via equity/macro provider."""
        return self.equity_provider.fetch_fx_rate(date_str=date_str, fallback_rate=fallback_rate)

    def refresh_portfolio_prices(self, state: PortfolioState) -> Dict[str, str]:
        """Batch refresh latest prices/NAVs for all holdings in a portfolio state."""
        targets = []
        results: Dict[str, str] = {}
        for h in state.holdings:
            if h.asset_type == "PVD" or h.symbol.upper().startswith("PVD_"):
                valuation = h.pvd_valuation
                results[h.symbol] = (
                    f"Statement NAV {valuation.statement_nav} THB ({valuation.as_of_date.isoformat()})"
                    if valuation is not None else "Statement NAV unavailable"
                )
                continue
            if h.asset_type == "Cash" or h.status == "archived" or h.units <= 0:
                continue
            is_fund = self._is_fund(h.symbol, h.asset_type)
            ccy = "USD" if (h.avg_cost_usd is not None and not is_fund) else "THB"
            targets.append((h, h.symbol, ccy, is_fund))

        if not targets:
            state.price_refresh_info = results
            return results

        fx_rate, fx_source = self.fetch_fx_rate()
        state.fx_rates["USDTHB"] = fx_rate
        results["USDTHB"] = f"{fx_rate:.4f} ({fx_source})"

        def _fetch_one(t: Tuple[Holding, str, str, bool]) -> Tuple[Holding, Optional[float], str, bool, Optional[str]]:
            holding, symbol, ccy, is_fund = t
            extra_info = None
            if is_fund:
                nav_data = self.fund_provider.fetch_nav(symbol)
                if nav_data and nav_data.nav > 0:
                    extra_info = nav_data.nav_date
                    return holding, nav_data.nav, "THB", True, extra_info
                return holding, None, "THB", True, None
            else:
                price = self.equity_provider.fetch_price(symbol, ccy)
                return holding, price, ccy, False, None

        max_workers = min(len(targets), 8)
        timeout_total = _TIMEOUT_PER_ITEM * max(len(targets), 1)

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(_fetch_one, t): t for t in targets}
            done, not_done = concurrent.futures.wait(futs.keys(), timeout=timeout_total)

            for f in done:
                try:
                    holding, price, ccy, is_fund, extra_info = f.result()
                    if price is not None and price > 0:
                        if ccy == "USD":
                            holding.current_price_usd = price
                            holding.current_price_thb = round(price * fx_rate, 2)
                        else:
                            holding.current_price_thb = price
                            holding.current_price_usd = round(price / fx_rate, 4 if is_fund else 2)
                        holding.fx_rate = fx_rate
                        
                        if is_fund and extra_info:
                            results[holding.symbol] = f"{price:.4f} THB (NAV {extra_info})"
                        else:
                            results[holding.symbol] = f"{price:.2f} {ccy}"
                    else:
                        results[holding.symbol] = "fetch failed (kept previous)"
                except Exception as e:
                    t = futs[f]
                    results[t[1]] = f"error: {e}"

            for f in not_done:
                f.cancel()
                t = futs[f]
                results[t[1]] = "timeout (kept previous)"

        state.price_refresh_info = results
        return results

    def fetch_fundamentals(self, state: PortfolioState, force: bool = False) -> Dict[str, str]:
        """Fetch equity fundamentals for stocks/ETFs, marking funds appropriately."""
        results: Dict[str, str] = {}
        equity_holdings = []

        for h in state.holdings:
            if h.asset_type == "Cash":
                continue
            if h.asset_type == "PVD" or h.symbol.upper().startswith("PVD_"):
                results[h.symbol] = "PVD statement valuation (no fundamentals)"
                continue
            if self._is_fund(h.symbol, h.asset_type):
                results[h.symbol] = "Fund (no stock ratios)"
            else:
                equity_holdings.append(h)

        if equity_holdings:
            # Pass a state containing only eligible equity holdings across the
            # provider boundary, then copy its derived fields back by symbol.
            filtered_state = state.model_copy(deep=True)
            filtered_state.holdings = [h for h in filtered_state.holdings if h.asset_type not in ("Cash", "PVD") and not h.symbol.upper().startswith("PVD_") and h.asset_type != "Fund"]
            equity_results = self.equity_provider.fetch_fundamentals(filtered_state, force=force)
            updated = {h.symbol: h for h in filtered_state.holdings}
            for holding in state.holdings:
                source = updated.get(holding.symbol)
                if source is None:
                    continue
                for field in ("company_name", "pe_ratio", "eps", "payout_ratio", "market_cap_value", "dividend_per_share", "dividend_yield", "fundamentals_updated_at"):
                    setattr(holding, field, getattr(source, field))
            # Merge equity results while keeping fund annotations
            for k, v in equity_results.items():
                if k not in results:
                    results[k] = v

        return results
