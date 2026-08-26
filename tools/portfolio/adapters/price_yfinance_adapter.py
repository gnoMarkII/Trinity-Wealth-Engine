import concurrent.futures
import time
from datetime import datetime, timedelta
from typing import Optional, Tuple, Literal, Dict

import yfinance as yf

from core.logger import get_logger
from core.retry import with_retry
from tools.portfolio.domain.constants import (
    CASH_THB_SYMBOL,
    CASH_USD_SYMBOL,
    _FLOAT_EPS,
    FUNDAMENTALS_TTL_SECONDS,
)
from tools.portfolio.domain.models import PortfolioState, Holding, _now_iso
from tools.portfolio.ports.price_port import MarketPricePort

log = get_logger(__name__)

_USDTHB_TICKER = "USDTHB=X"
_PRICE_FETCH_TIMEOUT = 6.0


def _yf_symbol(symbol: str, currency: str) -> str:
    """แปลง symbol -> ticker ที่ yfinance รู้จัก (THB -> เติม .BK)"""
    if currency == "THB" and not symbol.endswith(".BK"):
        return f"{symbol}.BK"
    return symbol


def _fetch_last_price(yf_symbol: str) -> Optional[float]:
    try:
        tk = yf.Ticker(yf_symbol)
        fi = tk.fast_info
        last = getattr(fi, "last_price", None)
        if last is not None and float(last) > 0:
            return float(last)
        hist = tk.history(period="1d")
        if not hist.empty:
            val = float(hist["Close"].iloc[-1])
            if val > 0:
                return val
    except Exception as e:
        log.warning("fetch price failed for %s: %s", yf_symbol, e)
    return None


def _fetch_fx_rate() -> Optional[float]:
    adapter = PriceYFinanceAdapter()
    rate, _ = adapter.fetch_fx_rate()
    return rate


class PriceYFinanceAdapter(MarketPricePort):
    """yfinance market price, FX rate, and fundamentals adapter with in-memory TTL caching."""

    def __init__(self, price_ttl_seconds: float = 60.0):
        self.price_ttl = price_ttl_seconds
        self._price_cache: Dict[str, Tuple[float, float]] = {}  # symbol -> (price, timestamp)

    def fetch_price(self, symbol: str, currency: Literal["THB", "USD"]) -> Optional[float]:
        if currency not in ("THB", "USD"):
            raise ValueError(f"currency ต้องเป็น 'THB' หรือ 'USD' (got '{currency}')")
        yf_sym = _yf_symbol(symbol, currency)
        return _fetch_last_price(yf_sym)

    def fetch_fx_rate(
        self, date_str: Optional[str] = None, fallback_rate: Optional[float] = None
    ) -> Tuple[float, Literal["historical", "live", "fallback"]]:
        default_fallback = fallback_rate if fallback_rate is not None and fallback_rate > 0 else 36.5
        today_str = datetime.now().strftime("%Y-%m-%d")

        if date_str and date_str.strip() and date_str.strip() < today_str:
            clean_date = date_str.strip()

            def _get_historical():
                try:
                    target_dt = datetime.strptime(clean_date, "%Y-%m-%d")
                except Exception:
                    return None
                start_dt = target_dt - timedelta(days=5)
                end_dt = target_dt + timedelta(days=1)
                df = yf.download(
                    _USDTHB_TICKER,
                    start=start_dt.strftime("%Y-%m-%d"),
                    end=end_dt.strftime("%Y-%m-%d"),
                    progress=False,
                )
                if df is not None and not df.empty:
                    close = df["Close"]
                    if hasattr(close, "columns"):
                        close = close.iloc[:, 0]
                    if hasattr(close.index, "tz") and close.index.tz is not None:
                        close.index = close.index.tz_localize(None)
                    val = close.asof(target_dt)
                    if val is not None:
                        val_float = float(val)
                        if val_float > 0 and val_float == val_float:
                            return round(val_float, 4)
                return None

            try:
                rate = with_retry(_get_historical)
                if rate is not None:
                    return rate, "historical"
            except Exception:
                pass
            return default_fallback, "fallback"

        try:
            live = _fetch_last_price(_USDTHB_TICKER)
            if live is not None and live > 0:
                return round(live, 4), "live"
        except Exception:
            pass

        return default_fallback, "fallback"

    def refresh_portfolio_prices(self, state: PortfolioState) -> Dict[str, str]:

        targets = []
        for h in state.holdings:
            if h.asset_type == "Cash":
                continue
            if h.avg_cost_usd is not None:
                targets.append((h, h.symbol, "USD"))
            elif h.avg_cost_thb is not None:
                targets.append((h, h.symbol, "THB"))

        if not targets:
            return {}

        results: Dict[str, str] = {}
        fx_rate, fx_source = self.fetch_fx_rate()
        state.fx_rates["USDTHB"] = fx_rate
        results["USDTHB"] = f"{fx_rate:.4f} ({fx_source})"

        def _fetch_one(t: Tuple[Holding, str, str]) -> Tuple[Holding, Optional[float], str]:
            holding, symbol, ccy = t
            yf_sym = _yf_symbol(symbol, ccy)
            price = self.fetch_price(symbol, ccy)  # type: ignore
            return holding, price, ccy

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(targets), 8)) as ex:
            futs = {ex.submit(_fetch_one, t): t for t in targets}
            done, not_done = concurrent.futures.wait(futs.keys(), timeout=_PRICE_FETCH_TIMEOUT * len(targets))

            for f in done:
                try:
                    holding, price, ccy = f.result()
                    if price is not None and price > 0:
                        if ccy == "USD":
                            holding.current_price_usd = price
                            holding.current_price_thb = round(price * fx_rate, 2)
                        else:
                            holding.current_price_thb = price
                            holding.current_price_usd = round(price / fx_rate, 2)
                        holding.fx_rate = fx_rate
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
        now = time.time()
        results: Dict[str, str] = {}

        def _process_one(h: Holding):
            if h.asset_type == "Cash":
                return
            if not force and h.fundamentals_updated_at is not None:
                if now - h.fundamentals_updated_at < FUNDAMENTALS_TTL_SECONDS:
                    results[h.symbol] = "cached (TTL valid)"
                    return

            ccy = "USD" if h.avg_cost_usd is not None else "THB"
            yf_sym = _yf_symbol(h.symbol, ccy)
            try:
                tk = yf.Ticker(yf_sym)
                info = tk.info or {}

                pe = info.get("trailingPE") or info.get("peRatio")
                eps = info.get("trailingEps") or info.get("epsTrailingTwelveMonths")
                payout = info.get("payoutRatio")
                mcap = info.get("marketCap")
                div_rate = info.get("dividendRate") or info.get("trailingAnnualDividendRate")
                div_yield_raw = info.get("dividendYield")
                trailing_yield_raw = info.get("trailingAnnualDividendYield")
                long_name = info.get("longName") or info.get("shortName")

                if long_name and isinstance(long_name, str):
                    h.company_name = long_name
                if pe is not None:
                    h.pe_ratio = float(pe)
                if eps is not None:
                    h.eps = float(eps)
                if payout is not None and payout > 0:
                    h.payout_ratio = float(payout * 100) if payout <= 1.0 else float(payout)
                if mcap is not None and mcap > 0:
                    h.market_cap_value = float(mcap)
                if div_rate is not None and div_rate >= 0:
                    h.dividend_per_share = float(div_rate)

                div_yield_val = None
                if div_yield_raw is not None and div_yield_raw >= 0:
                    div_yield_val = float(div_yield_raw * 100) if div_yield_raw <= 1.0 else float(div_yield_raw)
                elif trailing_yield_raw is not None and trailing_yield_raw >= 0:
                    div_yield_val = float(trailing_yield_raw * 100) if trailing_yield_raw <= 1.0 else float(trailing_yield_raw)

                if div_yield_val is not None:
                    h.dividend_yield = round(div_yield_val, 2)

                h.fundamentals_updated_at = now
                results[h.symbol] = "ok"
            except Exception as e:
                log.warning("fetch fundamentals failed for %s: %s", h.symbol, e)
                results[h.symbol] = f"error: {e}"

        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
            list(ex.map(_process_one, state.holdings))

        return results
