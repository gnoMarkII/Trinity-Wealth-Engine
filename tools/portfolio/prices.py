from typing import Optional, Tuple, Literal, Dict
import yfinance as yf
from tools.portfolio import get_default_service
from tools.portfolio.domain.models import PortfolioState
from tools.portfolio.adapters.markdown.repository_adapter import _get_portfolio_lock
from tools.portfolio.adapters.price_yfinance_adapter import (
    _USDTHB_TICKER,
    _yf_symbol,
    _fetch_last_price as _yf_fetch_last_price,
    _fetch_fx_rate as _yf_fetch_fx_rate,
    _PRICE_FETCH_TIMEOUT as _DEFAULT_TIMEOUT,
)
from langchain_core.tools import tool

_PRICE_FETCH_TIMEOUT = _DEFAULT_TIMEOUT

from datetime import datetime, timedelta

def _fetch_last_price(symbol: str) -> Optional[float]:
    return _yf_fetch_last_price(symbol)

def _fetch_fx_rate() -> Optional[float]:
    return _fetch_last_price(_USDTHB_TICKER)

def fetch_fx_rate(
    date_str: Optional[str] = None, fallback_rate: Optional[float] = None
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
            from core.retry import with_retry
            rate = with_retry(_get_historical)
            if rate is not None:
                return rate, "historical"
        except Exception:
            pass
        return default_fallback, "fallback"

    try:
        live = _fetch_fx_rate()
        if live is not None and live > 0:
            return round(live, 4), "live"
    except Exception:
        pass

    return default_fallback, "fallback"

def fetch_latest_price(symbol: str, currency: Literal["THB", "USD"] = "THB") -> Optional[float]:
    if currency not in ("THB", "USD"):
        raise ValueError("currency ต้องเป็น 'THB' หรือ 'USD'")
    if currency == "THB":
        try:
            svc = get_default_service()
            if svc and hasattr(svc, "price_provider") and hasattr(svc.price_provider, "fund_provider"):
                if svc.price_provider.fund_provider.has_fund(symbol):
                    return svc.price_provider.fetch_price(symbol, currency)
        except Exception:
            pass
    yf_sym = _yf_symbol(symbol, currency)
    return _fetch_last_price(yf_sym)

def _legacy_refresh_prices(state: PortfolioState) -> Dict[str, str]:
    from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
    results: Dict[str, str] = {}
    tasks = {}

    with ThreadPoolExecutor(max_workers=5) as executor:
        for h in state.holdings:
            if h.asset_type == "Cash" or h.status == "archived" or h.units <= 0:
                continue
            curr = "USD" if h.avg_cost_usd is not None else "THB"
            yf_sym = _yf_symbol(h.symbol, curr)
            future = executor.submit(_fetch_last_price, yf_sym)
            tasks[future] = (h, curr)

        timeout = globals().get("_PRICE_FETCH_TIMEOUT", _DEFAULT_TIMEOUT)
        for future in tasks:
            h, curr = tasks[future]
            try:
                price = future.result(timeout=timeout)
                if price is not None:
                    if curr == "USD":
                        h.current_price_usd = price
                    else:
                        h.current_price_thb = price
                    results[h.symbol] = "ok"
                else:
                    results[h.symbol] = "no_data"
            except TimeoutError:
                results[h.symbol] = "timeout"
            except Exception as e:
                results[h.symbol] = f"error: {e}"

    return results


def _refresh_prices(state: PortfolioState) -> Dict[str, str]:
    if _fetch_last_price != _yf_fetch_last_price:
        return _legacy_refresh_prices(state)

    try:
        svc = get_default_service()
        if svc and hasattr(svc, "price_provider") and svc.price_provider:
            provider = svc.price_provider
            while hasattr(provider, "_delegate"):
                provider = provider._delegate
            return provider.refresh_portfolio_prices(state)
    except Exception:
        pass
    return _legacy_refresh_prices(state)

def _fetch_fundamentals(state: PortfolioState, force: bool = False) -> Dict[str, str]:
    return get_default_service().price_provider.fetch_fundamentals(state, force=force)

def _sync_market_prices_impl(portfolio_id: str = "default") -> str:
    from filelock import Timeout
    from tools.tool_errors import LOCK_TIMEOUT
    from tools.portfolio.domain.validator import validate_portfolio_id

    pid = validate_portfolio_id(portfolio_id)
    try:
        lock = _get_portfolio_lock(pid)
        with lock:
            service = get_default_service()
            return service.sync_market_prices(portfolio_id=pid)
    except Timeout:
        return LOCK_TIMEOUT.format(detail=f"portfolio lock '{pid}'")
    except Exception as e:
        return f"Error: {e}"


@tool
def sync_market_prices(portfolio_id: str = "default") -> str:
    """ดึงราคาตลาดล่าสุดของทุกสินทรัพย์ในพอร์ตโฟลิโอ"""
    return _sync_market_prices_impl(portfolio_id=portfolio_id)


