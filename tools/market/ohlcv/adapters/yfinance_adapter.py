"""YFinance Adapter for OHLCV Market Data and Corporate Actions."""
import logging
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import yfinance as yf

from tools.market.ohlcv.ports.ohlcv_port import OhlcvProviderPort, CorporateActionProviderPort
from core.retry import with_retry as _with_retry
from tools.market.earnings import fetch_earnings_dates as _core_fetch_earnings

log = logging.getLogger(__name__)


class YFinanceOhlcvAdapter(OhlcvProviderPort, CorporateActionProviderPort):
    """Concrete driven adapter using yfinance."""

    def __init__(self, yf_module=None):
        self._yf = yf_module or yf

    def fetch_history(
        self,
        symbol: str,
        period: str,
        interval: str,
        auto_adjust: bool = True,
    ) -> pd.DataFrame:
        tk = self._yf.Ticker(symbol)
        try:
            df = _with_retry(lambda: tk.history(period=period, interval=interval, auto_adjust=auto_adjust))
            return df if df is not None else pd.DataFrame()
        except Exception as exc:
            log.warning("Failed to fetch OHLCV from yfinance for %s (period=%s, interval=%s): %s", symbol, period, interval, exc)
            return pd.DataFrame()

    def fetch_dividends(self, symbol: str) -> Tuple[List[Dict[str, Any]], str]:
        tk = self._yf.Ticker(symbol)
        raw_dividends: list[dict] = []
        try:
            div_s = _with_retry(lambda: tk.dividends)
            if div_s is not None and not div_s.empty:
                for ts, amount in div_s.items():
                    if pd.notna(amount) and float(amount) > 0:
                        date_str = ts.strftime("%Y-%m-%d") if hasattr(ts, "strftime") else str(ts)[:10]
                        raw_dividends.append({
                            "date_str": date_str,
                            "timestamp_ms": int(ts.timestamp() * 1000) if hasattr(ts, "timestamp") else 0,
                            "dividend_amount": round(float(amount), 4),
                        })
                return raw_dividends, "ok" if raw_dividends else "empty"
            return [], "empty"
        except Exception as exc:
            log.warning("Dividends fetch failed for %s: %s", symbol, exc)
            return [], "failed"

    def fetch_splits(self, symbol: str) -> Tuple[List[Dict[str, Any]], str]:
        tk = self._yf.Ticker(symbol)
        raw_splits: list[dict] = []
        try:
            splits_s = _with_retry(lambda: tk.splits)
            if splits_s is not None and not splits_s.empty:
                for ts, ratio in splits_s.items():
                    if pd.notna(ratio) and float(ratio) > 0:
                        ratio_f = float(ratio)
                        if ratio_f >= 1.0:
                            num = ratio_f
                            den = 1.0
                            formatted = f"{int(num) if num.is_integer() else num}-for-1 forward split"
                        else:
                            num = 1.0
                            den = round(1.0 / ratio_f, 4)
                            formatted = f"1-for-{int(den) if den.is_integer() else den} reverse split"

                        date_str = ts.strftime("%Y-%m-%d") if hasattr(ts, "strftime") else str(ts)[:10]
                        raw_splits.append({
                            "date_str": date_str,
                            "timestamp_ms": int(ts.timestamp() * 1000) if hasattr(ts, "timestamp") else 0,
                            "split_numerator": num,
                            "split_denominator": den,
                            "split_formatted": formatted,
                        })
                return raw_splits, "ok" if raw_splits else "empty"
            return [], "empty"
        except Exception as exc:
            log.warning("Splits fetch failed for %s: %s", symbol, exc)
            return [], "failed"

    def fetch_earnings(self, symbol: str, tz_name: str) -> Tuple[List[Dict[str, Any]], str, Optional[str]]:
        try:
            earn_res = _core_fetch_earnings(symbol, tz_name)
            return earn_res.rows, earn_res.status, earn_res.source_as_of
        except Exception as exc:
            log.warning("Earnings fetch failed for %s: %s", symbol, exc)
            return [], "failed", None
