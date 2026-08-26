"""Dividend adapter implementing DividendHistoryPort via yfinance."""
import logging
from datetime import datetime
from typing import Dict, List
import pandas as pd
import yfinance as yf

from tools.portfolio.ports.dividend_port import DividendHistoryPort

log = logging.getLogger(__name__)


def _normalize_yf_symbol(symbol: str) -> str:
    s = symbol.strip().upper()
    if s in {"THB", "USD", "CASH_THB", "CASH_USD"}:
        return s
    return s


class DividendYFinanceAdapter(DividendHistoryPort):
    """Adapter fetching dividend history from yfinance."""

    def fetch_dividend_history(self, symbols: List[str]) -> Dict[str, List[Dict]]:
        results: Dict[str, List[Dict]] = {}

        for sym in symbols:
            clean_sym = sym.strip().upper()
            yf_sym = _normalize_yf_symbol(clean_sym)
            try:
                ticker = yf.Ticker(yf_sym)
                divs = ticker.dividends
                if divs is None or divs.empty:
                    results[clean_sym] = []
                    continue

                div_list = []
                for dt, amount in divs.items():
                    date_str = pd.Timestamp(dt).strftime("%Y-%m-%d")
                    div_list.append({
                        "date": date_str,
                        "amount": float(amount),
                        "currency": "USD" if not clean_sym.endswith(".BK") else "THB",
                    })
                div_list.sort(key=lambda x: x["date"])
                results[clean_sym] = div_list
            except Exception as e:
                log.warning("Failed to fetch dividend history for %s: %s", clean_sym, e)
                results[clean_sym] = []

        return results
