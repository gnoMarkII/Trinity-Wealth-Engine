"""Dividend adapter implementing DividendHistoryPort via yfinance."""
import logging
from datetime import datetime
from typing import Dict, List
import pandas as pd
import yfinance as yf

from tools.portfolio.ports.dividend_port import DividendHistoryPort
from tools.portfolio.prices import _yf_symbol

log = logging.getLogger(__name__)


class DividendYFinanceAdapter(DividendHistoryPort):
    """Adapter fetching dividend history from yfinance."""

    def fetch_dividend_history(self, symbols: List[str]) -> Dict[str, List[Dict]]:
        results: Dict[str, List[Dict]] = {}

        for sym in symbols:
            clean_sym = sym.strip().upper()
            yf_sym = _yf_symbol(clean_sym)
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
