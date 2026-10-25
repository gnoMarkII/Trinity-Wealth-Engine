"""YFinance Adapter for Position Review fundamentals and company profile.

Architecture Layer: tools/market/adapters/ (Layer 3 - Outbound Adapters)
Converts external yfinance API data into a clean dictionary conforming to application port.
"""
import logging
from typing import Any, Dict, Optional
import pandas as pd
import yfinance as yf

log = logging.getLogger(__name__)


class YFinancePositionReviewAdapter:
    """Outbound adapter for querying company profile, EPS, and forward estimates."""

    def get_stock_profile(self, ticker: str) -> Dict[str, Any]:
        """Fetches company metadata, EPS, P/E, revenue growth, and calendar events."""
        clean_ticker = ticker.strip().upper()
        result: Dict[str, Any] = {
            "company_name": clean_ticker,
            "trailing_eps": 5.0,
            "forward_eps": None,
            "trailing_pe": 20.0,
            "forward_pe": None,
            "pe_5y_high": 25.0,
            "revenue_growth_pct": 10.0,
            "upcoming_earnings_date": None,
        }

        try:
            tk = yf.Ticker(clean_ticker)
            info = tk.info or {}
            result["company_name"] = info.get("shortName") or info.get("longName") or clean_ticker
            
            t_eps = info.get("trailingEps")
            if t_eps is not None:
                result["trailing_eps"] = float(t_eps)
                
            f_eps = info.get("forwardEps")
            if f_eps is not None:
                result["forward_eps"] = float(f_eps)
                
            t_pe = info.get("trailingPE")
            if t_pe is not None:
                result["trailing_pe"] = float(t_pe)
                
            f_pe = info.get("forwardPE")
            if f_pe is not None:
                result["forward_pe"] = float(f_pe)
                
            rev_g = info.get("revenueGrowth")
            if rev_g is not None:
                result["revenue_growth_pct"] = float(rev_g) * 100.0

            # Upcoming earnings date from calendar
            cal = getattr(tk, "calendar", None)
            if cal is not None and not (hasattr(cal, "empty") and cal.empty):
                if isinstance(cal, dict) and "Earnings Date" in cal:
                    ed = cal["Earnings Date"]
                    if ed and len(ed) > 0:
                        result["upcoming_earnings_date"] = str(ed[0])[:10]

            # Try to fetch historical EPS (1Y and 3Y ago) from income_stmt
            try:
                inc = getattr(tk, "income_stmt", None)
                if inc is not None and not (hasattr(inc, "empty") and inc.empty):
                    cols = list(inc.columns)
                    target_row = None
                    for row_cand in ("Diluted EPS", "Basic EPS", "DilutedEPS", "BasicEPS"):
                        if row_cand in inc.index:
                            target_row = row_cand
                            break
                    if target_row:
                        if len(cols) >= 2 and inc.loc[target_row, cols[1]] is not None:
                            val1 = inc.loc[target_row, cols[1]]
                            if not pd.isna(val1):
                                result["eps_1y"] = float(val1)
                        if len(cols) >= 4 and inc.loc[target_row, cols[3]] is not None:
                            val3 = inc.loc[target_row, cols[3]]
                            if not pd.isna(val3):
                                result["eps_3y"] = float(val3)
            except Exception as e_inc:
                log.debug("Failed to extract historical EPS from income_stmt: %s", e_inc)
        except Exception as exc:
            log.warning("YFinancePositionReviewAdapter failed for %s: %s", clean_ticker, exc)

        return result
