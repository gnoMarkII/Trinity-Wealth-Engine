"""External insider-history adapter.

Only this adapter knows how to read yfinance's insider transaction table.
The application and SQLite layers receive normalized plain mappings.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import yfinance as yf

from application.equity.ports import InsiderHistoryProviderPort


TRANSACTION_CODE_WEIGHTS = {
    "P": 1.0,
    "S": 0.8,
    "M": 0.3,
    "A": 0.1,
    "F": 0.05,
    "G": 0.05,
}


def _text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    text = str(value)
    return default if text.lower() in {"nan", "nat", "none"} else text


def _number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if result == result and abs(result) != float("inf") else default


class YFinanceInsiderHistoryAdapter(InsiderHistoryProviderPort):
    """Fetch and normalize the bounded yfinance insider transaction table."""

    def __init__(self, ticker_factory=None) -> None:
        self._ticker_factory = ticker_factory or yf.Ticker

    def fetch(self, ticker: str) -> List[Dict[str, Any]]:
        table = self._ticker_factory(ticker).insider_transactions
        if table is None or getattr(table, "empty", True):
            return []

        records: List[Dict[str, Any]] = []
        for index, row in table.head(30).iterrows():
            insider_name = _text(row.get("Insider"), "Unknown")
            position = _text(row.get("Position"))
            tx_text = _text(row.get("Text")).lower()
            date_value = row.get("Start Date") or row.get("Date")
            if hasattr(date_value, "strftime"):
                transaction_date = date_value.strftime("%Y-%m-%d")
            else:
                transaction_date = _text(
                    date_value,
                    datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                )[:10]

            shares = _number(row.get("Shares"))
            total_value = _number(row.get("Value"))
            price_per_share = round(total_value / shares, 2) if shares > 0 and total_value > 0 else 0.0
            acquired_or_disposed = "D" if "sale" in tx_text or "sold" in tx_text else "A"
            transaction_code = "S" if acquired_or_disposed == "D" else "P"
            accession = f"yf_{ticker}_{transaction_date}_{index}"
            records.append(
                {
                    "accession_number": accession,
                    "issuer_cik": "0000000000",
                    "ticker": ticker.upper(),
                    "filing_url": f"https://www.sec.gov/edgar/searchedgar/companysearch?company={ticker}",
                    "filed_at": transaction_date,
                    "reporting_owner_cik": None,
                    "reporting_owner_name": insider_name,
                    "is_director": "director" in position.lower(),
                    "is_officer": any(value in position.lower() for value in ("officer", "ceo", "cfo")),
                    "is_ten_percent_owner": "10%" in position.lower(),
                    "officer_title": position,
                    "raw_xml_payload": None,
                    "is_amendment": False,
                    "amends_accession_number": None,
                    "transactions": [
                        {
                            "transaction_id": f"{accession}_tx_0",
                            "transaction_date": transaction_date,
                            "transaction_code": transaction_code,
                            "shares": shares,
                            "price_per_share": price_per_share,
                            "acquired_or_disposed": acquired_or_disposed,
                            "shares_owned_following": None,
                            "ownership_nature": "D",
                            "is_derivative": False,
                            "normalized_weight": TRANSACTION_CODE_WEIGHTS.get(transaction_code, 1.0),
                        }
                    ],
                }
            )
        return records


__all__ = ["YFinanceInsiderHistoryAdapter"]
