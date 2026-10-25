"""Master Fund / ETF Look-Through Catalog.

Provides curated top holdings and sector distributions for common Master ETFs and
global funds utilized by Thai Feeder Funds and Provident Funds (PVD).
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Optional, List, Tuple


@dataclass(frozen=True)
class MasterFundEntry:
    master_name: str
    master_ticker: str
    top_holdings: List[Tuple[str, str, Decimal]]  # (ticker, company_name, weight_pct)
    sector_breakdown: List[Tuple[str, Decimal]]   # (gics_sector_en, weight_pct)
    as_of_date: str = "2026-03-31"
    is_curated: bool = True


# Catalog of recognized Master ETFs and Active Funds
_MASTER_FUND_CATALOG: list[MasterFundEntry] = [
    # 1. iShares Core S&P 500 ETF (IVV / SPY)
    MasterFundEntry(
        master_name="iShares Core S&P 500 ETF",
        master_ticker="IVV",
        top_holdings=[
            ("MSFT", "Microsoft Corporation", Decimal("7.1")),
            ("AAPL", "Apple Inc.", Decimal("6.9")),
            ("NVDA", "NVIDIA Corporation", Decimal("6.1")),
            ("AMZN", "Amazon.com Inc.", Decimal("3.7")),
            ("META", "Meta Platforms Inc.", Decimal("2.5")),
            ("GOOGL", "Alphabet Inc. Class A", Decimal("2.1")),
            ("GOOG", "Alphabet Inc. Class C", Decimal("1.8")),
            ("BRK.B", "Berkshire Hathaway Inc.", Decimal("1.7")),
            ("AVGO", "Broadcom Inc.", Decimal("1.6")),
            ("JPM", "JPMorgan Chase & Co.", Decimal("1.3")),
        ],
        sector_breakdown=[
            ("Information Technology", Decimal("31.5")),
            ("Financials", Decimal("13.0")),
            ("Health Care", Decimal("11.5")),
            ("Consumer Discretionary", Decimal("10.2")),
            ("Communication Services", Decimal("9.0")),
            ("Industrials", Decimal("8.3")),
            ("Consumer Staples", Decimal("5.8")),
            ("Energy", Decimal("3.6")),
            ("Utilities", Decimal("2.5")),
            ("Real Estate", Decimal("2.3")),
            ("Materials", Decimal("2.3")),
        ],
    ),
    # 2. JPMorgan Nasdaq Equity Premium Income ETF (JEPQ / UCITS)
    MasterFundEntry(
        master_name="JPMorgan Nasdaq Equity Premium Income ETF",
        master_ticker="JEPQ",
        top_holdings=[
            ("MSFT", "Microsoft Corporation", Decimal("7.8")),
            ("AAPL", "Apple Inc.", Decimal("7.5")),
            ("NVDA", "NVIDIA Corporation", Decimal("6.5")),
            ("AMZN", "Amazon.com Inc.", Decimal("5.2")),
            ("META", "Meta Platforms Inc.", Decimal("4.6")),
            ("GOOGL", "Alphabet Inc. Class A", Decimal("3.8")),
            ("AVGO", "Broadcom Inc.", Decimal("3.4")),
            ("TSLA", "Tesla Inc.", Decimal("2.8")),
            ("COST", "Costco Wholesale Corp.", Decimal("2.1")),
            ("NFLX", "Netflix Inc.", Decimal("1.9")),
        ],
        sector_breakdown=[
            ("Information Technology", Decimal("48.0")),
            ("Communication Services", Decimal("16.0")),
            ("Consumer Discretionary", Decimal("14.0")),
            ("Consumer Staples", Decimal("6.0")),
            ("Health Care", Decimal("6.0")),
            ("Industrials", Decimal("5.0")),
            ("Financials", Decimal("3.0")),
            ("Utilities", Decimal("2.0")),
        ],
    ),
    # 3. iShares Edge MSCI World Value Factor UCITS ETF (IWVL)
    MasterFundEntry(
        master_name="iShares Edge MSCI World Value Factor UCITS ETF",
        master_ticker="IWVL",
        top_holdings=[
            ("CSCO", "Cisco Systems Inc.", Decimal("2.4")),
            ("INTC", "Intel Corporation", Decimal("2.1")),
            ("IBM", "International Business Machines Corp.", Decimal("2.0")),
            ("PFE", "Pfizer Inc.", Decimal("1.9")),
            ("BMY", "Bristol-Myers Squibb Co.", Decimal("1.8")),
            ("C", "Citigroup Inc.", Decimal("1.7")),
            ("VZ", "Verizon Communications Inc.", Decimal("1.6")),
            ("T", "AT&T Inc.", Decimal("1.5")),
            ("CMCSA", "Comcast Corporation", Decimal("1.4")),
            ("CVS", "CVS Health Corporation", Decimal("1.3")),
        ],
        sector_breakdown=[
            ("Financials", Decimal("28.0")),
            ("Health Care", Decimal("16.0")),
            ("Industrials", Decimal("12.0")),
            ("Information Technology", Decimal("11.0")),
            ("Consumer Staples", Decimal("9.0")),
            ("Energy", Decimal("8.0")),
            ("Consumer Discretionary", Decimal("6.0")),
            ("Communication Services", Decimal("5.0")),
            ("Materials", Decimal("5.0")),
        ],
    ),
    # 4. Wellington Global Quality Growth Fund
    MasterFundEntry(
        master_name="Wellington Global Quality Growth Fund",
        master_ticker="WELLINGTON-GQG",
        top_holdings=[
            ("MSFT", "Microsoft Corporation", Decimal("5.8")),
            ("GOOGL", "Alphabet Inc.", Decimal("4.9")),
            ("AMZN", "Amazon.com Inc.", Decimal("4.2")),
            ("NVDA", "NVIDIA Corporation", Decimal("4.0")),
            ("META", "Meta Platforms Inc.", Decimal("3.5")),
            ("V", "Visa Inc.", Decimal("3.1")),
            ("MA", "Mastercard Inc.", Decimal("2.9")),
            ("ASML", "ASML Holding NV", Decimal("2.6")),
            ("UNH", "UnitedHealth Group Inc.", Decimal("2.4")),
            ("LLY", "Eli Lilly and Co.", Decimal("2.3")),
        ],
        sector_breakdown=[
            ("Information Technology", Decimal("36.0")),
            ("Financials", Decimal("18.0")),
            ("Health Care", Decimal("14.0")),
            ("Consumer Discretionary", Decimal("12.0")),
            ("Communication Services", Decimal("10.0")),
            ("Industrials", Decimal("8.0")),
            ("Consumer Staples", Decimal("2.0")),
        ],
    ),
    # 5. Wellington Global Health Care Equity Fund
    MasterFundEntry(
        master_name="Wellington Global Health Care Equity Fund",
        master_ticker="WELLINGTON-GHC",
        top_holdings=[
            ("LLY", "Eli Lilly and Co.", Decimal("8.2")),
            ("UNH", "UnitedHealth Group Inc.", Decimal("7.5")),
            ("NVO", "Novo Nordisk A/S", Decimal("6.8")),
            ("ABBV", "AbbVie Inc.", Decimal("5.4")),
            ("MRK", "Merck & Co. Inc.", Decimal("5.1")),
            ("TMO", "Thermo Fisher Scientific Inc.", Decimal("4.8")),
            ("AZN", "AstraZeneca PLC", Decimal("4.5")),
            ("ISRG", "Intuitive Surgical Inc.", Decimal("3.9")),
            ("DHR", "Danaher Corporation", Decimal("3.6")),
            ("BSX", "Boston Scientific Corp.", Decimal("3.2")),
        ],
        sector_breakdown=[
            ("Health Care", Decimal("100.0")),
        ],
    ),
    # 6. Invesco Asian Equity Fund
    MasterFundEntry(
        master_name="Invesco Asian Equity Fund",
        master_ticker="INVESCO-ASIA",
        top_holdings=[
            ("TSM", "Taiwan Semiconductor Manufacturing", Decimal("9.2")),
            ("0700.HK", "Tencent Holdings Ltd.", Decimal("7.5")),
            ("005930.KS", "Samsung Electronics Co. Ltd.", Decimal("6.4")),
            ("9988.HK", "Alibaba Group Holding Ltd.", Decimal("4.8")),
            ("IBN", "ICICI Bank Ltd.", Decimal("3.9")),
            ("2317.TW", "Hon Hai Precision Industry", Decimal("3.2")),
            ("RELIANCE.NS", "Reliance Industries Ltd.", Decimal("3.0")),
            ("1299.HK", "AIA Group Ltd.", Decimal("2.8")),
            ("INFYS", "Infosys Ltd.", Decimal("2.5")),
            ("035420.KS", "NAVER Corp.", Decimal("2.1")),
        ],
        sector_breakdown=[
            ("Information Technology", Decimal("32.0")),
            ("Financials", Decimal("24.0")),
            ("Consumer Discretionary", Decimal("16.0")),
            ("Communication Services", Decimal("11.0")),
            ("Industrials", Decimal("7.0")),
            ("Materials", Decimal("5.0")),
            ("Energy", Decimal("3.0")),
            ("Consumer Staples", Decimal("2.0")),
        ],
    ),
    # 7. Vanguard Total International Stock ETF (VXUS)
    MasterFundEntry(
        master_name="Vanguard Total International Stock ETF",
        master_ticker="VXUS",
        top_holdings=[
            ("TSM", "Taiwan Semiconductor Manufacturing", Decimal("2.1")),
            ("ASML", "ASML Holding NV", Decimal("1.8")),
            ("NVO", "Novo Nordisk A/S", Decimal("1.7")),
            ("0700.HK", "Tencent Holdings Ltd.", Decimal("1.3")),
            ("SAP", "SAP SE", Decimal("1.1")),
            ("005930.KS", "Samsung Electronics Co. Ltd.", Decimal("1.0")),
            ("AZN", "AstraZeneca PLC", Decimal("1.0")),
            ("NESN", "Nestle SA", Decimal("0.9")),
            ("ROG", "Roche Holding AG", Decimal("0.9")),
            ("SHEL", "Shell PLC", Decimal("0.8")),
        ],
        sector_breakdown=[
            ("Financials", Decimal("21.0")),
            ("Industrials", Decimal("16.0")),
            ("Information Technology", Decimal("14.0")),
            ("Health Care", Decimal("11.0")),
            ("Consumer Discretionary", Decimal("10.0")),
            ("Consumer Staples", Decimal("7.0")),
            ("Materials", Decimal("7.0")),
            ("Energy", Decimal("5.0")),
            ("Communication Services", Decimal("4.0")),
            ("Utilities", Decimal("3.0")),
            ("Real Estate", Decimal("2.0")),
        ],
    ),
]

# Aliases and search terms mapping to catalog entries
_SEARCH_PATTERNS: list[tuple[list[str], MasterFundEntry]] = [
    (
        ["ISHARES CORE S&P 500", "ISHARES S&P 500", "IVV", "SPY", "SCBS&P500", "S&P 500"],
        _MASTER_FUND_CATALOG[0],
    ),
    (
        ["JPM NQ EQ PREM", "JEPQ", "TLNDQINCOME", "NASDAQ 100 EQUITY PREMIUM"],
        _MASTER_FUND_CATALOG[1],
    ),
    (
        ["ISHARES EDGE MSCI WORLD VALUE", "MSCI WORLD VALUE", "IWVL", "SCBGVALUE"],
        _MASTER_FUND_CATALOG[2],
    ),
    (
        ["WELLINGTON GLOBAL QUALITY", "WELLINGTON QUALITY", "B-GLOBAL", "PVD_SINSA_GLOB"],
        _MASTER_FUND_CATALOG[3],
    ),
    (
        ["WELLINGTON GLOBAL HEALTH", "WELLINGTON HEALTH", "BCARE", "PVD_SINSA_CARE"],
        _MASTER_FUND_CATALOG[4],
    ),
    (
        ["INVESCO ASIAN", "B-ASIA", "PVD_SINSA_ASIA"],
        _MASTER_FUND_CATALOG[5],
    ),
    (
        ["VANGUARD TOTAL INTL", "VXUS", "TLWORLDXUS"],
        _MASTER_FUND_CATALOG[6],
    ),
]


def find_master_fund(query: str) -> Optional[MasterFundEntry]:
    """Find matching Master Fund entry by name, ticker, or holding description."""
    if not query:
        return None

    norm_query = " ".join(query.strip().upper().replace("-", " ").replace("_", " ").split())

    # Exact or keyword substring search
    for keywords, entry in _SEARCH_PATTERNS:
        for kw in keywords:
            clean_kw = " ".join(kw.strip().upper().replace("-", " ").replace("_", " ").split())
            if clean_kw in norm_query or norm_query == clean_kw:
                return entry

    return None
