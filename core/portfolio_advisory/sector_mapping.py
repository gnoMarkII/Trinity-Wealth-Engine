"""Three-Way Sector Mapping: SPDR ETF Ticker <-> SPDR Sector Name <-> Peer Sector / GICS.

Architecture Layer: core/portfolio_advisory/ (Pure Domain)
Provides canonical sector normalization and alignment across Macro Sector Rotation
and Micro Equity Quant signals.
"""
from __future__ import annotations

from typing import Optional


# Mapping of SPDR US Sector ETFs to their canonical sector names
SPDR_ETF_TO_CANONICAL_SECTOR: dict[str, str] = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLV": "Healthcare",
    "XLE": "Energy",
    "XLI": "Industrials",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLU": "Utilities",
    "XLRE": "Real Estate",
    "XLB": "Materials",
    "XLC": "Communication Services",
}

# Reverse mapping: Canonical sector name to SPDR ETF ticker
CANONICAL_SECTOR_TO_SPDR_ETF: dict[str, str] = {
    v: k for k, v in SPDR_ETF_TO_CANONICAL_SECTOR.items()
}

# Extensive alias map covering Yahoo Finance, GICS, Bloomberg, SET, and Thai/English variations
CANONICAL_SECTOR_MAP: dict[str, str] = {
    # Technology
    "technology": "Technology",
    "tech": "Technology",
    "information technology": "Technology",
    "it": "Technology",
    "semiconductors": "Technology",
    "software": "Technology",
    "hardware": "Technology",
    "xlk": "Technology",
    # Financials
    "financials": "Financials",
    "financial services": "Financials",
    "financial": "Financials",
    "finance": "Financials",
    "bank": "Financials",
    "banking": "Financials",
    "insurance": "Financials",
    "xlf": "Financials",
    # Healthcare
    "healthcare": "Healthcare",
    "health care": "Healthcare",
    "pharma": "Healthcare",
    "pharmaceuticals": "Healthcare",
    "biotechnology": "Healthcare",
    "biotech": "Healthcare",
    "medical": "Healthcare",
    "xlv": "Healthcare",
    # Energy
    "energy": "Energy",
    "oil & gas": "Energy",
    "oil and gas": "Energy",
    "petroleum": "Energy",
    "xle": "Energy",
    # Industrials
    "industrials": "Industrials",
    "industrial": "Industrials",
    "manufacturing": "Industrials",
    "aerospace": "Industrials",
    "defense": "Industrials",
    "logistics": "Industrials",
    "transportation": "Industrials",
    "xli": "Industrials",
    # Consumer Discretionary
    "consumer discretionary": "Consumer Discretionary",
    "consumer cyclical": "Consumer Discretionary",
    "discretionary": "Consumer Discretionary",
    "cyclical": "Consumer Discretionary",
    "automotive": "Consumer Discretionary",
    "retail": "Consumer Discretionary",
    "xly": "Consumer Discretionary",
    # Consumer Staples
    "consumer staples": "Consumer Staples",
    "consumer defensive": "Consumer Staples",
    "staples": "Consumer Staples",
    "food & beverage": "Consumer Staples",
    "food and beverage": "Consumer Staples",
    "f&b": "Consumer Staples",
    "commerce": "Consumer Staples",
    "xlp": "Consumer Staples",
    # Utilities
    "utilities": "Utilities",
    "utility": "Utilities",
    "power": "Utilities",
    "clean energy": "Utilities",
    "xlu": "Utilities",
    # Real Estate
    "real estate": "Real Estate",
    "reit": "Real Estate",
    "property": "Real Estate",
    "prop": "Real Estate",
    "xlre": "Real Estate",
    # Materials
    "materials": "Materials",
    "basic materials": "Materials",
    "chemicals": "Materials",
    "metals & mining": "Materials",
    "xlb": "Materials",
    # Communication Services
    "communication services": "Communication Services",
    "communication": "Communication Services",
    "telecommunication": "Communication Services",
    "telecommunications": "Communication Services",
    "telecom": "Communication Services",
    "media": "Communication Services",
    "xlc": "Communication Services",
}


def normalize_to_canonical_sector(name_or_ticker: Optional[str]) -> Optional[str]:
    """Normalizes any sector name, alias, or ETF ticker to its canonical GICS Sector string.

    Returns canonical string (e.g. 'Technology', 'Financials') or None if unrecognized.
    """
    if not name_or_ticker or not isinstance(name_or_ticker, str):
        return None
    raw = name_or_ticker.strip().lower()
    if raw in CANONICAL_SECTOR_MAP:
        return CANONICAL_SECTOR_MAP[raw]
    # Check partial contains with word boundaries for short aliases (<= 3 chars)
    for alias, canonical in CANONICAL_SECTOR_MAP.items():
        if len(alias) <= 3:
            import re
            if re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", raw):
                return canonical
        else:
            if alias in raw:
                return canonical
    return name_or_ticker.strip().title()


def map_etf_to_canonical_sector(ticker: Optional[str]) -> Optional[str]:
    """Maps SPDR Sector ETF ticker (e.g. 'XLK') to canonical sector string ('Technology')."""
    if not ticker:
        return None
    cleaned = ticker.strip().upper()
    return SPDR_ETF_TO_CANONICAL_SECTOR.get(cleaned)


def is_sector_match(sector_a: Optional[str], sector_b: Optional[str]) -> bool:
    """Checks whether two sector identifiers map to the same canonical sector."""
    if not sector_a or not sector_b:
        return False
    can_a = normalize_to_canonical_sector(sector_a)
    can_b = normalize_to_canonical_sector(sector_b)
    if can_a and can_b:
        return can_a.lower() == can_b.lower()
    return sector_a.strip().lower() == sector_b.strip().lower()
