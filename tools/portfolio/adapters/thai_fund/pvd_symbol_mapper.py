"""PVD Symbol Mapper for BBLAM Provident Funds.

Translates internal PVD fund identifiers (e.g. from port_b9daad statements)
into publicly tradable Thai Mutual Fund symbols or underlying master representations.
"""
from __future__ import annotations

from typing import Tuple

_PVD_TO_MUTUAL_FUND_MAP: dict[str, str] = {
    "PVD_SINSA_GLOB": "B-GLOBAL",
    "PVD_SINSA_CARE": "BCARE",
    "PVD_SINSA_ASIA": "B-ASIA",
    "PVD_SINSA_GTO": "B-GTO",
    "PVD_SINSA_BTP": "BTP",
    "PVD_SINSA_GOLD": "GOLD-99.99",
    "PVD_SINSA_FI": "B-FIXED",
    "PVD_SINSA1": "B-TREASURY",
}


def is_pvd_symbol(symbol: str) -> bool:
    """Check if symbol corresponds to a known PVD scheme code."""
    norm = symbol.strip().upper()
    return norm.startswith("PVD_") or norm in _PVD_TO_MUTUAL_FUND_MAP


def resolve_pvd_symbol(symbol: str) -> Tuple[str, bool]:
    """Resolve PVD symbol to underlying Thai mutual fund symbol.
    
    Returns (resolved_symbol, is_pvd_mapped).
    If symbol is not a PVD identifier, returns (symbol, False).
    """
    clean = symbol.strip().upper()
    if clean in _PVD_TO_MUTUAL_FUND_MAP:
        return _PVD_TO_MUTUAL_FUND_MAP[clean], True
    return clean, False
