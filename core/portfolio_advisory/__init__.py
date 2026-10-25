"""Core Domain: 2-Tier Hybrid Portfolio Advisory Engine.

Pure domain layer isolating:
- Data & Temporal Integrity Guardrails (T1–T5, I1–I7)
- Real-World Market Lot Sizing & Dual-Currency Cash Guard
- Conviction-Adjusted Advisory Engine (<50ms deterministic calculation)
- Post-LLM Strategic Narrative Validator
"""

from core.portfolio_advisory.sector_mapping import (
    CANONICAL_SECTOR_MAP,
    SPDR_ETF_TO_CANONICAL_SECTOR,
    is_sector_match,
    normalize_to_canonical_sector,
)

__all__ = [
    "SPDR_ETF_TO_CANONICAL_SECTOR",
    "CANONICAL_SECTOR_MAP",
    "normalize_to_canonical_sector",
    "is_sector_match",
]
