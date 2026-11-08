"""Evidence Taxonomy & Anti-Citation-Laundering Domain Primitives.

Architecture Rule (Layer 1 - Core):
Pure Python standard library only — strictly zero external dependencies
per Hexagonal Architecture AST Domain Layer Isolation rules.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class EvidenceTier(str, Enum):
    """Four-tier hierarchy of evidence provenance for institutional research."""
    DOCUMENTED = "DOCUMENTED"         # Primary filing (SEC 10-K/10-Q, 8-K, official PR, audited report)
    DIRECTLY_TESTED = "DIRECTLY_TESTED" # Deterministic Python code execution, Decimal math, audited SQL
    INFERRED = "INFERRED"             # Analytical deduction, statistical model, AI analytical synthesis
    UNVERIFIED = "UNVERIFIED"         # Speculation, unverified rumors, social media sentiment


@dataclass(frozen=True)
class EvidenceProvenance:
    """Immutable audit metadata attributing a financial fact or thesis claim to its origin."""
    tier: EvidenceTier
    source_locator: str               # e.g. "SEC 10-Q 2026-08-15 p.24" or "Python Sandbox v1"
    accession_number: Optional[str] = None
    url: Optional[str] = None
    verified_at_utc: Optional[str] = None
    verification_hash: Optional[str] = None

    def to_badge_tag(self) -> str:
        """Converts provenance to an institutional Markdown tag."""
        if self.tier == EvidenceTier.DOCUMENTED:
            return f"[DOCUMENTED: {self.source_locator}]"
        elif self.tier == EvidenceTier.DIRECTLY_TESTED:
            return f"[TESTED: {self.source_locator}]"
        elif self.tier == EvidenceTier.INFERRED:
            return f"[INFERRED: {self.source_locator}]" if self.source_locator else "[INFERRED]"
        else:
            return f"[UNVERIFIED: {self.source_locator}]" if self.source_locator else "[UNVERIFIED]"

