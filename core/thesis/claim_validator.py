"""Anti-Citation-Laundering & Quantitative Claim Validator.

Architecture Rule (Layer 1 - Core):
Pure Python standard library only — strictly zero external dependencies (no FastAPI, Pydantic,
SQLite, or network I/O) per Hexagonal Architecture AST Domain Layer Isolation rules.

Reference:
Claude Code for Hedge Funds: Build a Reproducible As-Of Research State
Chapter 9 & Appendix H (Anti-Citation Laundering & Evidence Taxonomy)
"""
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core.thesis.evidence_taxonomy import EvidenceTier, EvidenceProvenance


@dataclass(frozen=True)
class ClaimRecord:
    """Quantitative or structural claim attributed to a research source."""
    claim_id: str
    ticker: str
    statement: str
    metric_name: Optional[str] = None
    claimed_value: Optional[Decimal] = None
    evidence_tier: EvidenceTier = EvidenceTier.DOCUMENTED
    source_locator: str = ""
    accession_number: Optional[str] = None
    url: Optional[str] = None
    tolerance_bps: int = 10  # 10 bps default tolerance for rounding differences (0.10%)
    unit: str = ""

    def to_provenance(self) -> EvidenceProvenance:
        return EvidenceProvenance(
            tier=self.evidence_tier,
            source_locator=self.source_locator,
            accession_number=self.accession_number,
            url=self.url,
        )


@dataclass(frozen=True)
class ClaimValidationResult:
    """Outcome of validating a claim against verified primary source facts."""
    is_valid: bool
    claim_id: str
    ticker: str
    original_tier: EvidenceTier
    effective_tier: EvidenceTier
    status: str  # "VERIFIED", "CITATION_MISMATCH", "MISSING_PRIMARY_FACT", "UNVERIFIED_SOURCE", "INFERRED_OPINION"
    explanation: str
    claimed_value: Optional[Decimal] = None
    source_fact_value: Optional[Decimal] = None
    delta_bps: Optional[Decimal] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "claim_id": self.claim_id,
            "ticker": self.ticker,
            "original_tier": self.original_tier.value,
            "effective_tier": self.effective_tier.value,
            "status": self.status,
            "explanation": self.explanation,
            "claimed_value": str(self.claimed_value) if self.claimed_value is not None else None,
            "source_fact_value": str(self.source_fact_value) if self.source_fact_value is not None else None,
            "delta_bps": str(self.delta_bps) if self.delta_bps is not None else None,
        }


def _coerce_decimal(value: Any) -> Optional[Decimal]:
    if value is None:
        return None
    if isinstance(value, Decimal):
        return value
    try:
        return Decimal(str(value).strip().replace(",", ""))
    except (InvalidOperation, ValueError, TypeError):
        return None


def validate_claim_evidence(
    claim: ClaimRecord,
    primary_source_facts: Dict[str, Any],
) -> ClaimValidationResult:
    """Asserts that claimed numbers match primary disclosure sources exactly or within tolerance.
    
    Prevents LLM Citation Laundering: A cited source link does not validate a claim
    unless the underlying quantitative figure in that source agrees with the statement.
    """
    # 1. Unverified tier claims
    if claim.evidence_tier == EvidenceTier.UNVERIFIED:
        return ClaimValidationResult(
            is_valid=False,
            claim_id=claim.claim_id,
            ticker=claim.ticker,
            original_tier=claim.evidence_tier,
            effective_tier=EvidenceTier.UNVERIFIED,
            status="UNVERIFIED",
            explanation=f"Claim {claim.claim_id} is unverified: lacking authoritative primary source.",
            claimed_value=claim.claimed_value,
        )

    # 2. Inferred tier claims (analytical deductions, hypotheses)
    if claim.evidence_tier == EvidenceTier.INFERRED:
        return ClaimValidationResult(
            is_valid=True,
            claim_id=claim.claim_id,
            ticker=claim.ticker,
            original_tier=claim.evidence_tier,
            effective_tier=EvidenceTier.INFERRED,
            status="INFERRED_OPINION",
            explanation=f"Claim {claim.claim_id} is classified as an analytical hypothesis or inference, not an empirical fact.",
            claimed_value=claim.claimed_value,
        )

    # 3. Directly tested tier claims (Python tools, audited SQL, deterministic calculations)
    if claim.evidence_tier == EvidenceTier.DIRECTLY_TESTED:
        if not claim.source_locator.strip():
            return ClaimValidationResult(
                is_valid=False,
                claim_id=claim.claim_id,
                ticker=claim.ticker,
                original_tier=claim.evidence_tier,
                effective_tier=EvidenceTier.UNVERIFIED,
                status="MISSING_EXECUTION_LOCATOR",
                explanation=f"Directly tested claim {claim.claim_id} lacks execution engine/formula locator.",
                claimed_value=claim.claimed_value,
            )
        # If fact is present in tool output dictionary, verify calculation
        if claim.metric_name and claim.metric_name in primary_source_facts:
            tested_val = _coerce_decimal(primary_source_facts[claim.metric_name])
            if tested_val is not None and claim.claimed_value is not None:
                diff = abs(claim.claimed_value - tested_val)
                # Strict 10^-6 tolerance for tested math
                if diff > Decimal("0.000001"):
                    return ClaimValidationResult(
                        is_valid=False,
                        claim_id=claim.claim_id,
                        ticker=claim.ticker,
                        original_tier=claim.evidence_tier,
                        effective_tier=EvidenceTier.UNVERIFIED,
                        status="TEST_MATH_MISMATCH",
                        explanation=(
                            f"Directly tested calculation mismatch for '{claim.metric_name}': "
                            f"claimed {claim.claimed_value} vs engine computed {tested_val}"
                        ),
                        claimed_value=claim.claimed_value,
                        source_fact_value=tested_val,
                    )
        return ClaimValidationResult(
            is_valid=True,
            claim_id=claim.claim_id,
            ticker=claim.ticker,
            original_tier=claim.evidence_tier,
            effective_tier=EvidenceTier.DIRECTLY_TESTED,
            status="VERIFIED",
            explanation=f"Claim {claim.claim_id} verified via deterministic test harness ({claim.source_locator}).",
            claimed_value=claim.claimed_value,
        )

    # 4. Documented tier claims (Primary disclosures, SEC 10-K/Q, 8-K, Press Releases)
    if not claim.source_locator.strip():
        return ClaimValidationResult(
            is_valid=False,
            claim_id=claim.claim_id,
            ticker=claim.ticker,
            original_tier=claim.evidence_tier,
            effective_tier=EvidenceTier.UNVERIFIED,
            status="UNVERIFIED_SOURCE",
            explanation=f"Documented claim {claim.claim_id} lacks source locator or accession number.",
            claimed_value=claim.claimed_value,
        )

    # If claim doesn't assert a specific metric, check locator existence
    if not claim.metric_name:
        return ClaimValidationResult(
            is_valid=True,
            claim_id=claim.claim_id,
            ticker=claim.ticker,
            original_tier=claim.evidence_tier,
            effective_tier=EvidenceTier.DOCUMENTED,
            status="VERIFIED_QUALITATIVE",
            explanation=f"Qualitative documented claim verified with locator '{claim.source_locator}'.",
            claimed_value=claim.claimed_value,
        )

    # Quantitative claim check against primary source facts
    if claim.metric_name not in primary_source_facts:
        return ClaimValidationResult(
            is_valid=False,
            claim_id=claim.claim_id,
            ticker=claim.ticker,
            original_tier=claim.evidence_tier,
            effective_tier=EvidenceTier.UNVERIFIED,
            status="MISSING_PRIMARY_FACT",
            explanation=(
                f"Anti-Citation Laundering Alert: Claimed metric '{claim.metric_name}' is not found in primary "
                f"facts extracted from {claim.source_locator}. Demoted to UNVERIFIED."
            ),
            claimed_value=claim.claimed_value,
        )

    source_raw = primary_source_facts[claim.metric_name]
    source_val = _coerce_decimal(source_raw)
    if source_val is None:
        return ClaimValidationResult(
            is_valid=False,
            claim_id=claim.claim_id,
            ticker=claim.ticker,
            original_tier=claim.evidence_tier,
            effective_tier=EvidenceTier.UNVERIFIED,
            status="NON_NUMERIC_PRIMARY_FACT",
            explanation=f"Primary fact for '{claim.metric_name}' is not a valid number: {source_raw}",
            claimed_value=claim.claimed_value,
        )

    if claim.claimed_value is None:
        # Claim specifies metric name but no number
        return ClaimValidationResult(
            is_valid=True,
            claim_id=claim.claim_id,
            ticker=claim.ticker,
            original_tier=claim.evidence_tier,
            effective_tier=EvidenceTier.DOCUMENTED,
            status="VERIFIED",
            explanation=f"Metric '{claim.metric_name}' confirmed present in {claim.source_locator}.",
            source_fact_value=source_val,
        )

    # Check delta in basis points or absolute difference
    if source_val == Decimal("0"):
        diff = abs(claim.claimed_value - source_val)
        delta_bps = Decimal("0") if diff == Decimal("0") else Decimal("999999")
    else:
        # delta_bps = |claimed - source| / |source| * 10,000
        delta_bps = (abs(claim.claimed_value - source_val) / abs(source_val) * Decimal("10000")).quantize(Decimal("0.01"))

    # Check tolerance
    if delta_bps > Decimal(str(claim.tolerance_bps)):
        return ClaimValidationResult(
            is_valid=False,
            claim_id=claim.claim_id,
            ticker=claim.ticker,
            original_tier=claim.evidence_tier,
            effective_tier=EvidenceTier.UNVERIFIED,
            status="CITATION_MISMATCH",
            explanation=(
                f"Anti-Citation Laundering VIOLATION: Claimed {claim.metric_name}={claim.claimed_value} "
                f"does not match documented primary source fact {source_val} from '{claim.source_locator}' "
                f"(delta: {delta_bps} bps exceeds tolerance {claim.tolerance_bps} bps). "
                f"Demoted to UNVERIFIED."
            ),
            claimed_value=claim.claimed_value,
            source_fact_value=source_val,
            delta_bps=delta_bps,
        )

    return ClaimValidationResult(
        is_valid=True,
        claim_id=claim.claim_id,
        ticker=claim.ticker,
        original_tier=claim.evidence_tier,
        effective_tier=EvidenceTier.DOCUMENTED,
        status="VERIFIED",
        explanation=(
            f"Fact tie-out confirmed: {claim.metric_name}={claim.claimed_value} matches primary source "
            f"fact {source_val} ({claim.source_locator}, delta: {delta_bps} bps)."
        ),
        claimed_value=claim.claimed_value,
        source_fact_value=source_val,
        delta_bps=delta_bps,
    )


def batch_validate_claims(
    claims: Sequence[ClaimRecord],
    primary_source_facts: Dict[str, Any],
) -> Tuple[List[ClaimValidationResult], bool]:
    """Validates a batch of thesis claims against primary source disclosure facts."""
    results: List[ClaimValidationResult] = []
    all_valid = True
    for claim in claims:
        res = validate_claim_evidence(claim, primary_source_facts)
        results.append(res)
        if not res.is_valid:
            all_valid = False
    return results, all_valid

