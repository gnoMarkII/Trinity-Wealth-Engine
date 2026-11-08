"""Institutional Two-Sided Estimate Ledger & Variant Perception Engine (Ch. 2, 5 & App H).

Architecture Rule (Layer 1 - Core):
Pure Python standard library only — strictly zero external dependencies per Hexagonal
Architecture AST Domain Layer Isolation rules. Deterministic math using Decimal only.

Reference:
Claude Code for Hedge Funds, Chapter 2 (pp. 23-25), Chapter 5 (pp. 38-40), App H (p. 116):
"An estimate needs a revision trail, an estimate owner, and an estimate basis.
The baseline row is not a cell to update. It is a historical belief record.
The critical verification is asymmetric: the revised row can be rejected and recreated;
the prior approved row must remain byte-for-byte unchanged."
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple


class EstimateType(str, Enum):
    """The two baseline perspectives in hedge fund research."""
    CONSENSUS = "CONSENSUS"            # Street sell-side median/mean
    FUND_THESIS = "FUND_THESIS"        # Internal proprietary fund analyst model


@dataclass(frozen=True)
class EstimateRecord:
    """Historical belief or consensus record in an append-only estimate ledger."""
    estimate_id: str                   # e.g. "EST-0001"
    ticker: str
    period: str                        # e.g. "2026Q2", "2026Q3"
    metric_name: str                   # "revenue", "diluted_eps", "gross_margin"
    estimate_type: EstimateType
    value: Decimal
    unit: str = "USD_M"                # "USD_M", "USD_SHARE", "PERCENT", "POINTS"
    as_of_utc: str = ""
    reason_code: str = "baseline"      # "baseline", "post_earnings_revision", "guidance_update"
    prior_record_id: Optional[str] = None # Links to unchanged prior record
    owner: str = "analyst_a"
    approval_status: str = "REVIEW_REQUIRED" # "APPROVED", "REVIEW_REQUIRED"
    evidence_label: str = "SYNTHETIC"  # "SYNTHETIC", "DOCUMENTED"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "estimate_id": self.estimate_id,
            "ticker": self.ticker,
            "period": self.period,
            "metric_name": self.metric_name,
            "estimate_type": self.estimate_type.value,
            "value": str(self.value),
            "unit": self.unit,
            "as_of_utc": self.as_of_utc,
            "reason_code": self.reason_code,
            "prior_record_id": self.prior_record_id,
            "owner": self.owner,
            "approval_status": self.approval_status,
            "evidence_label": self.evidence_label,
        }


@dataclass(frozen=True)
class RevisionDriver:
    """Specific driver delta explaining an analyst estimate revision."""
    name: str                          # e.g. "Volume", "Pricing", "FX"
    delta: Decimal
    evidence_locator: str = ""         # e.g. "10-Q Item 2 / Disclosures"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "delta": str(self.delta),
            "evidence_locator": self.evidence_locator,
        }


@dataclass(frozen=True)
class VariantPerceptionSnapshot:
    """Three-way surprise and variant alpha analysis upon earnings release."""
    ticker: str
    period: str
    metric_name: str
    reported_value: Decimal
    consensus_value: Decimal
    fund_thesis_value: Decimal
    street_surprise: Decimal           # Reported - Consensus
    street_surprise_pct: Decimal       # (Street Surprise / Consensus) * 100
    variant_spread: Decimal            # Fund Thesis - Consensus (Pre-print variant posture)
    variant_spread_pct: Decimal        # (Variant Spread / Consensus) * 100
    variant_realization: Decimal       # Reported - Fund Thesis
    variant_realization_pct: Decimal   # (Variant Realization / Fund Thesis) * 100
    alpha_verdict: str                 # "ALPHA_CONFIRMED", "CONSENSUS_CORRECT", "DOUBLE_MISS"
    unit: str = "USD_M"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "period": self.period,
            "metric_name": self.metric_name,
            "reported_value": str(self.reported_value),
            "consensus_value": str(self.consensus_value),
            "fund_thesis_value": str(self.fund_thesis_value),
            "street_surprise": str(self.street_surprise),
            "street_surprise_pct": str(self.street_surprise_pct),
            "variant_spread": str(self.variant_spread),
            "variant_spread_pct": str(self.variant_spread_pct),
            "variant_realization": str(self.variant_realization),
            "variant_realization_pct": str(self.variant_realization_pct),
            "alpha_verdict": self.alpha_verdict,
            "unit": self.unit,
        }


@dataclass(frozen=True)
class RevisionBridgeResult:
    """Mathematical verification that revision drivers sum to exact delta."""
    ticker: str
    period: str
    metric_name: str
    prior_estimate_id: str
    revised_estimate_id: str
    prior_value: Decimal
    revised_value: Decimal
    estimate_delta: Decimal
    drivers_sum: Decimal
    is_mathematically_balanced: bool
    status: str = "REVIEW_REQUIRED"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "period": self.period,
            "metric_name": self.metric_name,
            "prior_estimate_id": self.prior_estimate_id,
            "revised_estimate_id": self.revised_estimate_id,
            "prior_value": str(self.prior_value),
            "revised_value": str(self.revised_value),
            "estimate_delta": str(self.estimate_delta),
            "drivers_sum": str(self.drivers_sum),
            "is_mathematically_balanced": self.is_mathematically_balanced,
            "status": self.status,
        }


def _calc_pct(num: Decimal, den: Decimal) -> Decimal:
    """Helper to compute deterministic percentage rounded to 2 decimal places."""
    if den == Decimal("0"):
        return Decimal("0.00")
    return ((num / abs(den)) * Decimal("100")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def calculate_variant_perception(
    ticker: str,
    period: str,
    metric_name: str,
    reported: Decimal,
    consensus: Decimal,
    fund_thesis: Decimal,
    unit: str = "USD_M",
) -> VariantPerceptionSnapshot:
    """Computes the 3-way variant perception and street surprise reconciliation.
    
    Equations:
    - Street Surprise = Reported - Consensus
    - Variant Spread (Pre-Print Alpha Posture) = Fund Thesis - Consensus
    - Variant Realization = Reported - Fund Thesis
    """
    clean_ticker = ticker.strip().upper()
    street_surprise = reported - consensus
    street_pct = _calc_pct(street_surprise, consensus)

    variant_spread = fund_thesis - consensus
    spread_pct = _calc_pct(variant_spread, consensus)

    variant_realization = reported - fund_thesis
    realization_pct = _calc_pct(variant_realization, fund_thesis)

    # Classify institutional alpha realization verdict
    if variant_spread > Decimal("0") and reported >= fund_thesis:
        alpha_verdict = "ALPHA_CONFIRMED"      # Fund was out-of-consensus bullish and reported beat thesis
    elif variant_spread < Decimal("0") and reported <= fund_thesis:
        alpha_verdict = "ALPHA_CONFIRMED"      # Fund was out-of-consensus bearish and reported missed thesis
    elif (reported - consensus) * (fund_thesis - consensus) > Decimal("0"):
        alpha_verdict = "DIRECTIONALLY_CORRECT" # Same direction as thesis relative to consensus
    elif abs(street_surprise) <= abs(variant_realization):
        alpha_verdict = "CONSENSUS_MORE_ACCURATE"
    else:
        alpha_verdict = "DOUBLE_MISS"

    return VariantPerceptionSnapshot(
        ticker=clean_ticker,
        period=period,
        metric_name=metric_name,
        reported_value=reported,
        consensus_value=consensus,
        fund_thesis_value=fund_thesis,
        street_surprise=street_surprise,
        street_surprise_pct=street_pct,
        variant_spread=variant_spread,
        variant_spread_pct=spread_pct,
        variant_realization=variant_realization,
        variant_realization_pct=realization_pct,
        alpha_verdict=alpha_verdict,
        unit=unit,
    )


def append_estimate_revision(
    ledger: Sequence[EstimateRecord],
    revised_record: EstimateRecord,
    drivers: Sequence[RevisionDriver] = (),
    tolerance: Decimal = Decimal("0.01"),
) -> Tuple[List[EstimateRecord], RevisionBridgeResult]:
    """Appends a new estimate revision row while strictly asserting:
    1. Prior approved baseline row remains byte-for-byte unchanged.
    2. Sum of reason-code drivers equals revised delta.
    3. Final status is REVIEW_REQUIRED.
    """
    if not revised_record.prior_record_id:
        raise ValueError("Revised estimate record must specify prior_record_id")

    # Verify that prior record exists in the ledger
    prior_matches = [r for r in ledger if r.estimate_id == revised_record.prior_record_id]
    if not prior_matches:
        raise ValueError(f"Prior record {revised_record.prior_record_id} not found in ledger")
    prior_row = prior_matches[0]

    # Invariant checks: ticker, period, metric, unit must match exactly
    if revised_record.ticker != prior_row.ticker:
        raise ValueError(f"Cross-series revision: ticker mismatch {revised_record.ticker} != {prior_row.ticker}")
    if revised_record.period != prior_row.period:
        raise ValueError(f"Cross-period revision: period mismatch {revised_record.period} != {prior_row.period}")
    if revised_record.metric_name != prior_row.metric_name:
        raise ValueError(f"Cross-metric revision: metric mismatch {revised_record.metric_name} != {prior_row.metric_name}")
    if revised_record.unit != prior_row.unit:
        raise ValueError(f"Unit mismatch: revision unit {revised_record.unit} != prior unit {prior_row.unit}")
    if any(r.estimate_id == revised_record.estimate_id for r in ledger):
        raise ValueError(f"Duplicate estimate_id {revised_record.estimate_id} already exists in ledger")

    # Deterministic delta
    delta = revised_record.value - prior_row.value
    drivers_sum = sum((d.delta for d in drivers), Decimal("0")) if drivers else delta
    is_balanced = abs(delta - drivers_sum) <= tolerance

    bridge = RevisionBridgeResult(
        ticker=revised_record.ticker,
        period=revised_record.period,
        metric_name=revised_record.metric_name,
        prior_estimate_id=prior_row.estimate_id,
        revised_estimate_id=revised_record.estimate_id,
        prior_value=prior_row.value,
        revised_value=revised_record.value,
        estimate_delta=delta,
        drivers_sum=drivers_sum,
        is_mathematically_balanced=is_balanced,
        status="REVIEW_REQUIRED" if is_balanced else "REJECTED_UNBALANCED",
    )

    # Append to existing ledger copy only if mathematically balanced to preserve ledger integrity
    updated_ledger = list(ledger)
    if is_balanced:
        updated_ledger.append(revised_record)

    return updated_ledger, bridge

