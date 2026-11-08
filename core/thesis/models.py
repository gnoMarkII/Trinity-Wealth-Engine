"""Structured Thesis, Falsifiers & As-Of Research State Domain Models.

Architecture Rule (Layer 1 - Core):
Pure Python standard library only — strictly zero external dependencies (no FastAPI, Pydantic,
SQLite, or yfinance) per Hexagonal Architecture AST Domain Layer Isolation rules.
"""
from dataclasses import dataclass, field
from enum import Enum
from decimal import Decimal
import hashlib
import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core.thesis.evidence_taxonomy import EvidenceTier


@dataclass(frozen=True)
class FalsifierRule:
    """An explicit quantitative or structural condition that invalidates an active investment thesis."""
    falsifier_id: str
    metric_name: str                     # e.g. "datacenter_revenue", "gross_margin", "operating_cash_flow"
    operator: str                        # "<", "<=", ">", ">=", "==", "!=", "pct_drop_exceeds"
    threshold_value: Decimal             # Deterministic Decimal threshold
    unit: str = "USD"                    # "USD", "THB", "PERCENT", "BPS", "RATIO"
    period_basis: str = "QUARTERLY"      # "QUARTERLY", "TTM", "ANNUAL", "DAILY"
    invalidation_impact: str = "BREAKS_BUY_CASE" # "BREAKS_BUY_CASE", "FORCES_PORTFOLIO_TRIM", "SETTLES_BUY_VS_SELL"
    narrative_explanation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "falsifier_id": self.falsifier_id,
            "metric_name": self.metric_name,
            "operator": self.operator,
            "threshold_value": str(self.threshold_value),
            "unit": self.unit,
            "period_basis": self.period_basis,
            "invalidation_impact": self.invalidation_impact,
            "narrative_explanation": self.narrative_explanation,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FalsifierRule":
        return cls(
            falsifier_id=data["falsifier_id"],
            metric_name=data["metric_name"],
            operator=data["operator"],
            threshold_value=Decimal(str(data["threshold_value"])),
            unit=data.get("unit", "USD"),
            period_basis=data.get("period_basis", "QUARTERLY"),
            invalidation_impact=data.get("invalidation_impact", "BREAKS_BUY_CASE"),
            narrative_explanation=data.get("narrative_explanation", ""),
        )


class CatalystStatus(str, Enum):
    """Strict institutional status vocabulary for thesis catalysts (Ch. 7, p. 50)."""
    PENDING = "PENDING"
    CONFIRMED = "CONFIRMED"
    DELAYED = "DELAYED"
    MISSED = "MISSED"
    BREACHED = "BREACHED"


class CatalystDirection(str, Enum):
    """Factual surprise direction for observed catalyst outcomes."""
    INLINE = "INLINE"
    AHEAD = "AHEAD"
    BEHIND = "BEHIND"
    NOT_APPLICABLE = "N/A"


@dataclass(frozen=True)
class CatalystEvent:
    """A verifiable forward-looking milestone that will confirm or falsify the thesis."""
    catalyst_id: str
    event_name: str
    expected_date_str: str               # YYYY-MM-DD or YYYY-QN
    confidence_window_days: int = 7      # Acceptable scheduling margin
    source_locator: str = ""             # e.g. "SEC 8-K 2026-08-15 / IR Calendar"
    status: str = "PENDING"              # "PENDING", "CONFIRMED", "DELAYED", "MISSED", "BREACHED"
    observed_date_str: Optional[str] = None
    direction: str = "N/A"               # "INLINE", "AHEAD", "BEHIND", "N/A"

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "catalyst_id": self.catalyst_id,
            "event_name": self.event_name,
            "expected_date_str": self.expected_date_str,
            "confidence_window_days": self.confidence_window_days,
            "source_locator": self.source_locator,
            "status": self.status,
            "observed_date_str": self.observed_date_str,
        }
        if self.direction and self.direction != "N/A":
            d["direction"] = self.direction
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CatalystEvent":
        return cls(
            catalyst_id=data["catalyst_id"],
            event_name=data["event_name"],
            expected_date_str=data["expected_date_str"],
            confidence_window_days=int(data.get("confidence_window_days", 7)),
            source_locator=data.get("source_locator", ""),
            status=data.get("status", "PENDING"),
            observed_date_str=data.get("observed_date_str"),
            direction=data.get("direction", "N/A"),
        )


def compute_thesis_checksum(
    ticker: str,
    version: int,
    prior_claim_id: Optional[str],
    as_of_utc: str,
    claim_statement: str,
    falsifiers: Tuple[FalsifierRule, ...],
    linked_catalysts: Tuple[CatalystEvent, ...],
    status: str,
    evidence_tier: EvidenceTier,
    author_role: str,
) -> str:
    """Computes a deterministic SHA-256 hash for tamper-proof append-only integrity."""
    payload = {
        "ticker": ticker.strip().upper(),
        "version": version,
        "prior_claim_id": prior_claim_id,
        "as_of_utc": as_of_utc,
        "claim_statement": claim_statement.strip(),
        "falsifiers": [f.to_dict() for f in falsifiers],
        "linked_catalysts": [c.to_dict() for c in linked_catalysts],
        "status": status,
        "evidence_tier": evidence_tier.value,
        "author_role": author_role,
    }
    canonical_json = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ThesisClaimRecord:
    """Versioned, immutable claim record in the institutional research ledger."""
    claim_id: str                        # e.g. "TH-NVDA-001-v1"
    ticker: str
    version: int
    as_of_utc: str                       # ISO-8601 UTC timestamp
    claim_statement: str
    falsifiers: Tuple[FalsifierRule, ...] = ()
    linked_catalysts: Tuple[CatalystEvent, ...] = ()
    status: str = "ACTIVE"               # "ACTIVE", "REVISED", "FALSIFIED", "RETIRED"
    evidence_tier: EvidenceTier = EvidenceTier.DOCUMENTED
    author_role: str = "HUMAN_PM"        # "HUMAN_PM", "AI_AGENT", "CONSENSUS"
    prior_claim_id: Optional[str] = None
    checksum_sha256: str = ""

    def verify_integrity(self) -> bool:
        """Verifies that the record has not been altered since generation."""
        computed = compute_thesis_checksum(
            ticker=self.ticker,
            version=self.version,
            prior_claim_id=self.prior_claim_id,
            as_of_utc=self.as_of_utc,
            claim_statement=self.claim_statement,
            falsifiers=self.falsifiers,
            linked_catalysts=self.linked_catalysts,
            status=self.status,
            evidence_tier=self.evidence_tier,
            author_role=self.author_role,
        )
        return computed == self.checksum_sha256

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "ticker": self.ticker,
            "version": self.version,
            "prior_claim_id": self.prior_claim_id,
            "as_of_utc": self.as_of_utc,
            "claim_statement": self.claim_statement,
            "falsifiers": [f.to_dict() for f in self.falsifiers],
            "linked_catalysts": [c.to_dict() for c in self.linked_catalysts],
            "status": self.status,
            "evidence_tier": self.evidence_tier.value,
            "author_role": self.author_role,
            "checksum_sha256": self.checksum_sha256,
        }

    @classmethod
    def create(
        cls,
        claim_id: str,
        ticker: str,
        version: int,
        as_of_utc: str,
        claim_statement: str,
        falsifiers: Sequence[FalsifierRule] = (),
        linked_catalysts: Sequence[CatalystEvent] = (),
        status: str = "ACTIVE",
        evidence_tier: EvidenceTier = EvidenceTier.DOCUMENTED,
        author_role: str = "HUMAN_PM",
        prior_claim_id: Optional[str] = None,
    ) -> "ThesisClaimRecord":
        f_tuple = tuple(falsifiers)
        c_tuple = tuple(linked_catalysts)
        chk = compute_thesis_checksum(
            ticker=ticker,
            version=version,
            prior_claim_id=prior_claim_id,
            as_of_utc=as_of_utc,
            claim_statement=claim_statement,
            falsifiers=f_tuple,
            linked_catalysts=c_tuple,
            status=status,
            evidence_tier=evidence_tier,
            author_role=author_role,
        )
        return cls(
            claim_id=claim_id,
            ticker=ticker.strip().upper(),
            version=version,
            prior_claim_id=prior_claim_id,
            as_of_utc=as_of_utc,
            claim_statement=claim_statement.strip(),
            falsifiers=f_tuple,
            linked_catalysts=c_tuple,
            status=status,
            evidence_tier=evidence_tier,
            author_role=author_role,
            checksum_sha256=chk,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ThesisClaimRecord":
        falsifiers = tuple(FalsifierRule.from_dict(f) for f in data.get("falsifiers", []))
        catalysts = tuple(CatalystEvent.from_dict(c) for c in data.get("linked_catalysts", []))
        tier = EvidenceTier(data.get("evidence_tier", "DOCUMENTED"))
        return cls(
            claim_id=data["claim_id"],
            ticker=data["ticker"],
            version=int(data["version"]),
            prior_claim_id=data.get("prior_claim_id"),
            as_of_utc=data["as_of_utc"],
            claim_statement=data["claim_statement"],
            falsifiers=falsifiers,
            linked_catalysts=catalysts,
            status=data.get("status", "ACTIVE"),
            evidence_tier=tier,
            author_role=data.get("author_role", "HUMAN_PM"),
            checksum_sha256=data.get("checksum_sha256", ""),
        )


@dataclass(frozen=True)
class ThesisExceptionRow:
    """Result of an automated watchdog scan comparing observed reality to active thesis claims."""
    exception_id: str
    claim_id: str
    ticker: str
    detected_at_utc: str
    exception_type: str                  # "FALSIFIER_BREACHED", "DATE_CHANGED", "STALE", "NEW_SOURCE", "NO_CHANGE"
    metric_name: Optional[str] = None
    expected_value: Optional[str] = None
    observed_value: Optional[str] = None
    source_citation: Optional[str] = None
    delta_impact: Optional[str] = None
    human_action: str = "REVIEW_REQUIRED" # Institutional Hard Rule: AI cannot unilaterally dismiss or approve
    status: str = "PENDING"              # "PENDING", "ACKNOWLEDGED", "REVISED"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "exception_id": self.exception_id,
            "claim_id": self.claim_id,
            "ticker": self.ticker,
            "detected_at_utc": self.detected_at_utc,
            "exception_type": self.exception_type,
            "metric_name": self.metric_name,
            "expected_value": self.expected_value,
            "observed_value": self.observed_value,
            "source_citation": self.source_citation,
            "delta_impact": self.delta_impact,
            "human_action": self.human_action,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ThesisExceptionRow":
        return cls(
            exception_id=data["exception_id"],
            claim_id=data["claim_id"],
            ticker=data["ticker"],
            detected_at_utc=data["detected_at_utc"],
            exception_type=data["exception_type"],
            metric_name=data.get("metric_name"),
            expected_value=data.get("expected_value"),
            observed_value=data.get("observed_value"),
            source_citation=data.get("source_citation"),
            delta_impact=data.get("delta_impact"),
            human_action=data.get("human_action", "REVIEW_REQUIRED"),
            status=data.get("status", "PENDING"),
        )


@dataclass(frozen=True)
class AsOfResearchSnapshot:
    """Reproducible point-in-time container uniting thesis claims, financials, and macro state."""
    snapshot_id: str
    ticker: str
    as_of_utc: str
    active_claims: Tuple[ThesisClaimRecord, ...] = ()
    observed_metrics: Dict[str, str] = field(default_factory=dict)
    known_catalysts: Tuple[CatalystEvent, ...] = ()
    macro_regime: Optional[str] = None
    holding_units: Optional[float] = None
    holding_weight_pct: Optional[float] = None
    checksum_sha256: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "ticker": self.ticker,
            "as_of_utc": self.as_of_utc,
            "active_claims": [c.to_dict() for c in self.active_claims],
            "observed_metrics": self.observed_metrics,
            "known_catalysts": [k.to_dict() for k in self.known_catalysts],
            "macro_regime": self.macro_regime,
            "holding_units": self.holding_units,
            "holding_weight_pct": self.holding_weight_pct,
            "checksum_sha256": self.checksum_sha256,
        }
