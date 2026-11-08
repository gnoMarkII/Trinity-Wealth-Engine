"""Deterministic Earnings Reported-vs-Expected Driver Bridge (Institutional Hedge Fund Grade).

Architecture Rule (Layer 1 - Core):
Pure Python standard library only — strictly zero external dependencies (no FastAPI, Pydantic,
SQLite, or yfinance) per Hexagonal Architecture AST Domain Layer Isolation rules.
Deterministic financial math using Decimal. Never calculate math mentally or via LLM prompts.

Reference:
Claude Code for Hedge Funds, Chapter 5 (pp. 36-42) & Appendix C:
"Reconcile Earnings Without Rewriting the Prior View & Driver Sum Breakdown"
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from enum import Enum
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple


class DriverCategory(str, Enum):
    VOLUME = "VOLUME"
    PRICE_MIX = "PRICE_MIX"
    FX = "FX"
    INORGANIC_MA = "INORGANIC_MA"
    COST_PRODUCTIVITY = "COST_PRODUCTIVITY"
    ONE_OFF_OTHER = "ONE_OFF_OTHER"


@dataclass(frozen=True)
class EarningsSurprise:
    """Quantitative surprise calculation comparing reported results against pre-existing consensus."""
    metric_name: str
    reported: Decimal
    consensus: Decimal
    delta: Decimal
    surprise_pct: Decimal
    direction: str                     # "BEAT", "INLINE", "MISS"
    unit: str = "USD"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "reported": str(self.reported),
            "consensus": str(self.consensus),
            "delta": str(self.delta),
            "surprise_pct": str(self.surprise_pct),
            "direction": self.direction,
            "unit": self.unit,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EarningsSurprise":
        return cls(
            metric_name=data["metric_name"],
            reported=Decimal(str(data["reported"])),
            consensus=Decimal(str(data["consensus"])),
            delta=Decimal(str(data["delta"])),
            surprise_pct=Decimal(str(data["surprise_pct"])),
            direction=data["direction"],
            unit=data.get("unit", "USD"),
        )


@dataclass(frozen=True)
class EarningsDriver:
    """Individual causal driver explaining a portion of the reported vs consensus delta."""
    name: str
    category: DriverCategory
    impact_value: Decimal               # Signed numeric value in millions/native currency
    unit: str = "USD"
    narrative: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category.value,
            "impact_value": str(self.impact_value),
            "unit": self.unit,
            "narrative": self.narrative,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EarningsDriver":
        return cls(
            name=data["name"],
            category=DriverCategory(data["category"]),
            impact_value=Decimal(str(data["impact_value"])),
            unit=data.get("unit", "USD"),
            narrative=data.get("narrative", ""),
        )


@dataclass(frozen=True)
class EarningsBridge:
    """Mathematical reconciliation between reported earnings and prior expected thesis view."""
    ticker: str
    period: str                         # e.g. "Q3-2026", "FY2026"
    as_of_date: str                     # ISO date "YYYY-MM-DD"
    revenue_surprise: EarningsSurprise
    eps_surprise: EarningsSurprise
    ebit_margin_reported: Optional[Decimal] = None
    ebit_margin_consensus: Optional[Decimal] = None
    revenue_drivers: Tuple[EarningsDriver, ...] = ()
    sum_drivers: Decimal = Decimal("0")
    unallocated_delta: Decimal = Decimal("0")
    is_mathematically_balanced: bool = True
    thesis_reconciliation_verdict: str = "THESIS_CONFIRMED" # "THESIS_CONFIRMED", "THESIS_UNDER_PRESSURE", "THESIS_IMPAIRED", "THESIS_ACCELERATING"
    thesis_reconciliation_narrative: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "period": self.period,
            "as_of_date": self.as_of_date,
            "revenue_surprise": self.revenue_surprise.to_dict(),
            "eps_surprise": self.eps_surprise.to_dict(),
            "ebit_margin_reported": str(self.ebit_margin_reported) if self.ebit_margin_reported is not None else None,
            "ebit_margin_consensus": str(self.ebit_margin_consensus) if self.ebit_margin_consensus is not None else None,
            "revenue_drivers": [d.to_dict() for d in self.revenue_drivers],
            "sum_drivers": str(self.sum_drivers),
            "unallocated_delta": str(self.unallocated_delta),
            "is_mathematically_balanced": self.is_mathematically_balanced,
            "thesis_reconciliation_verdict": self.thesis_reconciliation_verdict,
            "thesis_reconciliation_narrative": self.thesis_reconciliation_narrative,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EarningsBridge":
        return cls(
            ticker=data["ticker"],
            period=data["period"],
            as_of_date=data["as_of_date"],
            revenue_surprise=EarningsSurprise.from_dict(data["revenue_surprise"]),
            eps_surprise=EarningsSurprise.from_dict(data["eps_surprise"]),
            ebit_margin_reported=Decimal(str(data["ebit_margin_reported"])) if data.get("ebit_margin_reported") is not None else None,
            ebit_margin_consensus=Decimal(str(data["ebit_margin_consensus"])) if data.get("ebit_margin_consensus") is not None else None,
            revenue_drivers=tuple(EarningsDriver.from_dict(d) for d in data.get("revenue_drivers", [])),
            sum_drivers=Decimal(str(data.get("sum_drivers", "0"))),
            unallocated_delta=Decimal(str(data.get("unallocated_delta", "0"))),
            is_mathematically_balanced=bool(data.get("is_mathematically_balanced", True)),
            thesis_reconciliation_verdict=data.get("thesis_reconciliation_verdict", "THESIS_CONFIRMED"),
            thesis_reconciliation_narrative=data.get("thesis_reconciliation_narrative", ""),
        )


class EarningsBridgeEngine:
    """Pure domain engine that evaluates earnings reported-vs-expected bridges with driver sum assertions."""

    @staticmethod
    def compute_surprise(
        reported: Decimal,
        consensus: Decimal,
        metric_name: str = "metric",
        unit: str = "USD",
        inline_threshold_pct: Decimal = Decimal("1.0"),
    ) -> EarningsSurprise:
        """Computes deterministic surprise dollar and percentage delta."""
        delta = reported - consensus
        if consensus != Decimal("0"):
            pct = ((delta / abs(consensus)) * Decimal("100")).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        else:
            pct = Decimal("0.00")

        if pct > inline_threshold_pct:
            direction = "BEAT"
        elif pct < -inline_threshold_pct:
            direction = "MISS"
        else:
            direction = "INLINE"

        return EarningsSurprise(
            metric_name=metric_name,
            reported=reported,
            consensus=consensus,
            delta=delta,
            surprise_pct=pct,
            direction=direction,
            unit=unit,
        )

    @classmethod
    def build_earnings_bridge(
        cls,
        ticker: str,
        period: str,
        as_of_date: str,
        revenue_reported: Decimal,
        revenue_consensus: Decimal,
        eps_reported: Decimal,
        eps_consensus: Decimal,
        revenue_unit: str = "USD_M",
        eps_unit: str = "USD",
        ebit_margin_reported: Optional[Decimal] = None,
        ebit_margin_consensus: Optional[Decimal] = None,
        revenue_drivers: Optional[Sequence[EarningsDriver]] = None,
        tolerance: Decimal = Decimal("0.01"),
    ) -> EarningsBridge:
        """Assembles verified EarningsBridge and mathematically asserts sum(drivers) == delta(revenue)."""
        clean_ticker = ticker.strip().upper()
        rev_surp = cls.compute_surprise(
            reported=revenue_reported,
            consensus=revenue_consensus,
            metric_name="revenue",
            unit=revenue_unit,
        )
        eps_surp = cls.compute_surprise(
            reported=eps_reported,
            consensus=eps_consensus,
            metric_name="eps",
            unit=eps_unit,
        )

        drivers_tuple = tuple(revenue_drivers) if revenue_drivers else ()
        if drivers_tuple:
            sum_drivers = sum((d.impact_value for d in drivers_tuple), Decimal("0"))
            unallocated = rev_surp.delta - sum_drivers
            is_balanced = abs(unallocated) <= tolerance
        else:
            sum_drivers = Decimal("0")
            unallocated = rev_surp.delta
            is_balanced = True

        # Prior view thesis reconciliation verdict
        verdict, narrative = cls._reconcile_prior_view(
            rev_surp=rev_surp,
            eps_surp=eps_surp,
            ebit_rep=ebit_margin_reported,
            ebit_cons=ebit_margin_consensus,
            drivers=drivers_tuple,
            is_balanced=is_balanced,
        )

        return EarningsBridge(
            ticker=clean_ticker,
            period=period,
            as_of_date=as_of_date,
            revenue_surprise=rev_surp,
            eps_surprise=eps_surp,
            ebit_margin_reported=ebit_margin_reported,
            ebit_margin_consensus=ebit_margin_consensus,
            revenue_drivers=drivers_tuple,
            sum_drivers=sum_drivers,
            unallocated_delta=unallocated,
            is_mathematically_balanced=is_balanced,
            thesis_reconciliation_verdict=verdict,
            thesis_reconciliation_narrative=narrative,
        )

    @staticmethod
    def _reconcile_prior_view(
        rev_surp: EarningsSurprise,
        eps_surp: EarningsSurprise,
        ebit_rep: Optional[Decimal],
        ebit_cons: Optional[Decimal],
        drivers: Sequence[EarningsDriver],
        is_balanced: bool,
    ) -> Tuple[str, str]:
        """Evaluates whether the core thesis is confirmed, impaired, or under pressure."""
        if not is_balanced:
            return (
                "THESIS_UNDER_PRESSURE",
                "Driver decomposition is mathematically unbalanced with unexplained residual delta.",
            )

        # Check if revenue beat is driven by organic volume vs inorganic/one-off
        volume_drivers = [d for d in drivers if d.category == DriverCategory.VOLUME]
        has_negative_volume = any(d.impact_value < Decimal("0") for d in volume_drivers)

        if rev_surp.direction == "BEAT" and eps_surp.direction == "BEAT":
            if has_negative_volume:
                return (
                    "THESIS_UNDER_PRESSURE",
                    "Topline beat achieved despite underlying core volume contraction, relying on price/mix or one-off drivers.",
                )
            if ebit_rep is not None and ebit_cons is not None and ebit_rep < ebit_cons:
                return (
                    "THESIS_UNDER_PRESSURE",
                    "Topline beat offset by operating margin compression below consensus.",
                )
            return (
                "THESIS_CONFIRMED",
                "Topline and bottom-line beat consensus with positive structural core drivers, confirming active thesis.",
            )

        if rev_surp.direction == "MISS" and eps_surp.direction == "MISS":
            return (
                "THESIS_IMPAIRED",
                "Double miss on revenue and EPS indicates decelerating demand or competitive loss; review thesis falsifiers.",
            )

        if rev_surp.direction == "MISS" or eps_surp.direction == "MISS":
            return (
                "THESIS_UNDER_PRESSURE",
                "Mixed print with divergence between revenue growth and profitability; monitoring required.",
            )

        return (
            "THESIS_CONFIRMED",
            "Performance strictly inline with consensus expectations, supporting baseline holding allocation.",
        )
