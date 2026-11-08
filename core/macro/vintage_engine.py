"""Macro Point-in-Time (PIT) Vintage Engine & Subsequent Revision Walk.

Architecture Rule (Layer 1 - Core):
Pure domain models and deterministic math. Strictly standard library only per Hexagonal
Architecture AST Domain Layer Isolation rules. Zero external dependencies.

References:
Claude Code for Hedge Funds:
- Chapter 6: Reconstruct a Macro Vintage Without Look-Ahead (pp. 43–47)
- Appendix C: Calculation Record - Macro (p. 79)
- Appendix H: Macro As-Of Row Schema (p. 117, 129)
- Laboratory 6: Reconstruct a Macro Vintage (pp. 94, 105–106)

Core Invariants:
1. eligible rows = rows where vintage_date <= cutoff_date
2. as-of value = value from max(eligible vintage_date)
3. latest value = value from max(all vintage_date)
4. revision after cutoff = latest_value - asof_value
5. If no row has vintage_date <= cutoff_date: status is UNAVAILABLE_AS_OF_CUTOFF (never forward-fill).
6. Observation date must never be confused with vintage date.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple


class LookAheadContaminationError(ValueError):
    """Raised when a selected vintage date exceeds the strict temporal cutoff date."""

    def __init__(self, vintage_date: str, cutoff_date: str, series_id: str = ""):
        self.vintage_date = vintage_date
        self.cutoff_date = cutoff_date
        self.series_id = series_id
        super().__init__(
            f"Look-ahead bias detected in {series_id or 'macro series'}: selected vintage "
            f"'{vintage_date}' is strictly after cutoff '{cutoff_date}'."
        )


class ObservationConfusionError(ValueError):
    """Raised when observation date is mistakenly used in place of vintage date."""
    pass


class MissingHistoricalCoverageError(ValueError):
    """Raised when no vintage observation exists on or before the cutoff date."""
    pass


@dataclass(frozen=True)
class MacroVintageObservation:
    """An individual vintage disclosure observation for an economic series."""
    series_id: str                      # e.g., "SYN_PAYROLL", "PAYEMS", "CPIAUCSL"
    observation_date: str               # e.g., "2026-04-01" (the economic period described)
    vintage_date: str                   # e.g., "2026-05-08", "2026-06-05", "2026-07-03" (publication release date)
    value: Decimal                      # Numeric disclosed value
    unit: str = "thousands"             # Explicit measurement unit carried from manifest
    seasonal_adjustment: str = "SA"     # "SA" or "NSA"
    release_source: str = "BLS"         # e.g., "BLS", "BEA", "FRED"
    evidence_label: str = "SYNTHETIC"   # Institutional evidence provenance label


@dataclass(frozen=True)
class MacroAsOfRow:
    """Deterministic As-Of vintage selection with subsequent revision walk (Chapter 6 & App H)."""
    series_id: str
    observation_date: str
    cutoff_date: str
    selected_vintage_date: Optional[str]
    asof_value: Optional[Decimal]
    latest_vintage_date: Optional[str]
    latest_value: Optional[Decimal]
    revision_after_cutoff: Optional[Decimal]
    unit: str
    seasonal_adjustment: str
    status: str = "REVIEW_REQUIRED"       # "REVIEW_REQUIRED" or "UNAVAILABLE_AS_OF_CUTOFF"
    evidence_label: str = "SYNTHETIC"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "series_id": self.series_id,
            "observation_date": self.observation_date,
            "cutoff_date": self.cutoff_date,
            "selected_vintage_date": self.selected_vintage_date,
            "asof_value": float(self.asof_value) if self.asof_value is not None else None,
            "latest_vintage_date": self.latest_vintage_date,
            "latest_value": float(self.latest_value) if self.latest_value is not None else None,
            "revision_after_cutoff": float(self.revision_after_cutoff) if self.revision_after_cutoff is not None else None,
            "unit": self.unit,
            "seasonal_adjustment": self.seasonal_adjustment,
            "status": self.status,
            "evidence_label": self.evidence_label,
        }


def select_macro_as_of_vintage(
    observations: List[MacroVintageObservation],
    cutoff_date: str,
    fail_on_missing: bool = False,
) -> MacroAsOfRow:
    """Selects the point-in-time vintage observation for a single series and observation date.

    Chapter 6 Invariants:
    1. Filter: eligible rows where r.vintage_date <= cutoff_date.
    2. As-Of value: value of row with max(eligible rows.vintage_date).
    3. Latest value: value of row with max(all rows.vintage_date).
    4. Revision: latest_value - asof_value.
    5. If no row has vintage_date <= cutoff_date:
       - If fail_on_missing is True: raises MissingHistoricalCoverageError.
       - Else: returns MacroAsOfRow with status="UNAVAILABLE_AS_OF_CUTOFF", asof_value=None.
    """
    if not observations:
        raise ValueError("Cannot select vintage from empty observation set.")

    # Guard: verify all observations share identical series_id and observation_date
    first = observations[0]
    series_id = first.series_id
    obs_date = first.observation_date
    unit = first.unit
    seasonal_adj = first.seasonal_adjustment
    label = first.evidence_label

    for obs in observations[1:]:
        if obs.series_id != series_id or obs.observation_date != obs_date:
            raise ValueError(
                f"Mixed series or observation dates in single vintage group: "
                f"expected ({series_id}, {obs_date}), got ({obs.series_id}, {obs.observation_date})"
            )

    # 1. Eligible rows on or before cutoff date
    eligible_rows = [r for r in observations if r.vintage_date <= cutoff_date]

    # Latest row across ALL available vintages
    latest_row = max(observations, key=lambda r: r.vintage_date)

    if not eligible_rows:
        if fail_on_missing:
            raise MissingHistoricalCoverageError(
                f"No vintage row exists for series '{series_id}' on or before cutoff '{cutoff_date}'."
            )
        return MacroAsOfRow(
            series_id=series_id,
            observation_date=obs_date,
            cutoff_date=cutoff_date,
            selected_vintage_date=None,
            asof_value=None,
            latest_vintage_date=latest_row.vintage_date,
            latest_value=latest_row.value,
            revision_after_cutoff=None,
            unit=unit,
            seasonal_adjustment=seasonal_adj,
            status="UNAVAILABLE_AS_OF_CUTOFF",
            evidence_label=label,
        )

    # 2. Selected row: maximum eligible vintage
    selected_row = max(eligible_rows, key=lambda r: r.vintage_date)

    # Safety Invariant Assertion: selected vintage date MUST NOT exceed cutoff date
    if selected_row.vintage_date > cutoff_date:
        raise LookAheadContaminationError(
            vintage_date=selected_row.vintage_date,
            cutoff_date=cutoff_date,
            series_id=series_id,
        )

    # 3. Subsequent revision walk
    revision = latest_row.value - selected_row.value

    return MacroAsOfRow(
        series_id=series_id,
        observation_date=obs_date,
        cutoff_date=cutoff_date,
        selected_vintage_date=selected_row.vintage_date,
        asof_value=selected_row.value,
        latest_vintage_date=latest_row.vintage_date,
        latest_value=latest_row.value,
        revision_after_cutoff=revision,
        unit=unit,
        seasonal_adjustment=seasonal_adj,
        status="REVIEW_REQUIRED",
        evidence_label=label,
    )


def build_macro_vintage_walk(
    observations: List[MacroVintageObservation],
    cutoff_date: str,
) -> List[MacroAsOfRow]:
    """Builds a complete macro vintage walk across all series and periods.

    Groups observations by (series_id, observation_date) and computes the exact
    point-in-time selection and revision walk for each period.
    """
    groups: Dict[Tuple[str, str], List[MacroVintageObservation]] = {}
    for obs in observations:
        key = (obs.series_id, obs.observation_date)
        if key not in groups:
            groups[key] = []
        groups[key].append(obs)

    results: List[MacroAsOfRow] = []
    # Sort groups deterministically by series_id, then observation_date
    for key in sorted(groups.keys()):
        obs_list = groups[key]
        row = select_macro_as_of_vintage(obs_list, cutoff_date=cutoff_date)
        results.append(row)

    return results

