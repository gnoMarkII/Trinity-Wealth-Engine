"""Application service for Macro Point-in-Time Vintage Analytics (Chapter 6 & App H).

Coordinates the retrieval and deterministic execution of the point-in-time
macro vintage selector without look-ahead contamination.
"""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, List, Optional

from core.macro.vintage_engine import (
    MacroAsOfRow,
    MacroVintageObservation,
    build_macro_vintage_walk,
    select_macro_as_of_vintage,
)


def get_default_institutional_vintage_fixture() -> List[MacroVintageObservation]:
    """Canonical synthetic macro vintage fixtures from Chapter 6 & Appendix H."""
    return [
        # Nonfarm Payrolls (SYN_PAYROLL) - April 2026 period (Chapter 6 Benchmark)
        MacroVintageObservation(
            series_id="SYN_PAYROLL",
            observation_date="2026-04-01",
            vintage_date="2026-05-08",
            value=Decimal("150"),
            unit="thousands",
            seasonal_adjustment="SA",
            release_source="BLS",
        ),
        MacroVintageObservation(
            series_id="SYN_PAYROLL",
            observation_date="2026-04-01",
            vintage_date="2026-06-05",
            value=Decimal("148"),
            unit="thousands",
            seasonal_adjustment="SA",
            release_source="BLS",
        ),
        MacroVintageObservation(
            series_id="SYN_PAYROLL",
            observation_date="2026-04-01",
            vintage_date="2026-07-03",
            value=Decimal("151"),
            unit="thousands",
            seasonal_adjustment="SA",
            release_source="BLS",
        ),
        # Nonfarm Payrolls (SYN_PAYROLL) - May 2026 period (Chapter 6 Benchmark)
        MacroVintageObservation(
            series_id="SYN_PAYROLL",
            observation_date="2026-05-01",
            vintage_date="2026-06-05",
            value=Decimal("120"),
            unit="thousands",
            seasonal_adjustment="SA",
            release_source="BLS",
        ),
        MacroVintageObservation(
            series_id="SYN_PAYROLL",
            observation_date="2026-05-01",
            vintage_date="2026-07-03",
            value=Decimal("126"),
            unit="thousands",
            seasonal_adjustment="SA",
            release_source="BLS",
        ),
        # Consumer Price Index MoM (SYN_CPI) - April 2026 period
        MacroVintageObservation(
            series_id="SYN_CPI",
            observation_date="2026-04-01",
            vintage_date="2026-05-13",
            value=Decimal("0.3"),
            unit="percent",
            seasonal_adjustment="SA",
            release_source="BLS",
        ),
        MacroVintageObservation(
            series_id="SYN_CPI",
            observation_date="2026-04-01",
            vintage_date="2026-06-12",
            value=Decimal("0.2"),
            unit="percent",
            seasonal_adjustment="SA",
            release_source="BLS",
        ),
        # US Real GDP QoQ (SYN_GDP) - 2026 Q1 period
        MacroVintageObservation(
            series_id="SYN_GDP",
            observation_date="2026-01-01",
            vintage_date="2026-04-30",  # Advance estimate
            value=Decimal("1.6"),
            unit="percent_annualized",
            seasonal_adjustment="SAAR",
            release_source="BEA",
        ),
        MacroVintageObservation(
            series_id="SYN_GDP",
            observation_date="2026-01-01",
            vintage_date="2026-05-28",  # Second estimate
            value=Decimal("1.4"),
            unit="percent_annualized",
            seasonal_adjustment="SAAR",
            release_source="BEA",
        ),
        MacroVintageObservation(
            series_id="SYN_GDP",
            observation_date="2026-01-01",
            vintage_date="2026-06-25",  # Third/Final estimate
            value=Decimal("1.5"),
            unit="percent_annualized",
            seasonal_adjustment="SAAR",
            release_source="BEA",
        ),
    ]


class MacroVintageApplicationService:
    """Manages macro vintage manifests and builds point-in-time revision walks."""

    def __init__(self, initial_observations: Optional[List[MacroVintageObservation]] = None):
        self._observations: List[MacroVintageObservation] = (
            list(initial_observations)
            if initial_observations is not None
            else get_default_institutional_vintage_fixture()
        )

    def add_observation(self, obs: MacroVintageObservation) -> None:
        """Appends a validated macro vintage observation to the memory store."""
        self._observations.append(obs)

    def get_vintage_walk(
        self,
        cutoff_date: Optional[str] = None,
        series_ids: Optional[List[str]] = None,
    ) -> List[MacroAsOfRow]:
        """Calculates point-in-time as-of values and subsequent revisions for matching series."""
        target_cutoff = cutoff_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")

        candidates = self._observations
        if series_ids:
            s_set = {s.strip().upper() for s in series_ids}
            candidates = [c for c in candidates if c.series_id.upper() in s_set]

        return build_macro_vintage_walk(candidates, cutoff_date=target_cutoff)

    def get_series_history(self, series_id: str) -> List[MacroVintageObservation]:
        """Returns all historical vintage disclosures for a given series ID."""
        clean_id = series_id.strip().upper()
        return [o for o in self._observations if o.series_id.upper() == clean_id]

