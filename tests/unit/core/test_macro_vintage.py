"""Unit tests for Macro Point-in-Time Vintage Engine & Revision Walk (Chapter 6).

Verifies the exact synthetic fixtures, invariants, and failure diagnoses from
'Claude Code for Hedge Funds' Chapter 6 (pp. 43–47) and Laboratory 6 (pp. 105–106):
- April Payroll: 150 -> 148 -> 151; As-of June 10 selects 148, revision +3.
- May Payroll: 120 -> 126; As-of June 10 selects 120, revision +6.
- Missing historical coverage returns UNAVAILABLE_AS_OF_CUTOFF (never forward-fills).
- Invariant tests against Look-ahead bias and observation/vintage confusion.
"""
from decimal import Decimal
import pytest

from core.macro.vintage_engine import (
    LookAheadContaminationError,
    MacroAsOfRow,
    MacroVintageObservation,
    MissingHistoricalCoverageError,
    build_macro_vintage_walk,
    select_macro_as_of_vintage,
)


@pytest.fixture
def chapter6_synthetic_observations():
    """Exact synthetic observations from Chapter 6, page 56 & Laboratory 6, page 106."""
    return [
        # April 2026 Payroll (3 vintages)
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
        # May 2026 Payroll (2 vintages)
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
    ]


def test_chapter6_reproduction_cutoff_june_10(chapter6_synthetic_observations):
    """Verifies Chapter 6 reproduction table at cutoff 2026-06-10:

    Obs. Date    As-of Vintage   As-of Value   Latest Vintage   Latest Value   Revision
    2026-04-01   2026-06-05      148           2026-07-03       151            +3
    2026-05-01   2026-06-05      120           2026-07-03       126            +6
    """
    cutoff = "2026-06-10"
    walk = build_macro_vintage_walk(chapter6_synthetic_observations, cutoff_date=cutoff)

    assert len(walk) == 2

    row_april = walk[0]
    assert row_april.series_id == "SYN_PAYROLL"
    assert row_april.observation_date == "2026-04-01"
    assert row_april.cutoff_date == cutoff
    assert row_april.selected_vintage_date == "2026-06-05"
    assert row_april.asof_value == Decimal("148")
    assert row_april.latest_vintage_date == "2026-07-03"
    assert row_april.latest_value == Decimal("151")
    assert row_april.revision_after_cutoff == Decimal("3")
    assert row_april.status == "REVIEW_REQUIRED"
    assert row_april.unit == "thousands"

    row_may = walk[1]
    assert row_may.series_id == "SYN_PAYROLL"
    assert row_may.observation_date == "2026-05-01"
    assert row_may.cutoff_date == cutoff
    assert row_may.selected_vintage_date == "2026-06-05"
    assert row_may.asof_value == Decimal("120")
    assert row_may.latest_vintage_date == "2026-07-03"
    assert row_may.latest_value == Decimal("126")
    assert row_may.revision_after_cutoff == Decimal("6")
    assert row_may.status == "REVIEW_REQUIRED"


def test_early_cutoff_may_20(chapter6_synthetic_observations):
    """At cutoff 2026-05-20:
    - April has only 2026-05-08 eligible (value 150).
    - May has NO eligible vintages (first released 2026-06-05) -> UNAVAILABLE_AS_OF_CUTOFF.
    """
    cutoff = "2026-05-20"
    walk = build_macro_vintage_walk(chapter6_synthetic_observations, cutoff_date=cutoff)

    assert len(walk) == 2

    # April observation
    row_april = walk[0]
    assert row_april.selected_vintage_date == "2026-05-08"
    assert row_april.asof_value == Decimal("150")
    assert row_april.latest_value == Decimal("151")
    assert row_april.revision_after_cutoff == Decimal("1")
    assert row_april.status == "REVIEW_REQUIRED"

    # May observation: not yet released as of May 20
    row_may = walk[1]
    assert row_may.selected_vintage_date is None
    assert row_may.asof_value is None
    assert row_may.latest_value == Decimal("126")
    assert row_may.revision_after_cutoff is None
    assert row_may.status == "UNAVAILABLE_AS_OF_CUTOFF"


def test_fail_on_missing_coverage_guard(chapter6_synthetic_observations):
    """Guards against silent imputation when fail_on_missing=True."""
    may_obs = [o for o in chapter6_synthetic_observations if o.observation_date == "2026-05-01"]
    with pytest.raises(MissingHistoricalCoverageError):
        select_macro_as_of_vintage(may_obs, cutoff_date="2026-05-20", fail_on_missing=True)


def test_observation_date_vs_vintage_date_invariance():
    """Diagnoses failure: Observation date confused with vintage date.

    An observation for 2026-04-01 cannot be claimed available on 2026-04-15
    if its first publication vintage date was 2026-05-08.
    """
    obs = [
        MacroVintageObservation(
            series_id="SYN_PAYROLL",
            observation_date="2026-04-01",
            vintage_date="2026-05-08",
            value=Decimal("150"),
        )
    ]
    # Cutoff is after observation date (2026-04-15 > 2026-04-01)
    # but BEFORE vintage release date (2026-04-15 < 2026-05-08)
    row = select_macro_as_of_vintage(obs, cutoff_date="2026-04-15")
    assert row.status == "UNAVAILABLE_AS_OF_CUTOFF"
    assert row.asof_value is None


def test_to_dict_serialization(chapter6_synthetic_observations):
    """Ensures serializability for REST DTO conversion."""
    walk = build_macro_vintage_walk(chapter6_synthetic_observations, cutoff_date="2026-06-10")
    d = walk[0].to_dict()
    assert d["series_id"] == "SYN_PAYROLL"
    assert d["asof_value"] == 148.0
    assert d["latest_value"] == 151.0
    assert d["revision_after_cutoff"] == 3.0
    assert d["status"] == "REVIEW_REQUIRED"

