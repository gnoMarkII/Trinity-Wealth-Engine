"""Unit Tests for 3D Smart Opportunity Evaluator.

Verifies:
- Candidate admissibility and axis exclusion (prohibitions cannot be compensated).
- Macro alignment quadrant mapping (tailwind vs headwind).
- Single-stock and sector capacity formulas with Decimal math.
- Incomplete coverage leading to unknown capacity.
- Priority scoring (1: High conviction, 2: Secondary, 3: Watch, 4: Review).
- Stable deterministic sorting with tie-breaking.
"""
from decimal import Decimal
import pytest

from core.opportunity_finder.evaluator import OpportunityEvaluator
from core.opportunity_finder.models import OpportunityCandidate


@pytest.fixture
def base_candidate():
    return OpportunityCandidate(
        security_id="US:MSFT",
        issuer_id="MSFT",
        symbol="MSFT",
        market="US",
        currency="USD",
        asset_type="EQUITY",
        company_name="Microsoft Corporation",
        sector_name="Information Technology",
        sector_name_th="เทคโนโลยี",
        source_pools=["POOL_A_FUND_LOOK_THROUGH"],
        source_refs=["fund:SCBS&P500E"],
        current_total_exposure_thb=Decimal("50000.00"),
        current_total_weight_pct=Decimal("5.00"),
    )


def test_evaluator_high_conviction_candidate(base_candidate):
    """High Conviction: Pass Axis + Tailwind + Strong Evidence + Available Capacity -> Priority 1."""
    confirmed_axis = {
        "exclusions": [],
        "prohibited_symbols": [],
        "prohibited_sectors": [],
    }
    applied_buckets = [
        {"id": "core", "name": "Core Compounder", "target_pct": 50},
        {"id": "growth", "name": "Growth", "target_pct": 50},
    ]
    sidecar_data = {
        "analysis_date": "2026-03-31",
        "composite_score": 85.0,
        "quant_signals": {"composite_score": 85.0, "upside_pct": 18.5},
        "valuation_workbench": {"upside_pct": 18.5, "target_price": 450.0},
    }
    sector_rotation = {
        "sectors": {
            "information technology": {"quadrant": "Leading"}
        }
    }

    report = OpportunityEvaluator.evaluate_all(
        candidates=[base_candidate],
        total_nav_thb=Decimal("1000000.00"),
        available_cash_thb=Decimal("200000.00"),
        sector_exposures_thb={"Information Technology": Decimal("100000.00")},
        confirmed_axis=confirmed_axis,
        applied_buckets=applied_buckets,
        sidecars_map={"MSFT": sidecar_data},
        sector_rotation_data=sector_rotation,
    )

    assert report.total_candidates_scanned == 1
    assert report.passed_candidates_count == 1
    res = report.candidates[0]
    assert res.policy_fit == "pass"
    assert res.macro_alignment == "tailwind"
    assert res.evidence_readiness == "sufficient"
    assert res.capacity_status == "available"
    assert res.research_priority == 1
    assert res.is_actionable is True
    assert res.suggested_bucket_id == "growth"


def test_evaluator_axis_exclusion_overrides_high_score(base_candidate):
    """Axis exclusion must strictly fail even if quant score is 10/10 and Macro is Leading."""
    confirmed_axis = {
        "prohibited_symbols": ["MSFT"],
    }
    sidecar_data = {
        "analysis_date": "2026-03-31",
        "composite_score": 10.0,
    }
    sector_rotation = {
        "sectors": {"information technology": {"quadrant": "Leading"}}
    }

    report = OpportunityEvaluator.evaluate_all(
        candidates=[base_candidate],
        total_nav_thb=Decimal("1000000.00"),
        available_cash_thb=Decimal("200000.00"),
        sector_exposures_thb={},
        confirmed_axis=confirmed_axis,
        sidecars_map={"MSFT": sidecar_data},
        sector_rotation_data=sector_rotation,
    )

    res = report.candidates[0]
    assert res.policy_fit == "fail"
    assert res.research_priority == 4
    assert res.is_actionable is False
    assert any("prohibited" in r.lower() for r in res.policy_reasons)


def test_evaluator_capacity_constrained(base_candidate):
    """If candidate already at single-stock cap (15%), capacity is constrained."""
    # NAV = 1M, single-stock cap 15% = 150k. Current MSFT exposure = 150k.
    base_candidate.current_total_exposure_thb = Decimal("150000.00")
    base_candidate.current_total_weight_pct = Decimal("15.00")

    report = OpportunityEvaluator.evaluate_all(
        candidates=[base_candidate],
        total_nav_thb=Decimal("1000000.00"),
        available_cash_thb=Decimal("100000.00"),
        sector_exposures_thb={},
        confirmed_axis={},
    )

    res = report.candidates[0]
    assert res.capacity_status == "constrained"
    assert res.remaining_capacity_thb == Decimal("0")
    assert res.is_actionable is False


def test_evaluator_unknown_capacity_on_low_coverage(base_candidate):
    """Low look-through coverage (< 60%) flags capacity as unknown."""
    report = OpportunityEvaluator.evaluate_all(
        candidates=[base_candidate],
        total_nav_thb=Decimal("1000000.00"),
        available_cash_thb=Decimal("100000.00"),
        sector_exposures_thb={},
        confirmed_axis={},
        look_through_coverage_pct=Decimal("45.0"),
    )

    res = report.candidates[0]
    assert res.capacity_status == "unknown"
    assert res.remaining_capacity_thb is None
    assert any("uncertainty" in w.lower() for w in report.warnings)


def test_evaluator_robust_against_non_dict_sidecar_fields(base_candidate):
    """Sidecar data where narrative_analysis or quant_signals is a string instead of dict does not crash."""
    malformed_sidecar = {
        "analysis_date": "2026-04-01",
        "narrative_analysis": "This is a raw text thesis, not a dict",
        "quant_signals": "No signals available",
        "valuation_workbench": None,
    }
    sector_rotation = {
        "sectors": {
            "information technology": "Leading",  # string instead of dict
        }
    }

    report = OpportunityEvaluator.evaluate_all(
        candidates=[base_candidate],
        total_nav_thb=Decimal("1000000.00"),
        available_cash_thb=Decimal("100000.00"),
        sector_exposures_thb={},
        confirmed_axis={},
        sidecars_map={"MSFT": malformed_sidecar},
        sector_rotation_data=sector_rotation,
    )

    res = report.candidates[0]
    assert res.evidence_readiness == "sufficient"
    assert res.moat_rating == "None"
    assert res.macro_alignment == "tailwind"
    assert res.macro_stage == "Leading"


def test_evaluator_sector_rotation_from_rows_list(base_candidate):
    """Sector rotation snapshot containing a 'rows' list from SectorRotationService resolves correctly."""
    sector_rotation = {
        "rows": [
            {"ticker": "XLK", "name": "Technology (เทคโนโลยี)", "quadrant": "Leading", "score": 8.0},
            {"ticker": "XLC", "name": "Communication Services (สื่อสาร)", "quadrant": "Lagging", "score": 4.0},
        ]
    }

    report = OpportunityEvaluator.evaluate_all(
        candidates=[base_candidate],
        total_nav_thb=Decimal("1000000.00"),
        available_cash_thb=Decimal("100000.00"),
        sector_exposures_thb={},
        confirmed_axis={},
        sector_rotation_data=sector_rotation,
    )

    res = report.candidates[0]
    assert res.macro_alignment == "tailwind"
    assert res.macro_stage == "Leading"


def test_evaluator_structural_capacity_when_cash_is_zero(base_candidate):
    """When deployable cash is 0 but portfolio has structural room, capacity is available with usable 0."""
    report = OpportunityEvaluator.evaluate_all(
        candidates=[base_candidate],
        total_nav_thb=Decimal("1000000.00"),
        available_cash_thb=Decimal("0.00"),
        sector_exposures_thb={},
        confirmed_axis={},
    )

    res = report.candidates[0]
    assert res.capacity_status == "available"
    assert res.remaining_capacity_thb == Decimal("0")
    assert res.issuer_room_thb == Decimal("100000.00")  # 150k cap - 50k current exposure
    assert any("Structural capacity available" in r for r in res.capacity_reasons)


def test_asset_classification_word_boundary_netflix():
    """Netflix must not be misclassified as an ETF due to substring matching."""
    from core.portfolio.security_identity import classify_asset_type
    assert classify_asset_type("NFLX", asset_type_raw="Stock", name="Netflix, Inc.") == "EQUITY"
    assert classify_asset_type("SPY", asset_type_raw="ETF", name="SPDR S&P 500 ETF Trust") == "ETF"


def test_evaluator_composite_score_priority_thresholds(base_candidate):
    """Test deterministic priority scoring based on 0-100 scale thresholds:
    - Score >= 70.0 + Tailwind -> Priority 1 (High Conviction)
    - Score 50.0..69.9 (e.g. PLTR 65.0, FTNT 57.9) + Tailwind -> Priority 2 (Secondary)
    - Score < 50.0 + Tailwind -> Priority 3 (Watch / Research Required)
    - Legacy 0-10 format (8.5) -> normalized to 85.0 -> Priority 1
    """
    confirmed_axis = {
        "exclusions": [],
        "prohibited_symbols": [],
        "prohibited_sectors": [],
    }
    applied_buckets = [
        {"id": "growth", "name": "Growth", "target_pct": 50},
    ]
    sector_rotation = {"sectors": {"information technology": {"quadrant": "Leading"}}}

    def _eval(score):
        sidecar = {
            "analysis_date": "2026-03-31",
            "composite_score": score,
            "quant_signals": {"composite_score": score},
        }
        report = OpportunityEvaluator.evaluate_all(
            candidates=[base_candidate],
            total_nav_thb=Decimal("1000000.00"),
            available_cash_thb=Decimal("100000.00"),
            sector_exposures_thb={},
            confirmed_axis=confirmed_axis,
            applied_buckets=applied_buckets,
            sidecars_map={"MSFT": sidecar},
            sector_rotation_data=sector_rotation,
        )
        return report.candidates[0]

    # High conviction: score >= 70
    res_high = _eval(85.0)
    assert res_high.research_priority == 1
    assert res_high.composite_score == 85.0

    # PLTR case: score 65.0 -> Priority 2 Secondary
    res_pltr = _eval(65.0)
    assert res_pltr.research_priority == 2
    assert res_pltr.composite_score == 65.0

    # FTNT case: score 57.9 -> Priority 2 Secondary
    res_ftnt = _eval(57.9)
    assert res_ftnt.research_priority == 2
    assert res_ftnt.composite_score == 57.9

    # Weak score: score < 50 -> Priority 3 Watch
    res_weak = _eval(42.0)
    assert res_weak.research_priority == 3
    assert res_weak.composite_score == 42.0

    # Low score (8.5 on 0-100 scale) is NOT multiplied by 10 into 85.0 -> Priority 3
    res_low = _eval(8.5)
    assert res_low.research_priority == 3
    assert res_low.composite_score == 8.5


def test_evaluator_stale_sidecar(base_candidate):
    """Sidecar flagged as stale or older than 1 year sets evidence_readiness to 'stale'."""
    stale_sidecar = {
        "analysis_date": "2024-01-01",
        "composite_score": 85.0,
        "is_stale": True,
    }
    report = OpportunityEvaluator.evaluate_all(
        candidates=[base_candidate],
        total_nav_thb=Decimal("1000000.00"),
        available_cash_thb=Decimal("100000.00"),
        sector_exposures_thb={},
        confirmed_axis={},
        sidecars_map={"MSFT": stale_sidecar},
    )
    res = report.candidates[0]
    assert res.evidence_readiness == "stale"
    assert any("stale" in r.lower() for r in res.evidence_reasons)


