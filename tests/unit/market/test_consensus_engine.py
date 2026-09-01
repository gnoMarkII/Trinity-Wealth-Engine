"""Unit tests for Best-Effort Consensus Contract & Expectation Gap Engine (Phase 2 & v3.1)."""
import pytest
from tools.market.consensus_engine import compute_horizon_matched_expectation_gap


def test_consensus_eps_momentum_calculation():
    mock_consensus = {
        "trend_0y": {
            "current_estimate": 2.50,
            "estimate_30d_ago": 2.38,
            "change_30d_pct": 5.04,
        }
    }
    
    gap, momentum_score, status, flags = compute_horizon_matched_expectation_gap(
        consensus_data=mock_consensus,
        market_implied_growth_pct=15.0,
    )

    assert momentum_score == 100.0  # +5% chg caps at 100.0
    assert status == "partial"
    assert "unmatched_consensus_horizon:expectation_gap_unavailable" in flags


def test_horizon_matched_expectation_gap_matched():
    mock_consensus = {
        "revenue_growth_next_year_pct": 12.0,
        "trend_0y": {
            "current_estimate": 3.00,
            "estimate_30d_ago": 3.00,
            "change_30d_pct": 0.0,
        }
    }

    # Market implies 18% growth, consensus is 12% -> expectation gap is +6%
    gap, momentum_score, status, flags = compute_horizon_matched_expectation_gap(
        consensus_data=mock_consensus,
        market_implied_growth_pct=18.0,
    )

    assert gap == 6.0
    assert momentum_score == 50.0  # 0% chg is baseline 50.0
    assert status == "available"
    assert len(flags) == 0


def test_sparse_coverage_policy():
    gap, momentum_score, status, flags = compute_horizon_matched_expectation_gap(
        consensus_data=None,
        market_implied_growth_pct=10.0,
    )

    assert gap is None
    assert momentum_score is None
    assert status == "unavailable"
    assert "missing_consensus_data" in flags
