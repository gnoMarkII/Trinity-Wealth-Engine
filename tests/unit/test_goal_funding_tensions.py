"""Unit tests for Goal-Based Wealth Planning Engine: Three Faces of Risk, R_req, Levers & Tension.

Verifies deterministic mathematical calculations (Bisection solver, 4 Levers, tension classification)
using pure Python Decimals according to Trinity Wealth Engine Hexagonal Architecture.
"""
from decimal import Decimal
import pytest

from core.investor_essence.funding import (
    calculate_required_return,
    calculate_goal_adjustment_levers,
    evaluate_goal_tension,
)
from core.investor_essence.models import GoalTensionLevel


def test_calculate_required_return_lump_sum_only():
    # PV: 100,000, FV: 200,000 in 120 months (10 years) with PMT = 0
    # Expected CAGR: (200000/100000)^(1/10) - 1 = 2^(0.1) - 1 = 7.177% -> 7.18%
    r = calculate_required_return(
        pv=Decimal("100000.00"),
        fv=Decimal("200000.00"),
        pmt=Decimal("0.00"),
        months=120,
    )
    assert r is not None
    assert r == Decimal("7.18")


def test_calculate_required_return_savings_sufficient():
    # PV: 100,000, PMT: 10,000, Months: 50 -> total savings capacity: 600,000 >= FV 500,000
    r = calculate_required_return(
        pv=Decimal("100000.00"),
        fv=Decimal("500000.00"),
        pmt=Decimal("10000.00"),
        months=50,
    )
    assert r == Decimal("0.00")


def test_calculate_required_return_annuity_growth():
    # PV: 0, PMT: 10,000, Months: 60 (5 years), FV: 1,000,000
    # Needs substantial return (>20% CAGR) to turn 600,000 savings into 1,000,000
    r = calculate_required_return(
        pv=Decimal("0.00"),
        fv=Decimal("1000000.00"),
        pmt=Decimal("10000.00"),
        months=60,
    )
    assert r is not None
    assert Decimal("20.00") < r < Decimal("23.00")
    assert r == Decimal("21.20")



def test_calculate_required_return_edge_cases():
    # Target already met
    assert calculate_required_return(Decimal("500.00"), Decimal("400.00"), months=12) == Decimal("0.00")
    # Invalid months
    assert calculate_required_return(Decimal("100.00"), Decimal("500.00"), months=0) is None
    assert calculate_required_return(Decimal("100.00"), Decimal("500.00"), months=None) is None
    # No capital and no savings
    assert calculate_required_return(Decimal("0.00"), Decimal("500.00"), pmt=Decimal("0.00"), months=12) is None


def test_calculate_goal_adjustment_levers():
    # PV: 100,000, FV: 1,000,000, PMT: 5,000, Months: 60 (5 years), Expected Return: 7%
    levers = calculate_goal_adjustment_levers(
        pv=Decimal("100000.00"),
        fv=Decimal("1000000.00"),
        pmt=Decimal("5000.00"),
        months=60,
        expected_annual_return=Decimal("7.00"),
        required_cagr=Decimal("26.50"),
    )
    # Check Lever 1 (Save more monthly): PMT needed must be substantially higher than 5,000
    assert levers.save_more_monthly_thb > Decimal("5000.00")
    assert levers.total_monthly_savings_needed_thb > Decimal("10000.00")

    # Check Lever 2 (Extend timeline): Total months needed at 7% and 5k/mo must be > 60
    assert levers.extend_timeline_months > 0
    assert levers.total_months_needed > 60

    # Check Lever 3 (Downsized target): What can be achieved in 60 months at 7% with 5k/mo
    assert levers.downsized_target_thb < Decimal("1000000.00")
    assert levers.downsized_target_thb > Decimal("100000.00")

    # Check Lever 4 (Required Risk)
    assert levers.required_risk_cagr_pct == Decimal("26.50")


def test_evaluate_goal_tension_levels():
    # 1. Insufficient data (no deadline)
    res_no_deadline = evaluate_goal_tension(
        current_amount=Decimal("100000.00"),
        target_amount=Decimal("1000000.00"),
        months_remaining=None,
    )
    assert res_no_deadline.tension_level == GoalTensionLevel.INSUFFICIENT_DATA

    # 2. Balanced (Needs goal, fully funded or low return demand)
    res_balanced = evaluate_goal_tension(
        current_amount=Decimal("400000.00"),
        target_amount=Decimal("500000.00"),
        monthly_savings=Decimal("5000.00"),
        months_remaining=24,  # 400k + 24*5k = 520k > 500k -> 0% return required
        priority_tier="needs",
    )
    assert res_balanced.tension_level == GoalTensionLevel.BALANCED
    assert res_balanced.required_return_cagr_pct == Decimal("0.00")

    # 3. Critical tension (Needs goal requiring > 8% return)
    # PV: 10,000, FV: 500,000 in 24 months with 2,000/mo -> requires extreme return
    res_crit = evaluate_goal_tension(
        current_amount=Decimal("10000.00"),
        target_amount=Decimal("500000.00"),
        monthly_savings=Decimal("2000.00"),
        months_remaining=24,
        priority_tier="needs",
    )
    assert res_crit.tension_level == GoalTensionLevel.CRITICAL_TENSION
    assert res_crit.levers.save_more_monthly_thb > Decimal("0.00")
    assert res_crit.levers.extend_timeline_months > 0

    # 4. Moderate tension (Wants goal requiring ~10% return)
    # PV: 100,000, FV: 250,000 in 60 months with PMT = 1,000
    res_mod = evaluate_goal_tension(
        current_amount=Decimal("100000.00"),
        target_amount=Decimal("250000.00"),
        monthly_savings=Decimal("1000.00"),
        months_remaining=60,
        expected_annual_return=Decimal("7.00"),
        priority_tier="wants",
    )
    assert res_mod.tension_level in (GoalTensionLevel.MODERATE_TENSION, GoalTensionLevel.CRITICAL_TENSION)

    # 5. Serialization dictionary
    d = res_mod.to_dict()
    assert "tension_level" in d
    assert "levers" in d
    assert isinstance(d["levers"], dict)


def test_calculate_required_return_long_horizon_no_overflow():
    # Long horizon: 40 years (480 months) with high FV growth
    r = calculate_required_return(
        pv=Decimal("1000.00"),
        fv=Decimal("1000000000.00"),
        pmt=Decimal("10.00"),
        months=480,
    )
    assert r is not None
    assert r > Decimal("30.00")

