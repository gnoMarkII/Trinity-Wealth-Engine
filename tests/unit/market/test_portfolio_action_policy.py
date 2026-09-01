"""Unit tests for Scoped Portfolio Policy & Multi-Constraint Action Engine (Phase 2 & v3.1 Hardened)."""
import pytest
from schemas.micro_quant_schemas import DeterministicScorecard, QuantSignals, ReverseDCFResult
from schemas.portfolio_policy_schemas import PortfolioPolicy
from tools.market.portfolio_action_engine import compute_portfolio_action


def test_portfolio_action_no_portfolio_connected():
    signals = QuantSignals(ticker="AAPL", market="US", evaluated_at="2026-08-28T12:00:00Z")
    scorecard = DeterministicScorecard(action_stance="ACCUMULATE_NOW")

    verdict = compute_portfolio_action(
        ticker="AAPL",
        market="US",
        scorecard=scorecard,
        quant_signals=signals,
        policy=None,
        portfolio_state=None,
    )

    assert verdict.action == "NO_PORTFOLIO_CONNECTED"
    assert verdict.security_stance == "ACCUMULATE_NOW"


def test_portfolio_action_adtv_liquidity_cap():
    policy = PortfolioPolicy(
        policy_id="pol_growth_01",
        portfolio_id="port_main",
        target_weight_pct=10.0,
        max_weight_pct=15.0,
        max_adtv_participation_rate=0.02,  # 2% ADTV limit
    )
    portfolio_state = {
        "total_nav": 1000000.0,  # $1M NAV -> 10% target = $100k
        "cash": 300000.0,
        "positions": {},
        "sector_allocations": {},
    }
    # 20D ADTV = $1,000,000 -> 2% limit = $20,000 maximum order
    signals = QuantSignals(
        ticker="TSLA",
        market="US",
        evaluated_at="2026-08-28T12:00:00Z",
        adtv_local_currency=1000000.0,
    )
    scorecard = DeterministicScorecard(action_stance="ACCUMULATE_NOW")

    verdict = compute_portfolio_action(
        ticker="TSLA",
        market="US",
        scorecard=scorecard,
        quant_signals=signals,
        policy=policy,
        portfolio_state=portfolio_state,
        current_price=200.0,
    )

    assert verdict.action == "ACCUMULATE_SCALED"
    assert verdict.proposed_order_notional <= 20000.0
    assert any("adtv_liquidity_cap" in c for c in verdict.binding_constraints)


def test_portfolio_action_bank_sector_exclusion_not_applicable():
    policy = PortfolioPolicy(
        policy_id="pol_value_01",
        portfolio_id="port_main",
        target_weight_pct=5.0,
        max_weight_pct=10.0,
    )
    portfolio_state = {"total_nav": 500000.0, "cash": 50000.0, "positions": {}}
    
    signals = QuantSignals(
        ticker="JPM",
        market="US",
        evaluated_at="2026-08-28T12:00:00Z",
        reverse_dcf_result=ReverseDCFResult(
            is_eligible=False,
            status="not_applicable",
            exclusion_reason="Financial Services Sector",
        ),
    )
    scorecard = DeterministicScorecard(action_stance="HOLD_WAIT")

    verdict = compute_portfolio_action(
        ticker="JPM",
        market="US",
        scorecard=scorecard,
        quant_signals=signals,
        policy=policy,
        portfolio_state=portfolio_state,
        current_price=200.0,
    )

    assert verdict.action == "NOT_APPLICABLE"


def test_portfolio_action_missing_price_fail_closed():
    policy = PortfolioPolicy(
        policy_id="pol_growth_01",
        portfolio_id="port_main",
        target_weight_pct=10.0,
        max_weight_pct=15.0,
    )
    portfolio_state = {"total_nav": 1000000.0, "cash": 300000.0, "positions": {}}
    signals = QuantSignals(ticker="NVDA", market="US", evaluated_at="2026-08-28T12:00:00Z")
    scorecard = DeterministicScorecard(action_stance="ACCUMULATE_NOW")

    # Missing price (None or <= 0)
    verdict = compute_portfolio_action(
        ticker="NVDA",
        market="US",
        scorecard=scorecard,
        quant_signals=signals,
        policy=policy,
        portfolio_state=portfolio_state,
        current_price=None,
    )

    assert verdict.action == "RESTRICTED_MISSING_PRICE"
    assert verdict.proposed_order_shares == 0
    assert verdict.status == "unavailable"


def test_portfolio_action_scope_mismatch_rejected():
    policy = PortfolioPolicy(
        policy_id="pol_asset_specific",
        portfolio_id="port_main",
        scope="asset",
        target_ticker="MSFT",
        target_weight_pct=8.0,
        max_weight_pct=10.0,
    )
    portfolio_state = {"total_nav": 1000000.0, "cash": 300000.0, "positions": {}}
    signals = QuantSignals(ticker="GOOGL", market="US", evaluated_at="2026-08-28T12:00:00Z")
    scorecard = DeterministicScorecard(action_stance="ACCUMULATE_NOW")

    # Evaluate GOOGL against a policy scoped strictly to MSFT
    verdict = compute_portfolio_action(
        ticker="GOOGL",
        market="US",
        scorecard=scorecard,
        quant_signals=signals,
        policy=policy,
        portfolio_state=portfolio_state,
        current_price=175.0,
    )

    assert verdict.action == "RESTRICTED_SCOPE_MISMATCH"
    assert verdict.proposed_order_shares == 0


def test_portfolio_action_neutral_stance_buy_gate():
    policy = PortfolioPolicy(
        policy_id="pol_growth_01",
        portfolio_id="port_main",
        target_weight_pct=10.0,
        max_weight_pct=15.0,
    )
    portfolio_state = {"total_nav": 1000000.0, "cash": 300000.0, "positions": {}}
    signals = QuantSignals(ticker="AAPL", market="US", evaluated_at="2026-08-28T12:00:00Z")
    scorecard = DeterministicScorecard(action_stance="HOLD_WAIT")

    # HOLD_WAIT stance must strictly prevent any new buy orders
    verdict = compute_portfolio_action(
        ticker="AAPL",
        market="US",
        scorecard=scorecard,
        quant_signals=signals,
        policy=policy,
        portfolio_state=portfolio_state,
        current_price=220.0,
    )

    assert verdict.action == "HOLD_UNCHANGED"
    assert verdict.proposed_order_shares == 0
    assert verdict.proposed_order_notional == 0.0
    assert any(cr.constraint_name == "security_stance_buy_gate" and not cr.passed for cr in verdict.constraint_results)
