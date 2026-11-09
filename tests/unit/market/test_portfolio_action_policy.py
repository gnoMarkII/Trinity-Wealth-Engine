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


def test_trading_service_pretrade_action_policy_thb_and_usd(isolated_portfolio):
    from tools.portfolio.bootstrap import (
        build_default_portfolio_dependencies,
        build_legacy_portfolio_service,
    )

    pt = isolated_portfolio
    pt._record_income_locked("Other", 1_000_000.0, None)
    svc = build_legacy_portfolio_service(build_default_portfolio_dependencies())
    trading_svc = svc._trading_service

    trading_svc.update_fx_rate(rate=35.0)
    policy_th = PortfolioPolicy(
        policy_id="pol_th",
        portfolio_id="default",
        target_weight_pct=20.0,
        max_weight_pct=25.0,
        minimum_cash_buffer_pct=10.0,
    )
    sig_th = QuantSignals(
        ticker="PTT.BK",
        market="TH",
        evaluated_at="2026-08-28T12:00:00Z",
        peer_sector="Energy",
        adtv_local_currency=50_000_000.0,
    )
    sc = DeterministicScorecard(action_stance="ACCUMULATE_NOW")
    verdict_th = trading_svc.evaluate_pretrade_action_policy(
        ticker="PTT.BK",
        market="TH",
        scorecard=sc,
        quant_signals=sig_th,
        policy=policy_th,
        current_price=35.0,
    )
    assert verdict_th.action == "ACCUMULATE_FULL"
    assert verdict_th.currency == "THB"
    assert verdict_th.proposed_order_shares == int(200_000.0 / 35.0)

    # US stock against the same THB-based portfolio converts THB NAV/cash with USDTHB=35.0
    sig_us = QuantSignals(
        ticker="AAPL",
        market="US",
        evaluated_at="2026-08-28T12:00:00Z",
        peer_sector="Technology",
        adtv_local_currency=10_000_000.0,
    )
    verdict_us = trading_svc.evaluate_pretrade_action_policy(
        ticker="AAPL",
        market="US",
        scorecard=sc,
        quant_signals=sig_us,
        policy=policy_th,
        current_price=200.0,
    )
    assert verdict_us.currency == "USD"
    # 1,000,000 THB / 35 = 28,571.43 USD -> 20% target ~ 5,714.29 USD -> 28 shares @ 200
    assert verdict_us.proposed_order_shares == 28


def test_advisory_service_evaluate_holding_action_policy():
    from application.portfolio.advisory_service import PortfolioAdvisoryService
    from tools.portfolio.domain.models import Holding, PortfolioState, Summary

    class _FakePortfolioReader:
        def get_current_portfolio_state(self, refresh_prices: bool = False, portfolio_id: str = "default"):
            return PortfolioState(
                last_updated="2026-08-28T12:00:00",
                base_currency="THB",
                fx_rates={"USDTHB": 35.0},
                summary=Summary(total_value_thb=700_000.0),
                holdings=[
                    Holding(symbol="CASH_THB", asset_type="Cash", units=350_000.0, market_value_thb=350_000.0),
                    Holding(symbol="PTT.BK", asset_type="Stock", units=10000.0, market_value_thb=350_000.0, current_price_thb=35.0),
                ],
            )

    class _Dummy:
        def __getattr__(self, item):
            return lambda *args, **kwargs: {}

    class _SidecarReader:
        def get_all_equity_sidecars(self):
            return {"PTT.BK": {"quant_signals": {"peer_sector": "Energy"}}}

        def get_watchlist_sidecars(self):
            return {}

    adv = PortfolioAdvisoryService(
        portfolio_reader=_FakePortfolioReader(),
        macro_reader=_Dummy(),
        sidecar_reader=_SidecarReader(),
        vault_port=_Dummy(),
        synthesizer_port=_Dummy(),
    )
    policy = PortfolioPolicy(
        policy_id="pol_adv",
        portfolio_id="default",
        target_weight_pct=60.0,
        max_weight_pct=70.0,
        max_sector_exposure_pct=60.0,
        minimum_cash_buffer_pct=10.0,
    )
    signals = QuantSignals(
        ticker="PTT.BK",
        market="TH",
        evaluated_at="2026-08-28T12:00:00Z",
        peer_sector="Energy",
        raw_analysis_price=35.0,
    )
    sc = DeterministicScorecard(action_stance="ACCUMULATE_NOW")

    verdict = adv.evaluate_holding_action_policy(
        ticker="PTT",
        scorecard=sc,
        quant_signals=signals,
        policy=policy,
        market="TH",
    )
    assert verdict.current_weight_pct == pytest.approx(50.0, abs=0.05)
    assert verdict.action == "ACCUMULATE_FULL"
    assert verdict.proposed_order_shares == int(70_000.0 / 35.0)
