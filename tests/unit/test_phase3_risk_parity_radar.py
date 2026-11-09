from decimal import Decimal
import pytest

from core.portfolio_advisory.risk_parity import (
    compute_bucket_active_share_pct,
    compute_risk_contributions_and_parity,
    RiskContributionSummary,
)
from core.portfolio_advisory.models import (
    PortfolioRiskRadar,
    HoldingAdvisoryItem,
    AdvisoryActionCode,
    AssetEvaluationTrack,
)
from core.portfolio_advisory.engine import evaluate_portfolio_advisory


def test_active_share_exact_cases():
    # 1. Identical weights -> 0% active share
    act_identical = {"growth": Decimal("60.0"), "dividend": Decimal("40.0")}
    tgt_identical = {"growth": Decimal("60.0"), "dividend": Decimal("40.0")}
    assert compute_bucket_active_share_pct(act_identical, tgt_identical) == Decimal("0.00")

    # 2. Completely disjoint -> 100% active share
    act_disjoint = {"growth": Decimal("100.0")}
    tgt_disjoint = {"cash": Decimal("100.0")}
    # 0.5 * (|100 - 0| + |0 - 100|) = 100.00%
    assert compute_bucket_active_share_pct(act_disjoint, tgt_disjoint) == Decimal("100.00")

    # 3. Partial drift
    act_drift = {"growth": Decimal("70.0"), "dividend": Decimal("30.0")}
    tgt_drift = {"growth": Decimal("50.0"), "dividend": Decimal("50.0")}
    # 0.5 * (|70 - 50| + |30 - 50|) = 0.5 * (20 + 20) = 20.00%
    assert compute_bucket_active_share_pct(act_drift, tgt_drift) == Decimal("20.00")


def test_risk_contributions_and_parity_closure():
    weights = {
        "PTT": Decimal("50.0"),
        "AAPL": Decimal("50.0"),
    }
    # AAPL has double volatility of PTT
    vols = {
        "PTT": Decimal("15.0"),
        "AAPL": Decimal("30.0"),
    }
    summary = compute_risk_contributions_and_parity(weights, vols)
    assert isinstance(summary, RiskContributionSummary)

    # Risk contribution closure invariant: sum must be exactly 100.00%
    total_rc = sum(summary.risk_contributions_pct.values())
    assert total_rc == Decimal("100.00")

    # Higher vol asset AAPL must contribute more risk than PTT
    assert summary.risk_contributions_pct["AAPL"] > summary.risk_contributions_pct["PTT"]

    # Inverse vol weight: PTT (15%) should get ~66.67% and AAPL (30%) should get ~33.33%
    total_iv = sum(summary.inverse_vol_weights_pct.values())
    assert total_iv == Decimal("100.00")
    assert summary.inverse_vol_weights_pct["PTT"] > summary.inverse_vol_weights_pct["AAPL"]


def test_risk_contributions_equal_assets():
    weights = {
        "STOCK_A": Decimal("50.0"),
        "STOCK_B": Decimal("50.0"),
    }
    vols = {
        "STOCK_A": Decimal("20.0"),
        "STOCK_B": Decimal("20.0"),
    }
    summary = compute_risk_contributions_and_parity(weights, vols)
    assert summary.risk_contributions_pct["STOCK_A"] == Decimal("50.00")
    assert summary.risk_contributions_pct["STOCK_B"] == Decimal("50.00")
    assert summary.inverse_vol_weights_pct["STOCK_A"] == Decimal("50.00")
    assert summary.inverse_vol_weights_pct["STOCK_B"] == Decimal("50.00")


def test_advisory_engine_radar_integration():
    """Verify evaluate_portfolio_advisory calculates active share, risk contribution, and friction"""
    class MockHolding:
        def __init__(self, symbol, asset_type, units, price, mkt_val, bucket_id, market="TH", currency="THB"):
            self.symbol = symbol
            self.asset_type = asset_type
            self.units = Decimal(str(units))
            self.price = Decimal(str(price))
            self.market_value_thb = Decimal(str(mkt_val))
            self.bucket_id = bucket_id
            self.market = market
            self.currency = currency
            self.status = "active"
            self.name = symbol

    class MockSummary:
        total_value_thb = Decimal("1000000.00")
        usd_thb_rate = Decimal("35.50")

    class MockBucket:
        def __init__(self, bucket_id, name, target_percent):
            self.bucket_id = bucket_id
            self.name = name
            self.target_percent = Decimal(str(target_percent))

    class MockState:
        holdings = [
            MockHolding("PTT", "equity", 10000, 50, 500000, "growth", market="TH", currency="THB"),
            MockHolding("AAPL", "equity", 100, 140, 500000, "growth", market="US", currency="USD"),
        ]
        summary = MockSummary()
        buckets = [
            MockBucket("growth", "Growth Bucket", Decimal("100.0")),
        ]

    report = evaluate_portfolio_advisory(
        portfolio_state=MockState(),
    )

    radar = report.risk_radar
    assert radar.bucket_active_share_pct is not None
    assert radar.bucket_active_share_pct == Decimal("0.00")  # Growth is 100% actual, 100% target
    assert "PTT" in radar.risk_contributions_pct
    assert "AAPL" in radar.risk_contributions_pct
    assert sum(radar.risk_contributions_pct.values()) == Decimal("100.00")
    assert "PTT" in radar.inverse_vol_weights_pct
    assert "AAPL" in radar.inverse_vol_weights_pct

    # Holdings advisory items have risk contribution populated
    for h in report.holdings_advisory:
        assert h.risk_contribution_pct is not None
        # Without sidecar volatility, engine must explicitly mark 20% fallback as estimated
        assert h.volatility_source == "estimated_default"
        assert h.is_volatility_estimated is True
        assert h.volatility_pct is None
        assert h.symbol in radar.volatility_estimated_symbols


def test_r09_zero_and_negative_correlation_symmetry_and_permutation_invariance():
    """R09: Explicit zero/negative correlation, key orientation, and input permutation invariance."""
    w = {"A": Decimal("50"), "B": Decimal("50")}
    vols = {"A": Decimal("20"), "B": Decimal("20")}

    # Zero correlation must NOT be treated as missing (0.30 fallback)
    # Var = 0.5^2 * 20^2 + 0.5^2 * 20^2 + 0 = 100 + 100 = 200.0000
    s1 = compute_risk_contributions_and_parity(
        weights=w,
        volatilities=vols,
        pairwise_correlations={("A", "B"): Decimal("0")},
        shrinkage_factor=Decimal("0"),
    )
    s2 = compute_risk_contributions_and_parity(
        weights={"B": Decimal("50"), "A": Decimal("50")},
        volatilities={"B": Decimal("20"), "A": Decimal("20")},
        pairwise_correlations={("B", "A"): Decimal("0")},
        shrinkage_factor=Decimal("0"),
    )
    assert s1.total_portfolio_variance == Decimal("200.0000")
    assert s2.total_portfolio_variance == Decimal("200.0000")
    assert s1.risk_contributions_pct == s2.risk_contributions_pct == {
        "A": Decimal("50.00"),
        "B": Decimal("50.00"),
    }

    # Asymmetric conflict must raise ValueError
    with pytest.raises(ValueError, match="Asymmetric correlation"):
        compute_risk_contributions_and_parity(
            weights=w,
            volatilities=vols,
            pairwise_correlations={("A", "B"): Decimal("0"), ("B", "A"): Decimal("0.5")},
        )

    # Negative correlation & permutation invariance across 3 assets
    w3 = {"X": Decimal("40"), "Y": Decimal("35"), "Z": Decimal("25")}
    v3 = {"X": Decimal("18"), "Y": Decimal("27"), "Z": Decimal("12")}
    corrs3 = {("X", "Y"): Decimal("-0.4"), ("Z", "X"): Decimal("0.0"), ("Y", "Z"): Decimal("0.2")}
    res_a = compute_risk_contributions_and_parity(w3, v3, corrs3, shrinkage_factor=Decimal("0.0"))
    res_b = compute_risk_contributions_and_parity(
        {"Z": Decimal("25"), "X": Decimal("40"), "Y": Decimal("35")},
        {"Z": Decimal("12"), "Y": Decimal("27"), "X": Decimal("18")},
        {("Y", "X"): Decimal("-0.4"), ("X", "Z"): Decimal("0.0"), ("Z", "Y"): Decimal("0.2")},
        shrinkage_factor=Decimal("0.0"),
    )
    assert res_a.total_portfolio_variance == res_b.total_portfolio_variance
    assert res_a.risk_contributions_pct == res_b.risk_contributions_pct
    assert res_a.inverse_vol_weights_pct == res_b.inverse_vol_weights_pct
    assert sum(res_a.risk_contributions_pct.values()) == Decimal("100.00")


def test_r09_zero_variance_hedge_unavailable_and_non_psd_rejected():
    """R09: Valid zero-variance hedge returns UNAVAILABLE_ZERO_VARIANCE (no 33.33=99.99 fallback); non-PSD raises."""
    # 3 equal assets with pairwise correlation -0.5 -> corr matrix eigenvalues [1.5, 1.5, 0], variance == 0!
    w3 = {"A": Decimal("1"), "B": Decimal("1"), "C": Decimal("1")}
    v3 = {"A": Decimal("20"), "B": Decimal("20"), "C": Decimal("20")}
    corrs_hedge = {
        ("A", "B"): Decimal("-0.5"),
        ("A", "C"): Decimal("-0.5"),
        ("B", "C"): Decimal("-0.5"),
    }
    res = compute_risk_contributions_and_parity(
        w3, v3, pairwise_correlations=corrs_hedge, shrinkage_factor=Decimal("0.0")
    )
    assert res.risk_contribution_status == "UNAVAILABLE_ZERO_VARIANCE"
    assert res.risk_contributions_pct == {}
    assert res.total_portfolio_variance == Decimal("0.0000")
    assert res.total_portfolio_volatility_pct == Decimal("0.00")
    assert sum(res.inverse_vol_weights_pct.values()) == Decimal("100.00")

    # Indefinite / non-PSD correlation matrix (e.g., all pairs -0.8 for 3 assets -> min eigenvalue = 1 - 1.6 = -0.6)
    bad_corrs = {
        ("A", "B"): Decimal("-0.8"),
        ("A", "C"): Decimal("-0.8"),
        ("B", "C"): Decimal("-0.8"),
    }
    with pytest.raises(ValueError, match="not positive semidefinite"):
        compute_risk_contributions_and_parity(
            w3, v3, pairwise_correlations=bad_corrs, shrinkage_factor=Decimal("0.0")
        )


def test_r01_nested_quant_signals_dict_and_model_risk_contributions_and_friction():
    """R01: PTT/SCB 10%/40% nested signals yield 11.34/88.66 risk contrib & positive market impact; rejects wrong market."""
    from schemas.micro_quant_schemas import (
        EquitySentimentContext,
        MicroQuantOutput,
        QuantSignals,
    )

    class H:
        def __init__(self, symbol, mkt_val, price, market="TH", currency="THB", action_hint=False):
            self.symbol = symbol
            self.asset_type = "Stock"
            self.units = Decimal(str(mkt_val / price))
            self.market_value_thb = Decimal(str(mkt_val))
            self.current_price_thb = Decimal(str(price)) if currency == "THB" else None
            self.current_price_usd = Decimal(str(price)) if currency == "USD" else None
            self.avg_cost_thb = Decimal(str(price)) if currency == "THB" else None
            self.avg_cost_usd = Decimal(str(price)) if currency == "USD" else None
            self.bucket_id = "core"
            self.market = market
            self.currency = currency
            self.status = "active"
            self.name = symbol

    class B:
        def __init__(self, bid, pct):
            self.bucket_id = bid
            self.name = bid
            self.target_percent = pct

    class S:
        holdings = [
            H("PTT.BK", 500_000, 50.0, market="TH", currency="THB"),
            H("SCB.BK", 500_000, 100.0, market="TH", currency="THB"),
        ]
        summary = type("Sum", (), {"usd_thb_rate": Decimal("35.5"), "cash_balance_thb": Decimal("0")})()
        buckets = [B("core", Decimal("100.0"))]

    # 1. Dict sidecar (with PTT normalized alias + ACCUMULATE_NOW lot sizing on SCB)
    dict_sidecars = {
        "PTT": {
            "ticker": "PTT.BK",
            "market": "TH",
            "quant_signals": {
                "raw_analysis_price": 50.0,
                "volatility_pct": 10.0,
                "adtv_local_currency": 10_000_000.0,
                "deterministic_scorecard": {"core_conviction_score": 8.0, "action_stance": "HOLD"},
            },
        },
        "SCB.BK": {
            "ticker": "SCB.BK",
            "market": "TH",
            "quant_signals": {
                "raw_analysis_price": 100.0,
                "volatility_pct": 40.0,
                "adtv_local_currency": 1_000_000.0,  # smaller ADTV -> noticeable market impact
                "deterministic_scorecard": {"core_conviction_score": 8.5, "action_stance": "ACCUMULATE_NOW"},
                "tactical_setup": {"buy_zone_max": 110.0, "buy_zone_min": 95.0},
            },
        },
    }
    rep_dict = evaluate_portfolio_advisory(portfolio_state=S(), equity_sidecars=dict_sidecars)
    assert rep_dict.risk_radar.risk_contributions_pct["PTT.BK"] == Decimal("11.34")
    assert rep_dict.risk_radar.risk_contributions_pct["SCB.BK"] == Decimal("88.66")
    assert rep_dict.risk_radar.risk_contribution_status == "AVAILABLE"
    assert rep_dict.risk_radar.total_portfolio_volatility_pct is not None
    assert rep_dict.risk_radar.volatility_estimated_symbols == []

    scb_item = next(h for h in rep_dict.holdings_advisory if h.symbol == "SCB.BK")
    assert scb_item.volatility_pct == Decimal("40.00")
    assert scb_item.volatility_source == "observed"
    assert scb_item.is_volatility_estimated is False
    # Explicit TH commission alone at order <= 29,761 is >16.8 bps (min 50 THB), with 40% vol & 1M ADTV impact adds significantly
    assert scb_item.estimated_friction_bps is not None
    assert scb_item.estimated_friction_bps > Decimal("25.0")

    # 2. Pydantic model sidecars give identical results
    sentiment = EquitySentimentContext(
        evaluated_at="2026-11-01T00:00:00Z",
        market_sentiment="neutral",
        sources_summary="test",
    )
    model_sidecars = {
        "PTT": MicroQuantOutput(
            ticker="PTT.BK",
            market="TH",
            analysis_date="2026-11-01",
            quant_signals=QuantSignals(
                ticker="PTT.BK",
                market="TH",
                raw_analysis_price=50.0,
                volatility_pct=10.0,
                adtv_local_currency=10_000_000.0,
                evaluated_at="2026-11-01T00:00:00Z",
            ),
            sentiment_context=sentiment,
            narrative_analysis="test",
            base_case_summary="test",
        ),
        "SCB": MicroQuantOutput(
            ticker="SCB.BK",
            market="TH",
            analysis_date="2026-11-01",
            quant_signals=QuantSignals(
                ticker="SCB.BK",
                market="TH",
                raw_analysis_price=100.0,
                volatility_pct=40.0,
                adtv_local_currency=1_000_000.0,
                evaluated_at="2026-11-01T00:00:00Z",
            ),
            sentiment_context=sentiment,
            narrative_analysis="test",
            base_case_summary="test",
        ),
    }
    rep_model = evaluate_portfolio_advisory(portfolio_state=S(), equity_sidecars=model_sidecars)
    assert rep_model.risk_radar.risk_contributions_pct["PTT.BK"] == Decimal("11.34")
    assert rep_model.risk_radar.risk_contributions_pct["SCB.BK"] == Decimal("88.66")

    # 3. Market mismatch sidecar (US ticker paired under alias key to TH holding) must be ignored
    mismatch_sidecars = {
        "PTT": {
            "ticker": "PTT",
            "market": "US",
            "quant_signals": {"volatility_pct": 90.0, "adtv_local_currency": 100.0},
        },
        "SCB.BK": dict_sidecars["SCB.BK"],
    }
    rep_mismatch = evaluate_portfolio_advisory(portfolio_state=S(), equity_sidecars=mismatch_sidecars)
    ptt_mm = next(h for h in rep_mismatch.holdings_advisory if h.symbol == "PTT.BK")
    assert ptt_mm.volatility_source == "estimated_default"
    assert ptt_mm.is_volatility_estimated is True
    assert ptt_mm.volatility_pct is None
    assert "PTT.BK" in rep_mismatch.risk_radar.volatility_estimated_symbols


def test_r14_bucket_active_share_includes_unassigned_unconfigured_unbucketed_and_handles_zero_nav():
    """R14: Unassigned holdings, unconfigured buckets, unbucketed cash, and zero NAV."""
    class H:
        def __init__(self, sym, mkt_val, bid, asset_type="Stock"):
            self.symbol = sym
            self.asset_type = asset_type
            self.units = Decimal("100")
            self.current_price_thb = Decimal(str(mkt_val / 100)) if mkt_val > 0 else Decimal("1")
            self.market_value_thb = Decimal(str(mkt_val))
            self.bucket_id = bid
            self.status = "active"

    class B:
        def __init__(self, bid, pct):
            self.bucket_id = bid
            self.name = bid
            self.target_percent = pct

    # Case A: 100% unassigned holding against 100% "core" target -> 100.00% active share
    state_unassigned = type("S", (), {
        "holdings": [H("A", 100_000, None)],
        "buckets": [B("core", Decimal("100"))],
        "summary": type("Sum", (), {"usd_thb_rate": Decimal("35.5"), "cash_balance_thb": Decimal("0")})(),
    })()
    rep_unassigned = evaluate_portfolio_advisory(state_unassigned)
    assert rep_unassigned.risk_radar.bucket_active_share_status == "AVAILABLE"
    assert rep_unassigned.risk_radar.bucket_active_share_pct == Decimal("100.00")

    # Case B: 60k in "core", 40k unbucketed summary cash ("cash" NOT in target buckets) vs 100% "core" -> 40.00%
    state_unbucketed_cash = type("S", (), {
        "holdings": [H("A", 60_000, "core")],
        "buckets": [B("core", Decimal("100"))],
        "summary": type("Sum", (), {"usd_thb_rate": Decimal("35.5"), "cash_balance_thb": Decimal("40000")})(),
    })()
    rep_ucash = evaluate_portfolio_advisory(state_unbucketed_cash)
    assert rep_ucash.risk_radar.bucket_active_share_pct == Decimal("40.00")

    # Case C: 50k in "core", 50k in unconfigured "legacy" vs targets {"core": 60, "cash": 40} -> 50.00%
    state_unconfigured = type("S", (), {
        "holdings": [H("A", 50_000, "core"), H("B", 50_000, "legacy")],
        "buckets": [B("core", Decimal("60")), B("cash", Decimal("40"))],
        "summary": type("Sum", (), {"usd_thb_rate": Decimal("35.5"), "cash_balance_thb": Decimal("0")})(),
    })()
    rep_unconf = evaluate_portfolio_advisory(state_unconfigured)
    assert rep_unconf.risk_radar.bucket_active_share_pct == Decimal("50.00")

    # Case D: Zero NAV -> UNAVAILABLE & None (not 50.00%)
    state_zero = type("S", (), {
        "holdings": [],
        "buckets": [B("core", Decimal("100"))],
        "summary": type("Sum", (), {"usd_thb_rate": Decimal("35.5"), "cash_balance_thb": Decimal("0")})(),
    })()
    rep_zero = evaluate_portfolio_advisory(state_zero)
    assert rep_zero.risk_radar.bucket_active_share_status == "UNAVAILABLE"
    assert rep_zero.risk_radar.bucket_active_share_pct is None


def test_r08_new_radar_holding_and_drawdown_fields_survive_dto_json_and_synthesis_payload():
    """R08 / W04: Domain -> DTO -> JSON and synthesis LLM payload preserve every new radar & holding field."""
    from application.portfolio.advisory_service import _convert_report_to_dto
    from core.portfolio_advisory.llm_payload_builder import build_advisory_llm_payload

    class H:
        def __init__(self, symbol, units, price, bucket_id):
            self.symbol = symbol
            self.asset_type = "Stock"
            self.units = Decimal(str(units))
            self.current_price_thb = Decimal(str(price))
            self.market_value_thb = Decimal(str(units * price))
            self.avg_cost_thb = Decimal(str(price))
            self.bucket_id = bucket_id
            self.market = "TH"
            self.currency = "THB"
            self.status = "active"
            self.name = symbol

    class B:
        def __init__(self, bid, pct):
            self.bucket_id = bid
            self.name = bid
            self.target_percent = pct

    class S:
        holdings = [
            H("PTT.BK", 5000, 100.0, "core"),
            H("SCB.BK", 5000, 100.0, "core"),
        ]
        summary = type("Sum", (), {"usd_thb_rate": Decimal("35.5"), "cash_balance_thb": Decimal("0")})()
        buckets = [B("core", Decimal("100.0"))]

    sidecars = {
        "PTT.BK": {
            "ticker": "PTT.BK",
            "market": "TH",
            "quant_signals": {
                "raw_analysis_price": 100.0,
                "volatility_pct": 10.0,
                "adtv_local_currency": 10_000_000.0,
                "deterministic_scorecard": {"core_conviction_score": 9.0, "action_stance": "ACCUMULATE_NOW"},
                "tactical_setup": {"buy_zone_max": 110.0, "buy_zone_min": 90.0},
            },
        },
        "SCB.BK": {
            "ticker": "SCB.BK",
            "market": "TH",
            "quant_signals": {
                "raw_analysis_price": 100.0,
                "volatility_pct": 40.0,
                "adtv_local_currency": 1_000_000.0,
                "deterministic_scorecard": {"core_conviction_score": 8.0, "action_stance": "HOLD"},
            },
        },
    }

    report = evaluate_portfolio_advisory(
        portfolio_state=S(),
        equity_sidecars=sidecars,
        peak_nav_thb=Decimal("1100000"),
    )
    dto = _convert_report_to_dto(report)
    json_payload = dto.model_dump()
    llm_payload = build_advisory_llm_payload(report)

    for radar_dict in (json_payload["risk_radar"], llm_payload["risk_radar"]):
        assert radar_dict["top_1_weight_pct"] == float(report.risk_radar.top_1_weight_pct)
        assert radar_dict["concentration_risk_level"] == report.risk_radar.concentration_risk_level
        assert radar_dict["bucket_active_share_pct"] == float(report.risk_radar.bucket_active_share_pct)
        assert radar_dict["risk_contributions_pct"]["PTT.BK"] == float(report.risk_radar.risk_contributions_pct["PTT.BK"])
        assert radar_dict["total_portfolio_volatility_pct"] == float(report.risk_radar.total_portfolio_volatility_pct)
        assert radar_dict["drawdown_posture_tier"] == report.risk_radar.drawdown_posture_tier
        assert radar_dict["drawdown_pct"] == float(report.risk_radar.drawdown_pct)

    ptt_dto = next(h for h in json_payload["holdings_advisory"] if h["symbol"] == "PTT.BK")
    ptt_llm = next(h for h in llm_payload["holdings_advisory"] if h["symbol"] == "PTT.BK")
    ptt_dom = next(h for h in report.holdings_advisory if h.symbol == "PTT.BK")
    for h_dict in (ptt_dto, ptt_llm):
        assert h_dict["risk_contribution_pct"] == float(ptt_dom.risk_contribution_pct)
        assert h_dict["estimated_friction_bps"] == float(ptt_dom.estimated_friction_bps)
        assert h_dict["volatility_pct"] == float(ptt_dom.volatility_pct)
        assert h_dict["volatility_source"] == "observed"
