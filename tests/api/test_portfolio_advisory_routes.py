"""Tests for Portfolio Advisory Endpoints (/api/portfolio/advisory and /api/portfolio/advisory/synthesize)."""
from unittest.mock import MagicMock, patch
import pytest

from schemas.portfolio_advisory_schemas import (
    DataAnomalyWarningDTO,
    DualCurrencyCashPlanDTO,
    HoldingAdvisoryDTO,
    PortfolioAdvisoryNarrativeDTO,
    PortfolioAdvisoryResponseDTO,
    PortfolioRiskRadarDTO,
)


@pytest.fixture
def mock_advisory_response():
    return PortfolioAdvisoryResponseDTO(
        evaluated_at="2026-10-18T10:00:00Z",
        data_mode="FULL_CROSS_SCALE",
        portfolio_state_hash="hash12345678",
        total_nav_thb=1_000_000.0,
        cash_plan=DualCurrencyCashPlanDTO(
            cash_thb_available=200_000.0,
            cash_usd_available=1_000.0,
            usd_thb_fx_rate=35.50,
            immediate_deploy_thb=50_000.0,
            immediate_deploy_usd=0.0,
            conditional_limit_order_thb=0.0,
            conditional_limit_order_usd=0.0,
            thb_surplus_after_immediate=150_000.0,
            usd_surplus_after_immediate=1_000.0,
            fx_conversion_needed_thb=0.0,
        ),
        risk_radar=PortfolioRiskRadarDTO(
            weighted_beta=1.05,
            sector_crowding={},
            fx_exposure_pct={"THB": 80.0, "USD": 20.0},
            purpose_bucket_drift={},
            unassigned_holdings_value_thb=0.0,
            unassigned_holdings_nav_pct=0.0,
            unassigned_drift_guard_active=False,
        ),
        holdings_advisory=[
            HoldingAdvisoryDTO(
                symbol="ADVANC.BK",
                asset_name="Advanced Info Service",
                asset_type="Stock",
                market="TH",
                native_currency="THB",
                evaluation_track="TRACK_1_DIRECT_EQUITIES",
                current_units=500.0,
                current_price_native=250.0,
                current_value_thb=125_000.0,
                current_weight_pct=12.5,
                target_weight_pct=12.5,
                weight_drift_pct=0.0,
                action_code="HOLD_WITHIN_BAND",
                action_rationale="สัดส่วนอยู่ในกรอบสมดุล",
            )
        ],
        quarantined_holdings=[],
        anomaly_warnings=[],
        cached_narrative=None,
    )


def test_get_portfolio_advisory_endpoint(authed_client, mock_advisory_response):
    """GET /api/portfolio/advisory returns status 200 and typed advisory payload."""
    from api.dependencies import get_portfolio_advisory_service
    from api.main import app

    mock_svc = MagicMock()
    mock_svc.get_portfolio_advisory.return_value = mock_advisory_response

    app.dependency_overrides[get_portfolio_advisory_service] = lambda: mock_svc
    try:
        r = authed_client.get("/api/portfolio/advisory?refresh_prices=false")
        assert r.status_code == 200
        data = r.json()
        assert data["data_mode"] == "FULL_CROSS_SCALE"
        assert data["total_nav_thb"] == 1_000_000.0
        assert len(data["holdings_advisory"]) == 1
        assert data["holdings_advisory"][0]["symbol"] == "ADVANC.BK"
        assert data["cash_plan"]["cash_thb_available"] == 200_000.0
    finally:
        app.dependency_overrides.pop(get_portfolio_advisory_service, None)


def test_synthesize_advisory_playbook_endpoint(authed_client):
    """POST /api/portfolio/advisory/synthesize returns synthesized playbook."""
    from api.dependencies import get_portfolio_advisory_service
    from api.main import app

    mock_narrative = PortfolioAdvisoryNarrativeDTO(
        synthesized_at="2026-10-18T10:05:00Z",
        portfolio_state_hash="hash12345678",
        macro_executive_summary="กลยุทธ์สมดุลภายใต้เศรษฐกิจ Expansion",
        immediate_action_playbook=["เคาะซื้อ ADVANC.BK ตามสัดส่วน"],
        conditional_dip_plan=[],
        risk_mitigation_and_hedging=[],
        consistency_verified=True,
    )

    mock_svc = MagicMock()
    mock_svc.synthesize_playbook.return_value = mock_narrative

    app.dependency_overrides[get_portfolio_advisory_service] = lambda: mock_svc
    try:
        r = authed_client.post("/api/portfolio/advisory/synthesize", json={"force_refresh": True})
        assert r.status_code == 200
        data = r.json()
        assert data["portfolio_state_hash"] == "hash12345678"
        assert "ADVANC.BK" in data["immediate_action_playbook"][0]
        assert data["consistency_verified"] is True
    finally:
        app.dependency_overrides.pop(get_portfolio_advisory_service, None)


def test_get_portfolio_advisory_end_to_end_new_radar_and_holding_fields_survive_json(authed_client):
    """R08: Real service + engine domain->DTO->JSON serialization carries risk/friction/active-share fields."""
    from decimal import Decimal
    from types import SimpleNamespace
    from api.dependencies import get_portfolio_advisory_service
    from api.main import app
    from application.portfolio.advisory_service import PortfolioAdvisoryService

    state = SimpleNamespace(
        holdings=[
            SimpleNamespace(
                symbol="PTT.BK",
                name="PTT PCL",
                asset_type="Stock",
                status="active",
                units=Decimal("5000"),
                current_price_thb=Decimal("100"),
                current_price_usd=None,
                market_value_thb=Decimal("500000"),
                avg_cost_thb=Decimal("95"),
                avg_cost_usd=None,
                bucket_id="core",
                market="TH",
                currency="THB",
                sector="Energy",
                unrealized_pnl_percent=Decimal("5.0"),
            ),
            SimpleNamespace(
                symbol="SCB.BK",
                name="SCB X",
                asset_type="Stock",
                status="active",
                units=Decimal("5000"),
                current_price_thb=Decimal("100"),
                current_price_usd=None,
                market_value_thb=Decimal("500000"),
                avg_cost_thb=Decimal("95"),
                avg_cost_usd=None,
                bucket_id="core",
                market="TH",
                currency="THB",
                sector="Financials",
                unrealized_pnl_percent=Decimal("5.0"),
            ),
        ],
        allocation_targets=[
            SimpleNamespace(bucket_id="core", name="Core Equities", target_percent=Decimal("100.0")),
        ],
        summary=SimpleNamespace(usd_thb_rate=Decimal("35.5"), cash_balance_thb=Decimal("0")),
    )
    sidecars = {
        "PTT": {
            "ticker": "PTT.BK",
            "market": "TH",
            "quant_signals": {
                "raw_analysis_price": 100.0,
                "volatility_pct": 10.0,
                "adtv_local_currency": 10_000_000.0,
                "deterministic_scorecard": {"core_conviction_score": 7.5, "action_stance": "HOLD"},
            },
        },
        "SCB.BK": {
            "ticker": "SCB.BK",
            "market": "TH",
            "quant_signals": {
                "raw_analysis_price": 100.0,
                "volatility_pct": 40.0,
                "adtv_local_currency": 1_000_000.0,
                "deterministic_scorecard": {"core_conviction_score": 8.5, "action_stance": "ACCUMULATE_NOW"},
                "tactical_setup": {"buy_zone_max": 105.0, "buy_zone_min": 95.0},
            },
        },
    }

    state_reader = MagicMock()
    state_reader.get_current_portfolio_state.return_value = state
    macro_reader = MagicMock()
    macro_reader.get_latest_macro_strategy.return_value = None
    macro_reader.get_latest_sector_rotation.return_value = None
    sidecar_reader = MagicMock()
    sidecar_reader.get_all_equity_sidecars.return_value = sidecars
    sidecar_reader.get_watchlist_sidecars.return_value = {}
    vault_port = MagicMock()
    vault_port.load_latest_advisory_sidecar.return_value = None

    real_svc = PortfolioAdvisoryService(
        portfolio_reader=state_reader,
        macro_reader=macro_reader,
        sidecar_reader=sidecar_reader,
        vault_port=vault_port,
        synthesizer_port=MagicMock(),
    )

    app.dependency_overrides[get_portfolio_advisory_service] = lambda: real_svc
    try:
        r = authed_client.get("/api/portfolio/advisory?refresh_prices=false")
        assert r.status_code == 200
        data = r.json()
        assert data["risk_radar"]["bucket_active_share_pct"] == 0.0
        assert data["risk_radar"]["bucket_active_share_status"] == "AVAILABLE"
        assert data["risk_radar"]["risk_contribution_status"] == "AVAILABLE"
        assert data["risk_radar"]["risk_contributions_pct"] == {"PTT.BK": 11.34, "SCB.BK": 88.66}
        assert data["risk_radar"]["inverse_vol_weights_pct"] == {"PTT.BK": 80.0, "SCB.BK": 20.0}
        assert data["risk_radar"]["total_portfolio_volatility_pct"] > 0
        scb = next(h for h in data["holdings_advisory"] if h["symbol"] == "SCB.BK")
        assert scb["risk_contribution_pct"] == 88.66
        assert scb["estimated_friction_bps"] > 25.0
        assert scb["volatility_pct"] == 40.0
        assert scb["volatility_source"] == "observed"
    finally:
        app.dependency_overrides.pop(get_portfolio_advisory_service, None)
