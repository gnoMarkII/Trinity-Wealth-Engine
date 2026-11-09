"""Tests for Portfolio Watchlist Opportunities Routes."""
from unittest.mock import MagicMock
import pytest
from fastapi.testclient import TestClient

from api.dependencies import get_portfolio_opportunity_service
from schemas.opportunity_schemas import (
    CandidateEvaluationResultDTO,
    OpportunityCandidateDTO,
    PortfolioOpportunitiesResponseDTO,
)


@pytest.fixture
def mock_opportunity_dto():
    candidate = OpportunityCandidateDTO(
        security_id="US:MSFT",
        issuer_id="MSFT",
        symbol="MSFT",
        market="US",
        currency="USD",
        asset_type="EQUITY",
        company_name="Microsoft Corporation",
        sector_name="Information Technology",
        sector_name_th="เทคโนโลยี",
        source_pools=["POOL_A_FUND_LOOK_THROUGH", "POOL_B_EQUITY_SIDECAR"],
        source_refs=["fund:SCBS&P500E", "sidecar:MSFT"],
        already_directly_held=False,
        already_watchlisted=False,
        current_direct_value_thb=0.0,
        current_indirect_value_thb=21000.0,
        current_total_exposure_thb=21000.0,
        current_total_weight_pct=2.1,
    )
    result = CandidateEvaluationResultDTO(
        candidate=candidate,
        policy_fit="pass",
        policy_reasons=["Admissible under confirmed Investor Axis rules."],
        suggested_bucket_id="growth",
        applied_bucket_name="Growth",
        evidence_readiness="sufficient",
        composite_score=8.7,
        valuation_upside_pct=15.0,
        target_price=450.0,
        target_price_basis="DCF",
        moat_rating="Wide",
        evidence_reasons=["Active sidecar verified."],
        macro_alignment="tailwind",
        macro_stage="Leading",
        macro_reasons=["Information Technology in Leading rotation quadrant."],
        capacity_status="available",
        remaining_capacity_thb=129000.0,
        issuer_room_thb=129000.0,
        sector_room_thb=250000.0,
        capacity_reasons=["Available under 15% Cap."],
        research_priority=1,
        priority_reasons=["High Conviction candidate."],
        is_actionable=True,
    )
    return PortfolioOpportunitiesResponseDTO(
        portfolio_id="default",
        evaluation_id="opp_20261019_001",
        evaluated_at="2026-10-19T10:00:00Z",
        readiness="ready",
        total_candidates_scanned=1,
        passed_candidates_count=1,
        candidates=[result],
        warnings=[],
        missing_requirements=[],
        snapshot_meta={"nav_thb": 1000000.0},
    )


def test_get_portfolio_opportunities_success(authed_client, mock_opportunity_dto):
    from api.dependencies import get_portfolio_opportunity_service
    from api.main import app

    mock_service = MagicMock()
    mock_service.get_opportunities.return_value = mock_opportunity_dto

    app.dependency_overrides[get_portfolio_opportunity_service] = lambda: mock_service

    try:
        response = authed_client.get("/api/portfolio/watchlist/opportunities?portfolio_id=default")
        assert response.status_code == 200
        data = response.json()
        assert data["portfolio_id"] == "default"
        assert data["readiness"] == "ready"
        assert len(data["candidates"]) == 1
        cand = data["candidates"][0]
        assert cand["candidate"]["symbol"] == "MSFT"
        assert cand["research_priority"] == 1
        assert cand["macro_alignment"] == "tailwind"
        assert cand["capacity_status"] == "available"
        assert cand["is_actionable"] is True
    finally:
        app.dependency_overrides.pop(get_portfolio_opportunity_service, None)


def test_get_portfolio_opportunity_service_provider():
    from api.dependencies import get_portfolio_opportunity_service
    from application.portfolio.opportunity_service import PortfolioOpportunityService

    svc = get_portfolio_opportunity_service()
    assert isinstance(svc, PortfolioOpportunityService)
    assert callable(getattr(svc, "get_opportunities", None))


def test_w02_opportunity_service_end_to_end_wires_fx_and_adtv_limit(authed_client):
    """W02: Real PortfolioOpportunityService extracts portfolio FX rate and sidecar ADTV -> JSON response."""
    from decimal import Decimal
    from types import SimpleNamespace
    from api.dependencies import get_portfolio_opportunity_service
    from api.main import app
    from application.portfolio.opportunity_service import PortfolioOpportunityService
    from core.opportunity_finder.models import OpportunityCandidate

    state = SimpleNamespace(
        summary=SimpleNamespace(market_value_thb=Decimal("1000000"), available_cash_thb=Decimal("200000"), usd_thb_rate=Decimal("35.0")),
        fx_rates={"USDTHB": 35.0},
        holdings=[],
        allocation_targets=[],
    )
    portfolio_reader = MagicMock()
    portfolio_reader.get_current_portfolio_state.return_value = state

    equity_reader = MagicMock()
    equity_reader.get_all_equity_sidecars.return_value = {
        "MSFT": {
            "market": "US",
            "analysis_date": "2026-10-15",
            "quant_signals": {
                "composite_score": 84.0,
                "adtv_local_currency": 40000.0,  # $40k * 35 = 1,400,000 THB -> 5% = 70,000 THB cap
            },
        }
    }
    policy_reader = MagicMock()
    policy_reader.get_confirmed_investment_axis.return_value = {"prohibited_symbols": []}
    policy_reader.get_applied_bucket_policy.return_value = []
    macro_reader = MagicMock()
    macro_reader.get_latest_sector_rotation.return_value = {
        "sectors": {"information technology": {"quadrant": "Leading"}}
    }
    candidate_pool = MagicMock()
    candidate_pool.discover_candidates.return_value = [
        OpportunityCandidate(
            security_id="US:MSFT",
            issuer_id="MSFT",
            symbol="MSFT",
            market="US",
            currency="USD",
            asset_type="EQUITY",
            company_name="Microsoft",
            sector_name="Information Technology",
            sector_name_th="เทคโนโลยี",
        )
    ]

    svc = PortfolioOpportunityService(
        portfolio_reader=portfolio_reader,
        fund_port=MagicMock(),
        equity_reader=equity_reader,
        policy_reader=policy_reader,
        macro_reader=macro_reader,
        watchlist_reader=MagicMock(),
        candidate_pool_service=candidate_pool,
    )

    app.dependency_overrides[get_portfolio_opportunity_service] = lambda: svc
    try:
        r = authed_client.get("/api/portfolio/watchlist/opportunities?portfolio_id=default&force_refresh=true")
        assert r.status_code == 200
        data = r.json()
        assert data["snapshot_meta"]["usd_thb_fx_rate"] == 35.0
        cand = data["candidates"][0]
        assert cand["liquidity_status"] == "verified"
        assert cand["adtv_thb"] == 1400000.0
        assert cand["liquidity_room_thb"] == 70000.0
        assert cand["remaining_capacity_thb"] == 70000.0
    finally:
        app.dependency_overrides.pop(get_portfolio_opportunity_service, None)

