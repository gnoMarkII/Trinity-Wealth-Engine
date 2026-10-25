"""Tests for Portfolio Funds & Look-Through Routes."""
from unittest.mock import MagicMock
import pytest
from fastapi.testclient import TestClient

from api.dependencies import get_fund_look_through_service
from schemas.portfolio_advisory_schemas import (
    FundConstituentsDTO,
    FundHoldingItemDTO,
    FundSectorItemDTO,
    LookThroughHoldingExposureDTO,
    PortfolioLookThroughReportDTO,
    TrueSectorExposureDTO,
)


@pytest.fixture
def mock_fund_dto():
    return FundConstituentsDTO(
        fund_id="F00000SP500",
        symbol="SCBS&P500E",
        as_of_date="2026-03-31",
        master_fund_name="iShares Core S&P 500 ETF",
        master_fund_ticker="IVV",
        top_holdings=[
            FundHoldingItemDTO(name="Microsoft Corporation", percent=7.1, ticker="MSFT", asset_type="EQUITY"),
            FundHoldingItemDTO(name="Apple Inc.", percent=6.9, ticker="AAPL", asset_type="EQUITY"),
        ],
        sector_breakdown=[
            FundSectorItemDTO(sector_name_th="เทคโนโลยี", sector_name_en="Information Technology", percent=31.5),
        ],
        asset_allocation={"equity": 99.04, "cash": 0.96},
    )


@pytest.fixture
def mock_look_through_dto():
    return PortfolioLookThroughReportDTO(
        portfolio_id="default",
        evaluated_at="2026-10-18T10:00:00Z",
        total_nav_thb=1000000.0,
        holdings_exposure=[
            LookThroughHoldingExposureDTO(
                ticker="MSFT",
                company_name="Microsoft Corporation",
                market="US",
                direct_value_thb=50000.0,
                direct_weight_pct=5.0,
                indirect_value_thb=21000.0,
                indirect_weight_pct=2.1,
                total_value_thb=71000.0,
                total_weight_pct=7.1,
                source_funds=[("SCBS&P500E", 21000.0)],
                is_cap_breached=False,
                is_hidden_concentration=False,
            )
        ],
        top_overlapped_stocks=[],
        sector_exposure=[
            TrueSectorExposureDTO(
                sector_name="Information Technology",
                sector_name_th="เทคโนโลยี",
                total_value_thb=350000.0,
                weight_pct=35.0,
                direct_stocks_pct=20.0,
                funds_pct=15.0,
                is_crowded=True,
            )
        ],
        hidden_concentration_warnings=[],
        cap_breached_count=0,
    )


def test_get_fund_constituents_success(authed_client, mock_fund_dto):
    from api.main import app
    mock_svc = MagicMock()
    mock_svc.get_fund_constituents_dto.return_value = mock_fund_dto
    app.dependency_overrides[get_fund_look_through_service] = lambda: mock_svc

    try:
        resp = authed_client.get("/api/portfolio/funds/SCBS&P500E/constituents")
        assert resp.status_code == 200
        data = resp.json()
        assert data["symbol"] == "SCBS&P500E"
        assert data["master_fund_ticker"] == "IVV"
        assert len(data["top_holdings"]) == 2
        assert data["top_holdings"][0]["ticker"] == "MSFT"
    finally:
        app.dependency_overrides.pop(get_fund_look_through_service, None)


def test_get_fund_constituents_not_found(authed_client):
    from api.main import app
    mock_svc = MagicMock()
    mock_svc.get_fund_constituents_dto.return_value = None
    app.dependency_overrides[get_fund_look_through_service] = lambda: mock_svc

    try:
        resp = authed_client.get("/api/portfolio/funds/UNKNOWN/constituents")
        assert resp.status_code == 404
    finally:
        app.dependency_overrides.pop(get_fund_look_through_service, None)


def test_get_portfolio_look_through_success(authed_client, mock_look_through_dto):
    from api.main import app
    mock_svc = MagicMock()
    mock_svc.get_portfolio_look_through_dto.return_value = mock_look_through_dto
    app.dependency_overrides[get_fund_look_through_service] = lambda: mock_svc

    try:
        resp = authed_client.get("/api/portfolio/default/look-through")
        assert resp.status_code == 200
        data = resp.json()
        assert data["portfolio_id"] == "default"
        assert len(data["holdings_exposure"]) == 1
        assert data["holdings_exposure"][0]["ticker"] == "MSFT"
        assert len(data["sector_exposure"]) == 1
        assert data["sector_exposure"][0]["is_crowded"] is True
    finally:
        app.dependency_overrides.pop(get_fund_look_through_service, None)
