from unittest.mock import MagicMock
import pytest
from fastapi.testclient import TestClient

from api.dependencies import get_portfolio_service
from tools.portfolio.ports.trade_ingestion_port import TradeDocumentMetadata
from tools.portfolio.domain.models import PortfolioState, Holding, TradeImportItem, TradeFeeBreakdown
from decimal import Decimal


def test_arr_gold_emails_endpoint(authed_client):
    mock_service = MagicMock()
    mock_sync_svc = MagicMock()
    mock_sync_svc.scan_emails.return_value = [
        TradeDocumentMetadata(
            message_id="msg-101",
            attachment_id="",
            subject="ยืนยันคำสั่งซื้อขายทองคำ ARR",
            sender="noreply@arrgoldtrading.com",
            received_at="2026-03-20T10:00:00Z",
            filename="body.html",
            size_bytes=1024,
        )
    ]
    mock_service._arr_gold_sync_service = mock_sync_svc

    from api.main import app
    app.dependency_overrides[get_portfolio_service] = lambda: mock_service

    try:
        resp = authed_client.get("/api/portfolio/arr-gold/emails")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["emails"]) == 1
        assert data["emails"][0]["message_id"] == "msg-101"
        assert data["emails"][0]["subject"] == "ยืนยันคำสั่งซื้อขายทองคำ ARR"
    finally:
        app.dependency_overrides.pop(get_portfolio_service, None)


def test_arr_gold_scan_email_endpoint(authed_client):
    mock_service = MagicMock()
    mock_sync_svc = MagicMock()
    mock_sync_svc.scan_and_stage_email.return_value = (
        "scan-xyz-123",
        [
            {
                "item_id": "item-1",
                "trade_date": "2026-03-20",
                "symbol": "GOLD-99.99",
                "action": "BUY",
                "units": 1.0,
                "price": 4221.53,
                "gross_amount": 4221.53,
                "fees": {"commission": 0.0, "vat": 0.0, "other_fees": 0.0, "fee_currency": "USD"},
                "net_amount": 4221.53,
                "currency": "USD",
                "confirmation_no": "B20260320001",
                "source": "ARR_GOLD",
                "fingerprint": "fp-1",
                "cash_adjusted": False,
                "asset_type": "Gold",
            }
        ],
        [],
    )
    mock_service._arr_gold_sync_service = mock_sync_svc

    from api.main import app
    app.dependency_overrides[get_portfolio_service] = lambda: mock_service

    try:
        resp = authed_client.post(
            "/api/portfolio/arr-gold/scan/email",
            json={"message_id": "msg-101", "subject": "ยืนยันคำสั่งซื้อขาย"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["scan_id"] == "scan-xyz-123"
        assert data["item_count"] == 1
        assert data["items"][0]["symbol"] == "GOLD-99.99"
        assert data["items"][0]["asset_type"] == "Gold"
        assert data["items"][0]["currency"] == "USD"
    finally:
        app.dependency_overrides.pop(get_portfolio_service, None)


def test_arr_gold_commit_endpoint(authed_client):
    mock_service = MagicMock()
    mock_sync_svc = MagicMock()
    state = PortfolioState(
        last_updated="2026-03-20T10:00:00Z",
        holdings=[
            Holding(symbol="GOLD-99.99", asset_type="Gold", units=1.0, avg_cost_usd=4221.53),
        ],
    )
    mock_sync_svc.commit_staged.return_value = {
        "status": "success",
        "imported_count": 1,
        "state": state,
    }
    mock_service._arr_gold_sync_service = mock_sync_svc

    from api.main import app
    app.dependency_overrides[get_portfolio_service] = lambda: mock_service

    try:
        resp = authed_client.post(
            "/api/portfolio/arr-gold/commit/scan-xyz-123",
            json={"portfolio_id": "default", "selected_item_ids": ["item-1"]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert data["imported_count"] == 1
        assert len(data["state"]["holdings"]) == 1
        assert data["state"]["holdings"][0]["symbol"] == "GOLD-99.99"
    finally:
        app.dependency_overrides.pop(get_portfolio_service, None)
