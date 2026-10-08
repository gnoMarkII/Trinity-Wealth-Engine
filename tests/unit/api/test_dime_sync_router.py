import io
from decimal import Decimal
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.auth import SESSION_COOKIE_NAME, _serializer
from tools.portfolio.domain.models import TradeImportItem, TradeFeeBreakdown, PortfolioState, Holding
from tools.portfolio.domain.errors import (
    StagedScanExpiredError,
    StagedScanForbiddenError,
    StagedScanNotFoundError,
    TradeDuplicateError,
)
from tools.portfolio.ports.trade_ingestion_port import TradeDocumentMetadata


@pytest.fixture
def auth_client():
    client = TestClient(app)
    # Generate signed session token
    token = _serializer().dumps({"authenticated": True, "sid": "test_session_123"})
    client.cookies.set(SESSION_COOKIE_NAME, token)
    return client


def _sample_trade_item():
    return TradeImportItem(
        item_id="item_001",
        trade_date="2026-09-01",
        symbol="AAPL",
        action="BUY",
        units=Decimal("10"),
        price=Decimal("150.00"),
        gross_amount=Decimal("1500.00"),
        fees=TradeFeeBreakdown(commission=Decimal("1.50"), vat=Decimal("0.11"), other_fees=Decimal("0.00"), fee_currency="USD"),
        net_amount=Decimal("1501.61"),
        currency="USD",
        confirmation_no="CONF_999",
        source="DIME",
        fingerprint="fp_sample_999",
        cash_adjusted=True,
    )


def test_dime_scan_upload_and_commit_flow(auth_client):
    fake_item = _sample_trade_item()
    fake_state = PortfolioState(
        last_updated="2026-09-01T00:00:00",
        holdings=[Holding(symbol="AAPL", asset_type="Stock", units=10.0, avg_cost_usd=150.16)],
    )

    with patch("tools.portfolio.service.PortfolioService.void_transaction") as mock_void:
        # Test scan upload endpoint
        with patch.object(
            app.dependency_overrides.get("get_portfolio_service", lambda: None),
            "_dime_sync_service",
            create=True,
        ):
            pass

    # Direct test via TestClient against routes
    from api.dependencies import get_portfolio_service
    mock_service = MagicMock()
    mock_sync = MagicMock()
    mock_service._dime_sync_service = mock_sync

    mock_sync.parse_and_stage_upload.return_value = ("scan_test_1", [fake_item])
    mock_sync.get_staged.return_value = [fake_item]
    mock_sync.commit_staged.return_value = fake_state

    app.dependency_overrides[get_portfolio_service] = lambda: mock_service
    try:
        # 1. Upload mock PDF
        pdf_bytes = b"%PDF-1.4 mock content"
        files = {"pdf_file": ("test.pdf", io.BytesIO(pdf_bytes), "application/pdf")}
        res = auth_client.post("/api/portfolio/dime/scan/upload", files=files)
        assert res.status_code == 200
        data = res.json()
        assert data["scan_id"] == "scan_test_1"
        assert data["item_count"] == 1
        assert data["items"][0]["symbol"] == "AAPL"

        # 2. Get staged items
        res_staged = auth_client.get("/api/portfolio/dime/staged/scan_test_1")
        assert res_staged.status_code == 200
        assert res_staged.json()["items"][0]["symbol"] == "AAPL"

        # 3. Commit staged items
        res_commit = auth_client.post("/api/portfolio/dime/commit/scan_test_1", json={"portfolio_id": "default"})
        assert res_commit.status_code == 200
        assert res_commit.json()["ok"] is True
        assert res_commit.json()["imported_count"] == 1
    finally:
        app.dependency_overrides.pop(get_portfolio_service, None)


def test_dime_sync_error_status_code_mappings(auth_client):
    from api.dependencies import get_portfolio_service
    mock_service = MagicMock()
    mock_sync = MagicMock()
    mock_service._dime_sync_service = mock_sync

    app.dependency_overrides[get_portfolio_service] = lambda: mock_service
    try:
        # 404 Not Found
        mock_sync.get_staged.side_effect = StagedScanNotFoundError("Not found")
        res_404 = auth_client.get("/api/portfolio/dime/staged/unknown_scan")
        assert res_404.status_code == 404

        # 410 Gone (Expired)
        mock_sync.get_staged.side_effect = StagedScanExpiredError("Expired")
        res_410 = auth_client.get("/api/portfolio/dime/staged/expired_scan")
        assert res_410.status_code == 410

        # 403 Forbidden (Session Mismatch)
        mock_sync.get_staged.side_effect = StagedScanForbiddenError("Session forbidden")
        res_403 = auth_client.get("/api/portfolio/dime/staged/other_session_scan")
        assert res_403.status_code == 403

        # 409 Conflict (Duplicate Trade)
        mock_sync.commit_staged.side_effect = TradeDuplicateError("Duplicate trade")
        mock_sync.get_staged.side_effect = None
        mock_sync.get_staged.return_value = [_sample_trade_item()]
        res_409 = auth_client.post("/api/portfolio/dime/commit/dup_scan", json={"portfolio_id": "default"})
        assert res_409.status_code == 409
    finally:
        app.dependency_overrides.pop(get_portfolio_service, None)


def test_void_and_deprecated_delete_endpoints(auth_client):
    from api.dependencies import get_portfolio_service
    mock_service = MagicMock()
    mock_state = PortfolioState(last_updated="2026-09-01T00:00:00")
    mock_service.void_transaction.return_value = mock_state

    app.dependency_overrides[get_portfolio_service] = lambda: mock_service
    try:
        # POST /void
        res_void = auth_client.post("/api/portfolio/actual/transactions/tx_123/void")
        assert res_void.status_code == 200
        mock_service.void_transaction.assert_called_with(tx_id="tx_123", portfolio_id="default")

        # DELETE /transactions/{tx_id} with deprecation header
        res_del = auth_client.delete("/api/portfolio/actual/transactions/tx_123")
        assert res_del.status_code == 200
        assert "X-Deprecation-Warning" in res_del.headers
        assert "use POST /api/portfolio/actual/transactions/{tx_id}/void instead" in res_del.headers["X-Deprecation-Warning"]
    finally:
        app.dependency_overrides.pop(get_portfolio_service, None)


def test_dime_sync_partial_commit_selection(auth_client):
    from api.dependencies import get_portfolio_service
    item1 = _sample_trade_item()
    item2 = TradeImportItem(
        item_id="item_002",
        trade_date="2026-09-02",
        symbol="MSFT",
        action="BUY",
        units=Decimal("5"),
        price=Decimal("300.00"),
        gross_amount=Decimal("1500.00"),
        fees=TradeFeeBreakdown(commission=Decimal("0"), vat=Decimal("0"), other_fees=Decimal("0"), fee_currency="USD"),
        net_amount=Decimal("1500.00"),
        currency="USD",
        confirmation_no="CONF_002",
        source="DIME",
        fingerprint="fp_sample_002",
        cash_adjusted=True,
    )
    fake_state = PortfolioState(
        last_updated="2026-09-02T00:00:00",
        holdings=[Holding(symbol="AAPL", asset_type="Stock", units=10.0, avg_cost_usd=150.16)],
    )

    mock_service = MagicMock()
    mock_sync = MagicMock()
    mock_service._dime_sync_service = mock_sync
    mock_sync.get_staged.return_value = [item1, item2]
    mock_sync.commit_staged.return_value = fake_state

    app.dependency_overrides[get_portfolio_service] = lambda: mock_service
    try:
        # Commit only item_001
        res = auth_client.post(
            "/api/portfolio/dime/commit/scan_partial",
            json={"portfolio_id": "default", "selected_item_ids": ["item_001"]},
        )
        assert res.status_code == 200
        assert res.json()["imported_count"] == 1
        mock_sync.commit_staged.assert_called_once_with(
            scan_id="scan_partial",
            session_id="test_session_123",
            portfolio_id="default",
            selected_item_ids=["item_001"],
        )
    finally:
        app.dependency_overrides.pop(get_portfolio_service, None)

