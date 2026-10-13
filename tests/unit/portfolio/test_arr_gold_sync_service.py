import pytest
from decimal import Decimal
from unittest.mock import MagicMock

from tools.portfolio.domain.models import TradeImportItem, TradeFeeBreakdown
from tools.portfolio.ports.trade_ingestion_port import TradeDocumentMetadata
from tools.portfolio.services.arr_gold_sync_service import ArrGoldSyncService


@pytest.fixture
def mock_deps():
    email_source = MagicMock()
    parser = MagicMock()
    staging = MagicMock()
    batch_import_service = MagicMock()
    batch_import_service.repo = None

    service = ArrGoldSyncService(
        email_source=email_source,
        parser=parser,
        staging=staging,
        batch_import_service=batch_import_service,
    )
    return service, email_source, parser, staging, batch_import_service


def test_scan_emails(mock_deps):
    service, email_source, _, _, _ = mock_deps
    mock_meta = TradeDocumentMetadata(
        message_id="123",
        attachment_id="123_att0",
        subject="Buy Gold (B26061208401)",
        sender="noreply@arrgoldtrading.com",
        received_at="2026-06-12",
        filename="arr_gold_123.html",
        size_bytes=2048,
    )
    email_source.search_arr_gold_emails.return_value = [mock_meta]

    res = service.scan_emails(limit=10)
    assert len(res) == 1
    assert res[0].subject == "Buy Gold (B26061208401)"
    email_source.search_arr_gold_emails.assert_called_once_with(query="", limit=10, since_date=None)


def test_scan_and_stage_email(mock_deps):
    service, email_source, parser, staging, _ = mock_deps

    email_source.fetch_email_html_body.return_value = "<html>dummy</html>"
    item = TradeImportItem(
        item_id="ARR_B26061208401",
        trade_date="2026-06-12",
        symbol="GOLD-99.99",
        action="BUY",
        units=Decimal("1.0"),
        price=Decimal("4221.53"),
        gross_amount=Decimal("4221.53"),
        fees=TradeFeeBreakdown(fee_currency="USD"),
        net_amount=Decimal("4221.53"),
        currency="USD",
        confirmation_no="B26061208401",
        order_id="B26061208401",
        source="ARR_GOLD",
        fingerprint="fp123",
        cash_adjusted=False,
        asset_type="Gold",
    )
    parser.parse_trade_html.return_value = [item]
    staging.stage_items.return_value = "scan_test_123"

    scan_id, previews, warnings = service.scan_and_stage_email(
        message_id="123",
        attachment_id="123_body",
        subject="Buy Gold (B26061208401)",
    )

    assert scan_id == "scan_test_123"
    assert len(previews) == 1
    assert previews[0]["confirmation_no"] == "B26061208401"
    assert previews[0]["symbol"] == "GOLD-99.99"
    staging.stage_items.assert_called_once()


def test_commit_staged(mock_deps):
    service, _, _, staging, batch_import_service = mock_deps

    item = TradeImportItem(
        item_id="ARR_B26061208401",
        trade_date="2026-06-12",
        symbol="GOLD-99.99",
        action="BUY",
        units=Decimal("1.0"),
        price=Decimal("4221.53"),
        gross_amount=Decimal("4221.53"),
        fees=TradeFeeBreakdown(fee_currency="USD"),
        net_amount=Decimal("4221.53"),
        currency="USD",
        confirmation_no="B26061208401",
        order_id="B26061208401",
        source="ARR_GOLD",
        fingerprint="fp123",
        cash_adjusted=False,
        asset_type="Gold",
    )
    staging.get_staged_items.return_value = [item]
    staging.pop_provenance.return_value = None
    batch_import_service.execute_batch_import.return_value = MagicMock()

    res = service.commit_staged(scan_id="scan_test_123", portfolio_id="default")
    assert res["status"] == "success"
    assert res["imported_count"] == 1
    batch_import_service.execute_batch_import.assert_called_once()
    staging.delete_staged.assert_called_once_with("scan_test_123", session_id=None)


def test_commit_staged_with_provenance(mock_deps, monkeypatch):
    service, _, _, staging, batch_import_service = mock_deps
    item = TradeImportItem(
        item_id="ARR_B26061208401",
        trade_date="2026-06-12",
        symbol="GOLD-99.99",
        action="BUY",
        units=Decimal("1.0"),
        price=Decimal("4221.53"),
        gross_amount=Decimal("4221.53"),
        fees=TradeFeeBreakdown(fee_currency="USD"),
        net_amount=Decimal("4221.53"),
        currency="USD",
        confirmation_no="B26061208401",
        order_id="B26061208401",
        source="ARR_GOLD",
        fingerprint="fp123",
        cash_adjusted=False,
        asset_type="Gold",
    )
    staging.get_staged_items.return_value = [item]
    staging.pop_provenance.return_value = {
        "account_email": "test@example.com",
        "emails": {
            "msg_1": {
                "status": "staged",
                "has_error": False,
                "subject": "Buy Gold",
                "confirmation_no": "B26061208401",
                "order_ids": ["B26061208401"],
            }
        },
    }
    batch_import_service.execute_batch_import.return_value = MagicMock()

    res = service.commit_staged(scan_id="scan_test_123", portfolio_id="default")
    assert res["status"] == "success"
    assert res["imported_count"] == 1


def test_commit_staged_not_found(mock_deps):
    service, _, _, staging, _ = mock_deps
    staging.get_staged_items.return_value = []

    with pytest.raises(ValueError, match="ไม่พบในระบบ"):
        service.commit_staged(scan_id="invalid_id")


def test_stream_batch_sync_stages_for_review(mock_deps, monkeypatch):
    service, email_source, parser, staging, batch_import_service = mock_deps
    monkeypatch.setenv("GMAIL_IMAP_USER", "test@example.com")

    mock_meta = TradeDocumentMetadata(
        message_id="msg_stream_1",
        attachment_id="att1",
        subject="Buy Gold (B26061208401)",
        sender="noreply@arrgoldtrading.com",
        received_at="2026-06-12",
        filename="test.html",
        size_bytes=1000,
    )
    email_source.search_arr_gold_emails.return_value = [mock_meta]
    email_source.fetch_email_html_body.return_value = "<html>test</html>"

    trade_item = TradeImportItem(
        item_id="ARR_B26061208401",
        trade_date="2026-06-12",
        symbol="GOLD-99.99",
        action="BUY",
        units=Decimal("1.0"),
        price=Decimal("4221.53"),
        gross_amount=Decimal("4221.53"),
        fees=TradeFeeBreakdown(fee_currency="USD"),
        net_amount=Decimal("4221.53"),
        currency="USD",
        confirmation_no="B26061208401",
        order_id="B26061208401",
        source="ARR_GOLD",
        fingerprint="fp123",
        cash_adjusted=False,
        asset_type="Gold",
    )
    parser.parse_trade_html.return_value = [trade_item]
    staging.stage_items.return_value = "scan_stream_123"

    events = list(service.stream_batch_sync(portfolio_id="default", force_rescan=True))
    complete_events = [ev for ev in events if "event: complete" in ev]
    assert len(complete_events) == 1

    import json
    data_line = [l for l in complete_events[0].split("\n") if l.startswith("data: ")][0]
    payload = json.loads(data_line[6:])
    assert payload["scan_id"] == "scan_stream_123"
    assert payload["item_count"] == 1
    assert payload["imported"] == 0
    assert len(payload["items"]) == 1
    assert payload["items"][0]["confirmation_no"] == "B26061208401"
    assert payload["items"][0]["asset_type"] == "Gold"

    # Batch import must NOT be executed during stream - requires user confirmation
    batch_import_service.execute_batch_import.assert_not_called()
    staging.stage_items.assert_called_once()

