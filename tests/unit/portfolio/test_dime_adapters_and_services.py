import tempfile
import time
from decimal import Decimal
from pathlib import Path
import pytest

from tools.portfolio.domain.models import (
    TradeImportItem,
    TradeFeeBreakdown,
    PortfolioState,
    Holding,
)
from tools.portfolio.domain.errors import (
    StagedScanExpiredError,
    StagedScanForbiddenError,
    StagedScanNotFoundError,
    TradeDuplicateError,
)
from tools.portfolio.adapters.dime.inmemory_staging_adapter import InMemoryStagingAdapter
from tools.portfolio.adapters.dime.isolated_parser_adapter import IsolatedDimePdfParserAdapter
from tools.portfolio.adapters.markdown.repository_adapter import MarkdownVaultRepositoryAdapter
from tools.portfolio.services.batch_trade_import_service import BatchTradeImportService


def _create_sample_item(symbol="AAPL", conf_no="CONF123", order_id="ORD1", fp=None, units="10", price="150.00", net="1500.00"):
    actual_fp = fp or f"fp_{conf_no}_{symbol}_{order_id}_{units}_{price}"
    return TradeImportItem(
        item_id=f"item_{conf_no}_{symbol}_{order_id}_{units}_{price}",
        trade_date="2026-09-01",
        symbol=symbol,
        action="BUY",
        units=Decimal(units),
        price=Decimal(price),
        gross_amount=Decimal(net),
        fees=TradeFeeBreakdown(commission=Decimal("0.00"), vat=Decimal("0.00"), other_fees=Decimal("0.00"), fee_currency="USD"),
        net_amount=Decimal(net),
        currency="USD",
        confirmation_no=conf_no,
        order_id=order_id,
        source="DIME",
        fingerprint=actual_fp,
        cash_adjusted=True,
    )



def test_inmemory_staging_adapter_session_isolation():
    staging = InMemoryStagingAdapter()
    items = [_create_sample_item()]

    # Stage bound to session_A
    scan_id = staging.stage_items(items, ttl_seconds=1800, session_id="session_A")

    # Correct session retrieves items
    retrieved = staging.get_staged_items(scan_id, session_id="session_A")
    assert len(retrieved) == 1
    assert retrieved[0].symbol == "AAPL"

    # Mismatched session raises StagedScanForbiddenError (403)
    with pytest.raises(StagedScanForbiddenError):
        staging.get_staged_items(scan_id, session_id="session_B")


def test_inmemory_staging_adapter_ttl_expiry():
    staging = InMemoryStagingAdapter()
    items = [_create_sample_item()]

    # 0 second TTL -> expires immediately
    scan_id = staging.stage_items(items, ttl_seconds=0, session_id="session_A")
    time.sleep(0.01)

    with pytest.raises(StagedScanExpiredError):
        staging.get_staged_items(scan_id, session_id="session_A")


def test_inmemory_staging_adapter_not_found():
    staging = InMemoryStagingAdapter()
    with pytest.raises(StagedScanNotFoundError):
        staging.get_staged_items("non_existent_scan_id")


def test_isolated_parser_validates_magic_header_and_size():
    parser = IsolatedDimePdfParserAdapter()

    # Invalid header
    with pytest.raises(ValueError, match="Header '%PDF-' ไม่ถูกต้อง"):
        parser.parse_confirmation_pdf(b"NOT_A_PDF_FILE")

    # Over 10MB
    huge_data = b"%PDF-" + b"0" * (10 * 1024 * 1024 + 10)
    with pytest.raises(ValueError, match="เกินขีดจำกัด 10MB"):
        parser.parse_confirmation_pdf(huge_data)


def test_batch_trade_import_service_intra_batch_duplicate_and_consolidation():
    with tempfile.TemporaryDirectory() as tmpdir:
        vault = Path(tmpdir)
        import os
        os.environ["OBSIDIAN_VAULT_PATH"] = str(vault)
        repo = MarkdownVaultRepositoryAdapter()
        svc = BatchTradeImportService(repo=repo)

        # Batch with two identical items (same confirmation_no and order_id) -> consolidated safely
        item1 = _create_sample_item(symbol="NVDA", conf_no="CONF_NVDA", order_id="ORD_NVDA_1", fp="fp_1")
        item2 = _create_sample_item(symbol="NVDA", conf_no="CONF_NVDA", order_id="ORD_NVDA_1", fp="fp_2")

        state = svc.execute_batch_import([item1, item2], "default")
        nvda = next(h for h in state.holdings if h.symbol == "NVDA")
        assert nvda.units == 10.0  # Consolidated into 10 units, not 20

        # Now test conflicting figures with the same order_id -> must raise TradeReconciliationError
        item3 = _create_sample_item(symbol="NVDA", conf_no="CONF_NVDA", order_id="ORD_NVDA_CONFLICT", units="10", price="100.00", net="1000.00")
        item4 = _create_sample_item(symbol="NVDA", conf_no="CONF_NVDA", order_id="ORD_NVDA_CONFLICT", units="20", price="100.00", net="2000.00")

        from tools.portfolio.domain.errors import TradeReconciliationError
        with pytest.raises(TradeReconciliationError, match="ตรวจพบ Conflict"):
            svc.execute_batch_import([item3, item4], "default")


def test_same_day_identical_orders_with_distinct_order_ids_both_preserved():
    """User Requirement 1: Two legitimate orders on same day with identical figures but different Order IDs must both be preserved."""
    with tempfile.TemporaryDirectory() as tmpdir:
        vault = Path(tmpdir)
        import os
        os.environ["OBSIDIAN_VAULT_PATH"] = str(vault)
        repo = MarkdownVaultRepositoryAdapter()
        svc = BatchTradeImportService(repo=repo)

        # Order 1: 10 shares NVDA @ $120
        order1 = _create_sample_item(symbol="NVDA", conf_no="CONF_DAY_1", order_id="ORD_1001", units="10", price="120.00", net="1200.00")
        # Order 2: 10 shares NVDA @ $120 (same date, same price, same units, but DISTINCT order_id)
        order2 = _create_sample_item(symbol="NVDA", conf_no="CONF_DAY_1", order_id="ORD_1002", units="10", price="120.00", net="1200.00")

        state = svc.execute_batch_import([order1, order2], "default")
        nvda = next(h for h in state.holdings if h.symbol == "NVDA")
        # Both must be preserved -> 10 + 10 = 20 units!
        assert nvda.units == 20.0

        with repo.unit_of_work("default") as uow:
            rows = uow.read_trade_log_locked()
            assert len(rows) == 2
            order_ids = [r["Order_ID"] for r in rows]
            assert "ORD_1001" in order_ids
            assert "ORD_1002" in order_ids


def test_conflict_detected_against_existing_ledger_row():
    """User Requirement 3: If new trade has same (confirmation_no, order_id) but different figures, conflict is raised."""
    with tempfile.TemporaryDirectory() as tmpdir:
        vault = Path(tmpdir)
        import os
        os.environ["OBSIDIAN_VAULT_PATH"] = str(vault)
        repo = MarkdownVaultRepositoryAdapter()
        svc = BatchTradeImportService(repo=repo)

        # Initial import: 10 shares AAPL @ 150
        initial = _create_sample_item(symbol="AAPL", conf_no="CONF_C1", order_id="ORD_C1", units="10", price="150.00", net="1500.00")
        svc.execute_batch_import([initial], "default")

        # Conflicting import: same confirmation_no and order_id, but units changed to 25
        conflicted = _create_sample_item(symbol="AAPL", conf_no="CONF_C1", order_id="ORD_C1", units="25", price="150.00", net="3750.00")

        from tools.portfolio.domain.errors import TradeReconciliationError
        with pytest.raises(TradeReconciliationError, match="ตรวจพบ Conflict กับ Ledger เดิม"):
            svc.execute_batch_import([conflicted], "default")


def test_dime_trade_missing_order_id_fails_closed():
    """User Requirement 2: Dime trades must never use fallback line_index, missing order_id must reject."""
    with tempfile.TemporaryDirectory() as tmpdir:
        vault = Path(tmpdir)
        import os
        os.environ["OBSIDIAN_VAULT_PATH"] = str(vault)
        repo = MarkdownVaultRepositoryAdapter()
        svc = BatchTradeImportService(repo=repo)

        # Trade with empty order_id
        bad_item = _create_sample_item(symbol="AAPL", conf_no="CONF_BAD", order_id="", units="10")

        with pytest.raises(ValueError, match="ห้ามใช้ fallback line_index สำหรับ Dime"):
            svc.execute_batch_import([bad_item], "default")


def test_batch_trade_import_service_deduplication_and_cash_adjustment():
    with tempfile.TemporaryDirectory() as tmpdir:
        vault = Path(tmpdir)
        import os
        os.environ["OBSIDIAN_VAULT_PATH"] = str(vault)
        repo = MarkdownVaultRepositoryAdapter()

        # Seed initial state with CASH_USD = 5000.00
        init_state = PortfolioState(
            last_updated="2026-09-01T00:00:00",
            holdings=[
                Holding(symbol="CASH_USD", asset_type="Cash", units=5000.0, market_value_thb=182500.0),
                Holding(symbol="CASH_THB", asset_type="Cash", units=50000.0, market_value_thb=50000.0),
            ],
            fx_rates={"USDTHB": 36.5},
        )
        with repo.unit_of_work("default") as uow:
            uow.commit(init_state)

        svc = BatchTradeImportService(repo=repo)

        # BUY 10 AAPL @ 150 = 1500 USD Net
        item = _create_sample_item(symbol="AAPL", conf_no="CONF_001", order_id="ORD_AAPL_1", fp="fp_aapl_1", units="10", price="150.00", net="1500.00")

        # First import
        s1 = svc.execute_batch_import([item], "default")
        cash_after_1 = next(h for h in s1.holdings if h.symbol == "CASH_USD").units
        assert cash_after_1 == 3500.0  # 5000 - 1500
        aapl = next(h for h in s1.holdings if h.symbol == "AAPL")
        assert aapl.units == 10.0

        # Second import with the exact same item -> must be skipped without error or double deduction
        s2 = svc.execute_batch_import([item], "default")
        cash_after_2 = next(h for h in s2.holdings if h.symbol == "CASH_USD").units
        assert cash_after_2 == 3500.0  # Still 3500!
        aapl_2 = next(h for h in s2.holdings if h.symbol == "AAPL")
        assert aapl_2.units == 10.0

        # Ledger should only have 1 trade row with Order_ID populated
        with repo.unit_of_work("default") as uow:
            rows = uow.read_trade_log_locked()
            assert len(rows) == 1
            assert rows[0]["Symbol"] == "AAPL"
            assert rows[0]["Order_ID"] == "ORD_AAPL_1"
            assert rows[0]["Cash_Adjusted"] == "YES"
            assert rows[0]["Net_Amount"] == "1500.00"


def test_dime_sync_service_sync_history_and_partial_quarantine():
    """User Requirements 2 & 3: Valid emails are committed & recorded in sync history, quarantined emails are excluded."""
    with tempfile.TemporaryDirectory() as tmpdir:
        vault = Path(tmpdir)
        import os
        os.environ["OBSIDIAN_VAULT_PATH"] = str(vault)
        repo = MarkdownVaultRepositoryAdapter()
        staging = InMemoryStagingAdapter()
        batch_svc = BatchTradeImportService(repo=repo)

        from unittest.mock import MagicMock
        from tools.portfolio.ports.trade_ingestion_port import TradeDocumentMetadata
        from tools.portfolio.services.dime_sync_service import DimeSyncService, _load_sync_history

        mock_email_source = MagicMock()
        mock_parser = MagicMock()

        email1 = TradeDocumentMetadata(
            message_id="msg_1",
            attachment_id="att_1",
            subject="Confirmation Note 1",
            sender="confirm@dime.co.th",
            received_at="2026-09-01",
            filename="conf1.pdf",
            size_bytes=1000,
            x_gm_msgid="gm_msg_1",
            uid="101",
            account_email="tester@example.com",
        )
        email2 = TradeDocumentMetadata(
            message_id="msg_2",
            attachment_id="att_2",
            subject="Confirmation Note 2 (Invalid)",
            sender="confirm@dime.co.th",
            received_at="2026-09-02",
            filename="conf2.pdf",
            size_bytes=1000,
            x_gm_msgid="gm_msg_2",
            uid="102",
            account_email="tester@example.com",
        )

        mock_email_source.search_dime_emails.return_value = [email1, email2]
        mock_email_source.fetch_pdf_attachment.side_effect = lambda message_id, attachment_id: b"%PDF-mock"

        item_valid = _create_sample_item(symbol="GOOGL", conf_no="CONF_G1", order_id="ORD_G1", units="5", price="150.00", net="750.00")
        item_invalid = _create_sample_item(symbol="META", conf_no="CONF_M2", order_id="", units="5", price="300.00", net="1500.00")

        mock_parse = MagicMock()
        mock_parse.call_count = 0

        def counting_parse(pdf_bytes, password=None):
            mock_parse.call_count += 1
            if mock_parse.call_count == 1:
                return [item_valid]
            return [item_invalid]

        mock_parser.parse_confirmation_pdf.side_effect = counting_parse

        dime_sync = DimeSyncService(
            email_source=mock_email_source,
            parser=mock_parser,
            staging=staging,
            batch_import_service=batch_svc,
        )

        # Run stream_batch_sync
        events = list(dime_sync.stream_batch_sync(password="pass", force_rescan=False, portfolio_id="default"))
        complete_event = next(e for e in events if e["event"] == "complete")

        # Must stage item_valid, quarantine item_invalid, generate warning
        assert complete_event["data"]["item_count"] == 1
        assert len(complete_event["data"]["warnings"]) == 1
        scan_id = complete_event["data"]["scan_id"]
        assert scan_id != ""

        # Before commit, sync history must be empty!
        hist_before = _load_sync_history("default", "tester@example.com")
        assert hist_before["synced_emails"] == {}

        # Commit staged
        dime_sync.commit_staged(scan_id=scan_id, portfolio_id="default")

        # After commit:
        # Email 1 (valid) must be in synced_emails
        # Email 2 (quarantined) must NOT be in synced_emails!
        hist_after = _load_sync_history("default", "tester@example.com")
        assert "gm_msg_1" in hist_after["synced_emails"]
        assert "gm_msg_2" not in hist_after["synced_emails"]
        assert hist_after["synced_emails"]["gm_msg_1"]["order_ids"] == ["ORD_G1"]

        # Run stream_batch_sync again without force_rescan -> email 1 must be skipped!
        mock_parse.call_count = 0
        events_2 = list(dime_sync.stream_batch_sync(password="pass", force_rescan=False, portfolio_id="default"))
        discovered_2 = next(e for e in events_2 if e["event"] == "discovered")
        assert discovered_2["data"]["already_synced"] == 1
        assert discovered_2["data"]["to_process"] == 1


