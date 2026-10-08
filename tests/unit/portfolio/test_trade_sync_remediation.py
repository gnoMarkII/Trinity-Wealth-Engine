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
from tools.portfolio.domain.calculations import extract_active_ledger_identities
from tools.portfolio.adapters.dime.inmemory_staging_adapter import InMemoryStagingAdapter
from tools.portfolio.adapters.markdown.repository_adapter import MarkdownVaultRepositoryAdapter
from tools.portfolio.services.batch_trade_import_service import BatchTradeImportService
from tools.portfolio.services.trading_service import PortfolioTradingService
from tools.portfolio.services.ledger_service import PortfolioLedgerService
from tools.portfolio.services.dime_sync_service import (
    DimeSyncService,
    _load_sync_history,
    _save_sync_history,
)
from tools.portfolio.ports.trade_ingestion_port import (
    TradeEmailSourcePort,
    TradeDocumentParserPort,
    TradeDocumentMetadata,
)


def _create_sample_item(symbol="AAPL", conf_no="CONF123", order_id="ORD1", units="10", price="150.00", net="1500.00"):
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
        fingerprint=f"fp_{conf_no}_{symbol}_{order_id}",
        cash_adjusted=True,
    )


class MockEmailSource(TradeEmailSourcePort):
    def __init__(self, emails=None, pdf_bytes=b"dummy"):
        self.emails = emails or []
        self.pdf_bytes = pdf_bytes

    def search_dime_emails(self, query="", limit=20, since_date=None):
        return self.emails

    def search_wealthx_emails(self, query="", limit=None, since_date=None):
        return self.emails

    def search_scbam_emails(self, query="", limit=None):
        return self.emails

    def fetch_pdf_attachment(self, message_id, attachment_id):
        return self.pdf_bytes

    def fetch_email_html_body(self, message_id):
        return "<html></html>"


class MockParser(TradeDocumentParserPort):
    def __init__(self, items=None):
        self.items = items or []

    def parse_confirmation_pdf(self, pdf_bytes, password=None):
        return list(self.items)


def test_inmemory_staging_adapter_provenance():
    staging = InMemoryStagingAdapter()
    items = [_create_sample_item()]
    scan_id = staging.stage_items(items, session_id="sess_1")

    # Staging provenance
    prov = {"account_email": "test@test.com", "emails": {"msg_1": {"status": "staged"}}}
    staging.stage_provenance(scan_id, prov, session_id="sess_1")

    # Get provenance
    retrieved = staging.get_provenance(scan_id, session_id="sess_1")
    assert retrieved == prov

    # Pop provenance
    popped = staging.pop_provenance(scan_id, session_id="sess_1")
    assert popped == prov

    # Second pop returns None
    assert staging.pop_provenance(scan_id, session_id="sess_1") is None


def test_extract_active_ledger_identities_filters_voided():
    rows = [
        # Active trade
        {"Transaction_ID": "tx_1", "Confirmation_No": "C1", "Order_ID": "O1", "Action": "BUY", "Symbol": "NVDA"},
        # Voided trade and its reversal
        {"Transaction_ID": "tx_2", "Confirmation_No": "C2", "Order_ID": "O2", "Action": "BUY", "Symbol": "AAPL"},
        {"Transaction_ID": "tx_3", "Confirmation_No": "C2", "Order_ID": "O2", "Action": "VOID_BUY", "Related_Transaction_ID": "tx_2", "Symbol": "AAPL"},
        # Active trade
        {"Transaction_ID": "tx_4", "Confirmation_No": "C3", "Order_ID": "O3", "Action": "BUY", "Symbol": "TSLA"},
    ]

    active = extract_active_ledger_identities(rows)
    assert ("C1", "O1") in active
    assert ("C3", "O3") in active
    # C2, O2 was voided, must NOT be in active map
    assert ("C2", "O2") not in active


def test_reimport_after_transaction_void():
    """Verify that a trade can be re-imported after being voided (Requirement: ลบแล้วดึงกลับมาใหม่ได้)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        vault = Path(tmpdir)
        import os
        os.environ["OBSIDIAN_VAULT_PATH"] = str(vault)

        repo = MarkdownVaultRepositoryAdapter()
        importer = BatchTradeImportService(repo=repo)
        ledger_svc = PortfolioLedgerService(repo=repo)

        item = _create_sample_item(symbol="AAPL", conf_no="CONF_AAPL", order_id="ORD_AAPL_1", units="10", price="150.00", net="1500.00")

        # 1. Initial import
        state = importer.execute_batch_import([item], "default")
        aapl = next(h for h in state.holdings if h.symbol == "AAPL")
        assert aapl.units == 10.0

        with repo.unit_of_work("default") as uow:
            rows = uow.read_trade_log_locked()
            tx_id = rows[0]["Transaction_ID"]

        # 2. Second import while active -> Skipped as duplicate
        state2 = importer.execute_batch_import([item], "default")
        with repo.unit_of_work("default") as uow:
            rows2 = uow.read_trade_log_locked()
            assert len(rows2) == 1  # Not duplicated

        # 3. User voids the transaction
        state3 = ledger_svc.void_transaction(tx_id=tx_id, portfolio_id="default")
        # Holdings for AAPL should be 0 (removed)
        assert not any(h.symbol == "AAPL" for h in state3.holdings)

        with repo.unit_of_work("default") as uow:
            rows3 = uow.read_trade_log_locked()
            assert len(rows3) == 2  # Original BUY + VOID_BUY

        # 4. User re-syncs and re-imports the item!
        # Because it was voided, extract_active_ledger_identities considers it vacated
        state4 = importer.execute_batch_import([item], "default")
        aapl_restored = next(h for h in state4.holdings if h.symbol == "AAPL")
        assert aapl_restored.units == 10.0  # Successfully restored to 10 units!

        with repo.unit_of_work("default") as uow:
            rows4 = uow.read_trade_log_locked()
            assert len(rows4) == 3  # Original BUY + VOID_BUY + New Restored BUY!


def test_dime_sync_dynamic_reconciliation_and_skip_active():
    """Verify that DimeSyncService skips active items and automatically re-syncs voided items."""
    with tempfile.TemporaryDirectory() as tmpdir:
        vault = Path(tmpdir)
        import os
        os.environ["OBSIDIAN_VAULT_PATH"] = str(vault)

        repo = MarkdownVaultRepositoryAdapter()
        staging = InMemoryStagingAdapter()
        importer = BatchTradeImportService(repo=repo)
        ledger_svc = PortfolioLedgerService(repo=repo)

        item1 = _create_sample_item(symbol="AAPL", conf_no="CONF_1", order_id="ORD_1")
        item2 = _create_sample_item(symbol="NVDA", conf_no="CONF_2", order_id="ORD_2")

        # Pre-import item1 into the ledger
        importer.execute_batch_import([item1], "default")

        email_meta1 = TradeDocumentMetadata(
            message_id="msg_1",
            attachment_id="att_1",
            subject="Dime Trade 1",
            sender="dime.co.th",
            received_at="2026-09-01",
            filename="dime1.pdf",
            size_bytes=1000,
            account_email="test@dime.co.th",
        )
        email_meta2 = TradeDocumentMetadata(
            message_id="msg_2",
            attachment_id="att_2",
            subject="Dime Trade 2",
            sender="dime.co.th",
            received_at="2026-09-02",
            filename="dime2.pdf",
            size_bytes=1000,
            account_email="test@dime.co.th",
        )

        email_source = MockEmailSource(emails=[email_meta1, email_meta2])

        class MultiParser(TradeDocumentParserPort):
            def parse_confirmation_pdf(self, pdf_bytes, password=None):
                # Return item1 or item2
                if pdf_bytes == b"att1_bytes":
                    return [item1]
                return [item2]

        parser = MockParser([item1]) # default

        # Sync service
        svc = DimeSyncService(
            email_source=email_source,
            parser=parser,
            staging=staging,
            batch_import_service=importer,
        )

        # Run stream_batch_sync: item1 is already active in ledger, so it should NOT be staged!
        events = list(svc.stream_batch_sync(portfolio_id="default"))
        complete_event = next(e for e in events if e.get("event") == "complete")
        data = complete_event["data"]

        # Item 1 was skipped because it's already active in ledger!
        assert data["item_count"] == 0
        assert data["already_in_portfolio_count"] > 0

        # Now test Void & Dynamic Reconciliation:
        # 1. Void item1
        with repo.unit_of_work("default") as uow:
            rows = uow.read_trade_log_locked()
            tx_id = rows[0]["Transaction_ID"]
        ledger_svc.void_transaction(tx_id, "default")

        # 2. Rescan: svc must detect that ORD_1 is now missing from active ledger and stage it!
        events2 = list(svc.stream_batch_sync(portfolio_id="default"))
        complete_event2 = next(e for e in events2 if e.get("event") == "complete")
        data2 = complete_event2["data"]

        assert data2["item_count"] == 1
        assert data2["items"][0]["order_id"] == "ORD_1"

        # 3. Commit staged: cross-request commit via scan_id
        # Creates a new DimeSyncService instance to simulate new FastAPI request
        new_svc_instance = DimeSyncService(
            email_source=email_source,
            parser=parser,
            staging=staging,  # shared singleton staging adapter
            batch_import_service=importer,
        )

        committed_state = new_svc_instance.commit_staged(scan_id=data2["scan_id"], portfolio_id="default")
        # Restored!
        aapl = next(h for h in committed_state.holdings if h.symbol == "AAPL")
        assert aapl.units == 10.0


def test_remove_holding_voids_transactions_and_allows_resync():
    """Verify that deleting a holding from Holdings tab voids its transactions and enables trade resync."""
    with tempfile.TemporaryDirectory() as tmpdir:
        vault = Path(tmpdir)
        import os
        os.environ["OBSIDIAN_VAULT_PATH"] = str(vault)

        repo = MarkdownVaultRepositoryAdapter()
        staging = InMemoryStagingAdapter()
        importer = BatchTradeImportService(repo=repo)
        trading_svc = PortfolioTradingService(repo=repo, price_provider=None, journal_provider=None)

        item = _create_sample_item(symbol="AAPL", conf_no="CONF_AAPL_1", order_id="ORD_AAPL_1", units="10", price="150.00", net="1500.00")

        # 1. Import trade into portfolio
        state = importer.execute_batch_import([item], "default")
        assert any(h.symbol == "AAPL" for h in state.holdings)

        with repo.unit_of_work("default") as uow:
            rows = uow.read_trade_log_locked()
            assert len(rows) == 1
            assert rows[0]["Action"] == "BUY"

        # 2. User deletes AAPL from Holdings tab (calls structured_remove_holding)
        state_after_delete = trading_svc.structured_remove_holding("AAPL", "default")
        assert not any(h.symbol == "AAPL" for h in state_after_delete.holdings)

        # 3. Assert transactions: hard-purged on holding removal!
        with repo.unit_of_work("default") as uow:
            rows_after_delete = uow.read_trade_log_locked()
            assert len(rows_after_delete) == 0

            # Active identities must now be empty!
            active_ids = extract_active_ledger_identities(rows_after_delete)
            assert ("CONF_AAPL_1", "ORD_AAPL_1") not in active_ids

        # 4. Now user re-syncs email: sync must detect ORD_AAPL_1 as missing and allow re-import!
        email_meta = TradeDocumentMetadata(
            message_id="msg_aapl",
            attachment_id="att_aapl",
            subject="Dime AAPL Trade",
            sender="dime.co.th",
            received_at="2026-09-01",
            filename="dime_aapl.pdf",
            size_bytes=1000,
            account_email="test@dime.co.th",
        )
        email_source = MockEmailSource(emails=[email_meta])
        parser = MockParser([item])
        sync_svc = DimeSyncService(
            email_source=email_source,
            parser=parser,
            staging=staging,
            batch_import_service=importer,
        )

        events = list(sync_svc.stream_batch_sync(portfolio_id="default"))
        complete_event = next(e for e in events if e.get("event") == "complete")
        data = complete_event["data"]

        # Item is now recognized as missing and staged!
        assert data["item_count"] == 1
        assert data["items"][0]["order_id"] == "ORD_AAPL_1"

        # 5. Commit staged restores the holding
        restored_state = sync_svc.commit_staged(data["scan_id"], portfolio_id="default")
        restored_aapl = next(h for h in restored_state.holdings if h.symbol == "AAPL")
        assert restored_aapl.units == 10.0

