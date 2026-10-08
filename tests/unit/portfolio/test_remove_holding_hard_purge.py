import tempfile
from decimal import Decimal
from pathlib import Path
import pytest

from tools.portfolio.domain.models import (
    TradeImportItem,
    TradeFeeBreakdown,
    Holding,
)
from tools.portfolio.adapters.markdown.repository_adapter import MarkdownVaultRepositoryAdapter
from tools.portfolio.services.batch_trade_import_service import BatchTradeImportService
from tools.portfolio.services.trading_service import PortfolioTradingService
from tools.portfolio.domain.calculations import recalc_all


def test_remove_holding_hard_purges_transactions_and_refunds_cash(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp_dir:
        vault = Path(tmp_dir)
        monkeypatch.setenv("OBSIDIAN_VAULT_PATH", str(vault))

        repo = MarkdownVaultRepositoryAdapter()
        trading_svc = PortfolioTradingService(repo=repo, price_provider=None, journal_provider=None)
        importer = BatchTradeImportService(repo=repo)

        # Deposit initial cash
        with repo.unit_of_work("default") as uow:
            st = uow.load_state()
            cash_thb = next((h for h in st.holdings if h.symbol == "CASH_THB"), None)
            if cash_thb:
                cash_thb.units = 50000.0
            else:
                st.holdings.append(Holding(symbol="CASH_THB", asset_type="Cash", units=50000.0, market_value_thb=50000.0))
            uow.commit(st)

        # Import 2 trades: one for KT-FUND, one for OTHER-STOCK
        item_kt = TradeImportItem(
            item_id="item_kt",
            trade_date="2026-09-01",
            symbol="KT-FUND",
            action="BUY",
            units=Decimal("100"),
            price=Decimal("100.00"),
            gross_amount=Decimal("10000.00"),
            fees=TradeFeeBreakdown(commission=Decimal("0.00"), vat=Decimal("0.00"), other_fees=Decimal("0.00"), fee_currency="THB"),
            net_amount=Decimal("10000.00"),
            currency="THB",
            confirmation_no="CONF_KT_1",
            order_id="ORD_KT_1",
            source="DIME",
            fingerprint="fp_kt",
            cash_adjusted=True,
        )
        item_other = TradeImportItem(
            item_id="item_other",
            trade_date="2026-09-01",
            symbol="OTHER",
            action="BUY",
            units=Decimal("10"),
            price=Decimal("500.00"),
            gross_amount=Decimal("5000.00"),
            fees=TradeFeeBreakdown(commission=Decimal("0.00"), vat=Decimal("0.00"), other_fees=Decimal("0.00"), fee_currency="THB"),
            net_amount=Decimal("5000.00"),
            currency="THB",
            confirmation_no="CONF_OTHER_1",
            order_id="ORD_OTHER_1",
            source="DIME",
            fingerprint="fp_other",
            cash_adjusted=True,
        )
        importer.execute_batch_import([item_kt, item_other], portfolio_id="default")

        # Verify before removal
        with repo.unit_of_work("default") as uow:
            st = uow.load_state()
            assert any(h.symbol == "KT-FUND" for h in st.holdings)
            assert any(h.symbol == "OTHER" for h in st.holdings)
            cash = next(h for h in st.holdings if h.symbol == "CASH_THB")
            assert cash.units == 35000.0  # 50,000 - 10,000 - 5,000
            rows = uow.read_trade_log_locked()
            assert len(rows) == 2

        # User removes KT-FUND from Holdings
        updated_state = trading_svc.structured_remove_holding("KT-FUND", portfolio_id="default")

        # Verify holding is gone
        assert not any(h.symbol == "KT-FUND" for h in updated_state.holdings)
        assert any(h.symbol == "OTHER" for h in updated_state.holdings)

        # Cash was refunded
        cash_after = next(h for h in updated_state.holdings if h.symbol == "CASH_THB")
        assert cash_after.units == 45000.0  # 35,000 + 10,000 refunded

        # Verify ledger: KT-FUND trade rows are completely hard-purged!
        with repo.unit_of_work("default") as uow:
            rows_after = uow.read_trade_log_locked()
            assert len(rows_after) == 1
            assert rows_after[0]["Symbol"] == "OTHER"
            assert not any(r.get("Symbol") == "KT-FUND" for r in rows_after)


def test_remove_holding_cleans_up_orphaned_transactions(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp_dir:
        vault = Path(tmp_dir)
        monkeypatch.setenv("OBSIDIAN_VAULT_PATH", str(vault))

        repo = MarkdownVaultRepositoryAdapter()
        trading_svc = PortfolioTradingService(repo=repo, price_provider=None, journal_provider=None)
        importer = BatchTradeImportService(repo=repo)

        item = TradeImportItem(
            item_id="item_orphan",
            trade_date="2026-09-01",
            symbol="ORPHAN",
            action="BUY",
            units=Decimal("10"),
            price=Decimal("100.00"),
            gross_amount=Decimal("1000.00"),
            fees=TradeFeeBreakdown(commission=Decimal("0.00"), vat=Decimal("0.00"), other_fees=Decimal("0.00"), fee_currency="THB"),
            net_amount=Decimal("1000.00"),
            currency="THB",
            confirmation_no="CONF_ORPHAN",
            order_id="ORD_ORPHAN",
            source="DIME",
            fingerprint="fp_orphan",
            cash_adjusted=False,
        )
        importer.execute_batch_import([item], portfolio_id="default")

        # Simulate holding removed manually from state.holdings (leaving orphaned trade in ledger)
        with repo.unit_of_work("default") as uow:
            st = uow.load_state()
            orphan_holding = next(h for h in st.holdings if h.symbol == "ORPHAN")
            st.holdings.remove(orphan_holding)
            uow.commit(st)

        # Call structured_remove_holding on the orphaned symbol
        state = trading_svc.structured_remove_holding("ORPHAN", portfolio_id="default")
        assert not any(h.symbol == "ORPHAN" for h in state.holdings)

        with repo.unit_of_work("default") as uow:
            rows = uow.read_trade_log_locked()
            assert len(rows) == 0
