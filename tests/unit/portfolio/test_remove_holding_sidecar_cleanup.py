import tempfile
from decimal import Decimal
from pathlib import Path
import pytest

from tools.portfolio.domain.models import (
    TradeImportItem,
    TradeFeeBreakdown,
)
from tools.portfolio.adapters.markdown.paths import get_holdings_dir
from tools.portfolio.adapters.markdown.repository_adapter import MarkdownVaultRepositoryAdapter
from tools.portfolio.services.batch_trade_import_service import BatchTradeImportService
from tools.portfolio.services.trading_service import PortfolioTradingService


def test_remove_holding_unlinks_sidecar(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp_dir:
        vault = Path(tmp_dir)
        monkeypatch.setenv("OBSIDIAN_VAULT_PATH", str(vault))

        repo = MarkdownVaultRepositoryAdapter()
        trading_svc = PortfolioTradingService(repo=repo, price_provider=None, journal_provider=None)
        importer = BatchTradeImportService(repo=repo)

        # Import trade for AAPL and MSFT
        item_aapl = TradeImportItem(
            item_id="item_aapl",
            trade_date="2026-09-01",
            symbol="AAPL",
            action="BUY",
            units=Decimal("10"),
            price=Decimal("150.00"),
            gross_amount=Decimal("1500.00"),
            fees=TradeFeeBreakdown(commission=Decimal("0.00"), vat=Decimal("0.00"), other_fees=Decimal("0.00"), fee_currency="USD"),
            net_amount=Decimal("1500.00"),
            currency="USD",
            confirmation_no="CONF_AAPL",
            order_id="ORD_AAPL",
            source="DIME",
            fingerprint="fp_aapl",
            cash_adjusted=True,
        )
        item_msft = TradeImportItem(
            item_id="item_msft",
            trade_date="2026-09-01",
            symbol="MSFT",
            action="BUY",
            units=Decimal("5"),
            price=Decimal("300.00"),
            gross_amount=Decimal("1500.00"),
            fees=TradeFeeBreakdown(commission=Decimal("0.00"), vat=Decimal("0.00"), other_fees=Decimal("0.00"), fee_currency="USD"),
            net_amount=Decimal("1500.00"),
            currency="USD",
            confirmation_no="CONF_MSFT",
            order_id="ORD_MSFT",
            source="DIME",
            fingerprint="fp_msft",
            cash_adjusted=True,
        )
        importer.execute_batch_import([item_aapl, item_msft], portfolio_id="default")

        holdings_dir = get_holdings_dir("default")
        aapl_file = holdings_dir / "AAPL.md"
        msft_file = holdings_dir / "MSFT.md"
        assert aapl_file.exists(), "AAPL sidecar should exist after import"
        assert msft_file.exists(), "MSFT sidecar should exist after import"

        # Explicitly remove AAPL
        trading_svc.structured_remove_holding("AAPL", portfolio_id="default")
        assert not aapl_file.exists(), "AAPL sidecar must be unlinked/deleted from disk upon structured_remove_holding"
        assert msft_file.exists(), "MSFT sidecar must remain intact"

        # Explicitly batch remove MSFT
        trading_svc.structured_batch_remove_holdings(["MSFT"], portfolio_id="default")
        assert not msft_file.exists(), "MSFT sidecar must be unlinked/deleted from disk upon batch remove"
