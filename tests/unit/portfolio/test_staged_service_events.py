"""Verify portfolio mutation services stage system journal entries atomically."""
from unittest.mock import MagicMock

from tools.portfolio.adapters.markdown.repository_adapter import MarkdownVaultRepositoryAdapter
from tools.portfolio.ports.journal_port import TradeJournalPort
from tools.portfolio.ports.price_port import MarketPricePort
from tools.portfolio.services.cash_flow_service import PortfolioCashFlowService
from tools.portfolio.services.trading_service import PortfolioTradingService


def _services(tmp_path, monkeypatch):
    monkeypatch.setenv("OBSIDIAN_VAULT_PATH", str(tmp_path / "vault"))
    repo = MarkdownVaultRepositoryAdapter()
    price = MagicMock(spec=MarketPricePort)
    # A failing direct system-journal implementation must not be consulted:
    # system events are staged in the PortfolioUnitOfWork instead.
    journal = MagicMock(spec=TradeJournalPort)
    journal.append_system_entry.side_effect = AssertionError("system journal bypassed UoW")
    return (
        PortfolioCashFlowService(repo, price, journal_provider=journal),
        PortfolioTradingService(repo, price, journal),
        repo,
    )


def test_cash_flow_and_trade_stage_journal_with_state(tmp_path, monkeypatch):
    cash_flow, trading, repo = _services(tmp_path, monkeypatch)

    cash_flow.structured_manage_cash_flow(1000.0, "deposit", "THB", date="2026-08-23")
    trading.structured_execute_trade(
        symbol="AAPL",
        asset_type="Stock",
        action="buy",
        units=2,
        price=100.0,
        currency="THB",
        date="2026-08-23",
        notes="staged",
    )

    # Resolve through the adapter's canonical path helper rather than relying
    # on any process-global path captured at import time.
    from tools.portfolio.adapters.markdown.paths import get_journal_filepath

    content = get_journal_filepath("default").read_text(encoding="utf-8")
    assert "CASH FLOW NOTE" in content
    assert "TRADE NOTE" in content
    assert "2026-08-23 12:00:00" in content
    assert "AAPL" in content
    assert trading.journal_provider.append_system_entry.call_count == 0
