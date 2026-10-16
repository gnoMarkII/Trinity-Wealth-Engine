from datetime import date
from decimal import Decimal
import hashlib

import pytest

from tools.archivist.vault_paths import VaultPaths


def _statement(as_of: str):
    from tools.portfolio.domain.pvd_models import PvdContributionSummary, PvdStatement, PvdSubFundHolding

    member_principal = Decimal("600.00")
    member_benefit = Decimal("200.00")
    employer_principal = Decimal("300.00")
    employer_benefit = Decimal("150.00")
    return PvdStatement(
        member_id="1234567890",
        member_hash=hashlib.sha256(b"1234567890").hexdigest(),
        masked_member_id="••••7890",
        as_of_date=date.fromisoformat(as_of),
        summary=PvdContributionSummary(
            member_principal=member_principal,
            member_benefit=member_benefit,
            employer_principal=employer_principal,
            employer_benefit=employer_benefit,
            total_contribution=Decimal("900.00"),
            total_nav=Decimal("1250.00"),
            employer_vesting_percent=Decimal("100"),
            member_vested_amount=Decimal("800.00"),
            employer_vested_amount=Decimal("450.00"),
        ),
        holdings=[PvdSubFundHolding(
            code="SINSA_FI", name="Fixed Income", units=Decimal("123.4567"), statement_nav=Decimal("10.1234"),
            statement_market_value=Decimal("1250.00"), weight_percent=Decimal("100"),
        )],
    )


class Parser:
    def __init__(self, statements):
        self.statements = iter(statements)

    def parse_statement(self, _pdf_bytes, password=None):
        assert password in (None, " 1234 ")
        return next(self.statements).model_copy(deep=True)


@pytest.fixture
def pvd_runtime(tmp_path, monkeypatch):
    # Legacy portfolio fixtures reload tools.portfolio modules between tests.
    # Compose every dependency at runtime so mutation/model class identities
    # come from the same module generation when this file follows those tests.
    from tools.portfolio.adapters.bblam_pvd.local_filesystem_pvd_archive_adapter import LocalFilesystemPvdArchiveAdapter
    from tools.portfolio.adapters.dime.inmemory_staging_adapter import InMemoryStagingAdapter
    from tools.portfolio.adapters.markdown.repository_adapter import MarkdownVaultRepositoryAdapter
    from tools.portfolio.transaction_store import PortfolioTransactionStore
    from tools.portfolio.transactional_repository import TransactionalPortfolioRepository

    vault = tmp_path / "vault"
    vault.mkdir()
    monkeypatch.setenv("OBSIDIAN_VAULT_PATH", str(vault))
    from tools.portfolio.adapters.markdown import paths
    monkeypatch.setattr(paths, "VAULT_PATH", vault)
    monkeypatch.setattr(paths, "PORTFOLIOS_DIR", vault / "20_Portfolio_Management/Current_Holdings/Portfolios")
    markdown = MarkdownVaultRepositoryAdapter()
    store = PortfolioTransactionStore(vault_paths=VaultPaths(vault), runtime_root=tmp_path / "runtime")
    repo = TransactionalPortfolioRepository(markdown, store=store)
    repo.create_portfolio("PVD test", portfolio_id="pvd_test")
    staging = InMemoryStagingAdapter()
    archive = LocalFilesystemPvdArchiveAdapter(root=tmp_path / "attachments" / "pvd")
    return tmp_path, repo, store, staging, archive


def test_statement_commit_reloads_exact_values_and_receipt_retry_returns_current_state(pvd_runtime):
    from tools.portfolio.services.pvd_sync_service import PvdSyncService

    tmp_path, repo, _store, staging, archive = pvd_runtime
    service = PvdSyncService(
        repo=repo,
        parser=Parser([_statement("2026-09-30")]),
        archive=archive,
        staging=staging,
    )
    pdf_bytes = b"%PDF-pvd-integration"
    scan = service.analyze_upload(pdf_bytes, portfolio_id="pvd_test", session_id="sid")
    committed = service.commit(scan.scan_id, portfolio_id="pvd_test", session_id="sid", idempotency_key="commit-1", vesting_percent=Decimal("50"))
    assert committed.status == "committed"

    restarted = type(repo)(repo.underlying_repo, store=repo.store)
    state = restarted.load_state("pvd_test")
    holding = next(h for h in state.holdings if h.symbol == "PVD_SINSA_FI")
    assert holding.units == 123.4567
    assert holding.market_value_thb == 1250.0
    assert state.summary.total_cost_basis_thb == 900.0
    assert state.pvd_statements[0].employer_vesting_percent == Decimal("50")
    assert state.pvd_statements[0].employer_vested_amount == Decimal("225.00")

    ledger_row = next(row for row in restarted.read_trade_log("pvd_test") if row.get("Action") == "PVD_SNAPSHOT")
    assert ledger_row["Cash_Adjusted"] == "NO"
    assert "1234567890" not in ledger_row["PVD_Metadata_JSON"]
    assert archive.read(portfolio_id="pvd_test", metadata=state.pvd_statements[0].model_dump(mode="json")) == pdf_bytes

    # The scan data has been deleted, but the durable receipt still makes the retry safe.
    retry_service = PvdSyncService(repo=restarted, parser=Parser([]), archive=archive, staging=staging)
    retry = retry_service.commit(scan.scan_id, portfolio_id="pvd_test", session_id="sid", idempotency_key="commit-1", vesting_percent=Decimal("50"))
    assert retry.status == "duplicate"
    assert retry.state["summary"]["total_value_thb"] == 1250.0


def test_full_reset_rotates_epoch_and_invalidates_staged_scan(pvd_runtime):
    from tools.portfolio.services.pvd_sync_service import PvdSyncService

    _tmp_path, repo, _store, staging, archive = pvd_runtime
    service = PvdSyncService(repo=repo, parser=Parser([_statement("2026-09-30")]), archive=archive, staging=staging)
    scan = service.analyze_upload(b"%PDF-before-reset", portfolio_id="pvd_test", session_id="sid")
    repo.backup_and_reset_clean_slate("pvd_test")
    with pytest.raises(ValueError, match="reset"):
        service.commit(scan.scan_id, portfolio_id="pvd_test", session_id="sid", idempotency_key="old-scan", vesting_percent=Decimal("100"))
    state = repo.load_state("pvd_test")
    assert state.pvd_import_epoch
    assert not state.pvd_statements
    assert not any(h.asset_type == "PVD" for h in state.holdings)


def _seed_mixed_portfolio(repo):
    from tools.portfolio.domain.calculations import recalc_all
    from tools.portfolio.domain.models import Holding
    from tools.portfolio.services.trading_service import PortfolioTradingService

    with repo.unit_of_work("pvd_test") as uow:
        state = uow.load_state()
        state.fx_rates["USDTHB"] = 35.0
        state.holdings = [
            Holding(symbol="CASH_THB", asset_type="Cash", units=10000),
            Holding(symbol="CASH_USD", asset_type="Cash", units=1000),
        ]
        recalc_all(state)
        uow.commit(state)
    # Explicit prices and FX keep this compatibility test offline.
    trading = PortfolioTradingService(repo, price_provider=None, journal_provider=None)
    trading.structured_execute_trade("PTT", "Stock", "buy", 100, 35, date="2026-09-01", portfolio_id="pvd_test")
    trading.structured_execute_trade("PTT", "Stock", "sell", 20, 40, date="2026-09-02", portfolio_id="pvd_test")
    trading.structured_execute_trade("AAPL", "Stock", "buy", 2, 200, currency="USD", exchange_rate=35, date="2026-09-03", portfolio_id="pvd_test")
    return trading


def test_two_pvd_imports_preserve_existing_thb_usd_trades_cash_and_state_contract(pvd_runtime):
    from api.schemas.portfolio import ActualPortfolioStateDTO, TransactionItemDTO
    from tools.portfolio.domain.calculations import recalc_all
    from tools.portfolio.services.pvd_sync_service import PvdSyncService

    _tmp_path, repo, store, staging, archive = pvd_runtime
    _seed_mixed_portfolio(repo)
    baseline = repo.load_state("pvd_test")
    baseline_holdings = {h.symbol: h.model_dump(mode="json") for h in baseline.holdings}
    baseline_rows = repo.read_trade_log("pvd_test")
    newer = _statement("2026-10-31")
    newer.summary.member_benefit = Decimal("350.00")
    newer.summary.employer_benefit = Decimal("250.00")
    newer.summary.member_vested_amount = Decimal("950.00")
    newer.summary.employer_vested_amount = Decimal("550.00")
    newer.summary.total_nav = Decimal("1500.00")
    newer.holdings[0].units = Decimal("150")
    newer.holdings[0].statement_nav = Decimal("10")
    newer.holdings[0].statement_market_value = Decimal("1500.00")
    service = PvdSyncService(repo=repo, parser=Parser([_statement("2026-09-30"), newer]), archive=archive, staging=staging)

    for index, expected_value in enumerate((1250.0, 1500.0), start=1):
        scan = service.analyze_upload(f"%PDF-mixed-{index}".encode(), portfolio_id="pvd_test", session_id="sid")
        result = service.commit(scan.scan_id, portfolio_id="pvd_test", session_id="sid", idempotency_key=f"mixed-{index}", vesting_percent=Decimal("100"))
        assert result.status == "committed"
        restarted = type(repo)(repo.underlying_repo, store=store)
        state = restarted.load_state("pvd_test")
        recalc_all(state)
        assert {h.symbol: h.model_dump(mode="json") for h in state.holdings if h.asset_type != "PVD"} == baseline_holdings
        assert state.summary.total_value_thb == baseline.summary.total_value_thb + expected_value
        assert state.summary.total_cost_basis_thb == baseline.summary.total_cost_basis_thb + 900.0
        assert state.summary.total_realized_profit_ytd == baseline.summary.total_realized_profit_ytd == 100.0
        assert next(h for h in state.holdings if h.symbol == "CASH_THB").units == 7300
        assert next(h for h in state.holdings if h.symbol == "CASH_USD").units == 600
        rows = restarted.read_trade_log("pvd_test")
        assert [row for row in rows if row["Action"] != "PVD_SNAPSHOT"] == baseline_rows
        assert sum(row["Action"] == "PVD_SNAPSHOT" for row in rows) == index
        wire_state = ActualPortfolioStateDTO.model_validate(state.model_dump(mode="json"))
        assert wire_state.summary.total_value_thb == state.summary.total_value_thb
        assert len(wire_state.pvd_statements) == index
        wire_pvd = next(h for h in wire_state.holdings if h.asset_type == "PVD")
        assert wire_pvd.pvd_valuation.statement_market_value == Decimal(str(expected_value))
        assert all(TransactionItemDTO.model_validate(row) for row in rows)


def test_existing_note_edit_void_and_cash_flow_preserve_committed_pvd(pvd_runtime):
    from tools.portfolio.services.cash_flow_service import PortfolioCashFlowService
    from tools.portfolio.services.ledger_service import PortfolioLedgerService
    from tools.portfolio.services.pvd_sync_service import PvdSyncService

    _tmp_path, repo, _store, staging, archive = pvd_runtime
    _seed_mixed_portfolio(repo)
    service = PvdSyncService(repo=repo, parser=Parser([_statement("2026-09-30")]), archive=archive, staging=staging)
    scan = service.analyze_upload(b"%PDF-mutations", portfolio_id="pvd_test", session_id="sid")
    service.commit(scan.scan_id, portfolio_id="pvd_test", session_id="sid", idempotency_key="mutations", vesting_percent=Decimal("50"))
    baseline = repo.load_state("pvd_test")
    baseline_pvd = next(h for h in baseline.holdings if h.asset_type == "PVD").model_dump(mode="json")
    baseline_pvd_rows = [row for row in repo.read_trade_log("pvd_test") if row["Action"] == "PVD_SNAPSHOT"]
    sell_id = next(row["Transaction_ID"] for row in repo.read_trade_log("pvd_test") if row["Action"] == "SELL")
    ledger = PortfolioLedgerService(repo)

    updated = ledger.update_trade_note(sell_id, "Keep ordinary portfolio workflow", portfolio_id="pvd_test")
    assert updated["Notes"] == "Keep ordinary portfolio workflow"
    edited = ledger.edit_transaction(sell_id, price=45, portfolio_id="pvd_test")
    assert next(h for h in edited.holdings if h.symbol == "CASH_THB").units == 7400
    assert edited.summary.total_realized_profit_ytd == 200
    voided = ledger.void_transaction(sell_id, portfolio_id="pvd_test")
    assert next(h for h in voided.holdings if h.symbol == "PTT").units == 100
    assert next(h for h in voided.holdings if h.symbol == "CASH_THB").units == 6500
    assert voided.summary.total_realized_profit_ytd == 0
    PortfolioCashFlowService(repo, price_provider=None).structured_manage_cash_flow(500, "deposit", portfolio_id="pvd_test")
    state = type(repo)(repo.underlying_repo, store=repo.store).load_state("pvd_test")
    assert next(h for h in state.holdings if h.symbol == "CASH_THB").units == 7000
    assert next(h for h in state.holdings if h.asset_type == "PVD").model_dump(mode="json") == baseline_pvd
    assert state.pvd_statements == baseline.pvd_statements
    assert [row for row in repo.read_trade_log("pvd_test") if row["Action"] == "PVD_SNAPSHOT"] == baseline_pvd_rows
    assert any(row["Action"] == "VOID_SELL" for row in repo.read_trade_log("pvd_test"))


def test_pvd_guards_leave_mixed_portfolio_unchanged(pvd_runtime):
    from tools.portfolio.services.ledger_service import PortfolioLedgerService
    from tools.portfolio.services.portfolio_state_service import PortfolioStateService
    from tools.portfolio.services.pvd_sync_service import PvdSyncService

    _tmp_path, repo, store, staging, archive = pvd_runtime
    trading = _seed_mixed_portfolio(repo)
    service = PvdSyncService(repo=repo, parser=Parser([_statement("2026-09-30")]), archive=archive, staging=staging)
    scan = service.analyze_upload(b"%PDF-guards", portfolio_id="pvd_test", session_id="sid")
    service.commit(scan.scan_id, portfolio_id="pvd_test", session_id="sid", idempotency_key="guards", vesting_percent=Decimal("100"))
    before = repo.load_state("pvd_test").model_dump(mode="json")
    before_rows = repo.read_trade_log("pvd_test")
    before_sequence = store.checkpoint("pvd_test").sequence
    pvd_id = next(row["Transaction_ID"] for row in before_rows if row["Action"] == "PVD_SNAPSHOT")
    ledger = PortfolioLedgerService(repo)
    state_service = PortfolioStateService(repo)

    operations = (
        lambda: trading.structured_execute_trade("PVD_SINSA_FI", "PVD", "sell", 1, 10, portfolio_id="pvd_test"),
        lambda: ledger.edit_transaction(pvd_id, price=11, portfolio_id="pvd_test"),
        lambda: ledger.void_transaction(pvd_id, portfolio_id="pvd_test"),
        lambda: trading.structured_batch_remove_holdings(["PTT", "PVD_SINSA_FI"], portfolio_id="pvd_test"),
        lambda: state_service.structured_batch_assign_holding_buckets(["PTT", "PVD_SINSA_FI"], "cash", portfolio_id="pvd_test"),
    )
    for operation in operations:
        with pytest.raises(ValueError, match="PVD"):
            operation()
        assert repo.load_state("pvd_test").model_dump(mode="json") == before
        assert repo.read_trade_log("pvd_test") == before_rows
        assert store.checkpoint("pvd_test").sequence == before_sequence
