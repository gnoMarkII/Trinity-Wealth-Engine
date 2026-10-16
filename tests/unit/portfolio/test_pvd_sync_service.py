from copy import deepcopy
from datetime import date
from decimal import Decimal
import hashlib

from tools.portfolio.domain.ledger_change import LedgerChange
from tools.portfolio.domain.models import PortfolioState, _now_iso
from tools.portfolio.domain.pvd_models import PvdContributionSummary, PvdStatement, PvdSubFundHolding
from tools.portfolio.services.pvd_sync_service import PvdSyncService


def make_statement(as_of, units, nav, value, contribution):
    total_nav = Decimal(value)
    contribution = Decimal(contribution)
    member_principal = contribution * Decimal("0.6")
    employer_principal = contribution - member_principal
    benefit = total_nav - contribution
    return PvdStatement(
        member_id="1234567890",
        member_hash=hashlib.sha256(b"1234567890").hexdigest(),
        masked_member_id="••••7890",
        as_of_date=date.fromisoformat(as_of),
        summary=PvdContributionSummary(
            member_principal=member_principal,
            member_benefit=benefit * Decimal("0.6"),
            employer_principal=employer_principal,
            employer_benefit=benefit * Decimal("0.4"),
            total_contribution=contribution,
            total_nav=total_nav,
            employer_vesting_percent=Decimal("100"),
            member_vested_amount=member_principal + benefit * Decimal("0.6"),
            employer_vested_amount=employer_principal + benefit * Decimal("0.4"),
        ),
        holdings=[PvdSubFundHolding(
            code="SINSA_FI", name="Fixed Income", units=Decimal(units), statement_nav=Decimal(nav),
            statement_market_value=total_nav, weight_percent=Decimal("100"),
        )],
    )


class FakeParser:
    def __init__(self, statements):
        self.statements = list(statements)

    def parse_statement(self, pdf_bytes, password=None):
        return deepcopy(self.statements.pop(0))


class FakeArchive:
    def __init__(self):
        self.documents = {}

    def archive(self, pdf_bytes, *, portfolio_id, statement):
        archive_id = hashlib.sha256(pdf_bytes).hexdigest()
        self.documents[(portfolio_id, archive_id)] = pdf_bytes
        return archive_id

    def read(self, *, portfolio_id, metadata):
        return self.documents[(portfolio_id, metadata["archive_id"])]


class FakeStore:
    def __init__(self):
        self.receipts = {}

    def get_command_receipt(self, portfolio_id, use_case, idempotency_key):
        return deepcopy(self.receipts.get((portfolio_id, use_case, idempotency_key)))


class FakeUow:
    def __init__(self, repo):
        self.repo = repo

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def load_state(self):
        return deepcopy(self.repo.state)

    def read_trade_log_locked(self):
        return deepcopy(self.repo.rows)

    def commit(self, state, ledger_change=None, command_receipt=None):
        self.repo.state = deepcopy(state)
        if isinstance(ledger_change, LedgerChange) and ledger_change.kind == "replace_all":
            self.repo.rows = deepcopy(ledger_change.rows or [])
        if command_receipt:
            key = ("default", command_receipt["use_case"], command_receipt["idempotency_key"])
            self.repo.store.receipts[key] = {
                "request_hash": command_receipt["request_hash"],
                "result": deepcopy(command_receipt["result"]),
            }

    def rollback(self):
        return None


class FakeRepo:
    def __init__(self):
        self.state = PortfolioState(last_updated=_now_iso())
        self.rows = []
        self.store = FakeStore()

    def portfolio_exists(self, portfolio_id):
        return portfolio_id == "default"

    def load_state(self, portfolio_id="default"):
        return deepcopy(self.state)

    def unit_of_work(self, portfolio_id="default"):
        return FakeUow(self)


def test_snapshots_replace_units_without_cash_effect_and_retry_returns_current_state():
    repo, archive = FakeRepo(), FakeArchive()
    parser = FakeParser([
        make_statement("2026-09-30", "100.25", "10", "1000.00", "900.00"),
        make_statement("2026-10-31", "110.5", "10", "1200.00", "1000.00"),
    ])
    from tools.portfolio.adapters.dime.inmemory_staging_adapter import InMemoryStagingAdapter
    staging = InMemoryStagingAdapter()
    service = PvdSyncService(repo=repo, parser=parser, archive=archive, staging=staging)

    first = service.analyze_upload(b"%PDF-first", portfolio_id="default", session_id="session")
    committed = service.commit(first.scan_id, portfolio_id="default", session_id="session", idempotency_key="first", vesting_percent=Decimal("50"))
    assert committed.status == "committed"
    assert repo.state.summary.total_value_thb == 1000.0
    assert all(h.asset_type != "Cash" for h in repo.state.holdings)
    assert repo.rows[0]["Action"] == "PVD_SNAPSHOT"
    assert repo.rows[0]["Cash_Adjusted"] == "NO"
    assert next(h for h in repo.state.holdings if h.symbol == "PVD_SINSA_FI").units == 100.25

    retry = service.commit(first.scan_id, portfolio_id="default", session_id="session", idempotency_key="first", vesting_percent=Decimal("50"))
    assert retry.status == "duplicate"
    assert retry.state["summary"]["total_value_thb"] == 1000.0

    second = service.analyze_upload(b"%PDF-second", portfolio_id="default", session_id="session")
    service.commit(second.scan_id, portfolio_id="default", session_id="session", idempotency_key="second", vesting_percent=Decimal("50"))
    latest = next(h for h in repo.state.holdings if h.symbol == "PVD_SINSA_FI")
    assert latest.units == 110.5
    assert latest.market_value_thb == 1200.0
    assert repo.state.summary.total_cost_basis_thb == 1000.0
    assert len(repo.rows) == 2


def test_scan_is_session_and_portfolio_bound():
    repo, staging = FakeRepo(), __import__("tools.portfolio.adapters.dime.inmemory_staging_adapter", fromlist=["InMemoryStagingAdapter"]).InMemoryStagingAdapter()
    service = PvdSyncService(repo=repo, parser=FakeParser([make_statement("2026-09-30", "1", "10", "10", "9")]), archive=FakeArchive(), staging=staging)
    scan = service.analyze_upload(b"%PDF-first", portfolio_id="default", session_id="session-a")
    try:
        service.commit(scan.scan_id, portfolio_id="default", session_id="session-b", idempotency_key="x", vesting_percent=Decimal("100"))
    except Exception as exc:
        assert "session" in str(exc).lower() or exc.__class__.__name__ == "StagedScanForbiddenError"
    else:
        raise AssertionError("expected cross-session staged access to fail")
