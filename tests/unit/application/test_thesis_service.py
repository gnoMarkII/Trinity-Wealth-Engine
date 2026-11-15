"""Unit tests for ThesisApplicationService."""
from decimal import Decimal
from typing import List, Optional
import pytest

from core.thesis.evidence_taxonomy import EvidenceTier
from core.thesis.models import (
    CatalystEvent,
    FalsifierRule,
    ThesisClaimRecord,
    ThesisExceptionRow,
)
from application.thesis.thesis_service import ThesisApplicationService


class InMemoryThesisStore:
    def __init__(self):
        self.claims: dict[str, list[ThesisClaimRecord]] = {}

    def save_claim(self, claim: ThesisClaimRecord) -> None:
        if claim.ticker not in self.claims:
            self.claims[claim.ticker] = []
        self.claims[claim.ticker].append(claim)

    def get_active_claim(self, ticker: str) -> Optional[ThesisClaimRecord]:
        t_claims = self.claims.get(ticker, [])
        active = [c for c in t_claims if c.status == "ACTIVE"]
        return active[-1] if active else None

    def load_claims(self, ticker: str) -> List[ThesisClaimRecord]:
        return self.claims.get(ticker, [])

    def list_all_active_claims(self) -> List[ThesisClaimRecord]:
        res = []
        for t, cl in self.claims.items():
            act = [c for c in cl if c.status == "ACTIVE"]
            if act:
                res.append(act[-1])
        return res


class InMemoryExceptionStore:
    def __init__(self):
        self.exceptions: list[ThesisExceptionRow] = []

    def insert_exception(self, exc: ThesisExceptionRow) -> None:
        self.exceptions.append(exc)

    def list_pending_exceptions(self, ticker: Optional[str] = None, limit: int = 50) -> List[ThesisExceptionRow]:
        if ticker:
            return [e for e in self.exceptions if e.ticker == ticker and e.status == "PENDING"][:limit]
        return [e for e in self.exceptions if e.status == "PENDING"][:limit]

    def acknowledge_exception(self, exception_id: str) -> None:
        for e in self.exceptions:
            if e.exception_id == exception_id:
                # Update status (reconstruct row or mutate status)
                pass


def test_record_new_and_revise_thesis():
    store = InMemoryThesisStore()
    exc_store = InMemoryExceptionStore()
    svc = ThesisApplicationService(store=store, exception_store=exc_store)

    rule1 = FalsifierRule("F1", "gross_margin", "<", Decimal("70.0"))
    cat1 = CatalystEvent("C1", "Q3 Earnings", "2026-11-20")

    claim1 = svc.record_new_thesis(
        ticker="NVDA",
        claim_statement="High growth AI hardware",
        falsifiers=[rule1],
        catalysts=[cat1],
        evidence_tier=EvidenceTier.DOCUMENTED,
        as_of_utc="2026-06-01T00:00:00Z",
    )
    assert claim1.version == 1
    assert claim1.claim_id == "TH-NVDA-v1"
    assert claim1.verify_integrity() is True

    # Revise thesis
    rule2 = FalsifierRule("F2", "gross_margin", "<", Decimal("72.0"))
    claim2 = svc.revise_thesis(
        ticker="NVDA",
        claim_statement="Higher margin bar with Blackwell",
        falsifiers=[rule2],
        catalysts=[cat1],
        as_of_utc="2026-10-01T00:00:00Z",
    )
    assert claim2.version == 2
    assert claim2.claim_id == "TH-NVDA-v2"
    assert claim2.prior_claim_id == "TH-NVDA-v1"

    # Time-travel query
    july_claim = svc.get_thesis_as_of("NVDA", "2026-07-01T00:00:00Z")
    assert july_claim is not None
    assert july_claim.version == 1

    nov_claim = svc.get_thesis_as_of("NVDA", "2026-11-01T00:00:00Z")
    assert nov_claim is not None
    assert nov_claim.version == 2


def test_run_watchdog_scan_detects_breach():
    store = InMemoryThesisStore()
    exc_store = InMemoryExceptionStore()
    svc = ThesisApplicationService(store=store, exception_store=exc_store)

    rule = FalsifierRule("F1", "gross_margin", "<", Decimal("70.0"))
    svc.record_new_thesis(
        ticker="NVDA",
        claim_statement="Margin stability",
        falsifiers=[rule],
        as_of_utc="2026-11-01T00:00:00Z",
    )

    # Margin drops to 65% -> breach!
    exceptions = svc.run_watchdog_scan(
        ticker="NVDA",
        as_of_date_str="2026-11-02",
        observed_metrics_override={"gross_margin": Decimal("65.0")},
    )
    assert len(exceptions) == 1
    assert exceptions[0].exception_type == "FALSIFIER_BREACHED"
    assert len(exc_store.exceptions) == 1


def test_get_thesis_coverage():
    store = InMemoryThesisStore()
    exc_store = InMemoryExceptionStore()
    svc = ThesisApplicationService(store=store, exception_store=exc_store)

    svc.record_new_thesis(
        ticker="NVDA",
        claim_statement="AI leadership",
        as_of_utc="2026-11-01T00:00:00Z",
    )
    svc.record_new_thesis(
        ticker="CPALL.BK",
        claim_statement="Retail dominance",
        as_of_utc="2026-11-01T00:00:00Z",
    )

    portfolio_tickers = ["NVDA", "CPALL.BK", "FTNT", "SCB"]
    cov = svc.get_thesis_coverage(portfolio_tickers)

    assert cov["total_holdings_count"] == 4
    assert cov["covered_count"] == 2
    assert cov["uncovered_count"] == 2
    assert cov["coverage_pct"] == 50.0
    assert "NVDA" in cov["covered_tickers"]
    assert "CPALL.BK" in cov["covered_tickers"]
    assert "FTNT" in cov["uncovered_tickers"]
    assert "SCB" in cov["uncovered_tickers"]

def test_catalyst_status_and_direction_enums():
    from core.thesis.models import CatalystDirection, CatalystEvent, CatalystStatus

    cat = CatalystEvent(
        catalyst_id="CAT-01",
        event_name="Q3 Earnings",
        expected_date_str="2026-11-15",
        status=CatalystStatus.CONFIRMED.value,
        direction=CatalystDirection.AHEAD.value,
    )
    assert cat.status == "CONFIRMED"
    assert cat.direction == "AHEAD"
    d = cat.to_dict()
    assert d["status"] == "CONFIRMED"
    assert d["direction"] == "AHEAD"
    restored = CatalystEvent.from_dict(d)
    assert restored.status == "CONFIRMED"
    assert restored.direction == "AHEAD"


def test_validate_ticker_allows_thai_mutual_fund_symbols_and_blocks_traversal():
    from core.security import validate_ticker
    import pytest

    assert validate_ticker("SCBGVALUE(E)") == "SCBGVALUE(E)"
    assert validate_ticker("SCBS&P500E") == "SCBS&P500E"
    assert validate_ticker("INTC") == "INTC"

    for bad in ("../INTC", "INTC/foo", "INTC\\bar", ".hidden", "INTC;rm -rf"):
        with pytest.raises(ValueError):
            validate_ticker(bad)


def test_thesis_file_store_lists_fund_symbols_and_skips_bad_files(tmp_path):
    from tools.thesis.thesis_file_store import ThesisFileStore
    from core.thesis.models import FalsifierRule, ThesisClaimRecord
    from core.thesis.evidence_taxonomy import EvidenceTier

    store = ThesisFileStore(data_dir=tmp_path / "data", vault_dir=tmp_path / "vault")
    claim_fund = ThesisClaimRecord.create(
        claim_id="TH-SCB-v1",
        ticker="SCBGVALUE(E)",
        version=1,
        as_of_utc="2026-11-12T10:00:00Z",
        claim_statement="Thai value equity fund thesis",
        evidence_tier=EvidenceTier.DOCUMENTED,
    )
    claim_intc = ThesisClaimRecord.create(
        claim_id="TH-INTC-v1",
        ticker="INTC",
        version=1,
        as_of_utc="2026-11-12T16:27:06Z",
        claim_statement="Intel turnaround watchlist thesis",
        falsifiers=[
            FalsifierRule(
                falsifier_id="FAL-INTC-PRICE",
                metric_name="current_price",
                operator="<",
                threshold_value=Decimal("54.17"),
                unit="USD",
            )
        ],
        evidence_tier=EvidenceTier.INFERRED,
    )
    store.save_claim(claim_fund)
    store.save_claim(claim_intc)
    # Write a corrupted file to verify list_all_active_claims does not abort remaining claims
    (tmp_path / "data" / "BROKEN_theses.json").write_text("{not valid json", encoding="utf-8")

    active = store.list_all_active_claims()
    tickers = {c.ticker for c in active}
    assert "SCBGVALUE(E)" in tickers
    assert "INTC" in tickers


def test_get_thesis_coverage_includes_monitored_theses_newest_first():
    store = InMemoryThesisStore()
    exc_store = InMemoryExceptionStore()
    svc = ThesisApplicationService(store=store, exception_store=exc_store)

    svc.record_new_thesis(
        ticker="NVDA",
        claim_statement="Portfolio stock thesis",
        as_of_utc="2026-11-01T00:00:00Z",
    )
    svc.record_new_thesis(
        ticker="INTC",
        claim_statement="Watchlist stock thesis",
        as_of_utc="2026-11-12T16:27:06Z",
    )

    cov = svc.get_thesis_coverage(["NVDA", "SCBGVALUE(E)"])
    assert cov["covered_count"] == 1
    monitored = cov["monitored_theses"]
    assert monitored[0]["ticker"] == "INTC"
    assert monitored[0]["in_portfolio"] is False
    assert monitored[1]["ticker"] == "NVDA"
    assert monitored[1]["in_portfolio"] is True

