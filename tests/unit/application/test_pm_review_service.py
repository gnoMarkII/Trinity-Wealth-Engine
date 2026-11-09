"""Unit tests for PMReviewApplicationService."""
from decimal import Decimal
from typing import Any, Dict, List, Optional, Sequence
import pytest

from application.manager.pm_review_service import PMReviewApplicationService
from core.thesis.models import FalsifierRule
from application.thesis.thesis_service import ThesisApplicationService
from tests.unit.application.test_thesis_service import InMemoryExceptionStore, InMemoryThesisStore


class InMemorySignoffRepo:
    def __init__(self):
        self.signoffs: dict[tuple[str, str], dict[str, Any]] = {}
        self.signoffs_by_packet: dict[tuple[str, str], dict[str, Any]] = {}
        self.packets: dict[tuple[str, str], dict[str, Any]] = {}

    def save_packet_snapshot(
        self, *, packet_id, portfolio_id, review_date, cutoff_utc,
        input_fingerprint_sha256, policy_version, packet_sha256, payload,
    ):
        key = (portfolio_id, packet_sha256)
        if key not in self.packets:
            revisions = [
                row["revision"] for (pid, _), row in self.packets.items()
                if pid == portfolio_id and row["review_date"] == review_date
            ]
            self.packets[key] = {
                "packet_id": packet_id, "portfolio_id": portfolio_id,
                "review_date": review_date, "revision": max(revisions, default=0) + 1,
                "cutoff_utc": cutoff_utc, "input_fingerprint_sha256": input_fingerprint_sha256,
                "policy_version": policy_version, "packet_sha256": packet_sha256,
                "payload": payload,
            }
        return dict(self.packets[key])

    def get_packet_snapshot(self, portfolio_id, packet_sha256):
        row = self.packets.get((portfolio_id, packet_sha256))
        return dict(row) if row else None

    def get_signoff(self, review_date: str, portfolio_id: str = "default", packet_sha256: Optional[str] = None) -> Optional[dict[str, Any]]:
        if packet_sha256:
            return self.signoffs_by_packet.get((portfolio_id, packet_sha256))
        return self.signoffs.get((portfolio_id, review_date))

    def save_signoff(
        self,
        signoff_id: str,
        review_date: str,
        signed_off_by: str,
        signed_off_at_utc: str,
        acknowledged_exceptions_count: int,
        notes: Optional[str],
        packet_sha256: str,
        portfolio_id: str = "default",
        verified_sample_ids: Optional[Sequence[str]] = None,
        acknowledged_exception_ids: Optional[Sequence[str]] = None,
        packet_id: Optional[str] = None,
        request_digest: Optional[str] = None,
        principal_subject_id: Optional[str] = None,
        principal_session_id: Optional[str] = None,
    ) -> dict[str, Any]:
        key = (portfolio_id, packet_sha256)
        if key in self.signoffs_by_packet:
            return dict(self.signoffs_by_packet[key])
        row = {
            "signoff_id": signoff_id,
            "portfolio_id": portfolio_id,
            "review_date": review_date,
            "signed_off_by": signed_off_by,
            "signed_off_at_utc": signed_off_at_utc,
            "acknowledged_exceptions_count": acknowledged_exceptions_count,
            "notes": notes,
            "packet_sha256": packet_sha256,
            "verified_sample_ids": list(verified_sample_ids or []),
            "acknowledged_exception_ids": list(acknowledged_exception_ids or []),
            "packet_id": packet_id,
            "request_digest": request_digest,
            "principal_subject_id": principal_subject_id,
            "principal_session_id": principal_session_id,
        }
        self.signoffs[(portfolio_id, review_date)] = row
        self.signoffs_by_packet[key] = row
        return dict(row)

    def list_signoffs(self, limit: int = 30, portfolio_id: Optional[str] = None) -> list[dict[str, Any]]:
        if portfolio_id:
            res = [v for k, v in self.signoffs.items() if k[0] == portfolio_id]
        else:
            res = list(self.signoffs.values())
        return res[:limit]


def test_build_morning_packet_review_required():
    signoff_repo = InMemorySignoffRepo()
    store = InMemoryThesisStore()
    exc_store = InMemoryExceptionStore()
    t_svc = ThesisApplicationService(store=store, exception_store=exc_store)

    pm_svc = PMReviewApplicationService(
        signoff_repo=signoff_repo,
        thesis_service=t_svc,
    )

    packet = pm_svc.build_morning_packet(review_date="2026-11-02")
    assert packet.review_date == "2026-11-02"
    assert packet.status == "REVIEW_REQUIRED"
    assert packet.packet_sha256 != ""
    assert "total_nav" in packet.portfolio_delta
    assert "yield_spread_10y_2y_bps" in packet.macro_delta


def test_sign_off_morning_review():
    signoff_repo = InMemorySignoffRepo()
    pm_svc = PMReviewApplicationService(signoff_repo=signoff_repo)

    current_packet = pm_svc.build_morning_packet(review_date="2026-11-02")
    sample_ids = [s["sample_id"] for s in current_packet.review_sample] if current_packet.review_sample else []

    result = pm_svc.sign_off(
        review_date="2026-11-02",
        signed_off_by="CHINO_PM",
        notes="Reviewed morning briefing. Allocations on track.",
        packet_sha256=current_packet.packet_sha256,
        verified_sample_ids=sample_ids,
    )
    assert result["status"] == "SIGNED_OFF"
    assert result["signed_off_by"] == "CHINO_PM"

    # Re-building packet now shows SIGNED_OFF
    packet = pm_svc.build_morning_packet(review_date="2026-11-02")
    assert packet.status == "SIGNED_OFF"
    assert packet.signoff_info is not None
    assert packet.signoff_info["signed_off_by"] == "CHINO_PM"


def test_build_morning_packet_with_advisory_and_coverage():
    from types import SimpleNamespace

    signoff_repo = InMemorySignoffRepo()
    store = InMemoryThesisStore()
    exc_store = InMemoryExceptionStore()
    t_svc = ThesisApplicationService(store=store, exception_store=exc_store)
    t_svc.record_new_thesis(ticker="NVDA", claim_statement="Active thesis", as_of_utc="2026-11-01T00:00:00Z")

    holding1 = SimpleNamespace(symbol="NVDA", market_value_thb=100000, unrealized_pnl_percent=12.5)
    holding2 = SimpleNamespace(symbol="FTNT", market_value_thb=50000, unrealized_pnl_percent=-3.2)
    fake_portfolio_state = SimpleNamespace(
        summary=SimpleNamespace(total_value_thb=150000),
        holdings=[holding1, holding2],
    )

    fake_advisory_dto = SimpleNamespace(
        holdings_advisory=[
            SimpleNamespace(
                symbol="FTNT",
                action_code="URGENT_REDUCE",
                action_rationale="Stop loss breached",
                is_immediate_execution=True,
            ),
            SimpleNamespace(
                symbol="NVDA",
                action_code="HOLD_WITHIN_BAND",
                action_rationale="Balanced",
                is_immediate_execution=False,
            ),
        ]
    )

    pm_svc = PMReviewApplicationService(
        signoff_repo=signoff_repo,
        thesis_service=t_svc,
        portfolio_reader=lambda portfolio_id="default", refresh_prices=False: fake_portfolio_state,
        advisory_reader=lambda portfolio_id="default": fake_advisory_dto,
    )

    packet = pm_svc.build_morning_packet(review_date="2026-11-02")
    assert len(packet.pending_decisions) == 1
    assert packet.pending_decisions[0]["ticker"] == "FTNT"
    assert packet.pending_decisions[0]["action_code"] == "URGENT_REDUCE"
    assert packet.pending_decisions[0]["urgency"] == "HIGH"

    assert packet.thesis_coverage is not None
    assert packet.thesis_coverage["total_holdings_count"] == 2
    assert packet.thesis_coverage["covered_count"] == 1
    assert packet.thesis_coverage["coverage_pct"] == 50.0
    assert "FTNT" in packet.thesis_coverage["uncovered_tickers"]

def test_build_morning_packet_with_quarantined_and_late_run():
    from types import SimpleNamespace

    signoff_repo = InMemorySignoffRepo()
    store = InMemoryThesisStore()
    exc_store = InMemoryExceptionStore()
    t_svc = ThesisApplicationService(store=store, exception_store=exc_store)

    broken_holding = SimpleNamespace(
        symbol="BROKEN",
        asset_type="Stock",
        units=100,
        current_price_thb=None,
        current_price_usd=0.0,
        current_price=None,
        market_value_thb=0.0,
        unrealized_pnl_percent=None,
    )
    ok_holding = SimpleNamespace(
        symbol="NVDA",
        asset_type="Stock",
        units=50,
        current_price_usd=130.0,
        market_value_thb=220000.0,
        unrealized_pnl_percent=15.0,
    )
    fake_portfolio_state = SimpleNamespace(
        summary=SimpleNamespace(total_value_thb=220000.0),
        holdings=[ok_holding, broken_holding],
    )

    pm_svc = PMReviewApplicationService(
        signoff_repo=signoff_repo,
        thesis_service=t_svc,
        portfolio_reader=lambda portfolio_id="default", refresh_prices=False: fake_portfolio_state,
    )

    packet = pm_svc.build_morning_packet(review_date="2026-11-04")
    assert len(packet.quarantined_holdings) >= 1
    quarantined_symbols = [q["symbol"] for q in packet.quarantined_holdings]
    assert "BROKEN" in quarantined_symbols
    assert packet.run_quality in ("ON_TIME", "LATE_RUN_STALE")



def test_factor_exposure_and_beta_in_morning_packet():
    """Verifies Chapter 8 factor exposure & beta integration in Quadrant 1."""
    from types import SimpleNamespace
    from core.equity_analytics.factor_exposure import ReturnObservation

    signoff_repo = InMemorySignoffRepo()
    h1 = SimpleNamespace(symbol="NVDA", units=100, market_value_thb=100000.0, unrealized_pnl_percent=5.0)
    h2 = SimpleNamespace(symbol="SHORT_HEDGE", units=-50, market_value_thb=-25000.0, unrealized_pnl_percent=-1.0)
    fake_portfolio_state = SimpleNamespace(
        summary=SimpleNamespace(total_value_thb=75000.0),
        holdings=[h1, h2],
    )

    fake_perf = [
        {"Date": "2026-11-01", "Total_NAV": 100000.0},
        {"Date": "2026-11-02", "Total_NAV": 102000.0},
        {"Date": "2026-11-03", "Total_NAV": 101000.0},
    ]

    mock_benchmark_reader = lambda perf: [
        ReturnObservation(date="2026-11-02", portfolio_return=Decimal("0.02"), factor_return=Decimal("0.01")),
        ReturnObservation(date="2026-11-03", portfolio_return=Decimal("-0.0098"), factor_return=Decimal("-0.005")),
    ]

    pm_svc = PMReviewApplicationService(
        signoff_repo=signoff_repo,
        portfolio_reader=lambda portfolio_id="default", refresh_prices=False: fake_portfolio_state,
        performance_reader=lambda days=30, portfolio_id="default": fake_perf,
        benchmark_return_reader=mock_benchmark_reader,
    )

    packet = pm_svc.build_morning_packet(review_date="2026-11-04")
    p_delta = packet.portfolio_delta

    # Gross = 100000 + 25000 = 125000
    assert p_delta["gross_market_value"] == 125000.0
    # Net = 100000 - 25000 = 75000
    assert p_delta["net_market_value"] == 75000.0
    assert p_delta["long_market_value"] == 100000.0
    assert p_delta["short_market_value"] == -25000.0
    # Exposure percentages
    assert p_delta["long_exposure_pct"] == 80.0
    assert p_delta["short_exposure_pct"] == 20.0
    assert p_delta["net_exposure_pct"] == 60.0
    # Beta
    assert "portfolio_beta" in p_delta
    assert p_delta["beta_benchmark"] == "SPY"


def test_state_integrity_tampered_triggers_quarantine():
    """State integrity failure immediately quarantines historical state baseline."""
    from core.portfolio_advisory.state_integrity import StateIntegrityReport

    signoff_repo = InMemorySignoffRepo()
    tampered_report = StateIntegrityReport(
        is_valid=False,
        state_id="as_of_2026_11_04",
        checked_files_count=3,
        mismatched_files=["prices_history.csv"],
        missing_files=[],
        tamper_details=[{"file": "prices_history.csv", "reason": "Hash mismatch"}],
        summary_status="INTEGRITY_TAMPERED",
        message="CRITICAL: Historical state tampered on prices_history.csv",
    )

    pm_svc = PMReviewApplicationService(
        signoff_repo=signoff_repo,
        state_integrity_checker=lambda: tampered_report,
    )

    packet = pm_svc.build_morning_packet(review_date="2026-11-04")
    assert packet.portfolio_delta.get("state_integrity") is not None
    assert packet.portfolio_delta["state_integrity"]["is_valid"] is False

    quarantined = [q for q in packet.quarantined_holdings if q["symbol"] == "STATE_INTEGRITY"]
    assert len(quarantined) == 1
    assert quarantined[0]["guardrail_code"] == "INTEGRITY_TAMPERED"


def test_pending_decisions_enriched_with_failure_atlas_smallest_repair():
    """Pending decisions are tagged with institutional failure class and smallest repair."""
    from types import SimpleNamespace

    signoff_repo = InMemorySignoffRepo()
    fake_advisory_dto = SimpleNamespace(
        holdings_advisory=[
            SimpleNamespace(
                symbol="FTNT",
                action_code="URGENT_REDUCE",
                action_rationale="Stop loss breached: current price $88.50 < threshold $92.00",
                is_immediate_execution=True,
            ),
        ]
    )

    pm_svc = PMReviewApplicationService(
        signoff_repo=signoff_repo,
        advisory_reader=lambda portfolio_id="default": fake_advisory_dto,
    )

    packet = pm_svc.build_morning_packet(review_date="2026-11-04")
    assert len(packet.pending_decisions) >= 1
    dec = packet.pending_decisions[0]
    assert "failure_class" in dec
    assert "smallest_repair" in dec
    assert dec["smallest_repair"] != ""


def test_build_morning_packet_includes_review_sample():
    """Morning review packet includes bounded 3-5 item risk-based review sample."""
    from types import SimpleNamespace

    signoff_repo = InMemorySignoffRepo()
    holdings = [
        SimpleNamespace(symbol="NVDA", market_value_thb=100000, unrealized_pnl_percent=5.0),
        SimpleNamespace(symbol="AAPL", market_value_thb=80000, unrealized_pnl_percent=-2.0),
        SimpleNamespace(symbol="MSFT", market_value_thb=70000, unrealized_pnl_percent=1.0),
        SimpleNamespace(symbol="GOOGL", market_value_thb=60000, unrealized_pnl_percent=0.5),
    ]
    fake_portfolio_state = SimpleNamespace(
        summary=SimpleNamespace(total_value_thb=310000),
        holdings=holdings,
    )

    pm_svc = PMReviewApplicationService(
        signoff_repo=signoff_repo,
        portfolio_reader=lambda *args, **kwargs: fake_portfolio_state,
    )

    packet = pm_svc.build_morning_packet(review_date="2026-11-05")
    assert hasattr(packet, "review_sample")
    assert len(packet.review_sample) >= 3
    assert len(packet.review_sample) <= 5

    sample_symbols = [s["symbol"] for s in packet.review_sample]
    assert len(set(sample_symbols)) == len(sample_symbols)
    assert all("sample_id" in s and "headline" in s for s in packet.review_sample)


def test_sign_off_records_verified_sample_ids():
    """Sign-off records verified sample IDs in the audit record and extracts them on retrieval."""
    from types import SimpleNamespace

    signoff_repo = InMemorySignoffRepo()
    holdings = [
        SimpleNamespace(
            symbol=symbol, asset_type="Stock", units=1, current_price_thb=100,
            current_price=None, current_price_usd=None, fx_rate=None,
            market_value_thb=100, unrealized_pnl_percent=None,
        )
        for symbol in ("AAA", "BBB", "CCC", "DDD")
    ]
    state = SimpleNamespace(summary=SimpleNamespace(total_value_thb=400), holdings=holdings)
    pm_svc = PMReviewApplicationService(
        signoff_repo=signoff_repo,
        portfolio_reader=lambda portfolio_id="default", refresh_prices=False: state,
    )

    # First inspect valid samples from the morning packet
    packet = pm_svc.build_morning_packet(review_date="2026-11-05")
    valid_samples = [s["sample_id"] for s in packet.review_sample]
    assert len(valid_samples) > 0

    # Forged sample ID rejected
    with pytest.raises(ValueError, match="Invalid verified sample ID"):
        pm_svc.sign_off(
            review_date="2026-11-05",
            signed_off_by="CHINO_PM",
            verified_sample_ids=["SMP-FORGED-999"],
            packet_sha256=packet.packet_sha256,
        )

    # Missing checksum rejected
    with pytest.raises(ValueError, match="Packet SHA-256 checksum is mandatory"):
        pm_svc.sign_off(
            review_date="2026-11-05",
            signed_off_by="CHINO_PM",
            packet_sha256="",
        )

    # Empty samples rejected when packet has review samples
    with pytest.raises(ValueError, match="Review sample verification required"):
        pm_svc.sign_off(
            review_date="2026-11-05",
            signed_off_by="CHINO_PM",
            packet_sha256=packet.packet_sha256,
            verified_sample_ids=[],
        )

    # Checksum mismatch rejected
    with pytest.raises(ValueError, match="Packet SHA-256 mismatch"):
        pm_svc.sign_off(
            review_date="2026-11-05",
            signed_off_by="CHINO_PM",
            packet_sha256="0" * 64,
            verified_sample_ids=valid_samples[:1],
        )

    # A partial sample list cannot approve the whole packet.
    with pytest.raises(ValueError, match="inspect every packet sample"):
        pm_svc.sign_off(
            review_date="2026-11-05",
            signed_off_by="CHINO_PM",
            packet_sha256=packet.packet_sha256,
            verified_sample_ids=valid_samples[:1],
        )

    # Reordered and duplicate evidence normalizes to the packet's unique canonical order.
    verified = list(reversed(valid_samples)) + [valid_samples[0]]
    result = pm_svc.sign_off(
        review_date="2026-11-05",
        signed_off_by="CHINO_PM",
        notes="Sample checks confirmed.",
        verified_sample_ids=verified,
        packet_sha256=packet.packet_sha256,
    )

    assert result["status"] == "SIGNED_OFF"
    assert result["verified_sample_ids"] == valid_samples

    # When re-building the packet, signoff_info extracts verified_sample_ids
    signed_packet = pm_svc.build_morning_packet(review_date="2026-11-05")
    assert signed_packet.status == "SIGNED_OFF"
    assert signed_packet.signoff_info is not None
    assert signed_packet.signoff_info.get("verified_sample_ids") == valid_samples


def test_sign_off_requires_and_persists_pending_exception_acknowledgements():
    from types import SimpleNamespace

    exception = SimpleNamespace(
        to_dict=lambda: {
            "exception_id": "EXC-PENDING-1",
            "ticker": "NVDA",
            "exception_type": "FALSIFIER_BREACHED",
            "status": "PENDING",
            "human_action": "REVIEW_REQUIRED",
        }
    )

    class StaticThesisService:
        def run_watchdog_scan(self, **kwargs):
            return [exception]

    repo = InMemorySignoffRepo()
    service = PMReviewApplicationService(signoff_repo=repo, thesis_service=StaticThesisService())
    packet = service.build_morning_packet(review_date="2026-11-05")
    sample_ids = [sample["sample_id"] for sample in packet.review_sample]

    with pytest.raises(ValueError, match="Exception acknowledgement required"):
        service.sign_off(
            review_date="2026-11-05",
            signed_off_by="CHINO_PM",
            packet_sha256=packet.packet_sha256,
            verified_sample_ids=sample_ids,
            acknowledged_exception_ids=[],
        )

    with pytest.raises(ValueError, match="not a pending exception"):
        service.sign_off(
            review_date="2026-11-05",
            signed_off_by="CHINO_PM",
            packet_sha256=packet.packet_sha256,
            verified_sample_ids=sample_ids,
            acknowledged_exception_ids=["EXC-FORGED"],
        )

    result = service.sign_off(
        review_date="2026-11-05",
        signed_off_by="CHINO_PM",
        packet_sha256=packet.packet_sha256,
        verified_sample_ids=sample_ids,
        acknowledged_exception_ids=["EXC-PENDING-1"],
    )
    assert result["acknowledged_exceptions_count"] == 1
    assert result["acknowledged_exception_ids"] == ["EXC-PENDING-1"]
    assert service.build_morning_packet(review_date="2026-11-05").status == "SIGNED_OFF"


def test_sign_off_fails_when_repository_cannot_persist_required_acknowledgements():
    from types import SimpleNamespace
    from application.manager.pm_review_service import SignoffPersistenceUnavailable

    exception = SimpleNamespace(
        to_dict=lambda: {
            "exception_id": "EXC-PENDING-2",
            "ticker": "NVDA",
            "exception_type": "NOT_EVALUABLE",
            "status": "PENDING",
            "human_action": "REVIEW_REQUIRED",
        }
    )

    class LegacyRepo(InMemorySignoffRepo):

        def save_signoff(
            self, signoff_id, review_date, signed_off_by, signed_off_at_utc,
            acknowledged_exceptions_count, notes, packet_sha256,
            portfolio_id="default", verified_sample_ids=None,
        ):
            raise AssertionError("legacy repository must not be called after rejecting unsupported evidence")

    class StaticThesisService:
        def run_watchdog_scan(self, **kwargs):
            return [exception]

    service = PMReviewApplicationService(signoff_repo=LegacyRepo(), thesis_service=StaticThesisService())
    packet = service.build_morning_packet(review_date="2026-11-05")
    sample_ids = [sample["sample_id"] for sample in packet.review_sample]
    with pytest.raises(SignoffPersistenceUnavailable, match="cannot durably store exception acknowledgements"):
        service.sign_off(
            review_date="2026-11-05",
            signed_off_by="CHINO_PM",
            packet_sha256=packet.packet_sha256,
            verified_sample_ids=sample_ids,
            acknowledged_exception_ids=["EXC-PENDING-2"],
        )


def test_sign_off_portfolio_isolation():
    """Sign-off of Portfolio A leaves Portfolio B in REVIEW_REQUIRED state."""
    signoff_repo = InMemorySignoffRepo()
    pm_svc = PMReviewApplicationService(signoff_repo=signoff_repo)

    packet_a_initial = pm_svc.build_morning_packet(portfolio_id="fund_alpha", review_date="2026-11-05")
    sample_ids = [s["sample_id"] for s in packet_a_initial.review_sample] if packet_a_initial.review_sample else []

    res_a = pm_svc.sign_off(
        review_date="2026-11-05",
        portfolio_id="fund_alpha",
        signed_off_by="CHINO_PM",
        notes="Alpha signed off",
        packet_sha256=packet_a_initial.packet_sha256,
        verified_sample_ids=sample_ids,
    )
    assert res_a["status"] == "SIGNED_OFF"

    # Packet for fund_alpha is SIGNED_OFF
    packet_alpha = pm_svc.build_morning_packet(portfolio_id="fund_alpha", review_date="2026-11-05")
    assert packet_alpha.status == "SIGNED_OFF"

    # Packet for fund_beta is STILL REVIEW_REQUIRED
    packet_beta = pm_svc.build_morning_packet(portfolio_id="fund_beta", review_date="2026-11-05")
    assert packet_beta.status == "REVIEW_REQUIRED"
    assert packet_beta.signoff_info is None

def test_usd_exposure_uses_holding_fx_and_quarantines_missing_fx():
    from types import SimpleNamespace

    usd_holding = SimpleNamespace(
        symbol="MSFT", asset_type="Stock", units=2,
        current_price_usd=100, current_price_thb=None, current_price=None,
        fx_rate=35, currency="USD", avg_cost_usd=90, avg_cost_thb=None,
        market_value_thb=0, unrealized_pnl_percent=None,
    )
    state = SimpleNamespace(
        summary=SimpleNamespace(total_value_thb=7000),
        holdings=[usd_holding],
    )
    service = PMReviewApplicationService(
        signoff_repo=InMemorySignoffRepo(),
        portfolio_reader=lambda portfolio_id="default", refresh_prices=False: state,
    )
    packet = service.build_morning_packet(review_date="2026-11-06")
    assert packet.portfolio_delta["gross_market_value"] == 7000.0
    assert packet.portfolio_delta["net_market_value"] == 7000.0
    assert packet.portfolio_delta["eligible_positions_count"] == 1
    assert not packet.quarantined_holdings

    usd_holding.fx_rate = None
    packet_without_fx = service.build_morning_packet(review_date="2026-11-06")
    assert packet_without_fx.portfolio_delta["gross_market_value"] == 0.0
    assert any(row["symbol"] == "MSFT" for row in packet_without_fx.quarantined_holdings)


def test_daily_pnl_requires_as_of_snapshots_and_explicit_external_flow():
    def build(rows, review_date="2026-11-06"):
        service = PMReviewApplicationService(
            signoff_repo=InMemorySignoffRepo(),
            performance_reader=lambda days=30, portfolio_id="default": rows,
        )
        return service.build_morning_packet(review_date=review_date).portfolio_delta

    missing_flow = build([
        {"Date": "2026-11-05", "Total_NAV": "100"},
        {"Date": "2026-11-06", "Total_NAV": "150"},
    ])
    assert missing_flow["daily_pnl_value"] is None
    assert missing_flow["daily_pnl_pct"] is None

    deposit = build([
        {"Date": "2026-11-05", "Total_NAV": "100", "Total_NAV_THB": "100", "Net_External_Flow_THB": "0", "Flow_Coverage_Status": "COMPLETE"},
        {"Date": "2026-11-06", "Total_NAV": "150", "Total_NAV_THB": "150", "Net_External_Flow_THB": "50", "Flow_Coverage_Status": "COMPLETE"},
    ])
    assert deposit["daily_pnl_value"] == 0.0
    assert deposit["daily_pnl_pct"] == 0.0

    market_gain_after_deposit = build([
        {"Date": "2026-11-05", "Total_NAV": "100", "Total_NAV_THB": "100", "Net_External_Flow_THB": "0", "Flow_Coverage_Status": "COMPLETE"},
        {"Date": "2026-11-06", "Total_NAV": "160", "Total_NAV_THB": "160", "Net_External_Flow_THB": "50", "Flow_Coverage_Status": "COMPLETE"},
    ])
    assert market_gain_after_deposit["daily_pnl_value"] == 10.0
    assert market_gain_after_deposit["daily_pnl_pct"] == 10.0

    withdrawal = build([
        {"Date": "2026-11-05", "Total_NAV": "100", "Total_NAV_THB": "100", "Net_External_Flow_THB": "0", "Flow_Coverage_Status": "COMPLETE"},
        {"Date": "2026-11-06", "Total_NAV": "70", "Total_NAV_THB": "70", "Net_External_Flow_THB": "-30", "Flow_Coverage_Status": "COMPLETE"},
    ])
    assert withdrawal["daily_pnl_value"] == 0.0
    assert withdrawal["daily_pnl_pct"] == 0.0

    future_snapshot = build([
        {"Date": "2026-11-06", "Total_NAV": "100", "Net_Cash_Flow": "0"},
        {"Date": "2026-11-07", "Total_NAV": "10000", "Net_Cash_Flow": "0"},
    ])
    assert future_snapshot["daily_pnl_value"] is None
    assert future_snapshot["daily_pnl_pct"] is None

    stale_snapshots = build([
        {"Date": "2026-10-01", "Total_NAV": "100", "Net_Cash_Flow": "0"},
        {"Date": "2026-11-01", "Total_NAV": "110", "Net_Cash_Flow": "0"},
    ])
    assert stale_snapshots["daily_pnl_value"] is None
    assert stale_snapshots["daily_pnl_pct"] is None

def test_pm_review_usd_fallback_requires_and_applies_holding_fx():
    from types import SimpleNamespace

    holding = SimpleNamespace(
        symbol="MSFT",
        asset_type="Stock",
        units=2,
        current_price_usd=100,
        current_price_thb=None,
        current_price=None,
        fx_rate=35,
        currency="USD",
        avg_cost_usd=90,
        avg_cost_thb=None,
        market_value_thb=0,
        unrealized_pnl_percent=None,
    )
    state = SimpleNamespace(summary=SimpleNamespace(total_value_thb=7000), holdings=[holding])
    service = PMReviewApplicationService(
        signoff_repo=InMemorySignoffRepo(),
        portfolio_reader=lambda portfolio_id="default", refresh_prices=False: state,
    )

    packet = service.build_morning_packet(review_date="2026-11-06")
    assert packet.portfolio_delta["gross_market_value"] == 7000.0
    assert packet.portfolio_delta["net_market_value"] == 7000.0
    assert packet.portfolio_delta["eligible_positions_count"] == 1
    assert not packet.quarantined_holdings

    holding.fx_rate = None
    packet = service.build_morning_packet(review_date="2026-11-06")
    assert packet.portfolio_delta["gross_market_value"] == 0.0
    assert any(row["symbol"] == "MSFT" for row in packet.quarantined_holdings)

def test_pm_review_mixed_currency_long_short_exposure_uses_decimal_fx():
    from types import SimpleNamespace

    holdings = [
        SimpleNamespace(
            symbol="USD_LONG", asset_type="Stock", units=2,
            current_price_usd=100, current_price_thb=None, current_price=None,
            fx_rate=35, currency="USD", avg_cost_usd=90, avg_cost_thb=None,
            market_value_thb=0, unrealized_pnl_percent=None,
        ),
        SimpleNamespace(
            symbol="USD_SHORT", asset_type="Stock", units=-1,
            current_price_usd=50, current_price_thb=None, current_price=None,
            fx_rate=35, currency="USD", avg_cost_usd=55, avg_cost_thb=None,
            market_value_thb=0, unrealized_pnl_percent=None,
        ),
        SimpleNamespace(
            symbol="TH_LONG.BK", asset_type="Stock", units=3,
            current_price_usd=None, current_price_thb=200, current_price=None,
            fx_rate=None, currency="THB", avg_cost_usd=None, avg_cost_thb=180,
            market_value_thb=0, unrealized_pnl_percent=None,
        ),
    ]
    state = SimpleNamespace(summary=SimpleNamespace(total_value_thb=5850), holdings=holdings)
    service = PMReviewApplicationService(
        signoff_repo=InMemorySignoffRepo(),
        portfolio_reader=lambda portfolio_id="default", refresh_prices=False: state,
    )

    delta = service.build_morning_packet(review_date="2026-11-06").portfolio_delta
    assert delta["long_market_value"] == 7600.0
    assert delta["short_market_value"] == -1750.0
    assert delta["gross_market_value"] == 9350.0
    assert delta["net_market_value"] == 5850.0
    assert delta["eligible_positions_count"] == 3


def test_r07_pm_review_uses_economic_nav_for_xd_continuity():
    """R07 regression: XD mechanical price drop offset by dividend receivable preserves return continuity."""
    perf_rows = [
        {
            "Date": "2026-11-05",
            "Total_NAV": "350000.00",
            "Total_NAV_THB": "350000.00",
            "Accrued_Dividend_Receivable_THB": "0.00",
            "Economic_NAV_THB": "350000.00",
            "Net_External_Flow_THB": "0.00",
            "Flow_Coverage_Status": "COMPLETE",
        },
        {
            "Date": "2026-11-06",
            "Total_NAV": "339200.00",
            "Total_NAV_THB": "339200.00",
            "Accrued_Dividend_Receivable_THB": "10800.00",
            "Economic_NAV_THB": "350000.00",
            "Net_External_Flow_THB": "0.00",
            "Flow_Coverage_Status": "COMPLETE",
        },
    ]
    service = PMReviewApplicationService(
        signoff_repo=InMemorySignoffRepo(),
        performance_reader=lambda days=5, portfolio_id="default", as_of_utc=None: perf_rows,
    )
    delta = service.build_morning_packet(review_date="2026-11-06").portfolio_delta
    assert delta["daily_pnl_status"] == "AVAILABLE"
    assert delta["daily_pnl_value"] == 0.0
    assert delta["daily_pnl_pct"] == 0.0
    assert delta["daily_pnl_nav_basis"] == "economic_nav"
