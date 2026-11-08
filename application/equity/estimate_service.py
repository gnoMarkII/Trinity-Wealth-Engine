"""Application use-case service for Two-Sided Estimate Ledger (Chapter 2, 5 & App H).

Architecture Rule (Layer 2 - Application):
Coordinates domain models (EstimateRecord, VariantPerceptionSnapshot, RevisionBridgeResult).
Zero direct third-party network frameworks or database transaction commits.
"""
from __future__ import annotations

from decimal import Decimal
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from application.institutional.ports import InstitutionalLedgerPort
from core.equity_analytics.estimate_ledger import (
    EstimateRecord,
    EstimateType,
    RevisionBridgeResult,
    RevisionDriver,
    VariantPerceptionSnapshot,
    append_estimate_revision,
    calculate_variant_perception,
)

log = logging.getLogger(__name__)


def _atomic_save_json(path: Path, data: Any) -> None:
    """Atomically persists JSON data using temp file swap to avoid corrupted writes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f".tmp_{os.getpid()}")
    try:
        content = json.dumps(data, indent=2, ensure_ascii=False)
        tmp_path.write_text(content, encoding="utf-8")
        tmp_path.replace(path)
    except Exception as exc:
        tmp_path.unlink(missing_ok=True)
        log.warning("Failed to atomically save state to %s: %s", path, exc)


class EstimateLedgerApplicationService:
    """Manages frozen baseline and revision rows across Street consensus and internal fund estimates."""

    def __init__(
        self,
        initial_ledger: Optional[Sequence[EstimateRecord]] = None,
        storage_path: Optional[Path] = None,
        ledger_repository: Optional[InstitutionalLedgerPort] = None,
    ) -> None:
        self._ledger: List[EstimateRecord] = []
        self._storage_path = Path(storage_path) if storage_path else None
        self._ledger_repository = ledger_repository

        if initial_ledger:
            self._ledger.extend(initial_ledger)
        else:
            self._seed_default_ledger()

        if self._ledger_repository is not None:
            persisted = self._ledger_repository.list_estimate_records()
            if persisted:
                self._ledger = [self._record_from_dict(row["payload"]) for row in persisted]
            else:
                for record in self._ledger:
                    self._ledger_repository.save_estimate_record(record.to_dict())
        elif self._storage_path and self._storage_path.exists():
            self._load_from_storage()

    @staticmethod
    def _record_from_dict(row: Dict[str, Any]) -> EstimateRecord:
        return EstimateRecord(
            estimate_id=row["estimate_id"], ticker=row["ticker"], period=row["period"],
            metric_name=row["metric_name"], estimate_type=EstimateType(row["estimate_type"]),
            value=Decimal(str(row["value"])), unit=row.get("unit", "USD_M"),
            as_of_utc=row.get("as_of_utc", ""), reason_code=row.get("reason_code", ""),
            prior_record_id=row.get("prior_record_id"), owner=row.get("owner", "analyst_a"),
            approval_status=row.get("approval_status", "REVIEW_REQUIRED"),
            evidence_label=row.get("evidence_label", "SYNTHETIC"),
        )

    def _load_from_storage(self) -> None:
        """Loads estimates from durable storage."""
        if not self._storage_path or not self._storage_path.exists():
            return
        try:
            raw = json.loads(self._storage_path.read_text(encoding="utf-8"))
            if isinstance(raw, list):
                loaded = []
                for row in raw:
                    if isinstance(row, dict) and "estimate_id" in row:
                        loaded.append(
                            EstimateRecord(
                                estimate_id=row["estimate_id"],
                                ticker=row["ticker"],
                                period=row["period"],
                                metric_name=row["metric_name"],
                                estimate_type=EstimateType(row["estimate_type"]),
                                value=Decimal(str(row["value"])),
                                unit=row.get("unit", "USD_M"),
                                as_of_utc=row["as_of_utc"],
                                reason_code=row.get("reason_code", ""),
                                prior_record_id=row.get("prior_record_id"),
                                owner=row.get("owner", "analyst_a"),
                                approval_status=row.get("approval_status", "REVIEW_REQUIRED"),
                                evidence_label=row.get("evidence_label", "SYNTHETIC"),
                            )
                        )
                loaded_ids = {r.estimate_id for r in loaded}
                remaining_defaults = [r for r in self._ledger if r.estimate_id not in loaded_ids]
                self._ledger = remaining_defaults + loaded
        except Exception as exc:
            log.warning("Could not load estimate ledger from %s: %s", self._storage_path, exc)

    def _persist_to_storage(self) -> None:
        """Flushes estimate ledger to durable storage."""
        if self._ledger_repository is not None or not self._storage_path:
            return
        data = [r.to_dict() for r in self._ledger]
        _atomic_save_json(self._storage_path, data)

    def _seed_default_ledger(self) -> None:
        """Seed representative consensus and fund thesis estimates."""
        defaults = [
            EstimateRecord(
                estimate_id="EST-CONS-GOOGL-Q2",
                ticker="GOOGL",
                period="2026Q2",
                metric_name="revenue",
                estimate_type=EstimateType.CONSENSUS,
                value=Decimal("86400.00"),
                unit="USD_M",
                as_of_utc="2026-07-20T13:30:00Z",
                reason_code="consensus_median",
                owner="street_consensus_demo",
                approval_status="APPROVED",
                evidence_label="SYNTHETIC",
            ),
            EstimateRecord(
                estimate_id="EST-FUND-GOOGL-Q2",
                ticker="GOOGL",
                period="2026Q2",
                metric_name="revenue",
                estimate_type=EstimateType.FUND_THESIS,
                value=Decimal("87500.00"),
                unit="USD_M",
                as_of_utc="2026-07-20T13:30:00Z",
                reason_code="cloud_acceleration_thesis",
                owner="analyst_alpha_demo",
                approval_status="APPROVED",
                evidence_label="SYNTHETIC",
            ),
            EstimateRecord(
                estimate_id="EST-0001",
                ticker="SYN-NSTAR",
                period="2026Q3",
                metric_name="revenue",
                estimate_type=EstimateType.FUND_THESIS,
                value=Decimal("1050.0"),
                unit="USD_M",
                as_of_utc="2026-08-06T13:30:00Z",
                reason_code="baseline",
                owner="analyst_a",
                approval_status="APPROVED",
                evidence_label="SYNTHETIC",
            ),
        ]
        self._ledger.extend(defaults)

    def get_variant_perception(
        self,
        ticker: str,
        period: str,
        metric_name: str,
        reported: Decimal,
        consensus_override: Optional[Decimal] = None,
        fund_thesis_override: Optional[Decimal] = None,
        as_of_utc: Optional[str] = None,
    ) -> VariantPerceptionSnapshot:
        """Calculates three-way surprise and variant alpha realization upon earnings print."""
        clean_ticker = ticker.strip().upper()
        unit = "USD_M"

        cons = consensus_override
        if cons is None:
            cons_matches = [
                r for r in self._ledger
                if r.ticker == clean_ticker and r.period == period and r.metric_name == metric_name
                and r.estimate_type == EstimateType.CONSENSUS
                and r.approval_status == "APPROVED"
                and (as_of_utc is None or r.as_of_utc <= as_of_utc)
            ]
            if not cons_matches:
                raise ValueError(f"No approved consensus estimate found for {clean_ticker} {period} {metric_name}")
            cons = cons_matches[-1].value
            unit = cons_matches[-1].unit

        fund = fund_thesis_override
        if fund is None:
            fund_matches = [
                r for r in self._ledger
                if r.ticker == clean_ticker and r.period == period and r.metric_name == metric_name
                and r.estimate_type == EstimateType.FUND_THESIS
                and r.approval_status == "APPROVED"
                and (as_of_utc is None or r.as_of_utc <= as_of_utc)
            ]
            if not fund_matches:
                raise ValueError(f"No approved fund thesis estimate found for {clean_ticker} {period} {metric_name}")
            fund = fund_matches[-1].value
            unit = fund_matches[-1].unit

        return calculate_variant_perception(
            ticker=clean_ticker,
            period=period,
            metric_name=metric_name,
            reported=reported,
            consensus=cons,
            fund_thesis=fund,
            unit=unit,
        )

    def list_estimates_for_ticker(self, ticker: str) -> List[EstimateRecord]:
        """Returns the chronological revision trail for a given ticker."""
        clean_ticker = ticker.strip().upper()
        return [r for r in self._ledger if r.ticker == clean_ticker]

    def record_estimate(self, record: EstimateRecord) -> None:
        """Appends a new estimate record and persists to durable storage."""
        if self._ledger_repository is not None:
            self._ledger_repository.save_estimate_record(record.to_dict())
        self._ledger.append(record)
        self._persist_to_storage()

    def record_revision(
        self,
        revised_record: EstimateRecord,
        drivers: Sequence[RevisionDriver] = (),
    ) -> RevisionBridgeResult:
        """Appends a new revision record while strictly asserting driver sum balance."""
        updated_ledger, bridge = append_estimate_revision(
            ledger=self._ledger,
            revised_record=revised_record,
            drivers=drivers,
        )
        if self._ledger_repository is not None:
            self._ledger_repository.save_estimate_record(revised_record.to_dict())
        self._ledger = updated_ledger
        self._persist_to_storage()
        return bridge

