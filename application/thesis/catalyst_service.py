"""Application use-case service for Catalyst Exception Queue (Chapter 7 & Lab 7).

Architecture Rule (Layer 2 - Application):
Coordinates domain models (CatalystRecord, CatalystException, scan_catalyst_queue).
No direct third-party network frameworks or database transaction commits.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from application.institutional.ports import InstitutionalLedgerPort
from core.thesis.catalyst_queue import (
    CatalystClassification,
    CatalystException,
    CatalystRecord,
    SourceCaptureRecord,
    evaluate_catalyst_exception,
    scan_catalyst_queue,
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


class CatalystApplicationService:
    """Manages active catalyst records, captured filings, and the factual exception queue."""

    def __init__(
        self,
        initial_catalysts: Optional[Sequence[CatalystRecord]] = None,
        storage_path: Optional[Path] = None,
        ledger_repository: Optional[InstitutionalLedgerPort] = None,
    ) -> None:
        self._catalysts: Dict[str, CatalystRecord] = {}
        self._captures: List[SourceCaptureRecord] = []
        self._reviewed_exceptions: Dict[str, CatalystException] = {}
        self._storage_path = Path(storage_path) if storage_path else None
        self._ledger_repository = ledger_repository

        if initial_catalysts:
            for cat in initial_catalysts:
                self._catalysts[cat.catalyst_id] = cat
        else:
            self._seed_default_catalysts()

        if self._ledger_repository is not None:
            self._refresh_repository_state()
            if not self._ledger_repository.list_latest_catalysts():
                for catalyst in self._catalysts.values():
                    self._ledger_repository.save_catalyst_revision(catalyst.to_dict())
        elif self._storage_path and self._storage_path.exists():
            self._load_from_storage()

    @staticmethod
    def _catalyst_from_dict(row: Dict[str, Any]) -> CatalystRecord:
        return CatalystRecord(
            catalyst_id=row["catalyst_id"], entity_id=row["entity_id"],
            event_name=row["event_name"], expected_date=row.get("expected_date") or "",
            source_id=row.get("source_id") or "", last_verification_utc=row.get("last_verification_utc") or "",
            linked_claim_id=row.get("linked_claim_id") or "", owner=row.get("owner", "analyst_a"),
            status=row.get("status", "ACTIVE"), evidence_label=row.get("evidence_label", "SYNTHETIC"),
        )

    @staticmethod
    def _capture_from_dict(row: Dict[str, Any]) -> SourceCaptureRecord:
        return SourceCaptureRecord(
            source_id=row["source_id"], entity_id=row.get("entity_id") or "",
            source_type=row.get("source_type") or row.get("document_type") or "UNKNOWN",
            captured_at_utc=row.get("captured_at_utc") or "",
            effective_period=row.get("effective_period") or row.get("fiscal_period") or "",
            source_locator=row.get("source_locator") or "",
            content_snippet=row.get("content_snippet") or "",
            content_hash=row.get("content_hash") or row.get("content_sha256") or "",
            is_untrusted_instruction=bool(row.get("is_untrusted_instruction", False)),
        )

    @staticmethod
    def _exception_from_dict(row: Dict[str, Any]) -> CatalystException:
        return CatalystException(
            exception_id=row["exception_id"], catalyst_id=row["catalyst_id"],
            entity_id=row["entity_id"], classification=CatalystClassification(row["classification"]),
            source_id=row["source_id"], observed_fact=row["observed_fact"],
            prior_date=row.get("prior_date"), new_date=row.get("new_date"),
            analyst_field=row.get("analyst_field", "materiality_assessment"),
            human_assessment=row.get("human_assessment"), status=row.get("status", "REVIEW_REQUIRED"),
            detected_utc=row.get("detected_utc", ""),
        )

    def _refresh_repository_state(self) -> None:
        if self._ledger_repository is None:
            return
        catalysts = self._ledger_repository.list_latest_catalysts()
        captures = self._ledger_repository.list_source_captures()
        assessments = self._ledger_repository.list_catalyst_assessments()
        if catalysts:
            self._catalysts = {
                item["catalyst_id"]: self._catalyst_from_dict(item.get("payload") or item)
                for item in catalysts
            }
        self._captures = [self._capture_from_dict(item) for item in captures]
        self._reviewed_exceptions = {}
        for row in assessments:
            payload = row.get("payload") or {}
            exception = payload.get("exception") if isinstance(payload, dict) else None
            if isinstance(exception, dict) and exception.get("exception_id"):
                record = self._exception_from_dict(exception)
                self._reviewed_exceptions[record.exception_id] = record

    def _load_from_storage(self) -> None:
        """Loads catalysts, captures, and acknowledged exceptions from storage."""
        if not self._storage_path or not self._storage_path.exists():
            return
        try:
            raw = json.loads(self._storage_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                for c_dict in raw.get("catalysts", []):
                    if isinstance(c_dict, dict) and "catalyst_id" in c_dict:
                        cat = CatalystRecord(
                            catalyst_id=c_dict["catalyst_id"],
                            entity_id=c_dict["entity_id"],
                            event_name=c_dict["event_name"],
                            expected_date=c_dict["expected_date"],
                            source_id=c_dict["source_id"],
                            last_verification_utc=c_dict["last_verification_utc"],
                            linked_claim_id=c_dict["linked_claim_id"],
                            owner=c_dict.get("owner", "analyst_a"),
                            status=c_dict.get("status", "ACTIVE"),
                            evidence_label=c_dict.get("evidence_label", "SYNTHETIC"),
                        )
                        self._catalysts[cat.catalyst_id] = cat

                for cap_dict in raw.get("captures", []):
                    if isinstance(cap_dict, dict) and "source_id" in cap_dict:
                        cap = SourceCaptureRecord(
                            source_id=cap_dict["source_id"],
                            entity_id=cap_dict["entity_id"],
                            source_type=cap_dict["source_type"],
                            captured_at_utc=cap_dict["captured_at_utc"],
                            effective_period=cap_dict["effective_period"],
                            source_locator=cap_dict["source_locator"],
                            content_snippet=cap_dict["content_snippet"],
                            content_hash=cap_dict.get("content_hash", ""),
                            is_untrusted_instruction=bool(cap_dict.get("is_untrusted_instruction", False)),
                        )
                        self._captures.append(cap)

                for ex_dict in raw.get("reviewed_exceptions", []):
                    if isinstance(ex_dict, dict) and "exception_id" in ex_dict:
                        ex = CatalystException(
                            exception_id=ex_dict["exception_id"],
                            catalyst_id=ex_dict["catalyst_id"],
                            entity_id=ex_dict["entity_id"],
                            classification=CatalystClassification(ex_dict["classification"]),
                            source_id=ex_dict["source_id"],
                            observed_fact=ex_dict["observed_fact"],
                            prior_date=ex_dict.get("prior_date"),
                            new_date=ex_dict.get("new_date"),
                            analyst_field=ex_dict.get("analyst_field", "materiality_assessment"),
                            human_assessment=ex_dict.get("human_assessment"),
                            status=ex_dict.get("status", "REVIEW_REQUIRED"),
                            detected_utc=ex_dict.get("detected_utc", ""),
                        )
                        self._reviewed_exceptions[ex.exception_id] = ex
        except Exception as exc:
            log.warning("Could not load catalyst state from %s: %s", self._storage_path, exc)

    def _persist_to_storage(self) -> None:
        """Flushes catalyst state to durable storage."""
        if self._ledger_repository is not None or not self._storage_path:
            return
        data = {
            "catalysts": [c.to_dict() for c in self._catalysts.values()],
            "captures": [c.to_dict() for c in self._captures],
            "reviewed_exceptions": [e.to_dict() for e in self._reviewed_exceptions.values()],
        }
        _atomic_save_json(self._storage_path, data)

    def _seed_default_catalysts(self) -> None:
        """Seed representative catalysts per Chapter 7 & Appendix H."""
        defaults = [
            CatalystRecord(
                catalyst_id="CAT-001",
                entity_id="SYN-NSTAR",
                event_name="Q3 Earnings Release",
                expected_date="2026-10-29",
                source_id="SYN-IR-001",
                last_verification_utc="2026-08-01T12:00:00Z",
                linked_claim_id="TH-001",
                owner="analyst_a",
            ),
            CatalystRecord(
                catalyst_id="CAT-002",
                entity_id="SYN-NSTAR",
                event_name="Product Launch",
                expected_date="2026-11-15",
                source_id="SYN-IR-002",
                last_verification_utc="2026-05-01T00:00:00Z",
                linked_claim_id="TH-002",
                owner="analyst_a",
            ),
            CatalystRecord(
                catalyst_id="CAT-003",
                entity_id="GOOGL",
                event_name="Cloud Next Keynote",
                expected_date="2026-09-22",
                source_id="GOOGL-IR-01",
                last_verification_utc="2026-08-01T00:00:00Z",
                linked_claim_id="TH-GOOGL-v1",
                owner="pm_hermes",
            ),
        ]
        for c in defaults:
            self._catalysts[c.catalyst_id] = c

    def register_catalyst(self, catalyst: CatalystRecord) -> None:
        """Appends a new immutable catalyst revision before updating local state."""
        if self._ledger_repository is not None:
            self._ledger_repository.save_catalyst_revision(catalyst.to_dict())
        self._catalysts[catalyst.catalyst_id] = catalyst
        self._persist_to_storage()

    def register_capture(self, capture: SourceCaptureRecord) -> None:
        """Persists and hash-verifies a source artifact before indexing its capture."""
        if self._ledger_repository is not None:
            stored = self._ledger_repository.capture_source(capture.to_dict())
            capture = self._capture_from_dict(stored)
        self._captures.append(capture)
        self._persist_to_storage()

    def scan_exceptions(
        self,
        date_overrides: Optional[Dict[str, str]] = None,
        as_of_utc: Optional[str] = None,
        stale_days_threshold: int = 60,
    ) -> List[CatalystException]:
        """Runs the factual exception queue scanner across registered catalysts."""
        self._refresh_repository_state()
        cats = list(self._catalysts.values())
        raw_exceptions = scan_catalyst_queue(
            catalysts=cats,
            captures=self._captures,
            date_overrides=date_overrides,
            as_of_utc=as_of_utc,
            stale_days_threshold=stale_days_threshold,
        )

        # Merge with any previously reviewed assessments
        final_exceptions: List[CatalystException] = []
        for ex in raw_exceptions:
            reviewed = self._reviewed_exceptions.get(ex.exception_id)
            if reviewed and reviewed.catalyst_id == ex.catalyst_id and reviewed.entity_id == ex.entity_id:
                final_exceptions.append(reviewed)
            else:
                final_exceptions.append(ex)

        return final_exceptions

    def review_exception(
        self,
        exception_id: str,
        human_assessment: str,
        reviewer: str = "PORTFOLIO_MANAGER_HERMES",
        date_overrides: Optional[Dict[str, str]] = None,
    ) -> Optional[CatalystException]:
        """PM enters human assessment (materiality, action) and acknowledges exception."""
        # Find exception across current scan
        exceptions = self.scan_exceptions(date_overrides=date_overrides)
        target = next((e for e in exceptions if e.exception_id == exception_id), None)
        if not target:
            target = self._reviewed_exceptions.get(exception_id)
        if not target:
            return None

        reviewed = CatalystException(
            exception_id=target.exception_id,
            catalyst_id=target.catalyst_id,
            entity_id=target.entity_id,
            classification=target.classification,
            source_id=target.source_id,
            observed_fact=target.observed_fact,
            prior_date=target.prior_date,
            new_date=target.new_date,
            analyst_field=target.analyst_field,
            human_assessment=f"[{reviewer}] {human_assessment}",
            status="ACKNOWLEDGED",
            detected_utc=target.detected_utc,
        )
        if self._ledger_repository is not None:
            semantic = {
                "exception": reviewed.to_dict(),
                "reviewer": reviewer,
                "assessment_text": human_assessment,
            }
            fingerprint = hashlib.sha256(
                json.dumps(target.to_dict(), sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            idempotency_key = hashlib.sha256(
                f"{fingerprint}:{reviewer}:{human_assessment}".encode("utf-8")
            ).hexdigest()
            capture = next((item for item in self._captures if item.source_id == target.source_id), None)
            self._ledger_repository.save_catalyst_assessment({
                "idempotency_key": idempotency_key,
                "portfolio_id": None,
                "catalyst_id": target.catalyst_id,
                "exception_fingerprint": fingerprint,
                "source_id": target.source_id,
                "source_revision": capture.content_hash if capture else None,
                "reviewer_subject_id": reviewer,
                "reviewer_session_id": None,
                "assessment_text": human_assessment,
                "payload": semantic,
            })
        self._reviewed_exceptions[exception_id] = reviewed
        self._persist_to_storage()
        return reviewed

