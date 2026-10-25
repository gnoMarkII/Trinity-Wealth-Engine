"""Application Layer Service for 8-Quarter Promise Check Orchestration.

Coordinates:
- Scraper (StockAnalysis live discovery & fetch)
- Vault Adapter (atomic cache of raw text & promise_ledger.json)
- Promise Extractor (LLM forward guidance extraction & resolution)
- Core Scoring (100% deterministic Python scoring, validation, guidance bias, milestone slips)
"""
from datetime import datetime
import logging
from typing import Any, Dict, List, Optional

from core.promise_check.scoring import (
    compute_promise_scores,
    compute_promise_delta_and_slip,
    validate_ledger,
    qkey,
    qlabel,
    FRAMEWORK_VERSION,
)
from application.earnings_call.ports import (
    PromiseExtractorPort,
    TranscriptScraperPort,
    TranscriptVaultPort,
)
from schemas.promise_schemas import (
    TranscriptCallDTO,
    PromiseItemDTO,
    PromiseScoresDTO,
    PromiseLedgerDTO,
    PromiseSyncResponseDTO,
)

log = logging.getLogger(__name__)


class PromiseApplicationService:
    """Orchestrates 8-quarter transcript fetching, promise extraction, and deterministic scoring."""

    def __init__(
        self,
        extractor_port: Optional[PromiseExtractorPort] = None,
        scraper_port: Optional[TranscriptScraperPort] = None,
        vault_port: Optional[TranscriptVaultPort] = None,
    ) -> None:
        self._extractor = extractor_port
        self._scraper = scraper_port
        self._vault = vault_port

    def _get_scraper(self) -> TranscriptScraperPort:
        if self._scraper:
            return self._scraper
        raise RuntimeError("TranscriptScraperPort was not injected into PromiseApplicationService")

    def _get_vault(self) -> TranscriptVaultPort:
        if self._vault:
            return self._vault
        raise RuntimeError("TranscriptVaultPort was not injected into PromiseApplicationService")

    def get_ledger(self, ticker: str) -> Optional[PromiseScoresDTO]:
        """Retrieve existing promise ledger and compute current scores."""
        clean_ticker = ticker.strip().upper()
        raw = self._get_vault().load_ledger(clean_ticker)
        if not raw:
            return None
        
        errors, warnings = validate_ledger(raw)
        if errors:
            log.warning("Ledger validation errors for %s: %s", clean_ticker, errors)
            
        scores = compute_promise_scores(raw)
        return PromiseScoresDTO.model_validate(scores)

    def get_sync_response(self, ticker: str) -> Optional[PromiseSyncResponseDTO]:
        """Retrieve existing promise ledger, compute current scores, and wrap in full DTO."""
        clean_ticker = ticker.strip().upper()
        raw = self._get_vault().load_ledger(clean_ticker)
        if not raw:
            return None
        
        errors, warnings = validate_ledger(raw)
        if errors:
            log.warning("Ledger validation errors for %s: %s", clean_ticker, errors)
            
        scores = compute_promise_scores(raw)
        scores_dto = PromiseScoresDTO.model_validate(scores)

        promise_dtos: List[PromiseItemDTO] = []
        for p in raw.get("promises", []):
            try:
                promise_dtos.append(PromiseItemDTO.model_validate(p))
            except Exception:
                pass

        ledger_dto = PromiseLedgerDTO(
            ticker=clean_ticker,
            company=raw.get("company", clean_ticker),
            framework_version=raw.get("framework_version", FRAMEWORK_VERSION),
            run_date=raw.get("run_date", ""),
            calls=[TranscriptCallDTO.model_validate(c) for c in raw.get("calls", [])],
            promises=promise_dtos,
        )

        return PromiseSyncResponseDTO(
            ledger=ledger_dto,
            scores=scores_dto,
            from_cache=True,
        )

    def sync_ledger(
        self,
        ticker: str,
        force_refresh: bool = False,
        limit_calls: int = 8,
    ) -> PromiseSyncResponseDTO:
        """Fetch transcripts, extract new commitments sequentially, resolve past promises, and persist ledger."""
        clean_ticker = ticker.strip().upper()
        
        # 1. Load existing ledger or initialize fresh one
        existing_raw = self._get_vault().load_ledger(clean_ticker) if not force_refresh else None
        ledger: Dict[str, Any] = existing_raw or {
            "framework_version": FRAMEWORK_VERSION,
            "ticker": clean_ticker,
            "company": clean_ticker,
            "run_date": datetime.utcnow().strftime("%Y-%m-%d"),
            "calls": [],
            "promises": [],
        }

        # 2. Discover calls on StockAnalysis
        discovered_calls = self._get_scraper().discover_calls(clean_ticker, limit=limit_calls)
        
        # Merge discovered calls with existing calls
        calls_by_q: Dict[str, Dict[str, Any]] = {c["quarter"]: c for c in ledger.get("calls", [])}
        for call_dto in discovered_calls:
            q = call_dto.quarter
            if q not in calls_by_q:
                calls_by_q[q] = call_dto.model_dump()
            else:
                # Update source url if newly discovered
                if call_dto.source_url and not calls_by_q[q].get("source_url"):
                    calls_by_q[q]["source_url"] = call_dto.source_url

        # Sort calls chronologically (oldest to newest)
        sorted_calls = sorted(calls_by_q.values(), key=lambda c: qkey(c.get("quarter")) or 0)
        # Limit to the latest 8 quarters
        if len(sorted_calls) > limit_calls:
            # Mark older calls as retired
            for c in sorted_calls[:-limit_calls]:
                c["retired"] = True
            active_calls = sorted_calls[-limit_calls:]
        else:
            active_calls = sorted_calls

        ledger["calls"] = sorted_calls
        
        new_promises_count = 0
        resolved_promises_count = 0

        # 3. Process each active call chronologically
        for call_dict in active_calls:
            quarter = call_dict["quarter"]
            date_str = call_dict.get("date", "")
            
            # Ensure transcript text is cached
            text = self._get_vault().load_transcript(clean_ticker, quarter)
            if not text and call_dict.get("source_url"):
                try:
                    call_dto = TranscriptCallDTO.model_validate(call_dict)
                    text, words = self._get_scraper().fetch_transcript(call_dto)
                    self._get_vault().save_transcript(clean_ticker, quarter, text)
                    call_dict["words"] = words
                    call_dict["status"] = "ok"
                except Exception as exc:
                    log.warning("Failed to fetch transcript text for %s %s: %s", clean_ticker, quarter, exc)
                    call_dict["status"] = "missing"
                    continue
            
            if not text:
                continue
            
            # Check if this call was already extracted
            existing_made_in = [p for p in ledger["promises"] if p.get("made_in") == quarter]
            
            # 3a. Extraction step: if no promises yet for this call and extractor is provided
            if not existing_made_in and self._extractor:
                open_promises = [p for p in ledger["promises"] if p.get("status") == "Pending"]
                extraction_result = self._extractor.extract_commitments_from_transcript(
                    company=ledger.get("company", clean_ticker),
                    ticker=clean_ticker,
                    fiscal_quarter=quarter,
                    call_date=date_str,
                    fy_end_month="December",
                    open_promises=open_promises,
                    transcript_text=text,
                )
                
                raw_extracted = extraction_result.get("promises", [])
                for idx, p in enumerate(raw_extracted[:12], start=1):
                    p["id"] = f"P-{clean_ticker}-{quarter}-{idx:02d}"
                    p["made_in"] = quarter
                    p["status"] = "Pending"
                    p["extraction_method"] = "direct_http"
                    ledger["promises"].append(p)
                    new_promises_count += 1

                # Reiterations tracking
                raw_reiterations = extraction_result.get("reiterations", [])
                for reit in raw_reiterations:
                    reit_id = (reit.get("existing_id") or reit.get("id")) if isinstance(reit, dict) else str(reit)
                    for p in ledger["promises"]:
                        if p.get("id") == reit_id:
                            reits = p.setdefault("reiterated_in", [])
                            if quarter not in reits:
                                reits.append(quarter)

                # Restatements tracking
                raw_restatements = extraction_result.get("restatements", [])
                for rest in raw_restatements:
                    if isinstance(rest, dict):
                        rest_id = rest.get("existing_id") or rest.get("id")
                        for p in ledger["promises"]:
                            if p.get("id") == rest_id and p.get("status") == "Pending":
                                p["status"] = "Restated"
                                p["resolved_in"] = quarter
                                p["restated_direction"] = rest.get("direction") or "lowered"
                                if rest.get("new_target"):
                                    p["notes"] = f"Restated in {quarter}: {rest.get('new_target')}"
                                resolved_promises_count += 1
            
            # 3b. Resolution step: check open pending promises due in or before this quarter
            call_k = qkey(quarter) or 0
            for p in ledger["promises"]:
                if p.get("status") != "Pending":
                    continue
                made_k = qkey(p.get("made_in")) or 0
                if made_k >= call_k:
                    # Cannot resolve in the same or earlier call than when the commitment was made
                    continue
                due_k = qkey(p.get("due")) or 0
                
                # Check if this call can resolve the promise (due <= current call)
                if due_k <= call_k and self._extractor:
                    resolution = self._extractor.resolve_promise(
                        promise=p,
                        resolving_quarter=quarter,
                        grace_quarter=None,
                        transcript_text=text,
                    )
                    status = resolution.get("status")
                    if status and status in ("Delivered", "Partial", "Missed", "Restated"):
                        p["status"] = status
                        p["resolved_in"] = quarter
                        p["evidence_quote"] = resolution.get("evidence_quote", "")
                        outcome_val = resolution.get("outcome_value")
                        p["outcome_value"] = outcome_val
                        p["actual_numeric"] = outcome_val
                        calc_delta, calc_slip = compute_promise_delta_and_slip(p, outcome_val, resolved_in=quarter)
                        p["delta_pct"] = calc_delta
                        p["slip_quarters"] = calc_slip
                        if resolution.get("restated_direction"):
                            p["restated_direction"] = resolution.get("restated_direction")
                        if resolution.get("notes"):
                            p["notes"] = resolution.get("notes")
                        if resolution.get("notes_th"):
                            p["notes_th"] = resolution.get("notes_th")
                        resolved_promises_count += 1
                    elif status == "Unverifiable":
                        p["status"] = "Unverifiable"
                        p["resolved_in"] = quarter
                        p["evidence_quote"] = resolution.get("evidence_quote", "")
                        p["notes"] = resolution.get("notes") or f"Searched calls {p.get('due')} through {quarter} transcripts but commitment was unverifiable."
                        p["notes_th"] = resolution.get("notes_th") or f"ตรวจสอบบทสนทนาตั้งแต่ {p.get('due')} ถึง {quarter} แต่ไม่สามารถยืนยันผลลัพธ์ได้"
                        resolved_promises_count += 1
                    elif due_k + 1 <= call_k:
                        # Grace period (due + 1) has expired without resolution -> Quietly dropped
                        p["status"] = "Quietly dropped"
                        p["resolved_in"] = quarter
                        p["notes"] = (
                            f"Searched calls {p.get('due')} through {quarter} transcripts "
                            "but commitment was omitted without restatement."
                        )
                        p["notes_th"] = (
                            f"ตรวจสอบบทสนทนาตั้งแต่ {p.get('due')} ถึง {quarter} "
                            "แต่ไม่มีการเอ่ยถึงหรือแถลงเป้าหมายใหม่"
                        )
                        resolved_promises_count += 1

        # 4. Atomic write of updated ledger
        self._get_vault().save_ledger(clean_ticker, ledger)

        # 5. Compute deterministic scores
        scores = compute_promise_scores(ledger)
        scores_dto = PromiseScoresDTO.model_validate(scores)

        # 6. Build promise items DTOs
        promise_dtos: List[PromiseItemDTO] = []
        for p in ledger.get("promises", []):
            try:
                promise_dtos.append(PromiseItemDTO.model_validate(p))
            except Exception:
                pass

        ledger_dto = PromiseLedgerDTO(
            ticker=clean_ticker,
            company=ledger.get("company", clean_ticker),
            framework_version=FRAMEWORK_VERSION,
            run_date=ledger.get("run_date", datetime.utcnow().strftime("%Y-%m-%d")),
            calls=[TranscriptCallDTO.model_validate(c) for c in ledger.get("calls", [])],
            promises=promise_dtos,
        )

        return PromiseSyncResponseDTO(
            ledger=ledger_dto,
            scores=scores_dto,
            from_cache=False,
        )
