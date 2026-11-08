"""Institutional Catalyst Exception Queue & Factual Occurrence Engine (Chapter 7 & App H).



Architecture Rule (Layer 1 - Core):

Pure Python standard library only — strictly zero external dependencies (no FastAPI, Pydantic,

SQLite, or requests) per Hexagonal Architecture AST Domain Layer Isolation rules.



Reference:

Claude Code for Hedge Funds, Chapter 7 (pp. 48-54), Lab 7 (p. 89), App H (pp. 118-119):

"Link catalysts to claims, not just dates. Use a small factual exception vocabulary:

NEW_SOURCE, DATE_CHANGED, STALE, NO_CHANGE, UNTRUSTED_INSTRUCTION.

A filing monitor detects occurrence, not significance. The analyst_field (e.g. materiality_assessment)

remains blank for the human reviewer."

"""

from __future__ import annotations



from dataclasses import dataclass, field

from datetime import datetime, timezone

from enum import Enum

import hashlib

from typing import Any, Dict, List, Optional, Sequence, Tuple





class CatalystClassification(str, Enum):

    """The 5 canonical factual classifications per Chapter 7 (p. 50)."""

    NEW_SOURCE = "NEW_SOURCE"

    DATE_CHANGED = "DATE_CHANGED"

    STALE = "STALE"

    NO_CHANGE = "NO_CHANGE"

    UNTRUSTED_INSTRUCTION = "UNTRUSTED_INSTRUCTION"





@dataclass(frozen=True)

class CatalystRecord:

    """Institutional catalyst calendar record linked to an investment claim."""

    catalyst_id: str

    entity_id: str                          # Ticker or synthetic entity (e.g. "GOOGL", "SYN-NSTAR")

    event_name: str                         # e.g. "Q3 Earnings Release", "FDA PDUFA Date"

    expected_date: str                      # Format: YYYY-MM-DD

    source_id: str                          # Links to source register (e.g. "SYN-IR-001")

    last_verification_utc: str              # ISO timestamp of last verified occurrence

    linked_claim_id: str                    # Links to thesis claim (e.g. "TH-001")

    owner: str = "analyst_a"

    status: str = "ACTIVE"                  # "ACTIVE" (under review), "RETIRED", "TRIGGERED"

    evidence_label: str = "SYNTHETIC"       # "SYNTHETIC" or "DOCUMENTED"



    def to_dict(self) -> Dict[str, Any]:

        return {

            "catalyst_id": self.catalyst_id,

            "entity_id": self.entity_id,

            "event_name": self.event_name,

            "expected_date": self.expected_date,

            "source_id": self.source_id,

            "last_verification_utc": self.last_verification_utc,

            "linked_claim_id": self.linked_claim_id,

            "owner": self.owner,

            "status": self.status,

            "evidence_label": self.evidence_label,

        }





# Type alias for cross-compatibility

CatalystEvent = CatalystRecord





@dataclass(frozen=True)

class SourceCaptureRecord:

    """Captured document, filing, or calendar snapshot before evaluation."""

    source_id: str

    entity_id: str

    source_type: str                        # "SEC_FILING", "IR_CALENDAR", "PRESS_RELEASE"

    captured_at_utc: str

    effective_period: str                   # e.g. "2026Q3"

    source_locator: str                     # URL or file path

    content_snippet: str

    content_hash: str = ""

    is_untrusted_instruction: bool = False  # Flagged by Hostile Input Sanitizer



    def to_dict(self) -> Dict[str, Any]:

        return {

            "source_id": self.source_id,

            "entity_id": self.entity_id,

            "source_type": self.source_type,

            "captured_at_utc": self.captured_at_utc,

            "effective_period": self.effective_period,

            "source_locator": self.source_locator,

            "content_snippet": self.content_snippet,

            "content_hash": self.content_hash,

            "is_untrusted_instruction": self.is_untrusted_instruction,

        }





@dataclass(frozen=True)

class CatalystException:

    """An exception recording an observed factual change without subjective judgment.

    

    Appendix H (p. 118):

    - Old and new dates both survive

    - Factual classification is testable

    - Analyst field (materiality_assessment) remains open/blank for the human reviewer

    - Status is always REVIEW_REQUIRED

    """

    exception_id: str

    catalyst_id: str

    entity_id: str

    classification: CatalystClassification

    source_id: str

    observed_fact: str

    prior_date: Optional[str] = None

    new_date: Optional[str] = None

    analyst_field: str = "materiality_assessment"

    human_assessment: Optional[str] = None  # Left blank for the PM per book rule

    status: str = "REVIEW_REQUIRED"

    detected_utc: str = ""



    def to_dict(self) -> Dict[str, Any]:

        return {

            "exception_id": self.exception_id,

            "catalyst_id": self.catalyst_id,

            "entity_id": self.entity_id,

            "classification": self.classification.value,

            "source_id": self.source_id,

            "observed_fact": self.observed_fact,

            "prior_date": self.prior_date,

            "new_date": self.new_date,

            "analyst_field": self.analyst_field,

            "human_assessment": self.human_assessment,

            "status": self.status,

            "detected_utc": self.detected_utc,

        }





def _parse_iso_utc(date_str: str) -> Optional[datetime]:

    """Helper to parse ISO-8601 or YYYY-MM-DD date string to UTC datetime."""

    clean = date_str.strip().replace("Z", "+00:00")

    if len(clean) == 10 and clean[4] == "-" and clean[7] == "-":

        clean += "T00:00:00+00:00"

    try:

        dt = datetime.fromisoformat(clean)

        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)

    except Exception:

        return None





def evaluate_catalyst_exception(
    catalyst: CatalystRecord,
    capture: Optional[SourceCaptureRecord] = None,
    new_expected_date: Optional[str] = None,
    as_of_utc: Optional[str] = None,
    stale_days_threshold: int = 60,
    exception_counter: int = 1,
) -> CatalystException:
    """Evaluates a catalyst against fresh capture and as-of cutoff.
    
    Invariants enforced:
    1. Hostile instructions in capture text -> UNTRUSTED_INSTRUCTION (quarantine).
    2. Date movement -> DATE_CHANGED (preserves both prior_date and new_date).
    3. New filing/capture attached without date change -> NEW_SOURCE.
    4. Verification date older than stale_days_threshold -> STALE.
    5. No change detected -> NO_CHANGE.
    6. Human field remains unpopulated; status is always REVIEW_REQUIRED.
    """
    now_iso = as_of_utc or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    d_key = now_iso[:10]

    # 1. Untrusted instruction gate
    if capture and capture.is_untrusted_instruction:
        cls_val = CatalystClassification.UNTRUSTED_INSTRUCTION
        src_id = capture.source_id
        fact = "Source text contained instruction-shaped content that was not executed"
        p_date = catalyst.expected_date
        n_date = catalyst.expected_date
        field_name = "security_review"
    # 2. Date changed gate
    elif new_expected_date and new_expected_date.strip() != catalyst.expected_date.strip():
        cls_val = CatalystClassification.DATE_CHANGED
        src_id = capture.source_id if capture else catalyst.source_id
        fact = f"Expected date changed from {catalyst.expected_date} to {new_expected_date.strip()}"
        p_date = catalyst.expected_date
        n_date = new_expected_date.strip()
        field_name = "materiality_assessment"
    # 3. New source captured gate
    elif capture and capture.source_id != catalyst.source_id:
        cls_val = CatalystClassification.NEW_SOURCE
        src_id = capture.source_id
        fact = f"New {capture.source_type} captured ({capture.source_id}): {capture.content_snippet[:80]}"
        p_date = catalyst.expected_date
        n_date = catalyst.expected_date
        field_name = "significance_assessment"
    else:
        last_dt = _parse_iso_utc(catalyst.last_verification_utc)
        curr_dt = _parse_iso_utc(now_iso) or datetime.now(timezone.utc)
        if last_dt and (curr_dt - last_dt).total_seconds() / 86400.0 > stale_days_threshold:
            days_diff = (curr_dt - last_dt).total_seconds() / 86400.0
            cls_val = CatalystClassification.STALE
            src_id = catalyst.source_id
            fact = f"Prior verification ({catalyst.last_verification_utc[:10]}) is older than the {stale_days_threshold}-day threshold ({int(days_diff)} days)"
            p_date = catalyst.expected_date
            n_date = catalyst.expected_date
            field_name = "source_reverification"
        else:
            cls_val = CatalystClassification.NO_CHANGE
            src_id = catalyst.source_id
            fact = f"Catalyst verified current as-of {d_key}"
            p_date = catalyst.expected_date
            n_date = catalyst.expected_date
            field_name = "routine_check"

    raw_key = f"{catalyst.entity_id}:{catalyst.catalyst_id}:{src_id}:{cls_val.value}:{p_date}:{n_date}:{d_key}"
    hash_id = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:8].upper()
    ex_id = f"EX-{catalyst.entity_id}-{hash_id}"

    return CatalystException(
        exception_id=ex_id,
        catalyst_id=catalyst.catalyst_id,
        entity_id=catalyst.entity_id,
        classification=cls_val,
        source_id=src_id,
        observed_fact=fact,
        prior_date=p_date,
        new_date=n_date,
        analyst_field=field_name,
        status="REVIEW_REQUIRED",
        detected_utc=now_iso,
    )


def scan_catalyst_queue(

    catalysts: Sequence[CatalystRecord],

    captures: Sequence[SourceCaptureRecord] = (),

    date_overrides: Optional[Dict[str, str]] = None,

    as_of_utc: Optional[str] = None,

    stale_days_threshold: int = 60,

) -> List[CatalystException]:

    """Batch-scans a sequence of catalysts and returns all non-NO_CHANGE exceptions or full queue."""

    overrides = date_overrides or {}

    # Filter captures strictly respecting temporal provenance: no look-ahead bias

    valid_captures = []

    as_of_dt = _parse_iso_utc(as_of_utc) if as_of_utc else None

    for c in captures:

        if as_of_dt:

            c_dt = _parse_iso_utc(c.captured_at_utc)

            if c_dt and c_dt <= as_of_dt:

                valid_captures.append(c)

        else:

            valid_captures.append(c)



    captures_by_entity: Dict[str, SourceCaptureRecord] = {c.entity_id: c for c in valid_captures}

    captures_by_source: Dict[str, SourceCaptureRecord] = {c.source_id: c for c in valid_captures}



    exceptions: List[CatalystException] = []

    counter = 1

    for cat in catalysts:

        cap = captures_by_source.get(cat.source_id) or captures_by_entity.get(cat.entity_id)

        new_date = overrides.get(cat.catalyst_id)

        ex = evaluate_catalyst_exception(

            catalyst=cat,

            capture=cap,

            new_expected_date=new_date,

            as_of_utc=as_of_utc,

            stale_days_threshold=stale_days_threshold,

            exception_counter=counter,

        )

        exceptions.append(ex)

        counter += 1



    return exceptions

