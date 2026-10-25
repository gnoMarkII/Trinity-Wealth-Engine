"""Deterministic scoring and validation engine for Promise Check ledgers.

Pure Domain Logic:
- No framework, network, sqlite, or API dependencies.
- Adheres to Framework v0.1 rules (points, weighting, guidance bias, transparency).
"""
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
import re
import statistics
from typing import Any, Dict, List, Optional, Tuple

FRAMEWORK_VERSION = "0.1"

STATUSES = [
    "Delivered",
    "Partial",
    "Missed",
    "Restated",
    "Quietly dropped",
    "Pending",
    "Unverifiable",
]

POINTS = {
    "Delivered": 100,
    "Partial": 50,
    "Restated-lowered": 40,
    "Missed": 15,
    "Quietly dropped": 0,
}

GRADES = [(85, "A"), (70, "B"), (55, "C"), (40, "D"), (0, "F")]

Q_RE = re.compile(r"^FY(\d{4})Q([1-4])$")
MAX_PER_CALL = 12
MIN_BIAS_N = 5


def qkey(q: Optional[str]) -> Optional[int]:
    """Convert fiscal label 'FY2026Q3' to a sortable integer (e.g. 2026*4 + 2)."""
    m = Q_RE.match(q or "")
    if not m:
        return None
    return int(m.group(1)) * 4 + int(m.group(2)) - 1


def qlabel(q: Optional[str]) -> str:
    """Format 'FY2026Q3' -> 'FY26 Q3'."""
    if not q:
        return ""
    m = Q_RE.match(q)
    return f"FY{m.group(1)[2:]} Q{m.group(2)}" if m else q


def word_count(s: Optional[str]) -> int:
    return len((s or "").split())


def weight_of(p: Dict[str, Any]) -> float:
    """Calculate point weight based on promise type, scope, and deadline precision."""
    if p.get("type") == "A_numeric" and p.get("target_kind") in ("range", "point", "floor", "ceiling"):
        w = 2.0 if p.get("scope") == "company" else 1.5
    else:  # direction-only guidance or milestone
        w = 1.0
    if p.get("due_precision") == "inferred":
        w *= 0.75
    return w


def score_key(p: Dict[str, Any]) -> Optional[str]:
    """Map promise status to its corresponding scoring bucket in POINTS, or None if un-scored."""
    s = p.get("status")
    if s == "Restated":
        return "Restated-lowered" if p.get("restated_direction") == "lowered" else None
    return s if s in POINTS else None


def validate_ledger(ledger: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """Validate ledger consistency and strict framework rules.
    
    Returns:
        (errors, warnings) lists of strings.
    """
    errors: List[str] = []
    warnings: List[str] = []

    calls = ledger.get("calls", [])
    call_q = {c.get("quarter") for c in calls if c.get("quarter")}
    ok_calls = [c for c in calls if c.get("status") == "ok"]

    if not ok_calls:
        errors.append("no usable calls in calls[]")
        return errors, warnings

    latest = max(qkey(c["quarter"]) for c in ok_calls if qkey(c.get("quarter")) is not None)

    if len(calls) != 8:
        warnings.append(f"calls[] has {len(calls)} entries, expected 8")

    for c in calls:
        if qkey(c.get("quarter")) is None:
            errors.append(f"bad quarter label in calls[]: {c.get('quarter')}")
        if c.get("status") == "missing":
            warnings.append(f"{c.get('quarter')} transcript missing — page must say so")

    promises = ledger.get("promises", [])
    ids = Counter(p.get("id") for p in promises if p.get("id"))
    for pid, n in ids.items():
        if n > 1:
            errors.append(f"duplicate id {pid}")

    per_call = Counter(p.get("made_in") for p in promises if not p.get("supersedes"))
    for q, n in per_call.items():
        if n > MAX_PER_CALL:
            errors.append(f"{q}: {n} promises exceeds cap of {MAX_PER_CALL}")

    for p in promises:
        pid = p.get("id", "UNKNOWN_ID")
        status = p.get("status")
        if status not in STATUSES:
            errors.append(f"{pid}: unknown status {status}")

        made_in = p.get("made_in")
        if made_in not in call_q:
            errors.append(f"{pid}: made_in {made_in} not in calls[]")

        mk, dk = qkey(made_in), qkey(p.get("due"))
        if dk is None:
            errors.append(f"{pid}: bad due {p.get('due')}")
            continue

        if mk is not None and dk < mk:
            errors.append(f"{pid}: due {p.get('due')} is before made_in {made_in}")

        if word_count(p.get("quote")) > 40:
            warnings.append(f"{pid}: quote is {word_count(p.get('quote'))} words (>40)")

        if status == "Pending":
            if dk <= latest and p.get("type") == "A_numeric":
                errors.append(f"{pid}: Pending but due {p.get('due')} is within the window")
            if dk <= latest and p.get("type") == "B_milestone" and dk + 1 <= latest:
                errors.append(f"{pid}: Pending milestone but due+1 is within the window")
        else:
            if status in ("Quietly dropped", "Unverifiable"):
                if not p.get("notes"):
                    errors.append(f"{pid}: {status} requires a note saying which transcripts were searched")
            elif not p.get("evidence_quote"):
                errors.append(f"{pid}: status {status} without evidence_quote")

            resolved_in = p.get("resolved_in")
            if not resolved_in:
                errors.append(f"{pid}: status {status} without resolved_in")
            else:
                if resolved_in not in call_q:
                    errors.append(f"{pid}: resolved_in {resolved_in} not in calls[]")
                rk = qkey(resolved_in)
                if rk is not None and mk is not None and rk <= mk:
                    errors.append(f"{pid}: resolved_in {resolved_in} is not after made_in {made_in}")

            if word_count(p.get("evidence_quote")) > 40:
                warnings.append(f"{pid}: evidence_quote is {word_count(p.get('evidence_quote'))} words (>40)")

        if status == "Restated" and p.get("restated_direction") not in ("raised", "lowered"):
            errors.append(f"{pid}: Restated without restated_direction")

        if status == "Quietly dropped" and dk + 1 > latest:
            errors.append(f"{pid}: Quietly dropped but grace period (due+1) has not expired in window")

        if p.get("extraction_method") == "webfetch":
            warnings.append(f"{pid}: extracted via WebFetch — quote not verified against saved text")

    return errors, warnings


def compute_promise_scores(ledger: Dict[str, Any]) -> Dict[str, Any]:
    """Compute deterministic credibility scores, delivery metrics, and guidance bias.
    
    Zero LLM arithmetic: 100% computed via strict Python math.
    """
    raw_promises = ledger.get("promises", [])
    raw_calls = ledger.get("calls", [])

    # Filter out retired rows from scoring
    promises = [p for p in raw_promises if not p.get("retired")]
    calls = sorted([c for c in raw_calls if not c.get("retired")], key=lambda c: qkey(c.get("quarter")) or 0)

    counts = {s: 0 for s in STATUSES}
    for p in promises:
        status = p.get("status")
        if status in counts:
            counts[status] += 1

    counts["Restated (raised)"] = sum(
        1 for p in promises if p.get("status") == "Restated" and p.get("restated_direction") == "raised"
    )
    counts["Restated (lowered)"] = sum(
        1 for p in promises if p.get("status") == "Restated" and p.get("restated_direction") == "lowered"
    )

    scored = [(p, score_key(p)) for p in promises if score_key(p) is not None]
    n_resolved = len(scored)

    if n_resolved > 0:
        num = sum(POINTS[k] * weight_of(p) for p, k in scored if k is not None)
        den = sum(weight_of(p) for p, _ in scored)
        delivery = num / den if den > 0 else 0.0
    else:
        delivery = None

    owned = counts["Missed"] + counts["Restated (lowered)"]
    dropped = counts["Quietly dropped"]
    transparency = 100.0 if (owned + dropped) == 0 else 100.0 * owned / (owned + dropped)
    credibility = None if delivery is None else (0.8 * delivery + 0.2 * transparency)
    grade = None if credibility is None else next(g for t, g in GRADES if credibility >= t)
    confidence = "Low" if n_resolved < 10 else "Medium" if n_resolved < 20 else "High"

    # Guidance Bias calculation (Type A numeric resolved with delta_pct)
    deltas = [
        p["delta_pct"] for p, k in scored
        if p.get("type") == "A_numeric" and p.get("delta_pct") is not None
    ]

    if len(deltas) >= MIN_BIAS_N:
        med = statistics.median(deltas)
        beat_rate = sum(1 for d in deltas if d > 0) / len(deltas)
        if med > 0.02 and beat_rate >= 0.8:
            bias, bias_code = "Sandbagger", "sandbagger"
            bias_msg = "They guide low and beat. Treat their guidance as a floor, not a forecast."
        elif abs(med) <= 0.02:
            bias, bias_code = "Straight shooter", "straight"
            bias_msg = "Their numbers land close to what they said. Take guidance at face value."
        elif med < -0.02:
            bias, bias_code = "Overpromiser", "overpromiser"
            bias_msg = "Actuals tend to come in below guidance. Discount what they tell you."
        else:
            bias, bias_code = "Mixed", "mixed"
            bias_msg = "No consistent lean — beats and misses roughly balance."
    else:
        med = beat_rate = None
        bias, bias_code = "Not enough numeric guidance", "insufficient"
        bias_msg = (
            f"Only {len(deltas)} resolved promise(s) carried a checkable number, too few to judge "
            "whether they guide high or low. This management team commits mostly to milestones "
            "and dates rather than figures — judge them on the timelines below."
        )

    # Average Milestone Slip
    slips = [
        p["slip_quarters"] for p, k in scored
        if p.get("type") == "B_milestone" and p.get("slip_quarters") is not None
    ]
    avg_slip = statistics.mean(slips) if len(slips) >= 3 else None

    # Trend by made_in cohort
    cohorts: List[Dict[str, Any]] = []
    by_q: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for p in promises:
        made_in = p.get("made_in")
        if made_in:
            by_q[made_in].append(p)

    for c in calls:
        quarter = c.get("quarter", "")
        rows = by_q.get(quarter, [])
        res = [p for p in rows if score_key(p) is not None]
        hit = sum(1 for p in res if p.get("status") == "Delivered")
        cohorts.append({
            "quarter": quarter,
            "label": qlabel(quarter),
            "date": c.get("date"),
            "status": c.get("status", "ok"),
            "made": len(rows),
            "resolved": len(res),
            "delivered": hit,
            "pending": sum(1 for p in rows if p.get("status") == "Pending"),
            "hit_rate": (hit / len(res)) if res else None,
        })

    usable = [c for c in cohorts if c["resolved"] >= 2]
    if len(usable) >= 4:
        half = len(usable) // 2
        older = statistics.mean(c["hit_rate"] for c in usable[:half] if c["hit_rate"] is not None)
        newer = statistics.mean(c["hit_rate"] for c in usable[-half:] if c["hit_rate"] is not None)
        diff = newer - older
        trend = "improving" if diff >= 0.10 else "deteriorating" if diff <= -0.10 else "stable"
    else:
        older = newer = None
        trend = "insufficient data"

    # Top 3 Kept & Broken promises
    sev = {"Quietly dropped": 0, "Missed": 1, "Restated-lowered": 2, "Partial": 3}

    def soft(p: Dict[str, Any]) -> int:
        return 1 if p.get("target_kind") == "direction" else 0

    kept = sorted(
        (p for p in promises if p.get("status") == "Delivered"),
        key=lambda p: (
            soft(p),
            -weight_of(p),
            p.get("slip_quarters") if p.get("slip_quarters") is not None else 0,
            -(p.get("delta_pct") or 0),
        ),
    )[:3]

    broken = sorted(
        (p for p, k in scored if k != "Delivered"),
        key=lambda p: (
            sev.get(score_key(p) or "", 9),
            soft(p),
            -weight_of(p),
            (p.get("delta_pct") or 0),
        ),
    )[:3]

    return {
        "framework_version": ledger.get("framework_version", FRAMEWORK_VERSION),
        "ticker": ledger.get("ticker", ""),
        "company": ledger.get("company", ""),
        "run_date": ledger.get("run_date", ""),
        "window": {
            "from": qlabel(calls[0]["quarter"]) if calls else "",
            "to": qlabel(calls[-1]["quarter"]) if calls else "",
            "calls": len(calls),
            "missing": [c["quarter"] for c in calls if c.get("status") == "missing"],
        },
        "credibility": None if credibility is None else round(credibility, 1),
        "grade": grade,
        "confidence": confidence,
        "delivery": None if delivery is None else round(delivery, 1),
        "transparency": round(transparency, 1),
        "n_promises": len(promises),
        "n_resolved": n_resolved,
        "counts": counts,
        "bias": {
            "label": bias,
            "code": bias_code,
            "message": bias_msg,
            "median_delta_pct": None if med is None else round(med * 100, 2),
            "beat_rate": None if beat_rate is None else round(beat_rate * 100, 1),
            "n": len(deltas),
            "min_n": MIN_BIAS_N,
        },
        "milestone_slip_quarters": None if avg_slip is None else round(avg_slip, 2),
        "milestone_slip_n": len(slips),
        "trend": {
            "verdict": trend,
            "older_half_hit_rate": None if older is None else round(older * 100, 1),
            "newer_half_hit_rate": None if newer is None else round(newer * 100, 1),
            "cohorts": cohorts,
        },
        "top_kept": [p["id"] for p in kept if p.get("id")],
        "top_broken": [p["id"] for p in broken if p.get("id")],
        "weights": {p["id"]: weight_of(p) for p in promises if p.get("id")},
    }


def compute_promise_delta_and_slip(
    promise: Dict[str, Any],
    outcome_value: Optional[float],
    resolved_in: Optional[str] = None,
) -> Tuple[Optional[float], Optional[int]]:
    """Deterministically computes fractional delta_pct and slip_quarters with Decimal precision."""
    slip_quarters: Optional[int] = None
    if resolved_in and promise.get("due"):
        rk = qkey(resolved_in)
        dk = qkey(promise.get("due"))
        if rk is not None and dk is not None:
            slip_quarters = max(0, rk - dk)

    delta_pct: Optional[float] = None
    if outcome_value is not None:
        try:
            outcome_dec = Decimal(str(outcome_value))
            target_point = promise.get("target_point")
            target_low = promise.get("target_low")
            target_high = promise.get("target_high")

            if target_point is not None:
                tgt_dec = Decimal(str(target_point))
                if tgt_dec != 0:
                    delta_pct = float((outcome_dec - tgt_dec) / abs(tgt_dec))
                else:
                    delta_pct = float(outcome_dec)
            elif target_low is not None and target_high is not None:
                low_dec = Decimal(str(target_low))
                high_dec = Decimal(str(target_high))
                if outcome_dec > high_dec and abs(high_dec) > 0:
                    delta_pct = float((outcome_dec - high_dec) / abs(high_dec))
                elif outcome_dec < low_dec and abs(low_dec) > 0:
                    delta_pct = float((outcome_dec - low_dec) / abs(low_dec))
                else:
                    delta_pct = 0.0
            elif target_low is not None:
                low_dec = Decimal(str(target_low))
                if outcome_dec < low_dec and abs(low_dec) > 0:
                    delta_pct = float((outcome_dec - low_dec) / abs(low_dec))
                else:
                    delta_pct = 0.0
            elif target_high is not None:
                high_dec = Decimal(str(target_high))
                if outcome_dec > high_dec and abs(high_dec) > 0:
                    delta_pct = float((outcome_dec - high_dec) / abs(high_dec))
                else:
                    delta_pct = 0.0
        except (InvalidOperation, ValueError, TypeError):
            delta_pct = None

    return delta_pct, slip_quarters

