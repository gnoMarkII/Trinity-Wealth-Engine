"""Pure deterministic settlement and benchmark alpha math for Roundtable scorecards.

Follows AGENTS.md rules:
- Pure domain module under ``core/roundtable/`` (zero external/framework/DB imports).
- Uses ``Decimal`` exclusively for financial returns and benchmark alpha.
- Prevents look-ahead bias via strict ``previous_date < settled_date`` enforcement.
"""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Optional

from core.roundtable.models import PastRoundtableScorecard


QUANTUM_PCT = Decimal("0.01")
HUNDRED = Decimal("100")


def default_benchmark_for_ticker(ticker: str) -> str:
    """Return canonical market benchmark ticker using asset suffix."""
    clean = ticker.strip().upper()
    if clean.endswith(".BK"):
        return "^SET.BK"
    return "SPY"


def _to_decimal(val: object) -> Optional[Decimal]:
    if val is None or isinstance(val, bool):
        return None
    try:
        d = Decimal(str(val).strip())
        if d.is_nan() or d.is_infinite():
            return None
        return d
    except (InvalidOperation, ValueError):
        return None


def compute_pct_return(start_price: Decimal, end_price: Decimal) -> Optional[Decimal]:
    """Compute percentage return ``((end / start) - 1) * 100`` rounded half-up to 0.01."""
    if start_price <= Decimal("0") or end_price <= Decimal("0"):
        return None
    raw = ((end_price / start_price) - Decimal("1")) * HUNDRED
    return raw.quantize(QUANTUM_PCT, rounding=ROUND_HALF_UP)


def compute_settlement_scorecard(
    *,
    previous_session_id: str,
    previous_created_at: float,
    previous_final_rating: str,
    primary_ticker: str,
    benchmark_ticker: Optional[str],
    entry_price: Optional[object],
    settled_price: Optional[object],
    settled_date_iso: Optional[str],
    benchmark_entry_price: Optional[object] = None,
    benchmark_settled_price: Optional[object] = None,
    reflection_lesson: Optional[str] = None,
) -> PastRoundtableScorecard:
    """Compute a look-ahead-safe closed-loop settlement scorecard for a prior roundtable."""
    prev_dt = datetime.fromtimestamp(float(previous_created_at), tz=timezone.utc).date()
    prev_date_iso = prev_dt.isoformat()
    bench = benchmark_ticker or default_benchmark_for_ticker(primary_ticker)

    entry_dec = _to_decimal(entry_price)
    settled_dec = _to_decimal(settled_price)

    if not settled_date_iso:
        return PastRoundtableScorecard(
            previous_session_id=previous_session_id,
            previous_date=prev_date_iso,
            primary_ticker=primary_ticker,
            benchmark_ticker=bench,
            entry_price=str(entry_dec) if entry_dec is not None else None,
            settlement_price=None,
            raw_return_pct=None,
            benchmark_return_pct=None,
            alpha_pct=None,
            previous_final_rating=previous_final_rating or "REVIEW",
            outcome_label="pending",
            reflection_lesson=reflection_lesson,
            settlement_status="pending",
        )

    settled_date_clean = settled_date_iso[:10]
    try:
        settled_dt = datetime.strptime(settled_date_clean, "%Y-%m-%d").date()
    except ValueError:
        settled_dt = prev_dt

    # Look-ahead bias guard: do not settle using same-day or earlier price bars
    if settled_dt <= prev_dt:
        return PastRoundtableScorecard(
            previous_session_id=previous_session_id,
            previous_date=prev_date_iso,
            primary_ticker=primary_ticker,
            benchmark_ticker=bench,
            entry_price=str(entry_dec) if entry_dec is not None else None,
            settlement_price=None,
            raw_return_pct=None,
            benchmark_return_pct=None,
            alpha_pct=None,
            previous_final_rating=previous_final_rating or "REVIEW",
            outcome_label="lookahead_skipped",
            reflection_lesson=reflection_lesson,
            settlement_status="lookahead_skipped",
        )

    if entry_dec is None or settled_dec is None:
        return PastRoundtableScorecard(
            previous_session_id=previous_session_id,
            previous_date=prev_date_iso,
            primary_ticker=primary_ticker,
            benchmark_ticker=bench,
            entry_price=str(entry_dec) if entry_dec is not None else None,
            settlement_price=str(settled_dec) if settled_dec is not None else None,
            raw_return_pct=None,
            benchmark_return_pct=None,
            alpha_pct=None,
            previous_final_rating=previous_final_rating or "REVIEW",
            outcome_label="unavailable",
            reflection_lesson=reflection_lesson,
            settlement_status="unavailable",
        )

    raw_ret = compute_pct_return(entry_dec, settled_dec)
    if raw_ret is None:
        return PastRoundtableScorecard(
            previous_session_id=previous_session_id,
            previous_date=prev_date_iso,
            primary_ticker=primary_ticker,
            benchmark_ticker=bench,
            entry_price=str(entry_dec),
            settlement_price=str(settled_dec),
            raw_return_pct=None,
            benchmark_return_pct=None,
            alpha_pct=None,
            previous_final_rating=previous_final_rating or "REVIEW",
            outcome_label="unavailable",
            reflection_lesson=reflection_lesson,
            settlement_status="unavailable",
        )

    b_entry_dec = _to_decimal(benchmark_entry_price)
    b_settled_dec = _to_decimal(benchmark_settled_price)
    bench_ret: Optional[Decimal] = None
    alpha: Optional[Decimal] = None
    if b_entry_dec is not None and b_settled_dec is not None:
        bench_ret = compute_pct_return(b_entry_dec, b_settled_dec)
        if bench_ret is not None:
            alpha = (raw_ret - bench_ret).quantize(QUANTUM_PCT, rounding=ROUND_HALF_UP)

    compare_metric = alpha if alpha is not None else raw_ret
    rating_upper = (previous_final_rating or "REVIEW").strip().upper()
    if rating_upper in ("BUY", "OVERWEIGHT"):
        if compare_metric > Decimal("0"):
            outcome = "outperformed"
        elif compare_metric < Decimal("0"):
            outcome = "lagged"
        else:
            outcome = "neutral"
    elif rating_upper in ("UNDERWEIGHT", "SELL"):
        if compare_metric < Decimal("0"):
            outcome = "outperformed"
        elif compare_metric > Decimal("0"):
            outcome = "lagged"
        else:
            outcome = "neutral"
    else:
        outcome = "neutral"

    return PastRoundtableScorecard(
        previous_session_id=previous_session_id,
        previous_date=prev_date_iso,
        primary_ticker=primary_ticker,
        benchmark_ticker=bench,
        entry_price=str(entry_dec.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)),
        settlement_price=str(settled_dec.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)),
        raw_return_pct=str(raw_ret),
        benchmark_return_pct=str(bench_ret) if bench_ret is not None else None,
        alpha_pct=str(alpha) if alpha is not None else None,
        previous_final_rating=rating_upper,
        outcome_label=outcome,
        reflection_lesson=reflection_lesson,
        settlement_status="settled",
    )

