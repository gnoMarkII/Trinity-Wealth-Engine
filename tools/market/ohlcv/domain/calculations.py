"""Domain Calculations for OHLCV Market Data, Warmup, Pivots, 52W and Corporate Action Mapping."""
import logging
from datetime import datetime
from typing import Optional, Literal
from zoneinfo import ZoneInfo
from dateutil.relativedelta import relativedelta
import pandas as pd

from tools.market.ohlcv.domain.models import (
    OHLCVCandleDTO,
    PivotLevelsDTO,
    CorporateActionEventDTO,
    IndicatorBurnInPolicyDTO,
    IndicatorWarmupDetailDTO,
)

log = logging.getLogger(__name__)


def get_fetch_period(range_str: str, interval_str: str) -> str:
    """Lookback Strategy: ขอข้อมูลย้อนหลังจาก Data Provider ด้วยระยะเวลาเป้าหมายที่ครอบคลุม Warm-up"""
    if interval_str == "15m":
        return "60d" if range_str == "1mo" else "1mo"
    if interval_str == "1h":
        if range_str in {"1y", "2y"}:
            return "730d"
        return "1y"
    if interval_str == "1d":
        if range_str in {"1mo", "3mo", "6mo"}:
            return "2y"
        if range_str == "1y":
            return "3y"
        if range_str == "5y":
            return "10y"
        return "max"
    if interval_str == "1wk":
        if range_str == "1y":
            return "5y"
        if range_str == "5y":
            return "10y"
        return "max"
    if interval_str == "1mo":
        return "max" if range_str == "max" else "10y"
    return range_str


def calculate_indicator_burn_in(
    indicator_key: str,
    required_bars: int,
    actual_warmup_bars: int,
    candles: list[OHLCVCandleDTO],
    display_start_ts: Optional[int],
) -> tuple[Literal["full", "partial", "unavailable"], int, Optional[int], Optional[int], IndicatorBurnInPolicyDTO]:
    total_bars = len(candles)
    if actual_warmup_bars >= required_bars:
        status: Literal["full", "partial", "unavailable"] = "full"
        burn_in_remaining = 0
        first_rel_idx = None
        first_rel_ts = display_start_ts
    elif total_bars >= required_bars:
        status = "partial"
        burn_in_remaining = max(0, required_bars - actual_warmup_bars)
        first_rel_idx = required_bars - 1
        first_rel_ts = candles[first_rel_idx].timestamp if first_rel_idx < total_bars else None
    else:
        status = "unavailable"
        burn_in_remaining = required_bars - total_bars
        first_rel_idx = None
        first_rel_ts = None

    seed_method = "sma_initial_period" if "EMA" in indicator_key else "wilder_rma_seed"
    algo_version = f"{indicator_key.lower()}_v1.0"
    policy = IndicatorBurnInPolicyDTO(
        algorithm_version=algo_version,
        seed_method=seed_method,
        convergence_tolerance_pct=0.01,
        required_burn_in_bars=required_bars,
        burn_in_bars_remaining=burn_in_remaining,
        first_reliable_timestamp=first_rel_ts,
        first_reliable_index=first_rel_idx,
    )
    return status, burn_in_remaining, first_rel_ts, first_rel_idx, policy


def calculate_warmup_metadata(
    candles: list[OHLCVCandleDTO],
    range_str: str,
    interval_str: str,
    tz_name: str,
) -> tuple[
    Optional[int],
    int,
    int,
    Literal["full", "partial", "unavailable", "sufficient", "insufficient", "not_applicable", "unknown"],
    dict[str, IndicatorWarmupDetailDTO],
]:
    if not candles:
        return None, 0, 200, "unknown", {}

    if range_str == "max":
        display_start_ts = candles[0].timestamp
        return display_start_ts, 0, 0, "not_applicable", {}

    try:
        latest_ts = candles[-1].timestamp / 1000.0
        latest_dt = datetime.fromtimestamp(latest_ts, tz=ZoneInfo(tz_name))

        if range_str == "5d":
            cutoff_dt = latest_dt - relativedelta(days=5)
        elif range_str == "1mo":
            cutoff_dt = latest_dt - relativedelta(months=1)
        elif range_str == "3mo":
            cutoff_dt = latest_dt - relativedelta(months=3)
        elif range_str == "6mo":
            cutoff_dt = latest_dt - relativedelta(months=6)
        elif range_str == "1y":
            cutoff_dt = latest_dt - relativedelta(years=1)
        elif range_str == "2y":
            cutoff_dt = latest_dt - relativedelta(years=2)
        elif range_str == "5y":
            cutoff_dt = latest_dt - relativedelta(years=5)
        else:
            cutoff_dt = datetime.fromtimestamp(candles[0].timestamp / 1000.0, tz=ZoneInfo(tz_name))

        cutoff_ts_ms = int(cutoff_dt.timestamp() * 1000)

        display_start_ts = None
        for c in candles:
            if c.timestamp >= cutoff_ts_ms:
                display_start_ts = c.timestamp
                break

        if display_start_ts is None:
            display_start_ts = candles[0].timestamp

        actual_warmup_bars = sum(1 for c in candles if c.timestamp < display_start_ts)

        indicator_specs = {
            "EMA200": 200,
            "EMA50": 50,
            "RSI14": 15,
            "ATR14": 15,
        }

        indicator_warmup: dict[str, IndicatorWarmupDetailDTO] = {}
        for ind_name, req_bars in indicator_specs.items():
            st, remaining, f_ts, f_idx, policy = calculate_indicator_burn_in(
                ind_name, req_bars, actual_warmup_bars, candles, display_start_ts
            )
            indicator_warmup[ind_name] = IndicatorWarmupDetailDTO(
                status=st,
                required_bars=req_bars,
                actual_warmup_bars=actual_warmup_bars,
                burn_in_bars_remaining=remaining,
                first_reliable_timestamp=f_ts,
                first_reliable_index=f_idx,
                burn_in_policy=policy,
            )

        legacy_status: Literal["full", "partial", "unavailable", "sufficient", "insufficient", "not_applicable", "unknown"] = (
            "sufficient" if actual_warmup_bars >= 200 else "insufficient"
        )

        return display_start_ts, actual_warmup_bars, 200, legacy_status, indicator_warmup
    except Exception as e:
        log.warning("Warmup metadata calculation error: %s", e)
        return candles[0].timestamp if candles else None, 0, 200, "unknown", {}


def calculate_pivot_levels(
    monthly_df: pd.DataFrame,
    tz_name: str,
) -> tuple[Optional[PivotLevelsDTO], Optional[str], Optional[str]]:
    if monthly_df is None or monthly_df.empty:
        return None, None, None

    try:
        now_in_market = datetime.now(ZoneInfo(tz_name))
        last_idx = len(monthly_df) - 1
        last_row = monthly_df.iloc[last_idx]

        ts = last_row.name
        if hasattr(ts, "tzinfo") and ts.tzinfo is not None:
            ts_market = ts.astimezone(ZoneInfo(tz_name))
        else:
            ts_market = ts.replace(tzinfo=ZoneInfo(tz_name))

        if (ts_market.year, ts_market.month) == (now_in_market.year, now_in_market.month):
            if len(monthly_df) >= 2:
                target_row = monthly_df.iloc[last_idx - 1]
            else:
                target_row = last_row
        else:
            target_row = last_row

        h = float(target_row["High"])
        l = float(target_row["Low"])
        c = float(target_row["Close"])

        if pd.isna(h) or pd.isna(l) or pd.isna(c) or (h == 0 and l == 0):
            return None, None, None

        pivot = (h + l + c) / 3.0
        r1 = 2.0 * pivot - l
        r2 = pivot + (h - l)
        r3 = h + 2.0 * (pivot - l)
        s1 = 2.0 * pivot - h
        s2 = pivot - (h - l)
        s3 = l - 2.0 * (h - pivot)
        s4 = s3 - (h - l)

        target_ts = target_row.name
        if hasattr(target_ts, "strftime"):
            pivot_as_of = target_ts.strftime("%Y-%m")
        else:
            pivot_as_of = str(target_ts)[:7]

        levels = PivotLevelsDTO(
            pivot=round(pivot, 4),
            r1=round(r1, 4),
            r2=round(r2, 4),
            r3=round(r3, 4),
            s1=round(s1, 4),
            s2=round(s2, 4),
            s3=round(s3, 4),
            s4=round(s4, 4),
        )
        return levels, "monthly", pivot_as_of
    except Exception as e:
        log.warning("Pivot calculation error: %s", e)
        return None, None, None


def calculate_52w(
    daily_candles: list[OHLCVCandleDTO],
    latest_dt: datetime,
    tz_name: str,
) -> tuple[Optional[float], Optional[float], int]:
    if not daily_candles:
        return None, None, 0

    try:
        cutoff_dt = latest_dt - relativedelta(years=1)
        cutoff_ts_ms = int(cutoff_dt.timestamp() * 1000)

        bars_1y = [c for c in daily_candles if c.timestamp >= cutoff_ts_ms]
        if not bars_1y:
            bars_1y = daily_candles

        w_high = max(c.high for c in bars_1y)
        w_low = min(c.low for c in bars_1y)

        earliest_ts = daily_candles[0].timestamp / 1000.0
        earliest_dt = datetime.fromtimestamp(earliest_ts, tz=ZoneInfo(tz_name))
        coverage_days = (latest_dt.date() - earliest_dt.date()).days

        return round(w_high, 4), round(w_low, 4), max(0, coverage_days)
    except Exception as e:
        log.warning("52W calculation error: %s", e)
        return None, None, 0


def map_corporate_actions(
    raw_earnings: list[dict],
    raw_dividends: list[dict],
    raw_splits: list[dict],
    candles: list[OHLCVCandleDTO],
    interval: str,
    currency: str,
    tz_name: str,
) -> list[CorporateActionEventDTO]:
    if not candles:
        return []

    latest_candle_ts = candles[-1].timestamp
    latest_candle_dt = datetime.fromtimestamp(latest_candle_ts / 1000.0, tz=ZoneInfo(tz_name))
    earliest_candle_ts = candles[0].timestamp

    candle_dates: list[tuple[datetime.date, int]] = []
    for c in candles:
        c_dt = datetime.fromtimestamp(c.timestamp / 1000.0, tz=ZoneInfo(tz_name))
        candle_dates.append((c_dt.date(), c.timestamp))

    curr_sym = "฿" if currency == "THB" else "$"
    events: list[CorporateActionEventDTO] = []

    def _find_session_for_date(event_date_str: str) -> tuple[Optional[int], Literal["reported_date", "next_session", "period_enclosing", "unknown"]]:
        try:
            e_dt = datetime.strptime(event_date_str, "%Y-%m-%d").date()
        except Exception:
            return None, "unknown"

        if e_dt > latest_candle_dt.date():
            return None, "unknown"

        if interval == "1d":
            for c_date, c_ts in candle_dates:
                if c_date == e_dt:
                    return c_ts, "reported_date"
            for c_date, c_ts in candle_dates:
                if c_date > e_dt:
                    return c_ts, "next_session"
            return None, "unknown"
        else:
            e_dt_full = datetime.strptime(event_date_str, "%Y-%m-%d").replace(tzinfo=ZoneInfo(tz_name))
            e_ms = int(e_dt_full.timestamp() * 1000)
            if e_ms < earliest_candle_ts or e_ms > latest_candle_ts + 31 * 86400 * 1000:
                return None, "unknown"

            chosen_ts = None
            for i, c in enumerate(candles):
                next_ts = candles[i + 1].timestamp if i + 1 < len(candles) else None
                if c.timestamp <= e_ms:
                    if next_ts is None or e_ms < next_ts:
                        chosen_ts = c.timestamp
                        break
            if chosen_ts is not None:
                return chosen_ts, "period_enclosing"
            return None, "unknown"

    for e in raw_earnings:
        target_ts, mapping_method = _find_session_for_date(e["date_str"])
        if target_ts is None:
            continue

        actual = e.get("eps_actual")
        estimate = e.get("eps_estimate")

        if actual is not None and estimate is not None:
            if actual > estimate:
                color: Literal["green", "red", "blue", "purple"] = "green"
                tooltip = f"Reported Date: {e['date_str']} (Mapped: {mapping_method}) | Earnings Beat: EPS {curr_sym}{actual:.2f} vs Est {curr_sym}{estimate:.2f}"
            elif actual < estimate:
                color = "red"
                tooltip = f"Reported Date: {e['date_str']} (Mapped: {mapping_method}) | Earnings Miss: EPS {curr_sym}{actual:.2f} vs Est {curr_sym}{estimate:.2f}"
            else:
                color = "blue"
                tooltip = f"Reported Date: {e['date_str']} (Mapped: {mapping_method}) | Earnings In-line: EPS {curr_sym}{actual:.2f}"
        else:
            color = "blue"
            tooltip = f"Reported Date: {e['date_str']} (Mapped: {mapping_method}) | Earnings Reported: EPS {curr_sym}{actual:.2f}" if actual is not None else f"Reported Date: {e['date_str']} (Mapped: {mapping_method}) | Earnings unavailable"

        events.append(
            CorporateActionEventDTO(
                event_type="earnings",
                timestamp=target_ts,
                date_str=e["date_str"],
                label="E",
                color=color,
                tooltip=tooltip,
                mapping_method=mapping_method,
                eps_actual=actual,
                eps_estimate=estimate,
            )
        )

    for d in raw_dividends:
        target_ts, mapping_method = _find_session_for_date(d["date_str"])
        if target_ts is None:
            continue

        amt = d.get("dividend_amount", 0.0)
        events.append(
            CorporateActionEventDTO(
                event_type="ex_dividend",
                timestamp=target_ts,
                date_str=d["date_str"],
                label="XD",
                color="blue",
                tooltip=f"Reported Date: {d['date_str']} (Mapped: {mapping_method}) | Ex-Dividend: {curr_sym}{amt:.2f}",
                mapping_method=mapping_method,
                dividend_amount=amt,
            )
        )

    for s in raw_splits:
        target_ts, mapping_method = _find_session_for_date(s["date_str"])
        if target_ts is None:
            continue

        formatted = s.get("split_formatted") or "Stock Split"
        events.append(
            CorporateActionEventDTO(
                event_type="split",
                timestamp=target_ts,
                date_str=s["date_str"],
                label="S",
                color="purple",
                tooltip=f"Reported Date: {s['date_str']} (Mapped: {mapping_method}) | {formatted}",
                mapping_method=mapping_method,
                split_numerator=s.get("split_numerator"),
                split_denominator=s.get("split_denominator"),
                split_formatted=formatted,
            )
        )

    events.sort(key=lambda x: x.timestamp)
    return events
