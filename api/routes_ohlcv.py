import logging
import re
import threading
import time
from datetime import datetime
from typing import Literal, Optional
from zoneinfo import ZoneInfo
from dateutil.relativedelta import relativedelta

from fastapi import APIRouter, Depends, HTTPException, Query
import pandas as pd
import yfinance as yf

from api.auth import require_session
from api.schemas import (
    CorporateActionEventDTO,
    CorporateActionsMetadataDTO,
    IndicatorBurnInPolicyDTO,
    IndicatorWarmupDetailDTO,
    OHLCVCandleDTO,
    OHLCVResponseDTO,
    PivotLevelsDTO,
)
from core.retry import with_retry as _with_retry
from tools.market.asset_resolver import resolve_asset
from tools.market.earnings import fetch_earnings_dates

log = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/equity",
    tags=["equity"],
    dependencies=[Depends(require_session)],
)

# In-Memory TTL Cache and Single-Flight per OHLCV key
_CACHE_TTL_SECONDS = 300.0
_CACHE: dict[str, tuple[float, OHLCVResponseDTO]] = {}
_CACHE_LOCK = threading.Lock()
_KEY_LOCKS: dict[str, threading.Lock] = {}

# Tiered Raw Action Cache and Per-Ticker Single-Flight Locks
_ACTION_CACHE: dict[str, dict[str, tuple[float, list[dict]]]] = {}
_ACTION_LOCK = threading.Lock()
_ACTION_KEY_LOCKS: dict[str, threading.Lock] = {}

EARNINGS_CACHE_TTL = 6 * 3600            # 6 Hours
DIVIDENDS_SPLITS_CACHE_TTL = 24 * 3600   # 24 Hours


def _get_key_lock(cache_key: str) -> threading.Lock:
    with _CACHE_LOCK:
        if cache_key not in _KEY_LOCKS:
            _KEY_LOCKS[cache_key] = threading.Lock()
        return _KEY_LOCKS[cache_key]


def _get_action_key_lock(ticker: str) -> threading.Lock:
    with _ACTION_LOCK:
        if ticker not in _ACTION_KEY_LOCKS:
            _ACTION_KEY_LOCKS[ticker] = threading.Lock()
        return _ACTION_KEY_LOCKS[ticker]


def _validate_ticker(ticker: str) -> str:
    ticker = ticker.upper().strip()
    if not re.match(r"^[A-Z0-9.\-_]+$", ticker):
        raise HTTPException(status_code=400, detail="Invalid ticker format")
    if ".." in ticker or "/" in ticker or "\\" in ticker:
        raise HTTPException(status_code=400, detail="Path traversal not allowed")
    return ticker


TIMEFRAME_CAPABILITIES: dict[str, list[str]] = {
    "15m": ["5d", "1mo"],
    "1h": ["1mo", "3mo", "6mo", "1y", "2y"],
    "1d": ["1mo", "3mo", "6mo", "1y", "5y", "max"],
    "1wk": ["1y", "5y", "max"],
    "1mo": ["5y", "max"],
}

ALLOWED_RANGES = {"5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "max"}
ALLOWED_INTERVALS = {"15m", "1h", "1d", "1wk", "1mo"}


def _get_fetch_period(range_str: str, interval_str: str) -> str:
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


def _calculate_indicator_burn_in(
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


def _calculate_warmup_metadata(
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
    """คำนวณ display_start_timestamp, actual_warmup_bars, indicator_warmup metadata
    ตาม Calendar-Exact Cutoff Arithmetic และ 3-Tier Indicator Convergence Semantics
    """
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

        # Invariant: display_start_timestamp คือ timestamp ของแท่งแรกสุดที่มี datetime >= cutoff_dt
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
            st, remaining, f_ts, f_idx, policy = _calculate_indicator_burn_in(
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

        # Global warmup status (anchored to standard 200-bar benchmark)
        global_status = indicator_warmup["EMA200"].status
        legacy_status: Literal["full", "partial", "unavailable", "sufficient", "insufficient", "not_applicable", "unknown"] = (
            "sufficient" if actual_warmup_bars >= 200 else "insufficient"
        )

        return display_start_ts, actual_warmup_bars, 200, legacy_status, indicator_warmup
    except Exception as e:
        log.warning("Warmup metadata calculation error: %s", e)
        return candles[0].timestamp if candles else None, 0, 200, "unknown", {}



def _calculate_pivot_levels(
    monthly_df: pd.DataFrame,
    tz_name: str,
) -> tuple[Optional[PivotLevelsDTO], Optional[str], Optional[str]]:
    """คำนวณ Pivot Point Classic จากแท่งเดือนที่ปิดสมบูรณ์แล้วล่าสุดตาม Market Timezone"""
    if monthly_df is None or monthly_df.empty:
        return None, None, None

    try:
        now_in_market = datetime.now(ZoneInfo(tz_name))
        last_idx = len(monthly_df) - 1
        last_row = monthly_df.iloc[last_idx]

        # Convert index timestamp to timezone
        ts = last_row.name
        if hasattr(ts, "tzinfo") and ts.tzinfo is not None:
            ts_market = ts.astimezone(ZoneInfo(tz_name))
        else:
            ts_market = ts.replace(tzinfo=ZoneInfo(tz_name))

        # Check if the last candle belongs to the current ongoing month
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


def _calculate_52w(
    daily_candles: list[OHLCVCandleDTO],
    latest_dt: datetime,
    tz_name: str,
) -> tuple[Optional[float], Optional[float], int]:
    """คำนวณ 52-Week High / Low และ Coverage Calendar Days จาก Daily Reference Candles"""
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


def _fetch_corporate_actions(
    ticker: str,
    provider_symbol: str,
    candles: list[OHLCVCandleDTO],
    interval: str,
    currency: str,
    tz_name: str,
) -> tuple[list[CorporateActionEventDTO], CorporateActionsMetadataDTO]:
    """ดึงและจัดการ Corporate Actions (Earnings, Dividends, Splits) พร้อม Caching และ Mapping"""
    if not candles:
        meta = CorporateActionsMetadataDTO(
            status="unavailable",
            earnings_status="empty",
            dividends_status="empty",
            splits_status="empty",
            missing_sources=["earnings", "dividends", "splits"],
        )
        return [], meta

    now_mono = time.monotonic()
    action_lock = _get_action_key_lock(provider_symbol)

    with action_lock:
        with _ACTION_LOCK:
            if provider_symbol not in _ACTION_CACHE:
                _ACTION_CACHE[provider_symbol] = {}
            ticker_cache = _ACTION_CACHE[provider_symbol]

        # 1. Earnings Fetch via Shared Service (6h TTL + single-flight + retry)
        earn_res = fetch_earnings_dates(provider_symbol, tz_name)
        raw_earnings: list[dict] = earn_res.rows
        earnings_status: Literal["ok", "failed", "empty"] = earn_res.status
        earnings_as_of: Optional[str] = earn_res.source_as_of

        # 2. Dividends Fetch & Cache Check
        raw_dividends: list[dict] = []
        dividends_status: Literal["ok", "failed", "empty"] = "empty"
        dividends_as_of: Optional[str] = None

        if "dividends" in ticker_cache and now_mono < ticker_cache["dividends"][0]:
            raw_dividends = ticker_cache["dividends"][1]
            dividends_status = "ok" if raw_dividends else "empty"
            dividends_as_of = ticker_cache["dividends"][2] if len(ticker_cache["dividends"]) > 2 else None
        else:
            try:
                tk = yf.Ticker(provider_symbol)
                div_s = _with_retry(lambda: tk.dividends)
                if div_s is not None and not div_s.empty:
                    for ts, amount in div_s.items():
                        if pd.notna(amount) and float(amount) > 0:
                            date_str = ts.strftime("%Y-%m-%d") if hasattr(ts, "strftime") else str(ts)[:10]
                            raw_dividends.append({
                                "date_str": date_str,
                                "timestamp_ms": int(ts.timestamp() * 1000) if hasattr(ts, "timestamp") else 0,
                                "dividend_amount": round(float(amount), 4),
                            })
                    dividends_status = "ok" if raw_dividends else "empty"
                else:
                    dividends_status = "empty"
                dividends_as_of = datetime.now(ZoneInfo(tz_name)).isoformat()
                with _ACTION_LOCK:
                    ticker_cache["dividends"] = (now_mono + DIVIDENDS_SPLITS_CACHE_TTL, raw_dividends, dividends_as_of)
            except Exception as e:
                log.warning("Dividends fetch failed for %s: %s", provider_symbol, e)
                dividends_status = "failed"

        # 3. Splits Fetch & Cache Check
        raw_splits: list[dict] = []
        splits_status: Literal["ok", "failed", "empty"] = "empty"
        splits_as_of: Optional[str] = None

        if "splits" in ticker_cache and now_mono < ticker_cache["splits"][0]:
            raw_splits = ticker_cache["splits"][1]
            splits_status = "ok" if raw_splits else "empty"
            splits_as_of = ticker_cache["splits"][2] if len(ticker_cache["splits"]) > 2 else None
        else:
            try:
                tk = yf.Ticker(provider_symbol)
                splits_s = _with_retry(lambda: tk.splits)
                if splits_s is not None and not splits_s.empty:
                    for ts, ratio in splits_s.items():
                        if pd.notna(ratio) and float(ratio) > 0:
                            ratio_f = float(ratio)
                            if ratio_f >= 1.0:
                                num = ratio_f
                                den = 1.0
                                formatted = f"{int(num) if num.is_integer() else num}-for-1 forward split"
                            else:
                                num = 1.0
                                den = round(1.0 / ratio_f, 4)
                                formatted = f"1-for-{int(den) if den.is_integer() else den} reverse split"

                            date_str = ts.strftime("%Y-%m-%d") if hasattr(ts, "strftime") else str(ts)[:10]
                            raw_splits.append({
                                "date_str": date_str,
                                "timestamp_ms": int(ts.timestamp() * 1000) if hasattr(ts, "timestamp") else 0,
                                "split_numerator": num,
                                "split_denominator": den,
                                "split_formatted": formatted,
                            })
                    splits_status = "ok" if raw_splits else "empty"
                else:
                    splits_status = "empty"
                splits_as_of = datetime.now(ZoneInfo(tz_name)).isoformat()
                with _ACTION_LOCK:
                    ticker_cache["splits"] = (now_mono + DIVIDENDS_SPLITS_CACHE_TTL, raw_splits, splits_as_of)
            except Exception as e:
                log.warning("Splits fetch failed for %s: %s", provider_symbol, e)
                splits_status = "failed"

    # Compute overall status, oldest as_of, and missing sources
    statuses = [earnings_status, dividends_status, splits_status]
    missing_sources = []
    if earnings_status == "failed":
        missing_sources.append("earnings")
    if dividends_status == "failed":
        missing_sources.append("dividends")
    if splits_status == "failed":
        missing_sources.append("splits")

    if all(s == "failed" for s in statuses):
        overall_status: Literal["available", "partial", "unavailable"] = "unavailable"
    elif any(s == "failed" for s in statuses):
        overall_status = "partial"
    else:
        overall_status = "available"

    available_timestamps = [ts for ts in [earnings_as_of, dividends_as_of, splits_as_of] if ts is not None]
    oldest_as_of = min(available_timestamps) if available_timestamps else None

    metadata_dto = CorporateActionsMetadataDTO(
        status=overall_status,
        as_of=oldest_as_of,
        earnings_status=earnings_status,
        earnings_as_of=earnings_as_of,
        dividends_status=dividends_status,
        dividends_as_of=dividends_as_of,
        splits_status=splits_status,
        splits_as_of=splits_as_of,
        missing_sources=missing_sources,
        data_provenance="Yahoo Finance (yfinance)",
    )

    # 4. Map raw events to candle sessions
    events = _map_corporate_actions(
        raw_earnings, raw_dividends, raw_splits, candles, interval, currency, tz_name
    )

    return events, metadata_dto


def _map_corporate_actions(
    raw_earnings: list[dict],
    raw_dividends: list[dict],
    raw_splits: list[dict],
    candles: list[OHLCVCandleDTO],
    interval: str,
    currency: str,
    tz_name: str,
) -> list[CorporateActionEventDTO]:
    """Map normalized raw events into candle sessions with non-inferential policy"""
    if not candles:
        return []

    latest_candle_ts = candles[-1].timestamp
    latest_candle_dt = datetime.fromtimestamp(latest_candle_ts / 1000.0, tz=ZoneInfo(tz_name))
    earliest_candle_ts = candles[0].timestamp

    # Pre-index candles for fast session matching
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

        # Policy: Future earnings scheduled after latest available trading date -> omit
        if e_dt > latest_candle_dt.date():
            return None, "unknown"

        if interval == "1d":
            # Exact date match
            for c_date, c_ts in candle_dates:
                if c_date == e_dt:
                    return c_ts, "reported_date"
            # If weekend/holiday, find next available session
            for c_date, c_ts in candle_dates:
                if c_date > e_dt:
                    return c_ts, "next_session"
            return None, "unknown"
        else:
            # interval in ("1wk", "1mo"): map to period enclosing event
            # Convert event date to ms
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

    # 1. Map Earnings
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

    # 2. Map Dividends
    for d in raw_dividends:
        target_ts, mapping_method = _find_session_for_date(d["date_str"])
        if target_ts is None:
            continue

        div_amt = d.get("dividend_amount", 0.0)
        tooltip = f"Ex-Dividend Date: {d['date_str']} (Mapped: {mapping_method}) | Dividend: {curr_sym}{div_amt:.2f}"

        events.append(
            CorporateActionEventDTO(
                event_type="ex_dividend",
                timestamp=target_ts,
                date_str=d["date_str"],
                label="XD",
                color="blue",
                tooltip=tooltip,
                mapping_method=mapping_method,
                dividend_amount=div_amt,
            )
        )

    # 3. Map Splits
    for s in raw_splits:
        target_ts, mapping_method = _find_session_for_date(s["date_str"])
        if target_ts is None:
            continue

        split_formatted = s.get("split_formatted", "Stock Split")
        tooltip = f"Stock Split Date: {s['date_str']} (Mapped: {mapping_method}) | {split_formatted} (Contextual Action)"

        events.append(
            CorporateActionEventDTO(
                event_type="split",
                timestamp=target_ts,
                date_str=s["date_str"],
                label="S",
                color="purple",
                tooltip=tooltip,
                mapping_method=mapping_method,
                split_numerator=s.get("split_numerator"),
                split_denominator=s.get("split_denominator"),
                split_formatted=split_formatted,
            )
        )

    # Sort events by timestamp, then event_type
    events.sort(key=lambda item: (item.timestamp, item.event_type))
    return events


@router.get("/{ticker}/ohlcv", response_model=OHLCVResponseDTO)
def get_equity_ohlcv(
    ticker: str,
    range: str = Query("6mo", description="Historical range based on interval capability matrix"),
    interval: str = Query("1d", description="Bar interval (15m, 1h, 1d, 1wk, 1mo)"),
) -> OHLCVResponseDTO:
    clean_ticker = _validate_ticker(ticker)
    allowed_for_interval = TIMEFRAME_CAPABILITIES.get(interval)
    if allowed_for_interval is None:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid interval '{interval}'. Allowed intervals are: {sorted(TIMEFRAME_CAPABILITIES.keys())}",
        )
    if range not in allowed_for_interval:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid interval and range combination '{interval}' with '{range}'. Allowed ranges for '{interval}' are: {allowed_for_interval}",
        )

    resolved = resolve_asset(clean_ticker)
    provider_symbol = resolved.provider_symbol or clean_ticker
    market: Literal["TH", "US"] = "TH" if (resolved.market == "TH" or provider_symbol.endswith(".BK")) else "US"
    currency: Literal["USD", "THB"] = "THB" if market == "TH" else "USD"
    tz_name = "Asia/Bangkok" if market == "TH" else "America/New_York"

    cache_key = f"{provider_symbol}:{range}:{interval}"
    now_mono = time.monotonic()

    # Check cache first
    with _CACHE_LOCK:
        if cache_key in _CACHE:
            expire_at, cached_dto = _CACHE[cache_key]
            if now_mono < expire_at:
                return cached_dto

    # Single-flight fetch per key
    key_lock = _get_key_lock(cache_key)
    with key_lock:
        # Double check cache inside lock
        with _CACHE_LOCK:
            if cache_key in _CACHE:
                expire_at, cached_dto = _CACHE[cache_key]
                if time.monotonic() < expire_at:
                    return cached_dto

        try:
            tk = yf.Ticker(provider_symbol)

            # 1. Fetch Chart History with Targeted Pre-roll (Explicit auto_adjust=True)
            fetch_period = _get_fetch_period(range, interval)
            try:
                chart_df = _with_retry(lambda: tk.history(period=fetch_period, interval=interval, auto_adjust=True))
            except Exception as e:
                if fetch_period != range:
                    log.warning("Extended fetch period %s failed for %s, falling back to %s: %s", fetch_period, provider_symbol, range, e)
                    chart_df = _with_retry(lambda: tk.history(period=range, interval=interval, auto_adjust=True))
                else:
                    raise
        except Exception as e:
            log.error("Failed to fetch OHLCV from yfinance for %s: %s", provider_symbol, e)
            raise HTTPException(status_code=502, detail=f"Failed to fetch market data for {clean_ticker}: {e}")

        if chart_df is None or chart_df.empty:
            raise HTTPException(status_code=404, detail=f"No OHLCV historical data found for {clean_ticker}")

        # 2. Build Candlestick DTO list & deduplicate
        seen_timestamps = set()
        candles: list[OHLCVCandleDTO] = []
        for ts, row in chart_df.iterrows():
            try:
                epoch_ms = int(ts.timestamp() * 1000)
                if epoch_ms in seen_timestamps:
                    continue
                seen_timestamps.add(epoch_ms)
                candles.append(
                    OHLCVCandleDTO(
                        timestamp=epoch_ms,
                        open=round(float(row["Open"]), 4),
                        high=round(float(row["High"]), 4),
                        low=round(float(row["Low"]), 4),
                        close=round(float(row["Close"]), 4),
                        volume=round(float(row.get("Volume", 0)), 2),
                    )
                )
            except Exception:
                continue

        candles.sort(key=lambda c: c.timestamp)

        # Calculate Warm-up and Cutoff metadata
        display_start_ts, avail_warmup, req_warmup, warmup_stat, indicator_warmup = _calculate_warmup_metadata(
            candles, range, interval, tz_name
        )

        # 3. Calculate Header Price, Price Change & 52W from Daily History Reference
        current_price: Optional[float] = None
        price_change: Optional[float] = None
        price_change_pct: Optional[float] = None
        price_as_of: Optional[str] = None
        daily_candles: list[OHLCVCandleDTO] = []

        try:
            if interval == "1d" and len(candles) >= 250:
                daily_candles = candles
            else:
                daily_df = _with_retry(lambda: tk.history(period="1y", interval="1d", auto_adjust=True))
                if daily_df is not None and not daily_df.empty:
                    for ts, row in daily_df.iterrows():
                        try:
                            daily_candles.append(
                                OHLCVCandleDTO(
                                    timestamp=int(ts.timestamp() * 1000),
                                    open=round(float(row["Open"]), 4),
                                    high=round(float(row["High"]), 4),
                                    low=round(float(row["Low"]), 4),
                                    close=round(float(row["Close"]), 4),
                                    volume=round(float(row.get("Volume", 0)), 2),
                                )
                            )
                        except Exception:
                            continue
                    daily_candles.sort(key=lambda c: c.timestamp)

            if daily_candles:
                last_daily = daily_candles[-1]
                current_price = last_daily.close
                last_ts = last_daily.timestamp / 1000.0
                last_dt = datetime.fromtimestamp(last_ts, tz=ZoneInfo(tz_name))
                price_as_of = last_dt.isoformat()

                if len(daily_candles) >= 2:
                    prev_close = daily_candles[-2].close
                    price_change = round(current_price - prev_close, 4)
                    if prev_close > 0:
                        price_change_pct = round((price_change / prev_close) * 100.0, 4)
                else:
                    price_change = 0.0
                    price_change_pct = 0.0
        except Exception as e:
            log.warning("Daily quote calculation fallback for %s: %s", provider_symbol, e)
            if candles:
                current_price = candles[-1].close

        # Calculate 52W High / Low on Daily Reference Basis
        latest_dt = datetime.fromtimestamp(candles[-1].timestamp / 1000.0, tz=ZoneInfo(tz_name))
        w52_high, w52_low, w52_coverage = _calculate_52w(daily_candles or candles, latest_dt, tz_name)

        # 4. Fetch Monthly History & Compute Pivot Points
        pivot_levels: Optional[PivotLevelsDTO] = None
        pivot_period: Optional[str] = None
        pivot_as_of: Optional[str] = None

        try:
            monthly_df = _with_retry(lambda: tk.history(period="2y", interval="1mo", auto_adjust=True))
            pivot_levels, pivot_period, pivot_as_of = _calculate_pivot_levels(monthly_df, tz_name)
        except Exception as e:
            log.warning("Monthly pivot fetch failed for %s: %s", provider_symbol, e)

        # 5. Fetch Corporate Actions (Earnings, Dividends, Splits) with Isolation
        events: list[CorporateActionEventDTO] = []
        events_metadata: Optional[CorporateActionsMetadataDTO] = None
        try:
            events, events_metadata = _fetch_corporate_actions(
                clean_ticker, provider_symbol, candles, interval, currency, tz_name
            )
        except Exception as e:
            log.warning("Corporate actions fetch error for %s: %s", provider_symbol, e)
            events_metadata = CorporateActionsMetadataDTO(
                status="unavailable",
                earnings_status="failed",
                dividends_status="failed",
                splits_status="failed",
                missing_sources=["earnings", "dividends", "splits"],
            )

        effective_caps = dict(TIMEFRAME_CAPABILITIES)
        cap_reasons: dict[str, str] = {}

        dto = OHLCVResponseDTO(
            ticker=clean_ticker,
            market=market,
            currency=currency,
            price_basis="provider_proportional_adj_close_ratio",
            provider_name="yfinance",
            provider_tier="best_effort",
            feed_latency_model="delayed_15m",
            current_price=current_price,
            price_change=price_change,
            price_change_pct=price_change_pct,
            price_as_of=price_as_of,
            candles=candles,
            pivot_levels=pivot_levels,
            pivot_period=pivot_period,
            pivot_as_of=pivot_as_of,
            requested_range=range,
            interval=interval,
            allowed_ranges=allowed_for_interval,
            effective_capabilities=effective_caps,
            capability_reasons=cap_reasons,
            display_start_timestamp=display_start_ts,
            available_warmup_bars=avail_warmup,
            required_warmup_bars=req_warmup,
            warmup_status=warmup_stat,
            indicator_warmup=indicator_warmup,
            events=events,
            events_metadata=events_metadata,
            week52_high=w52_high,
            week52_low=w52_low,
            week52_coverage_calendar_days=w52_coverage,
        )

        with _CACHE_LOCK:
            _CACHE[cache_key] = (time.monotonic() + _CACHE_TTL_SECONDS, dto)

        return dto

