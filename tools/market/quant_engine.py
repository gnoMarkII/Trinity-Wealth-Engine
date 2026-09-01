"""Deterministic price-based Quant Engine — Beta/Volatility/MDD/RSI/MACD คำนวณ local ทั้งหมดจาก yfinance history

ฟังก์ชันในไฟล์นี้เป็น internal helper (ไม่ใช่ @tool) — ถูกเรียกจาก tools/market/equity_quant_tool.py
เท่านั้น เพื่อประกอบ QuantSignals แบบ deterministic ไม่มี LLM แทรกแซงค่าตัวเลข
"""
import time
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, List, Tuple, Literal

import math
import numpy as np
import pandas as pd
import yfinance as yf
from langsmith import traceable
from pydantic import BaseModel

from core.logger import get_logger
from core.retry import with_retry as _with_retry
from schemas.micro_quant_schemas import AtomicMarketSnapshot
from tools.market.financial_autopsy import FinancialAutopsyPeriod

log = get_logger(__name__)

_MIN_TRADING_DAYS = 45  # เกณฑ์เดียวกับ Correlation Quality Guard ของ Macro (5.7 Pillar 4)


class PriceSeriesQuality(BaseModel):
    trading_days: int
    is_valid: bool
    stale_reason: Optional[str] = None


# --- Shared price-history cache (ลด .history() call ซ้ำระหว่าง beta/vol/mdd/technical ของ ticker เดียวกัน) ---
_HISTORY_CACHE: Dict[Tuple[str, str], Tuple[pd.DataFrame, float]] = {}
_HISTORY_ERROR_CACHE: Dict[Tuple[str, str], float] = {}
_HISTORY_SUCCESS_TTL_SECONDS = 6 * 3600
_HISTORY_ERROR_TTL_SECONDS = 60.0


def _get_price_history(provider_symbol: str, period: str) -> pd.DataFrame:
    """ดึงราคาย้อนหลังพร้อม retry + cache TTL (success 6h / error 60s) ป้องกัน rate-limit จาก .history()"""
    key = (provider_symbol, period)
    now = time.time()

    if key in _HISTORY_CACHE:
        df, ts = _HISTORY_CACHE[key]
        if now - ts < _HISTORY_SUCCESS_TTL_SECONDS:
            return df
        del _HISTORY_CACHE[key]

    if key in _HISTORY_ERROR_CACHE:
        ts = _HISTORY_ERROR_CACHE[key]
        if now - ts < _HISTORY_ERROR_TTL_SECONDS:
            raise RuntimeError(f"cached failure for {provider_symbol} ({period})")
        del _HISTORY_ERROR_CACHE[key]

    try:
        df = _with_retry(lambda: yf.Ticker(provider_symbol).history(period=period, auto_adjust=False))
    except Exception as e:
        log.warning("_get_price_history failed | %s (%s): %s", provider_symbol, period, e)
        _HISTORY_ERROR_CACHE[key] = now
        raise

    if df is not None and not df.empty and "Close" in df:
        df = df.dropna(subset=["Close"])

    _HISTORY_CACHE[key] = (df, now)
    return df


def _quality_from_df(df: Optional[pd.DataFrame]) -> PriceSeriesQuality:
    n = 0 if df is None else len(df)
    if n < _MIN_TRADING_DAYS:
        return PriceSeriesQuality(trading_days=n, is_valid=False, stale_reason="insufficient_trading_history")
    return PriceSeriesQuality(trading_days=n, is_valid=True)


def _fetch_error_quality() -> PriceSeriesQuality:
    return PriceSeriesQuality(trading_days=0, is_valid=False, stale_reason="fetch_error")


@traceable(run_type="tool")
def compute_beta(provider_symbol: str, benchmark: str = "^GSPC", period: str = "2y") -> Tuple[Optional[float], PriceSeriesQuality]:
    """คำนวณ Beta เทียบ benchmark จาก daily returns covariance/variance — ต้องมีวันทำการที่ overlap กันจริง ≥45 วัน"""
    try:
        stock_df = _get_price_history(provider_symbol, period)
        bench_df = _get_price_history(benchmark, period)
    except Exception:
        return None, _fetch_error_quality()

    if stock_df.empty or bench_df.empty or "Close" not in stock_df or "Close" not in bench_df:
        return None, PriceSeriesQuality(trading_days=0, is_valid=False, stale_reason="insufficient_trading_history")

    stock_ret = stock_df["Close"].pct_change().dropna()
    bench_ret = bench_df["Close"].pct_change().dropna()
    aligned = pd.concat([stock_ret, bench_ret], axis=1, join="inner")
    aligned.columns = ["stock", "bench"]
    overlapping_days = len(aligned)

    if overlapping_days < _MIN_TRADING_DAYS:
        return None, PriceSeriesQuality(trading_days=overlapping_days, is_valid=False, stale_reason="insufficient_trading_history")

    variance = aligned["bench"].var()
    if not variance or pd.isna(variance) or variance == 0:
        return None, PriceSeriesQuality(trading_days=overlapping_days, is_valid=False, stale_reason="zero_benchmark_variance")

    covariance = aligned["stock"].cov(aligned["bench"])
    beta = float(covariance / variance)
    return round(beta, 2), PriceSeriesQuality(trading_days=overlapping_days, is_valid=True)


@traceable(run_type="tool")
def compute_volatility(provider_symbol: str, period: str = "1y") -> Tuple[Optional[float], PriceSeriesQuality]:
    """Annualized Volatility (%) จาก std ของ daily returns * sqrt(252)"""
    try:
        df = _get_price_history(provider_symbol, period)
    except Exception:
        return None, _fetch_error_quality()

    quality = _quality_from_df(df)
    if not quality.is_valid or "Close" not in df or df["Close"].empty:
        return None, quality

    returns = df["Close"].pct_change().dropna()
    if returns.empty:
        return None, PriceSeriesQuality(trading_days=quality.trading_days, is_valid=False, stale_reason="insufficient_trading_history")

    vol_pct = float(returns.std() * (252 ** 0.5) * 100)
    return round(vol_pct, 2), quality


@traceable(run_type="tool")
def compute_mdd(provider_symbol: str, period: str = "3y") -> Tuple[Optional[float], PriceSeriesQuality]:
    """Maximum Drawdown (%) จาก running peak-to-trough ของราคาปิด"""
    try:
        df = _get_price_history(provider_symbol, period)
    except Exception:
        return None, _fetch_error_quality()

    quality = _quality_from_df(df)
    if not quality.is_valid or "Close" not in df or df["Close"].empty:
        return None, quality

    close = df["Close"]
    running_max = close.cummax()
    drawdown = (close - running_max) / running_max
    mdd_pct = float(drawdown.min() * 100)
    return round(mdd_pct, 2), quality


@traceable(run_type="tool")
def compute_technical_indicators(provider_symbol: str, period: str = "1y") -> Tuple[Optional[dict], PriceSeriesQuality]:
    """RSI(14) + MACD(12,26,9) จากราคาปิดย้อนหลัง — คืน {rsi_14, macd, macd_signal_line, macd_signal}"""
    try:
        df = _get_price_history(provider_symbol, period)
    except Exception:
        return None, _fetch_error_quality()

    quality = _quality_from_df(df)
    if not quality.is_valid or "Close" not in df or df["Close"].empty:
        return None, quality

    close = df["Close"]

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss.replace(0, float("nan"))
    rsi = 100 - (100 / (1 + rs))
    # avg_loss==0 ทำให้ rs หารด้วย NaN แล้วได้ NaN ทั้งที่ทางคณิตศาสตร์ RSI ต้องเป็นค่าปลายสุด
    # (ไม่มีวันขาดทุนเลยในช่วง 14 วัน = RSI 100, ราคาทรงตัวไม่ขยับเลย = RSI 50 เป็นกลาง)
    rsi = rsi.where(~((avg_loss == 0) & (avg_gain > 0)), 100.0)
    rsi = rsi.where(~((avg_loss == 0) & (avg_gain == 0)), 50.0)
    rsi_14 = float(rsi.iloc[-1]) if not rsi.empty and pd.notna(rsi.iloc[-1]) else None

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    macd_val = float(macd_line.iloc[-1]) if not macd_line.empty and pd.notna(macd_line.iloc[-1]) else None
    signal_val = float(signal_line.iloc[-1]) if not signal_line.empty and pd.notna(signal_line.iloc[-1]) else None

    if rsi_14 is None:
        return None, PriceSeriesQuality(trading_days=quality.trading_days, is_valid=False, stale_reason="insufficient_trading_history")

    macd_signal = None
    if macd_val is not None and signal_val is not None:
        macd_signal = "bullish" if macd_val > signal_val else "bearish"

    return {
        "rsi_14": round(rsi_14, 2),
        "macd": round(macd_val, 4) if macd_val is not None else None,
        "macd_signal_line": round(signal_val, 4) if signal_val is not None else None,
        "macd_signal": macd_signal,
    }, quality


def _fiscal_gap_days(date_a: str, date_b: str) -> Optional[int]:
    """หาจำนวนวันห่างระหว่าง fiscal_period_end 2 ค่า (string 'YYYY-MM-DD') — None ถ้า parse ไม่ได้"""
    try:
        return abs((datetime.fromisoformat(date_a) - datetime.fromisoformat(date_b)).days)
    except (TypeError, ValueError):
        return None


_YOY_GAP_MIN_DAYS = 300
_YOY_GAP_MAX_DAYS = 400


@traceable(run_type="parser")
def compute_growth_rates(periods: List[FinancialAutopsyPeriod]) -> Tuple[dict, List[str]]:
    """คำนวณ Revenue/Net Income YoY Growth จาก periods[0] (ล่าสุด) เทียบ periods[1] (ก่อนหน้า)

    Guard: ต้องมี ≥2 periods และห่างกันจริง 300-400 วัน ไม่งั้นไม่ใช่ 'YoY' ที่แท้จริง (อาจเป็น
    TTM/quarterly ปนมาจาก financial_autopsy's fiscal period clustering) — ไม่คำนวณถ้าไม่ผ่านเกณฑ์
    Guard: ค่า previous ≤ 0 ทำให้ % growth ไม่มีความหมาย (หารด้วยฐานติดลบ/ศูนย์) — ข้ามไป ไม่คำนวณ

    Returns:
        (result_dict, flags) — result_dict มี key revenue_growth_yoy_pct / net_income_growth_yoy_pct
        เป็น None เสมอถ้าคำนวณไม่ได้ ไม่มีการเดา/ประมาณค่าแทน
    """
    result = {"revenue_growth_yoy_pct": None, "net_income_growth_yoy_pct": None}

    if len(periods) < 2:
        return result, ["insufficient_periods:growth"]

    latest, previous = periods[0], periods[1]
    gap_days = _fiscal_gap_days(latest.fiscal_period_end, previous.fiscal_period_end)
    if gap_days is None or not (_YOY_GAP_MIN_DAYS <= gap_days <= _YOY_GAP_MAX_DAYS):
        return result, ["non_annual_period_gap:growth"]

    flags: List[str] = []

    if latest.total_revenue is not None and previous.total_revenue is not None:
        if previous.total_revenue > 0:
            result["revenue_growth_yoy_pct"] = round(
                (latest.total_revenue - previous.total_revenue) / previous.total_revenue * 100, 2
            )
        else:
            flags.append("base_year_negative:revenue_growth")

    if latest.net_income is not None and previous.net_income is not None:
        if previous.net_income > 0:
            result["net_income_growth_yoy_pct"] = round(
                (latest.net_income - previous.net_income) / previous.net_income * 100, 2
            )
        else:
            flags.append("base_year_negative:net_income_growth")

    return result, flags


_PRICE_PERCENTILE_MIN_DAYS = 250  # ~1 ปีเทรดจริง — เกณฑ์ขั้นต่ำให้คำนวณได้แม้ประวัติจะสั้นกว่า 5 ปีเต็ม


@traceable(run_type="tool")
def compute_price_percentile(provider_symbol: str, period: str = "5y") -> Tuple[Optional[float], Optional[float], PriceSeriesQuality]:
    """Percentile rank + Z-score ของราคาปิดล่าสุด เทียบ distribution ราคาปิดย้อนหลัง (period)

    หมายเหตุสำคัญ: นี่คือ Percentile ของ 'ราคา' ไม่ใช่ Valuation Multiple (P/E) — yfinance ไม่มี
    point-in-time fundamentals ให้คำนวณ Valuation Percentile ย้อนหลังได้จริง (ข้อจำกัดเดียวกับที่
    ทำให้ต้องเป็น Forward-tracking แทน Backtest ใน tools/market/quant_history.py)

    Atomic Pair Guard:
    1. แปลง Close เป็น numeric และตรวจ NaN / non-finite
    2. หาก latest close ใช้ไม่ได้ -> percentile=None, zscore=None, stale_reason="latest_close_unavailable"
    3. กรองค่า non-finite ออก และตรวจว่าต้องมีข้อมูลเหลือ ≥250 วัน
    4. ทั้ง percentile และ z-score ต้องได้ค่า finite พร้อมกัน หรือเป็น None พร้อมกันเสมอ
    """
    try:
        df = _get_price_history(provider_symbol, period)
    except Exception:
        return None, None, _fetch_error_quality()

    if df is None or "Close" not in df or df["Close"].empty:
        return None, None, PriceSeriesQuality(trading_days=0, is_valid=False, stale_reason="insufficient_trading_history")

    raw_close = pd.to_numeric(df["Close"], errors="coerce")
    if raw_close.empty:
        return None, None, PriceSeriesQuality(trading_days=0, is_valid=False, stale_reason="insufficient_trading_history")

    raw_latest = raw_close.iloc[-1]
    if pd.isna(raw_latest) or not math.isfinite(float(raw_latest)):
        return None, None, PriceSeriesQuality(
            trading_days=len(raw_close),
            is_valid=False,
            stale_reason="latest_close_unavailable",
        )

    # Clean internal non-finite values
    cleaned = raw_close.dropna()
    cleaned = cleaned[np.isfinite(cleaned)]
    trading_days = len(cleaned)

    if trading_days < _PRICE_PERCENTILE_MIN_DAYS:
        return None, None, PriceSeriesQuality(
            trading_days=trading_days,
            is_valid=False,
            stale_reason="insufficient_trading_history",
        )

    current_price = float(raw_latest)
    percentile_val = float((cleaned <= current_price).mean() * 100.0)

    std_val = float(cleaned.std())
    mean_val = float(cleaned.mean())
    zscore_val = 0.0 if std_val == 0 else float((current_price - mean_val) / std_val)

    if not (math.isfinite(percentile_val) and math.isfinite(zscore_val)):
        return None, None, PriceSeriesQuality(
            trading_days=trading_days,
            is_valid=False,
            stale_reason="calculation_failed",
        )

    return round(percentile_val, 2), round(zscore_val, 2), PriceSeriesQuality(trading_days=trading_days, is_valid=True)


def is_us_trading_day(dt: datetime) -> bool:
    """ตรวจสอบว่าเป็นวันทำการตลาดหุ้นสหรัฐ (NYSE/Nasdaq) หรือไม่ (จันทร์-ศุกร์ ยกเว้นวันหยุดราชการตลาดหลักทรัพย์)"""
    if dt.weekday() >= 5:  # Saturday = 5, Sunday = 6
        return False
    m, d, w = dt.month, dt.day, dt.weekday()
    # Fixed-date US market holidays
    if (m == 1 and d == 1) or (m == 6 and d == 19) or (m == 7 and d == 4) or (m == 12 and d == 25):
        return False
    # Holiday observed on Friday or Monday if falls on weekend
    if (m == 1 and d == 2 and w == 0) or (m == 7 and d == 5 and w == 0) or (m == 12 and d == 26 and w == 0):
        return False
    if (m == 7 and d == 3 and w == 4) or (m == 12 and d == 24 and w == 4):
        return False
    return True


def get_us_market_session_info(as_of_dt: datetime, actual_ohlcv_date_str: str) -> dict:
    """คำนวณ market session status, expected latest trading session date, และ missing trading sessions ตาม exchange calendar จริง"""
    try:
        if as_of_dt.tzinfo is not None:
            utc_dt = as_of_dt.astimezone(timezone.utc)
        else:
            utc_dt = as_of_dt.replace(tzinfo=timezone.utc)
    except Exception:
        utc_dt = datetime.now(timezone.utc)

    # US Eastern Time is UTC-4 (EDT) or UTC-5 (EST) - using EDT (-4h) as reference
    et_dt = utc_dt - timedelta(hours=4)
    et_hour = et_dt.hour + (et_dt.minute / 60.0)

    # 1. Market session status
    if et_dt.weekday() >= 5 or not is_us_trading_day(et_dt):
        market_session_status = "closed"
    elif 4.0 <= et_hour < 9.5:
        market_session_status = "pre_market"
    elif 9.5 <= et_hour < 16.0:
        market_session_status = "open"
    elif 16.0 <= et_hour < 20.0:
        market_session_status = "after_hours"
    else:
        market_session_status = "closed"

    # 2. Expected latest completed trading session
    candidate = et_dt.date()
    if not is_us_trading_day(et_dt) or et_hour < 16.0:
        candidate = candidate - timedelta(days=1)

    while not is_us_trading_day(datetime(candidate.year, candidate.month, candidate.day)):
        candidate = candidate - timedelta(days=1)

    expected_latest_session_date = candidate.strftime("%Y-%m-%d")

    # 3. Missing trading sessions
    missing_trading_sessions = 0
    try:
        actual_d = datetime.strptime(actual_ohlcv_date_str[:10], "%Y-%m-%d").date()
        if actual_d < candidate:
            cur = actual_d + timedelta(days=1)
            while cur <= candidate:
                if is_us_trading_day(datetime(cur.year, cur.month, cur.day)):
                    missing_trading_sessions += 1
                cur += timedelta(days=1)
    except Exception:
        missing_trading_sessions = 0

    if missing_trading_sessions == 0:
        data_freshness_status = "fresh"
    elif missing_trading_sessions == 1:
        data_freshness_status = "stale_one_session"
    else:
        data_freshness_status = "stale_multiple_sessions"

    return {
        "market_session_status": market_session_status,
        "data_freshness_status": data_freshness_status,
        "expected_latest_session_date": expected_latest_session_date,
        "actual_latest_session_date": actual_ohlcv_date_str[:10],
        "missing_trading_sessions": missing_trading_sessions,
    }


def create_atomic_market_snapshot(
    provider_symbol: str,
    df_1y: Optional[pd.DataFrame],
    info: dict,
    market: str = "US",
) -> Tuple[AtomicMarketSnapshot, List[str]]:
    """สร้าง Atomic Market Snapshot เป็นแหล่งข้อมูลราคาเดียว (Single Source of Truth) สำหรับ pipeline ทั้งหมด"""
    flags: List[str] = []
    now_dt = datetime.now(timezone.utc)
    now_iso = now_dt.isoformat()

    # 1. Latest OHLCV bar
    if df_1y is not None and not df_1y.empty and "Close" in df_1y and len(df_1y["Close"].dropna()) > 0:
        close_series = df_1y["Close"].dropna()
        latest_ohlcv_close = float(close_series.iloc[-1])
        last_idx = close_series.index[-1]
        if hasattr(last_idx, "strftime"):
            latest_ohlcv_date = last_idx.strftime("%Y-%m-%d")
        else:
            latest_ohlcv_date = str(last_idx)[:10]
        price_source: Literal["ohlcv_close", "verified_live_quote"] = "ohlcv_close"
    else:
        latest_ohlcv_close = float(info.get("currentPrice") or info.get("regularMarketPrice") or 0.0)
        latest_ohlcv_date = now_dt.strftime("%Y-%m-%d")
        flags.append("ohlcv_unavailable_fallback:market_data")
        price_source = "verified_live_quote"

    quote_price = info.get("currentPrice") or info.get("regularMarketPrice")
    shares_out = info.get("sharesOutstanding")

    # EOD analysis strictly defaults to latest OHLCV close
    analysis_price = latest_ohlcv_close
    analysis_price_as_of = latest_ohlcv_date

    # Calendar session analysis
    session_info = get_us_market_session_info(now_dt, latest_ohlcv_date)
    market_session_status = session_info["market_session_status"]
    data_freshness_status = session_info["data_freshness_status"]
    expected_session_date = session_info["expected_latest_session_date"]
    actual_session_date = session_info["actual_latest_session_date"]
    missing_sessions = session_info["missing_trading_sessions"]

    # Check sync & stale status
    price_sync_status: Literal["synced", "quote_ohlcv_mismatch", "stale"] = "synced"
    if data_freshness_status in ("stale_one_session", "stale_multiple_sessions"):
        flags.append(f"stale_ohlcv:{data_freshness_status}")
        flags.append("stale_ohlcv:market_data")
        if missing_sessions >= 2:
            price_sync_status = "stale"

    if price_sync_status != "stale" and quote_price is not None and latest_ohlcv_close > 0:
        diff_pct = abs(float(quote_price) - latest_ohlcv_close) / latest_ohlcv_close
        if diff_pct > 0.0005:  # > 0.05% difference between live quote and EOD close
            price_sync_status = "quote_ohlcv_mismatch"
            flags.append("quote_ohlcv_mismatch:market_data")

    if shares_out is not None and shares_out > 0:
        market_cap = round(analysis_price * shares_out, 2)
    else:
        market_cap = info.get("marketCap")

    freshness_status: Literal["fresh", "stale", "session_synced", "out_of_session"] = "fresh"
    if price_sync_status == "stale" or data_freshness_status == "stale_multiple_sessions":
        freshness_status = "stale"
    elif price_sync_status == "quote_ohlcv_mismatch":
        freshness_status = "out_of_session"

    snapshot = AtomicMarketSnapshot(
        analysis_price=round(analysis_price, 4),
        analysis_price_as_of=analysis_price_as_of,
        price_source=price_source,
        latest_ohlcv_close=round(latest_ohlcv_close, 4),
        latest_ohlcv_date=latest_ohlcv_date,
        shares_outstanding=shares_out,
        market_cap=market_cap,
        price_sync_status=price_sync_status,
        freshness_status=freshness_status,
        market_session_status=market_session_status,
        data_freshness_status=data_freshness_status,
        expected_latest_session_date=expected_session_date,
        actual_latest_session_date=actual_session_date,
        missing_trading_sessions=missing_sessions,
        retrieved_at=now_iso,
    )
    return snapshot, flags


