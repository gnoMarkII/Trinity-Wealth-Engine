from typing import Literal
import pandas as pd
import yfinance as yf
from core.retry import with_retry as _with_retry

Market = Literal["TH", "US"]

def _normalize_yf_ticker(ticker: str, market: Market = "US") -> str:
    """แปลง ticker → format ที่ yfinance รู้จัก
    TH: เติม .BK ถ้ายังไม่มี (SET listings เช่น PTT → PTT.BK)
    US: คืนตรงๆ (NYSE/Nasdaq ไม่ต้อง suffix)
    """
    sym = ticker.strip().upper()
    if market == "TH" and not sym.endswith(".BK"):
        return f"{sym}.BK"
    return sym

def _currency_for(market: Market) -> str:
    """Derive currency code ของรายงานราคา/งบจาก market"""
    return "THB" if market == "TH" else "USD"

def _yf_info(ticker: str) -> dict:
    """ดึง yf.Ticker(ticker).info พร้อม retry — wrap ทุก info call เพื่อกัน rate-limit"""
    return _with_retry(lambda: yf.Ticker(ticker).info)

def _yf_news(ticker: str) -> list:
    return _with_retry(lambda: yf.Ticker(ticker).news)

def _yf_financials(ticker: str):
    return _with_retry(lambda: yf.Ticker(ticker).financials)

def _fmt_number(value, fmt=".2f", suffix="") -> str:
    if value is None:
        return "N/A"
    try:
        v = float(value)
        if pd.isna(v):
            return "N/A"
        return f"{v:{fmt}}{suffix}"
    except (TypeError, ValueError):
        return "N/A"

def _fmt_large(value, currency_code: str = "USD") -> str:
    """แปลง marketCap เป็น B/M/T พร้อมระบุสกุล (default USD เพื่อ back-compat)"""
    if value is None:
        return "N/A"
    try:
        v = float(value)
        if pd.isna(v):
            return "N/A"
    except (TypeError, ValueError):
        return "N/A"
    if v >= 1e12:
        return f"{v / 1e12:.2f}T {currency_code}"
    if v >= 1e9:
        return f"{v / 1e9:.2f}B {currency_code}"
    if v >= 1e6:
        return f"{v / 1e6:.2f}M {currency_code}"
    return f"{v:.0f} {currency_code}"

def _fmt_fin(value, currency_code: str = "USD") -> str:
    """แปลงตัวเลขงบการเงินเป็น B/M พร้อมสกุล (default USD) — คืน N/A ถ้า None/NaN"""
    if value is None:
        return "N/A"
    try:
        v = float(value)
        if pd.isna(v):
            return "N/A"
        neg = v < 0
        a = abs(v)
        if a >= 1e12:
            s = f"{a / 1e12:.2f}T"
        elif a >= 1e9:
            s = f"{a / 1e9:.2f}B"
        elif a >= 1e6:
            s = f"{a / 1e6:.2f}M"
        else:
            s = f"{a:.0f}"
        return f"-{s} {currency_code}" if neg else f"{s} {currency_code}"
    except (TypeError, ValueError):
        return "N/A"
