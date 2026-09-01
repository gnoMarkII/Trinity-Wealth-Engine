from datetime import datetime, timedelta, timezone
from typing import Any, List, Optional, Tuple
import pandas as pd
import yfinance as yf
from langsmith import traceable
from core.logger import get_logger
from schemas.micro_quant_schemas import DataStatus, InsiderConviction, SmartMoneyFlags

_INSIDER_WINDOW_DAYS = 90

log = get_logger(__name__)


@traceable(run_type="parser")
def compute_smart_money_flags(ticker: str, info_dict: Optional[dict[str, Any]] = None) -> tuple[SmartMoneyFlags, list[str]]:
    """ประเมินสัญญาณ Smart Money (Insider buying/selling, Short Interest, Institutional Ownership)

    หมายเหตุ: คืนข้อมูลเชิงสัญญาณ (Flag) แยกจากคะแนนตัวเลข ไม่รวมใน Composite Score
    แนบ data_quality_flags '10b51_unfiltered:insider_signal' เสมอเนื่องจาก yfinance ไม่ได้แยก 10b5-1 plans
    """
    flags: list[str] = ["10b51_unfiltered:insider_signal"]

    if info_dict is None:
        try:
            tk = yf.Ticker(ticker)
            info_dict = tk.info or {}
        except Exception as e:
            log.warning("Failed to fetch info for ownership flags (%s): %s", ticker, e)
            info_dict = {}

    inst_pct = info_dict.get("heldPercentInstitutions")
    if inst_pct is not None:
        inst_pct = round(inst_pct * 100.0, 2)

    insider_pct = info_dict.get("heldPercentInsiders")
    if insider_pct is not None:
        insider_pct = round(insider_pct * 100.0, 2)

    short_pct = info_dict.get("shortPercentOfFloat")
    if short_pct is not None:
        short_pct = round(short_pct * 100.0, 2)

    short_squeeze_risk = bool(short_pct is not None and short_pct >= 15.0)

    # Insider transactions (90d)
    insider_buy_count = 0
    insider_sell_count = 0
    insider_signal = "neutral"

    try:
        tk = yf.Ticker(ticker)
        insiders = tk.insider_transactions
        if insiders is not None and not insiders.empty and "Transaction" in insiders.columns:
            date_col = next((c for c in ("Start Date", "Date") if c in insiders.columns), None)
            if date_col is not None:
                cutoff = datetime.now(timezone.utc) - timedelta(days=_INSIDER_WINDOW_DAYS)
                tx_dates = pd.to_datetime(insiders[date_col], errors="coerce", utc=True)
                recent = insiders[tx_dates >= cutoff]
            else:
                recent = insiders.iloc[0:0]
                flags.append("insider_date_unavailable:insider_signal")

            for _, row in recent.iterrows():
                trans = str(row.get("Transaction", "")).lower()
                if "buy" in trans or "purchase" in trans:
                    insider_buy_count += 1
                elif "sale" in trans or "sell" in trans:
                    insider_sell_count += 1

            if insider_buy_count > insider_sell_count:
                insider_signal = "buying"
            elif insider_sell_count > insider_buy_count:
                insider_signal = "selling"
    except Exception as e:
        log.debug("Could not fetch insider_transactions for %s: %s", ticker, e)

    # Overall Signal Flag
    if insider_signal == "buying" and not short_squeeze_risk:
        overall = "bullish_signal"
    elif insider_signal == "selling" or short_squeeze_risk:
        overall = "bearish_signal"
    else:
        overall = "neutral"

    res = SmartMoneyFlags(
        insider_signal=insider_signal,
        insider_buy_count_90d=insider_buy_count,
        insider_sell_count_90d=insider_sell_count,
        institutional_ownership_pct=inst_pct,
        insider_ownership_pct=insider_pct,
        short_interest_pct=short_pct,
        short_squeeze_risk=short_squeeze_risk,
        overall_smart_money_flag=overall,
    )
    return res, flags


@traceable(run_type="parser")
def compute_canonical_insider_conviction(
    ticker: str,
    market: str = "US",
    canonical_transactions: Optional[List[dict[str, Any]]] = None,
) -> Tuple[InsiderConviction, List[str]]:
    """คำนวณ Insider Conviction จาก Canonical SEC Form 4 Ledger
    - Code P (Open Market Purchase) สำหรับ Insider Timing
    - Code S (Sale) สำหรับ Contextual Selling Activity (ไม่หักคะแนนอัตโนมัติ)
    - C-Suite Weighting (CEO/CFO/COO)
    - หุ้นไทย (TH) -> not_applicable
    """
    flags: List[str] = []
    mkt = market.upper()

    if mkt == "TH":
        return InsiderConviction(
            status="not_applicable",
            data_status="not_applicable",
        ), ["form4_not_applicable_th_market:insider"]

    # If canonical_transactions not passed, try to fetch from yfinance as fallback
    txs = canonical_transactions
    if txs is None:
        try:
            tk = yf.Ticker(ticker)
            raw_insiders = tk.insider_transactions
            if raw_insiders is not None and not raw_insiders.empty:
                txs = []
                date_col = next((c for c in ("Start Date", "Date") if c in raw_insiders.columns), None)
                for _, r in raw_insiders.iterrows():
                    tx_type = str(r.get("Transaction", "")).strip().lower()
                    code = "P" if ("buy" in tx_type or "purchase" in tx_type) else ("S" if ("sale" in tx_type or "sell" in tx_type) else "O")
                    dt_val = str(r.get(date_col, "")) if date_col else ""
                    shares = float(r.get("Shares", 0) or 0)
                    price = float(r.get("Value", 0) or 0) / shares if shares > 0 else float(r.get("Price", 0) or 0)
                    text_title = str(r.get("Position", "") or r.get("Insider", "")).lower()
                    is_c_suite = any(t in text_title for t in ["ceo", "cfo", "coo", "chief", "president"])
                    txs.append({
                        "transaction_date": dt_val,
                        "transaction_code": code,
                        "shares": shares,
                        "price_per_share": price,
                        "officer_title": text_title,
                        "is_c_suite": is_c_suite,
                    })
        except Exception as e:
            log.warning("Failed to fetch insider transactions for %s: %s", ticker, e)
            return InsiderConviction(status="unavailable", data_status="unavailable"), ["form4_fetch_failed:insider"]

    if txs is None:
        return InsiderConviction(status="unavailable", data_status="unavailable"), ["form4_data_unavailable:insider"]

    cutoff = datetime.now(timezone.utc) - timedelta(days=_INSIDER_WINDOW_DAYS)
    p_count = 0
    p_value = 0.0
    s_count = 0
    s_value = 0.0
    c_suite_count = 0
    buy_prices: List[float] = []

    for tx in txs:
        dt_str = tx.get("transaction_date") or tx.get("filing_date") or ""
        try:
            tx_dt = pd.to_datetime(dt_str, utc=True)
            if tx_dt < cutoff:
                continue
        except Exception:
            pass  # If date parsing fails, include transaction within scope

        code = str(tx.get("transaction_code", "")).upper()
        shares = float(tx.get("shares", 0) or 0)
        price = float(tx.get("price_per_share", 0) or 0)
        val = shares * price

        if code == "P":
            p_count += 1
            p_value += val
            if price > 0:
                buy_prices.append(price)
            if tx.get("is_c_suite") or any(t in str(tx.get("officer_title", "")).lower() for t in ["ceo", "cfo", "coo", "chief", "president"]):
                c_suite_count += 1
        elif code == "S":
            s_count += 1
            s_value += val

    buy_min = min(buy_prices) if buy_prices else None
    buy_max = max(buy_prices) if buy_prices else None

    # Status Determination
    if c_suite_count >= 2 or (p_count >= 3 and p_value >= 100_000.0):
        status = "bullish_cluster"
    elif p_count >= 1:
        status = "moderate_buying"
    elif s_count > 0:
        status = "selling_activity"
    else:
        status = "neutral_no_signal"

    return InsiderConviction(
        open_market_p_count_90d=p_count,
        open_market_p_value_usd=round(p_value, 2),
        open_market_s_count_90d=s_count,
        open_market_s_value_usd=round(s_value, 2),
        c_suite_p_count=c_suite_count,
        insider_buy_range_min=round(buy_min, 2) if buy_min is not None else None,
        insider_buy_range_max=round(buy_max, 2) if buy_max is not None else None,
        status=status,
        data_status="available",
    ), flags
