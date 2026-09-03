from datetime import datetime, timedelta, timezone
from decimal import Decimal
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
    quarantine_meta: Optional[dict[str, Any]] = None,
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
                    shares_val = r.get("Shares")
                    shares = float(shares_val) if (shares_val is not None and pd.notna(shares_val)) else 0.0
                    val = r.get("Value")
                    value = float(val) if (val is not None and pd.notna(val)) else 0.0
                    pr = r.get("Price")
                    price_direct = float(pr) if (pr is not None and pd.notna(pr)) else 0.0
                    price = (value / shares) if (shares > 0 and value > 0) else price_direct
                    if pd.isna(price):
                        price = 0.0
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
    p_filing_accs = set()
    p_lot_count = 0
    p_cents = 0

    s_filing_accs = set()
    s_lot_count = 0
    s_cents = 0

    s_10b51_filing_accs = set()
    s_10b51_lot_count = 0
    s_10b51_cents = 0

    s_unflagged_filing_accs = set()
    s_unflagged_lot_count = 0
    s_unflagged_cents = 0

    tax_withhold_count = 0
    tax_withhold_cents = 0
    exercise_count = 0
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

        acc = tx.get("accession_number") or ""
        code = str(tx.get("transaction_code", "")).upper()
        tx_type = str(tx.get("transaction_type", code)).upper()
        is_10b51 = bool(tx.get("is_10b5_1_plan", False)) or tx_type == "S_10B5_1_FLAGGED"
        price_raw = tx.get("price_per_share")
        price = float(price_raw) if (price_raw is not None and pd.notna(price_raw)) else 0.0

        # Exact cents calculation
        lot_cents_raw = tx.get("lot_value_cents")
        if lot_cents_raw is None or pd.isna(lot_cents_raw):
            shares_raw = tx.get("shares")
            shares = float(shares_raw) if (shares_raw is not None and pd.notna(shares_raw)) else 0.0
            lot_cents = int(round(shares * price * 100))
        else:
            try:
                lot_cents = int(lot_cents_raw)
            except (ValueError, TypeError):
                lot_cents = 0

        if code == "P" or tx_type == "P":
            if acc:
                p_filing_accs.add(acc)
            p_lot_count += 1
            p_cents += lot_cents
            if price > 0:
                buy_prices.append(price)
            if tx.get("is_c_suite") or any(t in str(tx.get("officer_title", "")).lower() for t in ["ceo", "cfo", "coo", "chief", "president"]):
                c_suite_count += 1
        elif code == "S" or tx_type.startswith("S"):
            if acc:
                s_filing_accs.add(acc)
            s_lot_count += 1
            s_cents += lot_cents
            if is_10b51:
                if acc:
                    s_10b51_filing_accs.add(acc)
                s_10b51_lot_count += 1
                s_10b51_cents += lot_cents
            else:
                if acc:
                    s_unflagged_filing_accs.add(acc)
                s_unflagged_lot_count += 1
                s_unflagged_cents += lot_cents
        elif code == "F" or tx_type == "F_TAX_WITHHOLDING":
            tax_withhold_count += 1
            tax_withhold_cents += lot_cents
        elif code == "M" or tx_type == "M_EXERCISE":
            exercise_count += 1

    buy_min = min(buy_prices) if buy_prices else None
    buy_max = max(buy_prices) if buy_prices else None

    # Determine quarantine metadata
    q_meta = quarantine_meta or (getattr(canonical_transactions, "quarantine_meta", None) if canonical_transactions is not None else None)
    requires_review = q_meta.get("requires_review", False) if isinstance(q_meta, dict) else False
    quarantined_filings = q_meta.get("quarantined_filing_count_90d", 0) if isinstance(q_meta, dict) else 0
    quarantined_lots = q_meta.get("quarantined_lot_count_90d", 0) if isinstance(q_meta, dict) else 0

    # Status Determination
    if requires_review:
        status = "requires_review"
        signal_conf = "unavailable"
    elif c_suite_count >= 2 or (p_lot_count >= 3 and p_cents >= 100_000_00):
        status = "bullish_cluster"
        signal_conf = "high" if canonical_transactions is not None else "moderate"
    elif p_lot_count >= 1:
        status = "moderate_buying"
        signal_conf = "high" if canonical_transactions is not None else "moderate"
    elif s_lot_count > 0:
        status = "selling_activity"
        signal_conf = "high" if canonical_transactions is not None else "moderate"
    else:
        status = "neutral_no_signal"
        signal_conf = "high" if canonical_transactions is not None else "moderate"

    def _to_usd_str(cents: int) -> str:
        return f"{Decimal(cents) / Decimal(100):.2f}"

    return InsiderConviction(
        open_market_p_count_90d=p_lot_count,
        open_market_p_value_usd=float(Decimal(p_cents) / Decimal(100)),
        open_market_p_value_cents=p_cents,
        open_market_p_value_usd_str=_to_usd_str(p_cents),
        open_market_s_count_90d=s_lot_count,
        open_market_s_value_usd=float(Decimal(s_cents) / Decimal(100)),
        open_market_s_value_cents=s_cents,
        open_market_s_value_usd_str=_to_usd_str(s_cents),
        rule_10b5_1_s_count_90d=s_10b51_lot_count,
        rule_10b5_1_s_value_usd=float(Decimal(s_10b51_cents) / Decimal(100)),
        rule_10b5_1_s_value_cents=s_10b51_cents,
        rule_10b5_1_s_value_usd_str=_to_usd_str(s_10b51_cents),
        unflagged_s_count_90d=s_unflagged_lot_count,
        unflagged_s_value_usd=float(Decimal(s_unflagged_cents) / Decimal(100)),
        unflagged_s_value_cents=s_unflagged_cents,
        unflagged_s_value_usd_str=_to_usd_str(s_unflagged_cents),
        tax_withholding_count_90d=tax_withhold_count,
        tax_withholding_value_usd=float(Decimal(tax_withhold_cents) / Decimal(100)),
        exercise_count_90d=exercise_count,
        c_suite_p_count=c_suite_count,
        filing_count_90d=len(p_filing_accs | s_filing_accs),
        transaction_lot_count_90d=p_lot_count + s_lot_count,
        rule_10b5_1_filing_count_90d=len(s_10b51_filing_accs),
        rule_10b5_1_lot_count_90d=s_10b51_lot_count,
        quarantined_filing_count_90d=quarantined_filings,
        quarantined_lot_count_90d=quarantined_lots,
        insider_buy_range_min=round(buy_min, 2) if buy_min is not None else None,
        insider_buy_range_max=round(buy_max, 2) if buy_max is not None else None,
        signal_confidence=signal_conf,
        amendment_unresolved=requires_review,
        status=status,
        data_status="available" if not requires_review else "partial",
    ), flags
