from typing import Optional, Tuple, List, Literal
from langsmith import traceable
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
from langchain_core.tools import tool
from core.logger import get_logger
from core.retry import with_retry as _with_retry
from schemas.micro_quant_schemas import TacticalSetup
from tools.market.asset_resolver import resolve_asset
from .core import Market, _currency_for, _yf_info, _yf_news, _yf_financials, _fmt_number, _fmt_large, _fmt_fin

log = get_logger(__name__)

def _summarize_insider_transactions(tk: yf.Ticker) -> str:
    """สรุปการซื้อ/ขายหุ้นของคนวงในใน 6 เดือนล่าสุดจาก insider_transactions"""
    try:
        df = _with_retry(lambda: tk.insider_transactions)
        if df is None or df.empty:
            return "ไม่พบข้อมูล"

        date_col = next((c for c in ["startDate", "Start Date", "Date", "date"] if c in df.columns), None)
        tx_col = next((c for c in ["Transaction", "transaction", "Type", "type"] if c in df.columns), None)

        if date_col:
            cutoff = datetime.now() - timedelta(days=180)
            df = df.copy()
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df = df[df[date_col] >= cutoff]

        if df.empty:
            return "ไม่มีรายการใน 6 เดือนล่าสุด"

        if tx_col:
            tx_lower = df[tx_col].astype(str).str.lower()
            buys = int(tx_lower.str.contains(r"buy|purchase|acqui", na=False).sum())
            sells = int(tx_lower.str.contains(r"sell|sale|dispos", na=False).sum())
            if buys > sells:
                return f"ซื้อมากกว่าขาย ({buys} ซื้อ / {sells} ขาย — 6 เดือนล่าสุด)"
            if sells > buys:
                return f"ขายมากกว่าซื้อ ({sells} ขาย / {buys} ซื้อ — 6 เดือนล่าสุด)"
            return f"ซื้อและขายเท่ากัน ({buys} รายการ — 6 เดือนล่าสุด)"

        return f"มี {len(df)} รายการใน 6 เดือนล่าสุด (ไม่สามารถแยกประเภทได้)"
    except Exception:
        return "N/A"


@tool
def ingest_stock_momentum(ticker: str, market: Market = "US") -> str:
    """ดึงข้อมูลโมเมนตัมทางเทคนิคและสถิติราคา (Technical & Trading Momentum) จาก Yahoo Finance (รองรับ TH/US)

    [Usage/When to use]
    ใช้เมื่อต้องการวิเคราะห์แนวโน้มราคา ความเคลื่อนไหวของผู้ถือหุ้นใหญ่ และการชอร์ตเซล
    - ครอบคลุม: ราคาล่าสุด, Moving Averages (50D, 200D), High/Low 52 สัปดาห์, % Insider/Institution Hold,
      สรุปซื้อ/ขายของคนวงใน 6 เดือนล่าสุด, Short Ratio, Short % of Float
    - คำค้นที่เกี่ยวข้อง: "โมเมนตัม", "เทคนิค", "กราฟ", "ราคา", "Moving Average", "Insider", "Institution", "Short"

    [Caution]
    - หุ้นไทย (TH) อาจไม่มีข้อมูล Short หรือ Institution ครบถ้วนจาก Yahoo Finance
    - เครื่องมือนี้แค่ส่งคืนข้อความ Markdown (ไม่บันทึกไฟล์เอง)

    Args:
        ticker (str): Ticker symbol เช่น 'AAPL', 'PTT' (ห้ามมี .BK suffix — ระบบจะเติมให้)
        market (Market): 'TH' สำหรับหุ้นไทย (SET) หรือ 'US' สำหรับหุ้นอเมริกา (default)
    """
    resolved = resolve_asset(ticker, market_hint=market)
    display_sym = resolved.raw_symbol.removesuffix(".BK")
    yf_sym = resolved.provider_symbol
    currency = _currency_for(market)
    today = datetime.now().strftime("%Y-%m-%d")
    now_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        tk = yf.Ticker(yf_sym)
        info = _with_retry(lambda: tk.info)
    except Exception as e:
        log.warning("yfinance fetch failed | %s (%s): %s", display_sym, market, e)
        return f"ERROR: ไม่สามารถดึงข้อมูล {display_sym} ({market}) ได้: {e}"

    if not info or info.get("quoteType") is None:
        return f"ERROR: ไม่พบข้อมูลสำหรับ ticker '{display_sym}' market={market} — ตรวจสอบว่า Symbol ถูกต้อง"

    short_name = info.get("shortName") or display_sym
    cur = info.get("currentPrice")
    m50 = info.get("fiftyDayAverage")
    m200 = info.get("twoHundredDayAverage")

    current_price = _fmt_number(cur, fmt=",.2f", suffix=f" {currency}")
    ma50 = _fmt_number(m50, fmt=",.2f", suffix=f" {currency}")
    ma200 = _fmt_number(m200, fmt=",.2f", suffix=f" {currency}")
    high52 = _fmt_number(info.get("fiftyTwoWeekHigh"), fmt=",.2f", suffix=f" {currency}")
    low52 = _fmt_number(info.get("fiftyTwoWeekLow"), fmt=",.2f", suffix=f" {currency}")

    signal_parts = []
    if cur and m50:
        signal_parts.append("เหนือ MA50 ✓" if cur > m50 else "ใต้ MA50 ✗")
    if cur and m200:
        signal_parts.append("เหนือ MA200 ✓" if cur > m200 else "ใต้ MA200 ✗")
    signal_str = " | ".join(signal_parts) if signal_parts else "N/A"

    # Insider
    insider_held_raw = info.get("heldPercentInsiders")
    insider_held = _fmt_number(
        insider_held_raw * 100 if insider_held_raw is not None else None,
        fmt=".2f", suffix="%"
    )
    insider_tx_summary = _summarize_insider_transactions(tk)

    # Institution & Short Interest
    inst_held_raw = info.get("heldPercentInstitutions")
    inst_held = _fmt_number(
        inst_held_raw * 100 if inst_held_raw is not None else None,
        fmt=".2f", suffix="%"
    )
    short_ratio = _fmt_number(info.get("shortRatio"), fmt=".2f", suffix=" วัน")
    short_pct_raw = info.get("sharesPercentSharesOut")
    short_pct = _fmt_number(
        short_pct_raw * 100 if short_pct_raw is not None else None,
        fmt=".2f", suffix="%"
    )

    md_lines = [
        "---",
        "schema_version: 2",
        f"title: {display_sym} Momentum Insider {today}",
        "entity_type: Stock_Momentum",
        f"ticker: {display_sym}",
        f"market: {market}",
        f"date: {today}",
        f"last_updated: {now_time}",
        f"tags: [stock_momentum, {display_sym.lower()}, market_{market.lower()}, stock_analysis, technical]",
        "---",
        "",
        f"# โมเมนตัมราคาและคนวงใน: {short_name} ({display_sym}, {market})",
        "",
        "## สัญญาณเทคนิค (Technical Signals)",
        "",
        "| ดัชนี | ค่า | ความหมาย |",
        "|-------|-----|---------|",
        f"| **ราคาปัจจุบัน** | {current_price} | ราคาตลาดล่าสุด |",
        f"| **MA50** | {ma50} | ค่าเฉลี่ยเคลื่อนที่ 50 วัน |",
        f"| **MA200** | {ma200} | ค่าเฉลี่ยเคลื่อนที่ 200 วัน |",
        f"| **52W High** | {high52} | ราคาสูงสุดใน 52 สัปดาห์ |",
        f"| **52W Low** | {low52} | ราคาต่ำสุดใน 52 สัปดาห์ |",
        "",
        f"> **สัญญาณ:** {signal_str}",
        "",
        "## ข้อมูลคนวงใน (Insider Activity)",
        "",
        "| รายการ | ค่า |",
        "|--------|-----|",
        f"| **% Insider Hold** | {insider_held} |",
        f"| **ซื้อ/ขาย (6 เดือน)** | {insider_tx_summary} |",
        "",
        "## พฤติกรรมสถาบันและการชอร์ตเซล (Institution & Short Interest)",
        "",
        "| ดัชนี | ค่า | ความหมาย |",
        "|-------|-----|---------|",
        f"| **% Institution Hold** | {inst_held} | สัดส่วนหุ้นที่สถาบัน (กองทุน/บริษัท) ถือครอง — สูง = ความเชื่อมั่นสถาบัน |",
        f"| **Short Ratio** | {short_ratio} | จำนวนวันที่ต้องใช้ปิด Short ทั้งหมด — >5 วัน = ความเสี่ยง Short Squeeze สูง |",
        f"| **Short % of Float** | {short_pct} | % หุ้นที่ถูก Short เทียบหุ้นที่ซื้อขายได้ — >10% = มี Bearish Sentiment สูง |",
        "",
        "## Related",
        "",
        f"- {display_sym}",
        "",
        "## หมายเหตุ",
        "",
        "> ข้อมูลจาก Yahoo Finance — MA = Moving Average | Insider Hold = % หุ้นที่ผู้บริหารถือครอง",
        "> Institution Hold = กองทุน/บริษัทใหญ่ | Short Ratio = Days to Cover | Short % Float = Short Interest",
        "",
    ]

    return "\n".join(md_lines)


@traceable(run_type="parser")
def compute_tactical_setup(
    ticker: str,
    market: str = "US",
    price_history_df: Optional[pd.DataFrame] = None,
    current_price: Optional[float] = None,
) -> Tuple[TacticalSetup, list[str]]:
    """คำนวณแผนการเทรดเชิงระบบ (Tactical Setup): S/R, ATR, Price Stage, Buy Zone, Stop Loss, และ 1-3M Tactical R:R"""
    flags: list[str] = []
    df = price_history_df

    if df is None or df.empty:
        try:
            resolved = resolve_asset(ticker, market_hint=market)
            tk = yf.Ticker(resolved.provider_symbol)
            df = tk.history(period="1y", auto_adjust=False)
        except Exception as e:
            log.warning("Failed to fetch price history for tactical setup %s: %s", ticker, e)
            return TacticalSetup(status="unavailable"), ["price_history_unavailable:tactical"]

    if df is None or len(df) < 20:
        return TacticalSetup(status="partial"), ["insufficient_bars_for_tactical:tactical"]

    close = df["Close"].dropna()
    high = df["High"].dropna()
    low = df["Low"].dropna()

    curr_p = float(current_price) if current_price and current_price > 0 else float(close.iloc[-1])

    # 1. Moving Averages
    sma50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else None
    sma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else None

    # 2. ATR 14
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr14_series = tr.rolling(14).mean()
    atr14 = float(atr14_series.iloc[-1]) if len(atr14_series.dropna()) > 0 else max(0.5, curr_p * 0.02)

    # 3. Price Stage Determination
    stage = "UNKNOWN"
    if sma50 is not None and sma200 is not None:
        if curr_p > sma50 and sma50 > sma200:
            stage = "STAGE_2_MARKUP"
        elif curr_p < sma50 and sma50 < sma200:
            stage = "STAGE_4_MARKDOWN"
        elif curr_p < sma50 and sma50 > sma200:
            stage = "STAGE_3_DISTRIBUTION"
        else:
            stage = "STAGE_1_BASE"
    elif sma50 is not None:
        stage = "STAGE_2_MARKUP" if curr_p > sma50 else "STAGE_4_MARKDOWN"

    # 4. Support and Resistance Clustering (Past 60 bars)
    recent_lows = low.tail(60)
    recent_highs = high.tail(60)

    lows_below = recent_lows[recent_lows < curr_p * 0.99]
    if not lows_below.empty:
        key_support = float(lows_below.quantile(0.60))
    elif sma50 is not None and sma50 < curr_p:
        key_support = sma50
    elif sma200 is not None and sma200 < curr_p:
        key_support = sma200
    elif not recent_lows.empty:
        key_support = float(recent_lows.quantile(0.20))
    else:
        key_support = curr_p - (1.5 * atr14)

    highs_above = recent_highs[recent_highs > curr_p * 1.01]
    if not highs_above.empty:
        key_res = float(highs_above.quantile(0.40))
    elif sma50 is not None and sma50 > curr_p:
        key_res = sma50
    elif sma200 is not None and sma200 > curr_p:
        key_res = sma200
    elif not recent_highs.empty:
        key_res = float(recent_highs.quantile(0.85))
    else:
        key_res = curr_p + (2.5 * atr14)

    if key_res <= key_support:
        key_res = key_support + (1.5 * atr14)

    # 5. Pullback / Dip Setup Metrics
    buy_zone_min = round(max(0.1, key_support - (0.5 * atr14)), 2)
    buy_zone_max = round(key_support + (1.0 * atr14), 2)
    invalidation_stop = round(max(0.01, key_support - (0.75 * atr14)), 2)
    tactical_target = round(key_res, 2)
    is_in_buy_zone = (buy_zone_min <= curr_p <= buy_zone_max)

    current_rr: Optional[float] = None
    pullback_entry_status: Literal["below_stop", "in_buy_zone", "between_zone_and_target", "at_or_above_target", "unavailable"] = "unavailable"

    if curr_p <= invalidation_stop:
        pullback_entry_status = "below_stop"
        current_rr = None
    elif curr_p >= tactical_target:
        pullback_entry_status = "at_or_above_target"
        current_rr = None
    elif is_in_buy_zone:
        pullback_entry_status = "in_buy_zone"
        if invalidation_stop < curr_p < tactical_target:
            upside = tactical_target - curr_p
            downside = curr_p - invalidation_stop
            if downside > 0:
                current_rr = round(upside / downside, 2)
    elif buy_zone_max < curr_p < tactical_target:
        pullback_entry_status = "between_zone_and_target"
        upside = tactical_target - curr_p
        downside = curr_p - invalidation_stop
        if downside > 0:
            current_rr = round(upside / downside, 2)
    else:
        pullback_entry_status = "unavailable"

    bz_rr_min: Optional[float] = None
    if tactical_target > buy_zone_max and buy_zone_max > invalidation_stop:
        bz_rr_min = round((tactical_target - buy_zone_max) / (buy_zone_max - invalidation_stop), 2)

    bz_rr_max: Optional[float] = None
    if tactical_target > buy_zone_min and buy_zone_min > invalidation_stop:
        bz_rr_max = round((tactical_target - buy_zone_min) / (buy_zone_min - invalidation_stop), 2)

    # 6. Breakout Setup Metrics & Lifecycle
    avg_vol_20d: Optional[float] = None
    latest_vol: Optional[float] = None
    volume_ratio: Optional[float] = None
    volume_confirmed: Optional[bool] = None
    if "Volume" in df.columns:
        vol_clean = df["Volume"].dropna()
        if len(vol_clean) >= 21:
            avg_vol_20d = float(vol_clean.iloc[-21:-1].mean())
            latest_vol = float(vol_clean.iloc[-1])
        elif len(vol_clean) >= 2:
            avg_vol_20d = float(vol_clean.iloc[:-1].mean())
            latest_vol = float(vol_clean.iloc[-1])

        if avg_vol_20d is not None and avg_vol_20d > 0 and latest_vol is not None:
            volume_ratio = round(latest_vol / avg_vol_20d, 2)

    breakout_trig = round(key_res + (0.1 * atr14), 2)
    breakout_targ = round(key_res + (2.3 * atr14), 2)
    breakout_stop = round(key_res - (1.0 * atr14), 2)
    max_chase_price = round(breakout_trig + (0.5 * atr14), 2)
    breakout_planned_rr = round((breakout_targ - breakout_trig) / (breakout_trig - breakout_stop), 2) if (breakout_trig > breakout_stop) else None

    breakout_curr_rr: Optional[float] = None
    breakout_status: Literal["pre_trigger", "eligible", "chased", "expired"] = "pre_trigger"
    breakout_eligible: bool = False

    if curr_p < breakout_trig:
        # Pre-trigger: price has not crossed trigger level yet
        breakout_status = "pre_trigger"
        breakout_curr_rr = None
        breakout_eligible = False
        volume_confirmed = False
    elif breakout_trig <= curr_p <= max_chase_price:
        # Price is within actionable trigger window
        if avg_vol_20d is not None and latest_vol is not None and avg_vol_20d > 0:
            volume_confirmed = bool(latest_vol >= 1.5 * avg_vol_20d)

        if breakout_targ > curr_p and curr_p > breakout_stop:
            breakout_curr_rr = round((breakout_targ - curr_p) / (curr_p - breakout_stop), 2)
            # Volume confirmation: if volume data is available, require confirmation; otherwise rely on R:R
            vol_ok = volume_confirmed if volume_confirmed is not None else True
            if breakout_curr_rr >= 1.50 and vol_ok:
                breakout_status = "eligible"
                breakout_eligible = True
            else:
                breakout_status = "chased"
                breakout_eligible = False
        else:
            breakout_status = "expired"
            breakout_eligible = False
    elif curr_p > max_chase_price and curr_p < breakout_targ:
        # Price exceeded max chase tolerance
        breakout_curr_rr = round((breakout_targ - curr_p) / (curr_p - breakout_stop), 2)
        breakout_status = "chased"
        volume_confirmed = False
        breakout_eligible = False
    else:
        # Price above target or below stop loss
        breakout_status = "expired"
        breakout_curr_rr = None
        volume_confirmed = False
        breakout_eligible = False

    return TacticalSetup(
        price_stage=stage,
        current_price=round(curr_p, 2),
        sma_50=round(sma50, 2) if sma50 is not None else None,
        sma_200=round(sma200, 2) if sma200 is not None else None,
        atr_14=round(atr14, 2),
        key_support_level=round(key_support, 2),
        key_resistance_level=round(key_res, 2),
        buy_zone_min=buy_zone_min,
        buy_zone_max=buy_zone_max,
        invalidation_stop_loss=invalidation_stop,
        tactical_target_price=tactical_target,
        tactical_risk_reward_ratio=current_rr,
        current_rr_ratio=current_rr,
        buy_zone_rr_min=bz_rr_min,
        buy_zone_rr_max=bz_rr_max,
        is_in_buy_zone=is_in_buy_zone,
        pullback_entry_status=pullback_entry_status,
        breakout_trigger_price=breakout_trig,
        breakout_target_price=breakout_targ,
        breakout_stop_loss=breakout_stop,
        breakout_planned_rr=breakout_planned_rr,
        breakout_current_rr=breakout_curr_rr,
        max_breakout_chase_price=max_chase_price,
        breakout_entry_status=breakout_status,
        breakout_entry_eligible=breakout_eligible,
        breakout_volume_ratio=volume_ratio,
        breakout_volume_baseline=round(avg_vol_20d, 2) if avg_vol_20d is not None else None,
        breakout_volume_confirmed=volume_confirmed,
        horizon_timeframe="1-3M",
        status="available",
    ), flags



