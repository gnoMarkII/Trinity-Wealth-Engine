from langsmith import traceable
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
from langchain_core.tools import tool
from core.logger import get_logger
from core.retry import with_retry as _with_retry
from .core import Market, _normalize_yf_ticker, _currency_for, _yf_info, _yf_news, _yf_financials, _fmt_number, _fmt_large, _fmt_fin

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
@traceable(run_type="tool")
def ingest_stock_momentum(ticker: str, market: Market = "US") -> str:
    """ดึงข้อมูลสัญญาณเทคนิค/โมเมนตัมราคา ข้อมูลคนวงใน สถาบัน และ Short Interest (รองรับ TH/US)
    ครอบคลุม: MA50, MA200, 52W High/Low, % Insider/Institution Hold, Short Ratio, Short % Float
    Return เป็น Markdown พร้อม YAML frontmatter — ไม่บันทึกไฟล์ด้วยตัวเอง

    Args:
        ticker: Ticker symbol เช่น 'AAPL', 'PTT' (ห้ามมี .BK suffix — ระบบเติมให้)
        market: 'TH' (SET) หรือ 'US' (default) — TH อาจมี short/institution data น้อยกว่า
    """
    display_sym = ticker.strip().upper().removesuffix(".BK")
    yf_sym = _normalize_yf_ticker(display_sym, market)
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
        f"- [[{display_sym}]]",
        "",
        "## หมายเหตุ",
        "",
        "> ข้อมูลจาก Yahoo Finance — MA = Moving Average | Insider Hold = % หุ้นที่ผู้บริหารถือครอง",
        "> Institution Hold = กองทุน/บริษัทใหญ่ | Short Ratio = Days to Cover | Short % Float = Short Interest",
        "",
    ]

    return "\n".join(md_lines)


