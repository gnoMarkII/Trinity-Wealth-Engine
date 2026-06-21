from langsmith import traceable
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
from langchain_core.tools import tool
from core.logger import get_logger
from core.retry import with_retry as _with_retry
from .core import Market, _normalize_yf_ticker, _currency_for, _yf_info, _yf_news, _yf_financials, _fmt_number, _fmt_large, _fmt_fin

log = get_logger(__name__)

@tool
@traceable(run_type="tool")
def ingest_stock_consensus(ticker: str, market: Market = "US") -> str:
    """ดึงมุมมองนักวิเคราะห์ (Analyst Consensus) ของหุ้นรายตัวจาก Yahoo Finance (รองรับ TH/US)
    ครอบคลุม: ราคาเป้าหมาย (ต่ำ/เฉลี่ย/สูง), คำแนะนำ, จำนวนนักวิเคราะห์
    Return เป็น Markdown พร้อม YAML frontmatter — ไม่บันทึกไฟล์ด้วยตัวเอง

    Args:
        ticker: Ticker symbol เช่น 'AAPL', 'PTT' (ห้ามมี .BK suffix — ระบบเติมให้)
        market: 'TH' (SET) หรือ 'US' (default) — TH อาจไม่มี consensus จาก nyfinance
    """
    display_sym = ticker.strip().upper().removesuffix(".BK")
    yf_sym = _normalize_yf_ticker(display_sym, market)
    currency = _currency_for(market)
    today = datetime.now().strftime("%Y-%m-%d")
    now_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        info = _yf_info(yf_sym)
    except Exception as e:
        log.warning("yfinance fetch failed | %s (%s): %s", display_sym, market, e)
        return f"ERROR: ไม่สามารถดึงข้อมูล {display_sym} ({market}) ได้: {e}"

    if not info or info.get("quoteType") is None:
        return f"ERROR: ไม่พบข้อมูลสำหรับ ticker '{display_sym}' market={market} — ตรวจสอบว่า Symbol ถูกต้อง"

    target_low = _fmt_number(info.get("targetLowPrice"), fmt=",.2f", suffix=f" {currency}")
    target_mean = _fmt_number(info.get("targetMeanPrice"), fmt=",.2f", suffix=f" {currency}")
    target_high = _fmt_number(info.get("targetHighPrice"), fmt=",.2f", suffix=f" {currency}")
    current_price = _fmt_number(info.get("currentPrice"), fmt=",.2f", suffix=f" {currency}")
    recommendation = (info.get("recommendationKey") or "N/A").upper()
    num_analysts = info.get("numberOfAnalystOpinions")
    num_analysts_str = str(num_analysts) if num_analysts is not None else "N/A"

    _cur_raw = info.get("currentPrice")
    _mean_raw = info.get("targetMeanPrice")
    if _cur_raw and _mean_raw and float(_cur_raw) != 0:
        _upside_val = (float(_mean_raw) - float(_cur_raw)) / float(_cur_raw) * 100
        upside_pct = f"{_upside_val:+.1f}%"
    else:
        upside_pct = "N/A"

    short_name = info.get("shortName") or display_sym

    md_lines = [
        "---",
        f"title: {display_sym} Analyst Consensus {today}",
        "entity_type: Analyst_Consensus",
        f"ticker: {display_sym}",
        f"market: {market}",
        f"date: {today}",
        f"last_updated: {now_time}",
        f"tags: [analyst_consensus, {display_sym.lower()}, market_{market.lower()}, stock_analysis]",
        "---",
        "",
        f"# มุมมองนักวิเคราะห์: {short_name} ({display_sym}, {market})",
        "",
        "## Analyst Consensus",
        "",
        "| รายการ | ค่า | ความหมาย |",
        "|--------|-----|---------|",
        f"| **ราคาปัจจุบัน** | {current_price} | ราคาตลาดล่าสุด |",
        f"| **Target Low** | {target_low} | ราคาเป้าหมายต่ำสุดของนักวิเคราะห์ |",
        f"| **Target Mean** | {target_mean} | ราคาเป้าหมายเฉลี่ยของนักวิเคราะห์ |",
        f"| **Upside to Target** | {upside_pct} | % จากราคาปัจจุบันถึง Target Mean (+ = Upside, − = Downside) |",
        f"| **Target High** | {target_high} | ราคาเป้าหมายสูงสุดของนักวิเคราะห์ |",
        f"| **Recommendation** | {recommendation} | คำแนะนำรวม (BUY/HOLD/SELL) |",
        f"| **จำนวนนักวิเคราะห์** | {num_analysts_str} คน | จำนวนนักวิเคราะห์ที่ให้ความเห็น |",
        "",
        "## Related",
        "",
        f"- [[{display_sym}]]",
        "",
        "## หมายเหตุ",
        "",
        "> ข้อมูลจาก Yahoo Finance — Consensus มาจากนักวิเคราะห์ Wall Street ใช้ประกอบการตัดสินใจเท่านั้น",
        "",
    ]

    return "\n".join(md_lines)


