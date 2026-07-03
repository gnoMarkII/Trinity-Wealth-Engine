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
def ingest_financial_trends(ticker: str, market: Market = "US") -> str:
    """ดึงแนวโน้มงบการเงินย้อนหลัง (รายได้รวม + กำไรสุทธิ) ของหุ้นรายตัวจาก Yahoo Finance (รองรับ TH/US)

    [Usage/When to use]
    ใช้เมื่อต้องการดูข้อมูลทางการเงิน **ย้อนหลังหลายปี** (Time Series/Historical)
    - ครอบคลุม: Total Revenue และ Net Income ย้อนหลัง 4 ปีงบการเงิน
    - คำค้นที่เกี่ยวข้อง: "แนวโน้ม", "ย้อนหลัง", "รายปี", "4 ปี", "historical", "trends"
    - **สำคัญ**: หากผู้ใช้ใช้คำว่า "กระแสเงินสด" หรือ "สุขภาพการเงิน" ร่วมกับคำว่า "ย้อนหลัง" ให้เลือกใช้เครื่องมือนี้ (ingest_financial_trends) เป็นหลัก

    [Caution]
    - ห้ามใช้เครื่องมือนี้สำหรับดึง Snapshot ปัจจุบัน (ใช้ ingest_financial_health หรือ ingest_stock_fundamentals แทน)
    - เครื่องมือนี้แค่ส่งคืนข้อความ Markdown (ไม่บันทึกไฟล์เอง)

    Args:
        ticker (str): Ticker symbol เช่น 'AAPL', 'PTT' (ห้ามมี .BK suffix — ระบบจะเติมให้)
        market (Market): 'TH' สำหรับหุ้นไทย (SET) หรือ 'US' สำหรับหุ้นอเมริกา (default)
    """
    display_sym = ticker.strip().upper().removesuffix(".BK")
    yf_sym = _normalize_yf_ticker(display_sym, market)
    currency = _currency_for(market)
    today = datetime.now().strftime("%Y-%m-%d")
    now_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        tk = yf.Ticker(yf_sym)
        financials = _with_retry(lambda: tk.financials)
        info = _with_retry(lambda: tk.info)
        short_name = info.get("shortName") or display_sym
    except Exception as e:
        log.warning("yfinance fetch failed | %s (%s): %s", display_sym, market, e)
        return f"ERROR: ไม่สามารถดึงข้อมูล {display_sym} ({market}) ได้: {e}"

    if financials is None or financials.empty:
        return f"ERROR: ไม่พบข้อมูลงบการเงินสำหรับ {display_sym} ({market})"

    def _find_series(df: pd.DataFrame, candidates: list[str]):
        for key in candidates:
            if key in df.index:
                return df.loc[key]
        return None

    revenue_s = _find_series(financials, ["Total Revenue", "TotalRevenue"])
    net_income_s = _find_series(financials, [
        "Net Income", "NetIncome",
        "Net Income Common Stockholders", "NetIncomeCommonStockholders",
    ])

    cols = list(financials.columns[:4])

    md_lines = [
        "---",
        f"title: {display_sym} Financial Trends {today}",
        "entity_type: Financial_Trends",
        f"ticker: {display_sym}",
        f"market: {market}",
        f"date: {today}",
        f"last_updated: {now_time}",
        f"tags: [financial_trends, {display_sym.lower()}, market_{market.lower()}, stock_analysis]",
        "---",
        "",
        f"# แนวโน้มงบการเงิน: {short_name} ({display_sym}, {market})",
        "",
        "## รายได้รวมและกำไรสุทธิ (ย้อนหลัง 4 ปีงบการเงิน)",
        "",
        "| ปีงบการเงิน | รายได้รวม | กำไรสุทธิ |",
        "|------------|----------|----------|",
    ]

    for col in cols:
        year_str = col.strftime("%Y") if hasattr(col, "strftime") else str(col)[:4]
        rev = _fmt_fin(revenue_s[col] if revenue_s is not None else None, currency)
        ni = _fmt_fin(net_income_s[col] if net_income_s is not None else None, currency)
        md_lines.append(f"| {year_str} | {rev} | {ni} |")

    md_lines += [
        "",
        "## หมายเหตุ",
        "",
        f"> ข้อมูลจาก Yahoo Finance — B = พันล้าน {currency}, M = ล้าน {currency}, T = ล้านล้าน {currency}",
        "",
    ]

    notices = []
    if revenue_s is None:
        notices.append(f"ไม่พบแถว 'Total Revenue' — Index ที่มี: {list(financials.index[:8])}")
    if net_income_s is None:
        notices.append(f"ไม่พบแถว 'Net Income' — Index ที่มี: {list(financials.index[:8])}")

    if notices:
        md_lines += ["## ข้อผิดพลาด", ""]
        md_lines += [f"- {n}" for n in notices]
        md_lines.append("")

    md_lines += [
        "## Related",
        "",
        f"- [[{display_sym}]]",
        "",
    ]

    return "\n".join(md_lines)


