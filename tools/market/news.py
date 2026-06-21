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
def ingest_stock_news(ticker: str, market: Market = "US") -> str:
    """ดึงพาดหัวข่าวล่าสุด 5 ข่าวของหุ้นรายตัวจาก Yahoo Finance (รองรับ TH/US)
    แสดง title, publisher, และ link — Return Markdown พร้อม YAML frontmatter
    ไม่บันทึกไฟล์ด้วยตัวเอง

    Args:
        ticker: Ticker symbol เช่น 'AAPL', 'PTT' (ห้ามมี .BK suffix — ระบบเติมให้)
        market: 'TH' (SET) หรือ 'US' (default)
    """
    display_sym = ticker.strip().upper().removesuffix(".BK")
    yf_sym = _normalize_yf_ticker(display_sym, market)
    today = datetime.now().strftime("%Y-%m-%d")
    now_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        news_list = _yf_news(yf_sym)
    except Exception as e:
        log.warning("yfinance news fetch failed | %s (%s): %s", display_sym, market, e)
        return f"ERROR: ไม่สามารถดึงข่าวของ {display_sym} ({market}) ได้: {e}"

    if not news_list:
        return f"ERROR: ไม่พบข่าวสำหรับ {display_sym} ({market})"

    items = news_list[:5]

    md_lines = [
        "---",
        f"title: {display_sym} Latest News {today}",
        "entity_type: Company_News",
        f"ticker: {display_sym}",
        f"market: {market}",
        f"date: {today}",
        f"last_updated: {now_time}",
        f"tags: [news, {display_sym.lower()}, market_{market.lower()}, stock_analysis]",
        "---",
        "",
        f"# ข่าวล่าสุด: {display_sym} ({market})",
        "",
    ]

    for i, item in enumerate(items, start=1):
        content = item.get("content") or {}
        title = content.get("title") or item.get("title") or "ไม่มีชื่อข่าว"
        provider = (content.get("provider") or {}).get("displayName") or item.get("publisher") or "N/A"
        url = (content.get("canonicalUrl") or {}).get("url") or item.get("link") or ""
        link_md = f"[อ่านต่อ]({url})" if url else "N/A"
        md_lines += [
            f"{i}. **{title}**",
            f"   - ที่มา: {provider}",
            f"   - {link_md}",
            "",
        ]

    md_lines += [
        "## Related",
        "",
        f"- [[{display_sym}]]",
        "",
        "## หมายเหตุ",
        "",
        "> ข่าวจาก Yahoo Finance — ใช้ประกอบการวิเคราะห์เท่านั้น",
        "",
    ]

    return "\n".join(md_lines)


