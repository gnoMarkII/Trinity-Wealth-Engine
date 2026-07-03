from langsmith import traceable
from datetime import datetime, timedelta, timezone
import pandas as pd
import yfinance as yf
from langchain_core.tools import tool
from core.logger import get_logger
from core.retry import with_retry as _with_retry
from core.nlp_utils import group_similar_news, select_representative_news, calculate_freshness
from schemas.macro_schemas import ThemeCategory
from .core import Market, _normalize_yf_ticker, _currency_for, _yf_info, _yf_news, _yf_financials, _fmt_number, _fmt_large, _fmt_fin

log = get_logger(__name__)

@tool
def ingest_stock_news(ticker: str, market: Market = "US") -> str:
    """ดึงพาดหัวข่าวล่าสุด 5 ข่าวของหุ้นรายตัวจาก Yahoo Finance (รองรับ TH/US)

    [Usage/When to use]
    ใช้เมื่อต้องการทราบข่าวสารล่าสุดเกี่ยวกับบริษัทหรือหุ้นนั้นๆ โดยเฉพาะ
    - ครอบคลุม: Title, Publisher, และ Link

    [Caution]
    - ข่าวจะเป็นหัวข้อข่าวเท่านั้น หากต้องการเนื้อหาเต็มต้องใช้ `ingest_article_url` ภายหลัง
    - เครื่องมือนี้แค่ส่งคืนข้อความ Markdown (ไม่บันทึกไฟล์เอง)

    Args:
        ticker (str): Ticker symbol เช่น 'AAPL', 'PTT' (ห้ามมี .BK suffix — ระบบจะเติมให้)
        market (Market): 'TH' สำหรับหุ้นไทย (SET) หรือ 'US' สำหรับหุ้นอเมริกา (default)
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

    now_utc = datetime.now(timezone.utc)
    parsed_items = []
    
    for item in news_list:
        content = item.get("content") or {}
        title = content.get("title") or item.get("title") or "ไม่มีชื่อข่าว"
        provider = (content.get("provider") or {}).get("displayName") or item.get("publisher") or "N/A"
        url = (content.get("canonicalUrl") or {}).get("url") or item.get("link") or ""
        
        pub_time = content.get("pubDate") or item.get("providerPublishTime")
        published_at = None
        if isinstance(pub_time, (int, float)):
            published_at = datetime.fromtimestamp(pub_time, tz=timezone.utc)
        elif isinstance(pub_time, str):
            try:
                # Naive to UTC
                published_at = datetime.fromisoformat(pub_time.replace('Z', '+00:00'))
                if published_at.tzinfo is None:
                    published_at = published_at.replace(tzinfo=timezone.utc)
            except Exception:
                published_at = None
                
        if published_at:
            age_hours = int((now_utc - published_at).total_seconds() / 3600)
            freshness_score, freshness_reason = calculate_freshness(age_hours, ThemeCategory.RISK_SENTIMENT)
            is_stale = age_hours > 48
        else:
            age_hours = 9999
            freshness_score = 0.0
            freshness_reason = "Unknown age (parse failed)"
            is_stale = True
        
        parsed_items.append({
            "title": title,
            "source": provider,
            "link": url,
            "published_at": published_at,
            "age_hours": age_hours,
            "freshness_score": freshness_score,
            "freshness_reason": freshness_reason,
            "is_stale": is_stale
        })

    # Deduplicate
    clusters = group_similar_news(parsed_items, threshold=0.75)
    
    # Sort clusters by size then recency
    representatives = []
    for cluster in clusters:
        rep = select_representative_news(cluster)
        representatives.append(rep)
        
    # Sort representatives to put most confident / freshest first
    representatives.sort(key=lambda x: (x.get('sources_count', 1), -x.get('age_hours', 9999)), reverse=True)
    
    items = representatives[:5]

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
        title = item['title']
        provider = item['source']
        url = item['link']
        sources_count = item.get('sources_count', 1)
        freshness_reason = item.get('freshness_reason', 'Unknown age')
        is_stale = item.get('is_stale', False)
        stale_flag = "⚠️ [STALE]" if is_stale else ""
        
        link_md = f"[อ่านต่อ]({url})" if url else "N/A"
        md_lines += [
            f"{i}. **{title}** {stale_flag}",
            f"   - ที่มา: {provider} (Reported by {sources_count} sources)",
            f"   - อายุข่าว: {item.get('age_hours', 0)} ชั่วโมง ({freshness_reason})",
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


