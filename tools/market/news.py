from langsmith import traceable
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
import html
import re
import pandas as pd
import yfinance as yf
from langchain_core.tools import tool
from core.logger import get_logger
from core.retry import with_retry as _with_retry
from core.nlp_utils import group_similar_news, select_representative_news, calculate_freshness
from schemas.macro_schemas import ThemeCategory
from tools.market.asset_resolver import resolve_asset
from .core import Market, _currency_for, _yf_info, _yf_news, _yf_financials, _fmt_number, _fmt_large, _fmt_fin

log = get_logger(__name__)


def _rss_news_candidates(yf_sym: str, limit: int = 10):
    from tools.market.terminal_v2.adapters.rss_news_adapter import RssNewsDiscoveryAdapter
    rss_adapter = RssNewsDiscoveryAdapter()
    return rss_adapter.get_news_candidates(yf_sym, limit=limit)

@tool
def ingest_stock_news(ticker: str, market: Market = "US") -> str:
    """ดึงพาดหัวข่าวล่าสุด 8 ข่าวของหุ้นรายตัวจาก Yahoo Finance และ Multi-Source RSS (รองรับ TH/US)

    [Usage/When to use]
    ใช้เมื่อต้องการทราบข่าวสารล่าสุดเกี่ยวกับบริษัทหรือหุ้นนั้นๆ โดยเฉพาะ
    - ครอบคลุม: Title, Publisher, Link, และ Summary
    - Multi-source coverage: Yahoo Finance API + Google News RSS / Yahoo RSS

    [Caution]
    - ข่าวจะเป็นหัวข้อข่าวและบทคัดย่อเท่านั้น หากต้องการเนื้อหาเต็มต้องใช้ `ingest_article_url` ภายหลัง
    - เครื่องมือนี้แค่ส่งคืนข้อความ Markdown (ไม่บันทึกไฟล์เอง)

    Args:
        ticker (str): Ticker symbol เช่น 'AAPL', 'PTT' (ห้ามมี .BK suffix — ระบบจะเติมให้)
        market (Market): 'TH' สำหรับหุ้นไทย (SET) หรือ 'US' สำหรับหุ้นอเมริกา (default)
    """
    resolved = resolve_asset(ticker, market_hint=market)
    display_sym = resolved.raw_symbol.removesuffix(".BK")
    yf_sym = resolved.provider_symbol
    today = datetime.now().strftime("%Y-%m-%d")
    now_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    news_list = []
    yf_error: Exception | None = None
    try:
        news_list = _yf_news(yf_sym) or []
    except Exception as e:
        yf_error = e
        log.warning("yfinance news fetch failed | %s (%s): %s", display_sym, market, e)

    now_utc = datetime.now(timezone.utc)
    parsed_items = []

    for item in news_list:
        content = item.get("content") or {}
        title = content.get("title") or item.get("title") or "ไม่มีชื่อข่าว"
        provider = (content.get("provider") or {}).get("displayName") or item.get("publisher") or "N/A"
        url = (
            (content.get("canonicalUrl") or {}).get("url")
            or (content.get("clickThroughUrl") or {}).get("url")
            or item.get("link")
            or ""
        )

        raw_summary = content.get("summary") or content.get("description") or item.get("summary") or ""
        clean_summary = ""
        if raw_summary:
            clean_summary = html.unescape(re.sub(r"<[^>]+>", "", str(raw_summary))).strip()[:220]

        pub_time = content.get("pubDate") or item.get("providerPublishTime")
        published_at = None
        if isinstance(pub_time, (int, float)):
            published_at = datetime.fromtimestamp(pub_time, tz=timezone.utc)
        elif isinstance(pub_time, str):
            try:
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
            "is_stale": is_stale,
            "summary": clean_summary,
        })

    # Multi-source supplement: if Yahoo Finance returned fewer than 5 items, supplement from RSS Discovery
    if len(parsed_items) < 5:
        try:
            rss_snap = _rss_news_candidates(yf_sym, limit=10)
            for cand in rss_snap.items:
                cand_title = cand.headline.strip()
                if not cand_title:
                    continue
                if any(cand_title.lower() in p["title"].lower() or p["title"].lower() in cand_title.lower() for p in parsed_items):
                    continue
                cand_pub = None
                if cand.published_at:
                    try:
                        cand_pub = parsedate_to_datetime(cand.published_at)
                        if cand_pub.tzinfo is None:
                            cand_pub = cand_pub.replace(tzinfo=timezone.utc)
                    except Exception:
                        cand_pub = None
                if cand_pub:
                    cand_age = int((now_utc - cand_pub).total_seconds() / 3600)
                    f_score, f_reason = calculate_freshness(cand_age, ThemeCategory.RISK_SENTIMENT)
                    c_stale = cand_age > 48
                else:
                    cand_age = 24
                    f_score = 0.5
                    f_reason = "Recent candidate (estimated age)"
                    c_stale = False
                parsed_items.append({
                    "title": cand_title,
                    "source": cand.publisher or "RSS Discovery",
                    "link": cand.article_url or "",
                    "published_at": cand_pub,
                    "age_hours": cand_age,
                    "freshness_score": f_score,
                    "freshness_reason": f_reason,
                    "is_stale": c_stale,
                    "summary": "",
                })
        except Exception as exc:
            log.debug("RSS news supplement failed for %s: %s", yf_sym, exc)

    if not parsed_items:
        if yf_error is not None:
            return f"ERROR: ไม่สามารถดึงข่าวของ {display_sym} ({market}) ได้: {yf_error}"
        return f"ERROR: ไม่พบข่าวสำหรับ {display_sym} ({market})"

    # Deduplicate
    clusters = group_similar_news(parsed_items, threshold=0.75)

    # Sort clusters by size then recency
    representatives = []
    for cluster in clusters:
        rep = select_representative_news(cluster)
        representatives.append(rep)

    # Sort representatives to put most confident / freshest first
    representatives.sort(key=lambda x: (x.get('sources_count', 1), -x.get('age_hours', 9999)), reverse=True)

    items = representatives[:8]

    md_lines = [
        "---",
        "schema_version: 2",
        f"title: {display_sym} Latest News {today}",
        "entity_type: company_news",
        f"ticker: {display_sym}",
        f"tickers: [{display_sym}]",
        f"market: {market}",
        f"date: {today}",
        f"published_date: {today}",
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

        pub_iso = item['published_at'].isoformat() if item.get('published_at') else ""
        pub_comment = f" <!-- published_at: {pub_iso} -->" if pub_iso else ""

        link_md = f"[อ่านต่อ]({url})" if url else "N/A"
        md_lines += [
            f"{i}. **{title}** {stale_flag}{pub_comment}",
            f"   - ที่มา: {provider} (Reported by {sources_count} sources)",
            f"   - อายุข่าว: {item.get('age_hours', 0)} ชั่วโมง ({freshness_reason})",
        ]
        if item.get("summary"):
            md_lines.append(f"   - สรุปย่อ: {item['summary']}")
        md_lines += [
            f"   - {link_md}",
            "",
        ]

    md_lines += [
        "## Related",
        "",
        f"- {display_sym}",
        "",
        "## หมายเหตุ",
        "",
        "> ข่าวจาก Yahoo Finance & Multi-Source RSS — ใช้ประกอบการวิเคราะห์เท่านั้น",
        "",
    ]

    return "\n".join(md_lines)


def save_company_news_to_vault(news_markdown: str, ticker: str = "") -> str:
    """บันทึก Company News Markdown ลงใน Obsidian Vault V2 ทันที"""
    from tools.archivist.parser import extract_yaml_frontmatter_value
    from tools.archivist.writer import write_raw_markdown

    fm_ticker = extract_yaml_frontmatter_value(news_markdown, "ticker") or ""
    clean_ticker = (ticker or fm_ticker).strip().upper().removesuffix(".BK")
    today = extract_yaml_frontmatter_value(news_markdown, "date") or datetime.now().strftime("%Y-%m-%d")
    fm_title = extract_yaml_frontmatter_value(news_markdown, "title") or f"{clean_ticker} Latest News {today}"
    filename = fm_title if fm_title.startswith(today) else f"{today} {fm_title}"

    return write_raw_markdown.invoke({
        "content": news_markdown,
        "folder_path": "30_Knowledge_Base/News",
        "filename": filename,
    })
