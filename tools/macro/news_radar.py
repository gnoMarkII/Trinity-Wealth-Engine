import calendar
from datetime import datetime, timezone
import email.utils
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import feedparser
from langchain_core.tools import tool

from core.logger import get_logger
from core.nlp_utils import calculate_freshness, group_similar_news, select_representative_news
from core.retry import with_retry
from schemas.macro_schemas import ThemeCategory

log = get_logger(__name__)

_DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/rss+xml, application/rdf+xml, application/atom+xml, application/xml, text/xml, */*",
}

FEEDS = [
    {
        "name": "CNBC (Economy & Macro)",
        "url": "https://www.cnbc.com/id/20910258/device/rss/rss.html",
        "fallback_url": "https://www.cnbc.com/id/10000664/device/rss/rss.html",
    },
    {
        "name": "Google News (Global Macro & Fed)",
        "url": "https://news.google.com/rss/search?q=macroeconomics+OR+%22Federal+Reserve%22+OR+inflation&hl=en-US&gl=US&ceid=US:en",
    },
    {
        "name": "Yahoo Finance (Business/Macro - Reuters Backup)",
        "url": "https://feeds.finance.yahoo.com/rss/2.0/headline?s=^GSPC,^DJI,^IXIC,^TNX,CL=F,GC=F",
        "fallback_url": "https://finance.yahoo.com/news/rssindex",
    },
    {"name": "Prachachat (Finance & Macro Thailand)", "url": "https://www.prachachat.net/category/finance/feed"},
    {"name": "Bangkok Post (Business Thailand)", "url": "https://www.bangkokpost.com/rss/data/business.xml"},
]


def _get_news_dir() -> Path:
    return Path(os.getenv("OBSIDIAN_VAULT_PATH", "./memories")).resolve() / "30_Knowledge_Base" / "News"


def _is_url_fetched(url: str) -> bool:
    """ตรวจสอบว่า URL นี้เคยถูก fetch และ save ไว้ใน 30_Knowledge_Base/News แล้วหรือไม่"""
    news_dir = _get_news_dir()
    if not news_dir.exists():
        return False

    for md_file in news_dir.rglob("*.md"):
        if "Inbox" in md_file.parts or "Revisions" in md_file.parts:
            continue
        try:
            content = md_file.read_text(encoding="utf-8")
            if url in content:
                return True
        except Exception:
            continue
    return False


def _fetch_rss_with_retry(url: str):
    def _fetch():
        feed = feedparser.parse(
            url,
            agent=_DEFAULT_HEADERS["User-Agent"],
            request_headers=_DEFAULT_HEADERS,
        )
        status = getattr(feed, "status", 200)
        if status >= 400:
            raise ConnectionError(f"HTTP Error {status} for {url}")
        if getattr(feed, "bozo", 0) and isinstance(getattr(feed, "bozo_exception", None), Exception):
            if not feed.entries:
                raise feed.bozo_exception
        return feed

    return with_retry(_fetch)


def _fetch_feed_entries(feed: Dict[str, Any]) -> List[Any]:
    """Fetch entries from primary URL, falling back to backup URL on error."""
    urls_to_try = [feed["url"]]
    if "fallback_url" in feed:
        urls_to_try.append(feed["fallback_url"])

    for target_url in urls_to_try:
        try:
            feed_data = _fetch_rss_with_retry(target_url)
            if feed_data and getattr(feed_data, "entries", None):
                return feed_data.entries
        except Exception as e:
            log.warning("Error parsing feed %s (%s): %s", feed["name"], target_url, e)

    log.error("Failed to fetch feed %s after trying all endpoints", feed["name"])
    return []


def _parse_entry_published_at(entry: Any) -> Optional[datetime]:
    """Parse entry publication datetime from published_parsed tuple or published string."""
    try:
        parsed_time = getattr(entry, "published_parsed", None)
    except AttributeError:
        parsed_time = None

    if not parsed_time:
        try:
            parsed_time = getattr(entry, "updated_parsed", None)
        except AttributeError:
            parsed_time = None

    if parsed_time:
        try:
            return datetime.fromtimestamp(calendar.timegm(parsed_time), tz=timezone.utc)
        except Exception:
            pass

    try:
        pub_date_str = getattr(entry, "published", "")
    except AttributeError:
        pub_date_str = ""

    if not pub_date_str:
        try:
            pub_date_str = getattr(entry, "updated", "")
        except AttributeError:
            pub_date_str = ""

    if pub_date_str:
        try:
            dt = email.utils.parsedate_to_datetime(pub_date_str)
            return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)
        except Exception:
            pass

    return None


def get_news_candidates(max_items: int = 40) -> List[Dict[str, Any]]:
    """ดึงและ dedup ข่าวจากทุก RSS feed คืนเป็น list of dict ล้วนๆ ไม่มี side effect (ไม่เขียนไฟล์)

    ใช้เป็น candidate list สำหรับ human-in-the-loop approval (ก่อนสั่ง deep-dive จริง)
    แยกออกมาจาก generate_news_radar_daily เพื่อให้เรียกซ้ำได้อย่างปลอดภัย — สำคัญเพราะ
    LangGraph interrupt() รัน node ซ้ำจากต้นทุกครั้งที่ resume ฟังก์ชันนี้ต้อง idempotent
    """
    all_news_items: List[Dict[str, Any]] = []
    now_utc = datetime.now(timezone.utc)

    for feed in FEEDS:
        entries = _fetch_feed_entries(feed)
        for entry in entries[:50]:  # Increased to 50 for richer candidate pool
            try:
                title = entry.title.replace("|", "｜").replace("\n", " ").strip()
                link = entry.link
                published_at = _parse_entry_published_at(entry)
                pub_date_str = getattr(entry, "published", "")
                if not pub_date_str:
                    pub_date_str = getattr(entry, "updated", "")

                if published_at:
                    age_hours = max(0, int((now_utc - published_at).total_seconds() / 3600))
                    _, freshness_reason = calculate_freshness(age_hours, ThemeCategory.POLICY)
                else:
                    age_hours = 9999
                    freshness_reason = "Unknown age (parse failed)"

                summary_text = getattr(entry, "summary", getattr(entry, "description", ""))

                # Extract nested publisher from source if available (e.g. Google News)
                source_name = feed["name"]
                feed_source_title = getattr(getattr(entry, "source", None), "title", None)
                if feed_source_title:
                    source_name = f"{source_name} ({feed_source_title})"

                all_news_items.append({
                    "title": title,
                    "summary": summary_text,
                    "source": source_name,
                    "link": link,
                    "published_at": published_at,
                    "pub_date_str": pub_date_str,
                    "age_hours": age_hours,
                    "freshness_reason": freshness_reason,
                    "is_stale": age_hours > 48,
                })
            except Exception as e:
                log.error("Error processing entry for feed %s: %s", feed["name"], e)

    if not all_news_items:
        return []

    clusters = group_similar_news(all_news_items, threshold=0.75)
    representatives = [select_representative_news(cluster) for cluster in clusters]

    # Freshness-first priority sorting:
    # Tier 2: <= 24 hours (Ultra fresh)
    # Tier 1: 24 to 48 hours (Recent)
    # Tier 0: > 48 hours or unknown (Stale)
    # Within tier: sort by multi-source count descending, then age_hours ascending (youngest first)
    def _candidate_sort_key(item: Dict[str, Any]):
        age = item.get("age_hours", 9999)
        if age <= 24:
            tier = 2
        elif age <= 48:
            tier = 1
        else:
            tier = 0
        return (tier, item.get("sources_count", 1), -age)

    representatives.sort(key=_candidate_sort_key, reverse=True)

    candidates: List[Dict[str, Any]] = []
    for item in representatives[:max_items]:
        candidates.append({
            "title": item["title"],
            "summary": item.get("summary", ""),
            "link": item["link"],
            "source": item["source"],
            "sources_count": item.get("sources_count", 1),
            "age_hours": item.get("age_hours", 0),
            "freshness_reason": item.get("freshness_reason", "N/A"),
            "is_stale": item.get("is_stale", False),
            "is_fetched": _is_url_fetched(item["link"]),
        })
    return candidates


@tool
def generate_news_radar_daily() -> str:
    """สร้างเรดาร์ข่าวเศรษฐกิจมหภาครายวัน (News Radar) จาก RSS Feeds

    [Usage/When to use]
    ใช้เมื่อต้องการสรุปข่าวสารเศรษฐกิจมหภาค (Macro News) ล่าสุดจากแหล่งข่าวสำคัญ (เช่น Investing.com, Reuters, BOT)
    - สร้างเป็นตารางสรุปข่าวพร้อม URL อ้างอิง
    - เป็นการดึงข้อมูลจากแหล่งข่าว ไม่ใช่การดึงเนื้อหาเต็มของแต่ละข่าว

    [Caution]
    - เครื่องมือนี้แค่ส่งคืนข้อความ Markdown (ไม่บันทึกไฟล์เอง)

    Args:
        None

    Returns:
        str: รายงาน News Radar รายวันในรูปแบบ Markdown พร้อม YAML Frontmatter
    """
    try:
        today = datetime.now()

        md_lines = [
            "---",
            f"title: News Radar Daily {today.strftime('%Y-%m-%d')}",
            "entity_type: news_radar",
            "tags: [news, radar, inbox]",
            "---",
            "",
            f"# 📡 Macro News Radar ({today.strftime('%d/%m/%Y')})",
            f"อัปเดตเมื่อ: {today.strftime('%Y-%m-%d %H:%M:%S')} (UTC)",
            "",
            "ติ๊ก `[x]` ข่าวที่สนใจ จากนั้นสั่งให้ Agent เจาะลึกข่าว (Deep Diver) ได้เลย",
            "",
        ]

        candidates = get_news_candidates(max_items=25)

        if not candidates:
            md_lines.append("> 📭 ไม่มีข่าวใหม่ในวันนี้")
        else:
            md_lines.append("## 📰 Top Macro News (Deduplicated)")
            md_lines.append("| เลือก | หัวข้อข่าว | ความใหม่ | แหล่งข่าว |")
            md_lines.append("|:---:|---|---|---|")

            for item in candidates:
                title = item["title"]
                link = item["link"]
                checkbox = "[x]" if item["is_fetched"] else "[ ]"
                title_display = f"~~{title}~~" if item["is_fetched"] else title

                sources_count = item.get("sources_count", 1)
                source_display = f"{item['source']}" + (f" (+{sources_count-1})" if sources_count > 1 else "")

                stale_flag = " ⚠️" if item.get("is_stale") else ""
                age_h = item.get("age_hours", 0)
                freshness = f"Age: {age_h}h | {item.get('freshness_reason', 'N/A')}{stale_flag}"

                md_lines.append(f"| {checkbox} | [{title_display}]({link}) | {freshness} | {source_display} (Sources: {sources_count}) |")

            md_lines.append("")

        return "\n".join(md_lines)
    except Exception as e:
        log.error("Error generating news radar: %e", e)
        return f"Error: ไม่สามารถสร้าง News Radar ได้ ({str(e)})"


def ingest_news_candidates(candidates: list) -> List[Dict[str, Any]]:
    """แปลง Terminal V2 NewsCandidate ให้เข้าสู่รูปแบบ Candidate Dictionary ของ News Radar Funnel.

    ใช้สำหรับเชื่อมต่อผลการค้นพบข่าวของ Terminal V2 เข้ากับจุดต่อเดิมของ News Radar โดยคงระบบ
    Human-in-the-loop และการ dedup เดิมไว้ ไม่สร้างระบบเขียนข่าวหรือ workflow อนุมัติซ้ำ
    """
    normalized = []
    for item in candidates:
        headline = getattr(item, "headline", "")
        url = getattr(item, "article_url", "")
        pub = getattr(item, "publisher", "RSS Discovery")
        is_stale = getattr(item, "is_stale", False)
        normalized.append({
            "title": headline,
            "summary": f"{headline} (Publisher: {pub})",
            "link": url,
            "source": pub,
            "sources_count": 1,
            "age_hours": 0,
            "freshness_reason": "Terminal V2 Ticker RSS Discovery",
            "is_stale": is_stale,
            "is_fetched": _is_url_fetched(url) if url else False,
        })
    return normalized
