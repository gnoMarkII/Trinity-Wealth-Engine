import os
import re
from datetime import datetime, timezone
from pathlib import Path
import feedparser

from langchain_core.tools import tool
from core.logger import get_logger
from core.nlp_utils import group_similar_news, select_representative_news, calculate_freshness
from core.retry import with_retry
from schemas.macro_schemas import ThemeCategory
from tools._atomic_io import _atomic_write_to
import email.utils

log = get_logger(__name__)

def _fetch_rss_with_retry(url: str):
    def _fetch():
        import feedparser
        feed = feedparser.parse(url)
        status = getattr(feed, 'status', 200)
        if status >= 400:
            raise ConnectionError(f"HTTP Error {status} for {url}")
        if getattr(feed, 'bozo', 0) and isinstance(getattr(feed, 'bozo_exception', None), Exception):
            raise feed.bozo_exception
        return feed
    return with_retry(_fetch)

_SAVE_DIR = Path(__file__).resolve().parents[2] / "memories" / "30_Knowledge_Base" / "News" / "Inbox"
_NEWS_DIR = Path(__file__).resolve().parents[2] / "memories" / "30_Knowledge_Base" / "News"

FEEDS = [
    {"name": "Investing.com (Economic News)", "url": "https://www.investing.com/rss/news_285.rss"},
    {"name": "Yahoo Finance (Business/Macro - Reuters Backup)", "url": "https://finance.yahoo.com/news/rssindex"},
    {"name": "Prachachat (Finance & Macro Thailand)", "url": "https://www.prachachat.net/category/finance/feed"}
]

def _is_url_fetched(url: str) -> bool:
    """ตรวจสอบว่า URL นี้เคยถูก fetch และ save ไว้ใน 30_Knowledge_Base/News แล้วหรือไม่"""
    if not _NEWS_DIR.exists():
        return False
        
    for md_file in _NEWS_DIR.rglob("*.md"):
        # ไม่เช็คไฟล์ใน Inbox
        if "Inbox" in md_file.parts:
            continue
        try:
            content = md_file.read_text(encoding="utf-8")
            if url in content:
                return True
        except Exception:
            continue
    return False

def get_news_candidates(max_items: int = 15) -> list[dict]:
    """ดึงและ dedup ข่าวจากทุก RSS feed คืนเป็น list of dict ล้วนๆ ไม่มี side effect (ไม่เขียนไฟล์)

    ใช้เป็น candidate list สำหรับ human-in-the-loop approval (ก่อนสั่ง deep-dive จริง)
    แยกออกมาจาก generate_news_radar_daily เพื่อให้เรียกซ้ำได้อย่างปลอดภัย — สำคัญเพราะ
    LangGraph interrupt() รัน node ซ้ำจากต้นทุกครั้งที่ resume ฟังก์ชันนี้ต้อง idempotent
    """
    all_news_items: list[dict] = []
    now_utc = datetime.now(timezone.utc)

    for feed in FEEDS:
        try:
            feed_data = _fetch_rss_with_retry(feed['url'])
            if not feed_data.entries:
                continue

            for entry in feed_data.entries[:20]:  # Fetch more initially for dedup
                title = entry.title.replace("|", "｜").replace("\n", " ")
                link = entry.link

                pub_date_str = getattr(entry, "published", "")
                published_at = None
                parsed_time = getattr(entry, "published_parsed", None)
                if parsed_time:
                    import calendar
                    published_at = datetime.fromtimestamp(calendar.timegm(parsed_time), tz=timezone.utc)
                else:
                    if pub_date_str:
                        try:
                            parsed_tuple = email.utils.parsedate_tz(pub_date_str)
                            if parsed_tuple:
                                published_at = email.utils.to_datetime(parsed_tuple)
                                if published_at.tzinfo is None:
                                    published_at = published_at.replace(tzinfo=timezone.utc)
                        except Exception:
                            pass

                if published_at:
                    age_hours = int((now_utc - published_at).total_seconds() / 3600)
                    freshness_score, freshness_reason = calculate_freshness(age_hours, ThemeCategory.POLICY)
                else:
                    age_hours = 9999
                    freshness_reason = "Unknown age (parse failed)"

                all_news_items.append({
                    "title": title,
                    "source": feed['name'],
                    "link": link,
                    "published_at": published_at,
                    "pub_date_str": pub_date_str,
                    "age_hours": age_hours,
                    "freshness_reason": freshness_reason,
                    "is_stale": age_hours > 48
                })
        except Exception as e:
            log.error(f"Error parsing feed {feed['name']}: {e}")
            continue

    if not all_news_items:
        return []

    clusters = group_similar_news(all_news_items, threshold=0.75)
    representatives = [select_representative_news(cluster) for cluster in clusters]
    representatives.sort(key=lambda x: (x.get('sources_count', 1), -x.get('age_hours', 9999)), reverse=True)

    candidates = []
    for item in representatives[:max_items]:
        candidates.append({
            "title": item['title'],
            "link": item['link'],
            "source": item['source'],
            "sources_count": item.get('sources_count', 1),
            "age_hours": item.get('age_hours', 0),
            "freshness_reason": item.get('freshness_reason', 'N/A'),
            "is_stale": item.get('is_stale', False),
            "is_fetched": _is_url_fetched(item['link']),
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
        _SAVE_DIR.mkdir(parents=True, exist_ok=True)
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
            ""
        ]
        
        candidates = get_news_candidates(max_items=15)

        if not candidates:
            md_lines.append("> 📭 ไม่มีข่าวใหม่ในวันนี้")
        else:
            md_lines.append("## 📰 Top Macro News (Deduplicated)")
            md_lines.append("| เลือก | หัวข้อข่าว | ความใหม่ | แหล่งข่าว |")
            md_lines.append("|:---:|---|---|---|")

            for item in candidates:
                title = item['title']
                link = item['link']
                checkbox = "[x]" if item['is_fetched'] else "[ ]"
                title_display = f"~~{title}~~" if item['is_fetched'] else title

                sources_count = item.get('sources_count', 1)
                source_display = f"{item['source']}" + (f" (+{sources_count-1})" if sources_count > 1 else "")

                stale_flag = " ⚠️" if item.get('is_stale') else ""
                age_h = item.get('age_hours', 0)
                freshness = f"Age: {age_h}h | {item.get('freshness_reason', 'N/A')}{stale_flag}"

                md_lines.append(f"| {checkbox} | [{title_display}]({link}) | {freshness} | {source_display} (Sources: {sources_count}) |")

            md_lines.append("")

        content = "\n".join(md_lines)
        
        return content
    except Exception as e:
        log.error(f"Error generating news radar: {e}")
        return f"Error: ไม่สามารถสร้าง News Radar ได้ ({str(e)})"
