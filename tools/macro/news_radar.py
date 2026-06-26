import os
import re
from datetime import datetime
from pathlib import Path
import feedparser

from langchain_core.tools import tool
from core.logger import get_logger
from tools._atomic_io import _atomic_write_to

log = get_logger(__name__)

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

@tool
def generate_news_radar_daily() -> str:
    """สร้างไฟล์ News Radar (ตารางสรุปข่าวรายวัน) จาก RSS Feeds
    ดึงข่าวเศรษฐกิจมหภาค (Macro News) จากแหล่งต่างๆ เช่น Investing.com, Reuters, BOT
    และเซฟลงใน 30_Knowledge_Base/News/Inbox/News-Radar-Daily_{Date}.md
    """
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
    
    found_any = False
    
    for feed in FEEDS:
        md_lines.append(f"## 📰 {feed['name']}")
        
        try:
            parsed = feedparser.parse(feed['url'])
            if not parsed.entries:
                md_lines.append("> [!WARNING]\n> ไม่พบข่าว หรือดึงข้อมูลจาก RSS ไม่สำเร็จ\n")
                continue
                
            md_lines.append("| เลือก | หัวข้อข่าว | วันที่ |")
            md_lines.append("|:---:|---|:---:|")
            
            # ดึงมาแค่ 10 ข่าวล่าสุดต่อ Feed ก็พอ
            for entry in parsed.entries[:10]:
                title = entry.title.replace("|", "｜").replace("\n", " ")
                link = entry.link
                # บาง feed ไม่มี published
                pub_date = getattr(entry, "published", "")
                
                is_fetched = _is_url_fetched(link)
                checkbox = "[x]" if is_fetched else "[ ]"
                title_display = f"~~{title}~~" if is_fetched else title
                
                md_lines.append(f"| {checkbox} | [{title_display}]({link}) | {pub_date} |")
                
            md_lines.append("")
            found_any = True
            
        except Exception as e:
            log.error(f"Error parsing RSS for {feed['name']}: {e}")
            md_lines.append(f"> [!WARNING]\n> เกิดข้อผิดพลาดในการเชื่อมต่อ: {e}\n")
            
    if not found_any:
        md_lines.append("> 📭 ไม่มีข่าวใหม่ในวันนี้")
        
    content = "\n".join(md_lines)
    save_path = _SAVE_DIR / f"News-Radar-Daily_{today.strftime('%Y-%m-%d')}.md"
    
    _atomic_write_to(save_path, content)
    
    return f"สร้างไฟล์ News Radar สำเร็จแล้วที่: {save_path}"
