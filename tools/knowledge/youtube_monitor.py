import os
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path
from langchain_core.tools import tool
from tools._atomic_io import _atomic_write_to

TARGET_CHANNELS = {
    "@PiSecurities": "UCwhGe4-luVrfHN1bPDqBakQ",
    "@Finnomena": "UC_EgP5CYTAwJwU2wbnfN37w",
    "@bualuangsec": "UCLpPa3UthE1VlMlZ8Thm93w",
    "@pingprakit6949": "UCf2qSf_iiUuSPEHzme0g79w",
    "@TheStandardWealth": "UCcDjvLn1-qwPWL36erYwyUg",
    "@ksecuritieschannel": "UCgjKjt3dUjHCn2EskjjJYBg",
    "@Tam-Eig": "UCnj8uUh6SHdZi3FvWJj9Dyw",
    "@Wealthion": "UCKMeK-HGHfUFFArZ91rzv5A",
}

_SAVE_DIR = Path(__file__).resolve().parents[2] / "memories" / "30_Knowledge_Base" / "YouTube_Summaries" / "Inbox"

@tool
def generate_weekly_youtube_digest() -> str:
    """สร้างรายงานสรุปวิดีโอใหม่ในสัปดาห์ (Weekly Digest) จากช่อง YouTube ที่ติดตามไว้ผ่าน RSS

    [Usage/When to use]
    ใช้เมื่อต้องการอัปเดตรายการวิดีโอล่าสุดจากช่อง YouTube ที่เราสนใจ
    - ตรวจสอบรายการช่องที่ตั้งค่าไว้ (ในตัวแปร _CHANNELS)
    - ดึงข้อมูลจาก RSS ว่ามีคลิปไหนบ้างที่เผยแพร่ในสัปดาห์นี้
    - รวบรวมข้อมูลทั้งหมดเป็น Digest เดียว

    [Caution]
    - เครื่องมือนี้แค่ส่งคืนข้อความ Markdown (ไม่บันทึกไฟล์เอง)
    - **ต้อง** นำผลลัพธ์ที่ได้ไปส่งให้ Archivist บันทึกไฟล์ต่อด้วย `write_raw_markdown` ลงใน Inbox

    Returns:
        str: รายงาน Weekly Digest ในรูปแบบ Markdown พร้อม YAML Frontmatter
    """
    today = datetime.now(timezone.utc)
    seven_days_ago = today - timedelta(days=7)
    
    md_lines = [
        "---",
        f"title: YouTube Weekly Digest {today.strftime('%Y-%m-%d')}",
        "entity_type: youtube_digest",
        "tags: [youtube, inbox, digest]",
        "---",
        "",
        f"# 📺 YouTube Weekly Digest (รอบ 7 วันล่าสุด)",
        f"อัปเดตเมื่อ: {today.strftime('%Y-%m-%d %H:%M:%S')} (UTC)",
        "",
        "ติ๊ก `[x]` ช่องที่สนใจ จากนั้นคัดลอก Link ไปให้ Agent ดึง Transcript ได้เลย",
        ""
    ]
    
    ns = {
        'ns0': 'http://www.w3.org/2005/Atom',
        'media': 'http://search.yahoo.com/mrss/',
        'yt': 'http://www.youtube.com/xml/schemas/2015'
    }

    found_any = False

    for handle, channel_id in TARGET_CHANNELS.items():
        url = f"https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        try:
            xml_data = urllib.request.urlopen(req).read()
            root = ET.fromstring(xml_data)
            
            channel_title_element = root.find('ns0:title', ns)
            channel_title = channel_title_element.text if channel_title_element is not None else handle
            
            channel_videos = []
            
            for entry in root.findall('ns0:entry', ns):
                published_str = entry.find('ns0:published', ns).text
                if not published_str:
                    continue
                    
                try:
                    published_date = datetime.fromisoformat(published_str.replace('Z', '+00:00'))
                except ValueError:
                    continue
                    
                if published_date >= seven_days_ago:
                    video_title = entry.find('ns0:title', ns).text
                    video_link_element = entry.find('ns0:link', ns)
                    video_link = video_link_element.attrib.get('href', '') if video_link_element is not None else ''
                    
                    if "shorts" in video_link.lower() or "#shorts" in video_title.lower():
                        continue
                        
                    video_title = video_title.replace("|", "｜") # ใช้ Full-width pipe เพื่อไม่ให้ตาราง Markdown พัง
                    
                    thumbnail_url = ""
                    media_group = entry.find('media:group', ns)
                    if media_group is not None:
                        media_thumb = media_group.find('media:thumbnail', ns)
                        if media_thumb is not None:
                            thumbnail_url = media_thumb.attrib.get('url', '')

                    video_id_element = entry.find('yt:videoId', ns)
                    video_id = video_id_element.text if video_id_element is not None else None
                    is_fetched = False
                    if video_id:
                        summaries_dir = _SAVE_DIR.parent
                        if list(summaries_dir.rglob(f"YouTube_Insight_{video_id}_*.md")):
                            is_fetched = True

                    channel_videos.append({
                        "title": video_title,
                        "link": video_link,
                        "published": published_date.strftime("%Y-%m-%d"),
                        "thumbnail": thumbnail_url,
                        "is_fetched": is_fetched
                    })
                    
            if channel_videos:
                md_lines.append(f"## 📈 {channel_title}")
                md_lines.append("| เลือก | ภาพปก | ชื่อคลิป | วันที่เผยแพร่ |")
                md_lines.append("|:---:|:---:|---|:---:|")
                for v in channel_videos:
                    thumb_html = f'<img src="{v["thumbnail"]}" width="160" />' if v['thumbnail'] else ""
                    checkbox = "[x]" if v.get("is_fetched") else "[ ]"
                    title_display = f"~~{v['title']}~~" if v.get("is_fetched") else v['title']
                    md_lines.append(f"| {checkbox} | {thumb_html} | [{title_display}]({v['link']}) | {v['published']} |")
                md_lines.append("")
                found_any = True
                
        except Exception as e:
            md_lines.append(f"## 📈 {handle}")
            md_lines.append(f"> [!WARNING]\n> ดึงข้อมูลล้มเหลว: {e}\n")

    if not found_any:
        md_lines.append("> ไม่มีคลิปใหม่ในช่วง 7 วันที่ผ่านมา")

    content = "\n".join(md_lines)
    
    save_path = _SAVE_DIR / f"Weekly_Digest_{today.strftime('%Y-%m-%d')}.md"
    
    _atomic_write_to(save_path, content)
    
    return f"สร้างไฟล์ Weekly Digest สำเร็จแล้วที่: {save_path}"
