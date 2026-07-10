import os
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone, date
from pathlib import Path
from langchain_core.tools import tool
from core.retry import with_retry
from tools._atomic_io import _atomic_write_to

def _fetch_rss_with_retry(url: str) -> bytes:
    def _fetch():
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            return response.read()
    return with_retry(_fetch)

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

_YT_NS = {
    'ns0': 'http://www.w3.org/2005/Atom',
    'media': 'http://search.yahoo.com/mrss/',
    'yt': 'http://www.youtube.com/xml/schemas/2015',
}


def get_youtube_candidates(lookback_days: int = 30) -> list[dict]:
    """ดึงรายการคลิปใหม่จากทุกช่องที่ติดตาม คืนเป็น list of dict ล้วนๆ ไม่มี side effect (ไม่เขียนไฟล์)

    ใช้เป็น candidate list สำหรับ human-in-the-loop approval แยกออกมาจาก
    generate_weekly_youtube_digest เพื่อให้เรียกซ้ำได้อย่างปลอดภัย (idempotent) —
    LangGraph interrupt() รัน node ซ้ำจากต้นทุกครั้งที่ resume
    """
    today = datetime.now(timezone.utc)
    cutoff = today - timedelta(days=lookback_days)

    candidates: list[dict] = []
    for handle, channel_id in TARGET_CHANNELS.items():
        url = f"https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"
        try:
            xml_data = _fetch_rss_with_retry(url)
            root = ET.fromstring(xml_data)

            channel_title_element = root.find('ns0:title', _YT_NS)
            channel_title = channel_title_element.text if channel_title_element is not None else handle

            for entry in root.findall('ns0:entry', _YT_NS):
                published_str = entry.find('ns0:published', _YT_NS).text
                if not published_str:
                    continue
                try:
                    published_date = datetime.fromisoformat(published_str.replace('Z', '+00:00'))
                except ValueError:
                    continue
                if published_date < cutoff:
                    continue

                video_title = entry.find('ns0:title', _YT_NS).text
                video_link_element = entry.find('ns0:link', _YT_NS)
                video_link = video_link_element.attrib.get('href', '') if video_link_element is not None else ''

                if "shorts" in video_link.lower() or "#shorts" in video_title.lower():
                    continue

                video_title = video_title.replace("|", "｜")

                thumbnail_url = ""
                media_group = entry.find('media:group', _YT_NS)
                if media_group is not None:
                    media_thumb = media_group.find('media:thumbnail', _YT_NS)
                    if media_thumb is not None:
                        thumbnail_url = media_thumb.attrib.get('url', '')

                video_id_element = entry.find('yt:videoId', _YT_NS)
                video_id = video_id_element.text if video_id_element is not None else None
                is_fetched = False
                if video_id:
                    summaries_dir = _SAVE_DIR.parent
                    if list(summaries_dir.rglob(f"YouTube_Insight_{video_id}_*.md")):
                        is_fetched = True

                candidates.append({
                    "channel": channel_title,
                    "title": video_title,
                    "link": video_link,
                    "video_id": video_id,
                    "published": published_date.strftime("%Y-%m-%d"),
                    "thumbnail": thumbnail_url,
                    "is_fetched": is_fetched,
                })
        except Exception as e:
            from core.logger import get_logger
            get_logger(__name__).error(f"Error parsing channel {handle}: {e}")
            continue

    return candidates


@tool
def generate_weekly_youtube_digest() -> str:
    """สร้างรายงานสรุปวิดีโอใหม่ในรอบ 30 วันล่าสุด (Monthly Digest) จากช่อง YouTube ที่ติดตามไว้ผ่าน RSS

    [Usage/When to use]
    ใช้เมื่อต้องการอัปเดตรายการวิดีโอล่าสุดจากช่อง YouTube ที่เราสนใจ
    - ตรวจสอบรายการช่องที่ตั้งค่าไว้ (ในตัวแปร _CHANNELS)
    - ดึงข้อมูลจาก RSS ว่ามีคลิปไหนบ้างที่เผยแพร่ในช่วง 30 วันที่ผ่านมา
    - รวบรวมข้อมูลทั้งหมดเป็น Digest เดียว

    [Caution]
    - เครื่องมือนี้แค่ส่งคืนข้อความ Markdown (ไม่บันทึกไฟล์เอง)

    Args:
        None

    Returns:
        str: รายงาน Monthly Digest ในรูปแบบ Markdown พร้อม YAML Frontmatter
    """
    try:
        today = datetime.now(timezone.utc)

        md_lines = [
            "---",
            f"title: YouTube Weekly Digest {today.strftime('%Y-%m-%d')}",
            "entity_type: youtube_digest",
            "tags: [youtube, inbox, digest]",
            "---",
            "",
            f"# 📺 YouTube Weekly Digest (รอบ 30 วันล่าสุด)",
            f"อัปเดตเมื่อ: {today.strftime('%Y-%m-%d %H:%M:%S')} (UTC)",
            "",
            "ติ๊ก `[x]` ช่องที่สนใจ จากนั้นคัดลอก Link ไปให้ Agent ดึง Transcript ได้เลย",
            ""
        ]

        candidates = get_youtube_candidates(lookback_days=30)

        by_channel: dict[str, list[dict]] = {}
        for v in candidates:
            by_channel.setdefault(v["channel"], []).append(v)

        for channel_title, videos in by_channel.items():
            md_lines.append(f"## 📈 {channel_title}")
            md_lines.append("| เลือก | ภาพปก | ชื่อคลิป | วันที่เผยแพร่ |")
            md_lines.append("|:---:|:---:|---|:---:|")
            for v in videos:
                thumb_html = f'<img src="{v["thumbnail"]}" width="160" />' if v['thumbnail'] else ""
                checkbox = "[x]" if v.get("is_fetched") else "[ ]"
                title_display = f"~~{v['title']}~~" if v.get("is_fetched") else v['title']
                md_lines.append(f"| {checkbox} | {thumb_html} | [{title_display}]({v['link']}) | {v['published']} |")
            md_lines.append("")

        if not candidates:
            md_lines.append("> ไม่มีคลิปใหม่ในช่วง 30 วันที่ผ่านมา")

        content = "\n".join(md_lines)

        return content
    except Exception as e:
        return f"Error: ไม่สามารถสร้าง YouTube Digest ได้ ({str(e)})"


def load_recent_youtube_insights(
    lookback_days: int = 14,
    max_chars: int = 15_000,
    max_bullets_per_clip: int = 3,
    reference_date: os.PathLike | str | None = None, # Allowed signature flexible for testing or date passing
) -> str:
    """อ่าน YouTube Insight ล่าสุดจากคลัง Vault แบบ Deterministic พร้อม Pre-aggregation Condensation
    และ File-Boundary Truncation
    """
    from tools.archivist.parser import extract_yaml_frontmatter_value
    from core.logger import get_logger

    log = get_logger(__name__)

    vault_path = Path(os.getenv("OBSIDIAN_VAULT_PATH", "./memories")).resolve()
    summaries_dir = vault_path / "30_Knowledge_Base" / "YouTube_Summaries"
    if not summaries_dir.exists():
        return ""

    if reference_date and isinstance(reference_date, date):
        cutoff_date = reference_date - timedelta(days=lookback_days)
    elif reference_date and isinstance(reference_date, datetime):
        cutoff_date = reference_date.date() - timedelta(days=lookback_days)
    else:
        cutoff_date = datetime.now(timezone.utc).date() - timedelta(days=lookback_days)

    collected_clips = []

    for md_file in summaries_dir.glob("*.md"):
        try:
            content = md_file.read_text(encoding="utf-8")
            entity_type = extract_yaml_frontmatter_value(content, "entity_type")
            if entity_type != "youtube_insight":
                continue

            date_str = extract_yaml_frontmatter_value(content, "date")
            if not date_str:
                continue
            try:
                clip_date = datetime.strptime(date_str[:10], "%Y-%m-%d").date()
            except ValueError:
                continue

            if clip_date < cutoff_date:
                continue

            video_id = extract_yaml_frontmatter_value(content, "video_id")
            if not video_id or video_id.strip() == "" or video_id.lower() == "none":
                # Fallback to stem extraction
                stem_parts = md_file.stem.split(" ")
                if len(stem_parts) >= 3:
                    video_id = stem_parts[2]
                else:
                    video_id = md_file.stem

            channel = extract_yaml_frontmatter_value(content, "channel")
            if not channel or channel.strip() == "" or channel.lower() == "none":
                channel = "Unknown Channel"
            else:
                channel = channel.strip()

            # Extract ## ใจความสำคัญ
            bullets = []
            lines = content.splitlines()
            in_section = False
            for line in lines:
                stripped = line.strip()
                if stripped.startswith("## ใจความสำคัญ"):
                    in_section = True
                    continue
                elif in_section and stripped.startswith("## "):
                    break
                elif in_section:
                    if stripped.startswith("- ") or stripped.startswith("* "):
                        bullets.append(stripped)
                        if len(bullets) >= max_bullets_per_clip:
                            break

            if not bullets:
                continue

            clip_text = f"[{channel}: {video_id} | {date_str[:10]}]\n" + "\n".join(bullets)
            collected_clips.append((clip_date, clip_text))

        except Exception as e:
            log.warning("Error processing youtube insight file %s: %s", md_file, e)
            continue

    if not collected_clips:
        return ""

    # Sort Newest First
    collected_clips.sort(key=lambda x: x[0], reverse=True)

    # File-Boundary Truncation
    output_blocks = []
    total_chars = 0
    truncated = False

    for _, clip_text in collected_clips:
        block_len = len(clip_text) + (2 if output_blocks else 0)  # account for \n\n
        if total_chars + block_len > max_chars:
            if output_blocks:
                truncated = True
                break
            else:
                # If even the very first clip exceeds max_chars, take it up to max_chars
                output_blocks.append(clip_text[:max_chars])
                total_chars = len(output_blocks[0])
                truncated = True
                break
        output_blocks.append(clip_text)
        total_chars += block_len

    result = "\n\n".join(output_blocks)
    if truncated:
        result += f"\n... [ตัดทอน — แสดงเฉพาะคลิปที่ใหม่ที่สุด ไม่เกิน {max_chars:,} ตัวอักษร]"

    return result
