"""เครื่องมือค้นหาและสกัดข้อมูลจากคลัง YouTube Summaries สำหรับหาไอเดียทำคลิป"""
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple

from langchain_core.tools import tool

from core.logger import get_logger
from tools.archivist.parser import VAULT_PATH, _strip_frontmatter, extract_yaml_frontmatter_value

logger = get_logger(__name__)


def get_channel_with_fallback(content: str) -> str:
    """อ่านชื่อช่องจาก Frontmatter ก่อน หากไม่มีให้ Parse จากบรรทัด > แหล่งที่มา: ... | ช่อง: <ชื่อ> ในเนื้อหา"""
    channel = extract_yaml_frontmatter_value(content, "channel")
    if channel and channel.strip():
        return channel.strip()

    # Fallback: Parse จากเนื้อหา
    match = re.search(r'>\s*แหล่งที่มา:.*?\|\s*ช่อง:\s*([^|\r\n<]+)', content)
    if match:
        return match.group(1).strip()

    match_alt = re.search(r'ช่อง:\s*([^|\r\n<]+)', content)
    if match_alt:
        return match_alt.group(1).strip()

    return "ไม่ระบุช่อง"


def extract_first_bullet_of_key_takeaways(content: str, max_chars: int = 90) -> str:
    """ดึง Bullet แรกของ ## ใจความสำคัญ และตัดความยาวที่ max_chars ตัวอักษร สำหรับใช้ทำ Display Title"""
    body = _strip_frontmatter(content)
    # หา section ## ใจความสำคัญ
    match = re.search(r'^##\s+ใจความสำคัญ\s*\r?\n(.*?)(?=\r?\n##|\Z)', body, re.MULTILINE | re.DOTALL)
    if not match:
        return ""

    section_text = match.group(1)
    for line in section_text.splitlines():
        line_clean = line.strip()
        if line_clean.startswith(("- ", "* ", "+ ")):
            bullet_text = line_clean[2:].strip()
            # ตัด Markdown tags บางส่วนออกเพื่อความสะอาดในชื่อเรื่อง
            bullet_text = re.sub(r'\*\*(.*?)\*\*', r'\1', bullet_text)
            bullet_text = re.sub(r'\[\[(.*?)\]\]', r'\1', bullet_text)
            if len(bullet_text) > max_chars:
                return bullet_text[:max_chars] + "..."
            return bullet_text

    return ""


def extract_sections(content: str, target_headings: List[str]) -> Dict[str, str]:
    """สกัดเนื้อหาของแต่ละ Section ตาม Heading ที่กำหนด หากไฟล์ใดไม่มีบาง Section ให้คืนค่า string ว่าง (ไม่ Error)"""
    body = _strip_frontmatter(content)
    results = {h: "" for h in target_headings}

    for heading in target_headings:
        pattern = rf'^##\s+{re.escape(heading)}\s*\r?\n(.*?)(?=\r?\n##|\Z)'
        match = re.search(pattern, body, re.MULTILINE | re.DOTALL)
        if match:
            # ทำความสะอาด whitespace ข้างต้นและปลาย
            clean_text = match.group(1).strip()
            results[heading] = clean_text

    return results


def get_display_title(content: str, channel: str, fallback_title: str = "") -> str:
    """สร้าง Display Title จาก {channel} + bullet แรกของ ## ใจความสำคัญ (~90 chars)"""
    bullet = extract_first_bullet_of_key_takeaways(content, max_chars=90)
    if bullet:
        return f"[{channel}] {bullet}"
    if fallback_title:
        return f"[{channel}] {fallback_title}"
    return f"[{channel}] YouTube Insight"


@tool
def search_youtube_insights(
    query: str = "",
    channel: str = "",
    lookback_days: int = 30,
    max_results: int = 5,
) -> str:
    """ค้นหาข้อมูลและแก่นไอเดียการลงทุนจากคลังสรุป YouTube Summaries ตามเงื่อนไขที่กำหนด

    [Usage/When to use]
    ใช้เมื่อต้องการค้นหาข้อมูลหรือหาไอเดียทำคลิป YouTube จากคลังความรู้วิดีโอการลงทุนในอดีต
    สามารถกรองตามคำค้นหา (query), ชื่อช่อง (channel), และช่วงเวลา (lookback_days) ได้

    Args:
        query (str): คำค้นหาหรือหัวข้อที่สนใจ เช่น 'AI', 'หุ้นปันผล', 'BDMS', 'Sector Rotation' (ปล่อยว่างเพื่อดูทั้งหมดในช่วงเวลา)
        channel (str): กรองชื่อช่อง YouTube เช่น 'Pi Securities', 'Finnomena', 'Bualuang' (ปล่อยว่างเพื่อดูทุกช่อง)
        lookback_days (int): จำนวนวันย้อนหลังที่ต้องการสืบค้น (ค่าเริ่มต้น 30 วัน)
        max_results (int): จำนวนคลิปสูงสุดที่จะแสดง (ค่าเริ่มต้น 5 คลิป)

    Returns:
        str: รายงานสรุปข้อมูลคลิปที่ตรงเงื่อนไข พร้อม Section สำคัญ (ใจความสำคัญ, แนวคิดการลงทุน, หุ้นและสินทรัพย์)
    """
    summaries_dir = VAULT_PATH / "30_Knowledge_Base" / "YouTube_Summaries"
    if not summaries_dir.exists():
        return f"ไม่พบโฟลเดอร์คลังข้อมูล YouTube Summaries ที่: {summaries_dir}"

    cutoff_date = datetime.now(timezone.utc).date() - timedelta(days=lookback_days)
    candidates: List[Tuple[datetime, str, str, str, Dict[str, str]]] = []

    for md_file in summaries_dir.glob("*.md"):
        try:
            content = md_file.read_text(encoding="utf-8")
            entity_type = extract_yaml_frontmatter_value(content, "entity_type")
            if entity_type and entity_type != "youtube_insight":
                continue

            # วันที่
            date_str = extract_yaml_frontmatter_value(content, "date")
            published_at = None
            if date_str:
                try:
                    published_at = datetime.strptime(date_str[:10], "%Y-%m-%d")
                except ValueError:
                    pass
            if not published_at:
                mtime = datetime.fromtimestamp(md_file.stat().st_mtime)
                published_at = mtime

            if published_at.date() < cutoff_date:
                continue

            # ชื่อช่อง
            clip_channel = get_channel_with_fallback(content)
            if channel and channel.lower() not in clip_channel.lower():
                continue

            # ค้นหา Query
            if query:
                q_lower = query.lower()
                title_val = extract_yaml_frontmatter_value(content, "title") or ""
                if q_lower not in content.lower() and q_lower not in title_val.lower() and q_lower not in clip_channel.lower():
                    continue

            # สกัด Display Title และ Sections สำคัญ
            frontmatter_title = extract_yaml_frontmatter_value(content, "title") or md_file.stem
            display_title = get_display_title(content, clip_channel, fallback_title=frontmatter_title)
            source_url = extract_yaml_frontmatter_value(content, "source_url") or ""

            sections = extract_sections(content, ["ใจความสำคัญ", "แนวคิดการลงทุน", "หุ้นและสินทรัพย์"])
            candidates.append((published_at, display_title, clip_channel, source_url, sections))

        except Exception as e:
            logger.warning("Error reading file %s in search_youtube_insights: %s", md_file.name, e)
            continue

    if not candidates:
        return f"ไม่พบคลิป YouTube ในคลังที่ตรงเงื่อนไข (ค้นหา: '{query}', ช่อง: '{channel}', ย้อนหลัง: {lookback_days} วัน)"

    # เรียงลำดับจากใหม่ไปเก่า
    candidates.sort(key=lambda item: item[0], reverse=True)
    selected = candidates[:max_results]

    output_lines = [
        f"### ผลการค้นหาไอเดียจาก YouTube Summaries (พบ {len(selected)} จากทั้งหมด {len(candidates)} รายการตรงเงื่อนไข)",
        "",
    ]

    for pub_dt, disp_title, ch, url, sections in selected:
        pub_str = pub_dt.strftime("%Y-%m-%d")
        output_lines.append(f"#### {disp_title} ({pub_str})")
        if url:
            output_lines.append(f"- **URL:** {url}")
        output_lines.append(f"- **ช่อง:** {ch}")

        if sections["ใจความสำคัญ"]:
            output_lines.append("- **ใจความสำคัญ:**")
            for line in sections["ใจความสำคัญ"].splitlines()[:5]:  # แสดงไม่เกิน 5 bullet
                if line.strip():
                    output_lines.append(f"  {line.strip()}")

        if sections["แนวคิดการลงทุน"]:
            output_lines.append("- **แนวคิดการลงทุน:**")
            for line in sections["แนวคิดการลงทุน"].splitlines()[:5]:
                if line.strip():
                    output_lines.append(f"  {line.strip()}")

        if sections["หุ้นและสินทรัพย์"]:
            output_lines.append("- **หุ้น/สินทรัพย์ที่กล่าวถึง:**")
            for line in sections["หุ้นและสินทรัพย์"].splitlines()[:5]:
                if line.strip():
                    output_lines.append(f"  {line.strip()}")

        output_lines.append("")

    return "\n".join(output_lines)
