"""ส่งข่าวจาก News Funnel ไปยัง Discord ผ่าน Incoming Webhook

Failure-tolerant โดยตั้งใจ — ความล้มเหลวใด ๆ (ไม่ตั้ง webhook, network error, 4xx/5xx)
แค่ log แล้วคืน False ไม่ทำให้ pipeline ที่เรียกใช้พัง
"""
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List

import requests

from core.logger import get_logger

logger = get_logger(__name__)

_MAX_EMBEDS_PER_MESSAGE = 10
_REQUEST_TIMEOUT = 10
_RETRY_HTTP_CODES = {429, 500, 502, 503, 504}
_DEFAULT_RETRY_BACKOFF_SECONDS = 2.0

# Discord จำกัด description ต่อ embed ที่ 4,096 ตัวอักษร และผลรวมทุก field ในทุก embed
# ของ 1 ข้อความที่ 6,000 ตัวอักษร — message ของเรามีแค่ 1 embed เสมอ จึงเผื่อ margin ~300
# ตัวอักษรให้ title/fields/footer แล้วยังเหลือห่างจาก 6,000 มาก ไม่มีทางชน limit รวม
_MAX_INLINE_CONTENT_CHARS = 3800

# Discord จำกัดชื่อกระทู้ (thread_name ตอนสร้าง Forum post) ที่ 100 ตัวอักษร
_MAX_THREAD_NAME_CHARS = 100

_COLOR_RED = 0xFF0000
_COLOR_ORANGE = 0xFF8C00
_COLOR_YELLOW = 0xFFD700
# สีกลาง ใช้กับข่าวที่ ingest ผ่านช่องทาง manual (ไม่ผ่าน News Funnel triage จึงไม่มีคะแนน
# ให้เลือกสีตาม impact เหมือน _embed_color) — Discord blurple, โทนเป็นกลาง ไม่ implied ความสำคัญ
_COLOR_NEUTRAL = 0x5865F2

_PERIOD_LABEL = {"morning": "MORNING", "evening": "EVENING"}

# ช่องปลายทางเป็น Forum channel ที่ตั้ง required tags ไว้ — ทุกกระทู้ใหม่ต้องมี tag อย่างน้อย
# 1 อัน ไม่งั้น Discord ตอบ 400 ระดับ tag อิง threshold เดียวกับสี embed (_embed_color)
_TAG_ENV_VAR_BY_TIER = {
    "ultra": "DISCORD_TAG_ID_ULTRA",     # score >= 9
    "high": "DISCORD_TAG_ID_HIGH",       # score >= 8
    "warning": "DISCORD_TAG_ID_WARNING",  # score < 8 (ทุก event ที่มาถึงจุดนี้ผ่าน high-impact filter มาแล้ว คือ >= 7)
}

_IMPACT_BANNER_PATTERN = re.compile(r"^> \*\*Macro Impact:\*\*.*?\n\n", re.DOTALL)
_KEY_POINTS_HEADER = "ใจความสำคัญ"


def is_discord_enabled() -> bool:
    """ตรวจว่ามี DISCORD_WEBHOOK_URL ตั้งอยู่หรือไม่"""
    return bool(os.getenv("DISCORD_WEBHOOK_URL", "").strip())


def _embed_color(score: int) -> int:
    if score >= 9:
        return _COLOR_RED
    if score >= 8:
        return _COLOR_ORANGE
    return _COLOR_YELLOW


def _select_tag_id(score: int) -> str | None:
    """เลือก Forum tag ID ตามระดับ impact — คืน None ถ้าไม่ได้ตั้ง env var ของระดับนั้นไว้
    (เช่นช่องปลายทางไม่ใช่ Forum channel หรือยังไม่ได้ config tag)
    """
    if score >= 9:
        tier = "ultra"
    elif score >= 8:
        tier = "high"
    else:
        tier = "warning"
    tag_id = os.getenv(_TAG_ENV_VAR_BY_TIER[tier], "").strip()
    return tag_id or None


def _build_embed(ev: Dict[str, Any], description: str, period: str = "") -> Dict[str, Any]:
    macro_score = int(ev.get("macro_impact_score") or 0)
    asset_score = int(ev.get("asset_impact_score") or 0)
    title = ev.get("canonical_title") or "ข่าวไม่มีหัวข้อ"
    url = ev.get("canonical_url") or (ev.get("links") or [None])[0]
    period_label = _PERIOD_LABEL.get(period, period.upper() if period else "")

    fields = [
        {"name": "Macro Impact", "value": f"{macro_score}/10", "inline": True},
        {"name": "Asset Impact", "value": f"{asset_score}/10", "inline": True},
    ]
    embed: Dict[str, Any] = {
        "title": f"📰 {title}"[:256],
        "description": description,
        "color": _embed_color(max(macro_score, asset_score)),
        "fields": fields,
    }
    if url:
        embed["url"] = url
    if period_label:
        embed["footer"] = {"text": f"⏰ รอบ {period_label}"}
    return embed


def _strip_impact_banner(content: str) -> str:
    return _IMPACT_BANNER_PATTERN.sub("", content, count=1)


def _link_content(url: str | None) -> str | None:
    """ข้อความ plain text ที่โชว์บนหน้าโพสต์ เหนือ embed (Discord "content" field) — ครอบด้วย
    <> กัน Discord auto-unfurl ลิงก์เป็น preview การ์ดของตัวเอง ซึ่งจะซ้ำกับ embed ที่เราจัดรูปแบบเองแล้ว
    """
    return f"🔗 <{url}>" if url else None


def _build_simple_embed(title: str, description: str, url: str | None) -> Dict[str, Any]:
    """Embed แบบไม่มีคะแนน impact — ใช้กับข่าวที่ ingest ผ่านช่องทาง manual (ไม่ผ่าน News Funnel
    triage) จึงไม่มี Macro/Asset Impact fields และใช้สีกลางแทนสีตามคะแนน
    """
    embed: Dict[str, Any] = {
        "title": f"📄 {title}"[:256],
        "description": description,
        "color": _COLOR_NEUTRAL,
    }
    if url:
        embed["url"] = url
    return embed


def _frontmatter_value(content: str, key: str) -> str | None:
    """สกัดค่า YAML frontmatter key เดียว — เขียนแยกจาก tools.archivist.parser.extract_yaml_frontmatter_value
    (ที่ทำแบบเดียวกันเป๊ะ) โดยตั้งใจ เพื่อไม่ให้ core/discord_notifier.py ต้อง import โมดูลที่
    ลาก langchain_chroma/ML deps หนัก ๆ เข้ามาทั้งที่ต้องการแค่ regex เล็ก ๆ อันเดียว
    """
    if not content.strip().startswith("---"):
        return None
    parts = content.split("---", 2)
    if len(parts) < 3:
        return None
    pattern = rf"^\s*{re.escape(key)}\s*:\s*(?:[\"'](?P<qval>[^\"']+)[\"']|(?P<uval>[^\r\n#]+))"
    match = re.search(pattern, parts[1], re.MULTILINE)
    if not match:
        return None
    value = match.group("qval") or match.group("uval")
    return value.strip() if value else None


def _extract_article_body(content: str) -> str:
    """ตัด YAML frontmatter, บรรทัดหัวข้อ/แหล่งที่มา, และท้าย '## หมายเหตุ' ออกจาก
    _build_article_md() output (tools/knowledge/core.py) ให้เหลือแค่เนื้อหาบทความสำหรับ embed
    """
    if content.startswith("---"):
        parts = content.split("---", 2)
        body = parts[2].strip() if len(parts) >= 3 else content.strip()
    else:
        body = content.strip()
    body = re.sub(r"^#\s+.*\n", "", body, count=1)
    body = re.sub(r"^>\s*แหล่งที่มา:.*\n", "", body, count=1)
    body = re.sub(r"\n+## หมายเหตุ\n.*$", "", body, count=1, flags=re.DOTALL)
    return body.strip()


def _extract_section(markdown_content: str, header_text: str) -> str:
    """ดึงเนื้อหาใต้หัวข้อ '## {header_text}' จนถึงหัวข้อ ## ถัดไป (หรือจบข้อความ) — คืนค่าว่าง
    ถ้าไม่พบหัวข้อนั้นในเนื้อหา (เช่น LLM ข้ามส่วนนี้ไปเพราะไม่มีข้อมูล)
    """
    marker = f"## {header_text}"
    start = markdown_content.find(marker)
    if start == -1:
        return ""
    start += len(marker)
    next_header = re.search(r"\n##\s+", markdown_content[start:])
    end = start + next_header.start() if next_header else len(markdown_content)
    return markdown_content[start:end].strip()


def _post_with_retry(webhook_url: str, **kwargs: Any) -> bool:
    """ยิง requests.post ไป webhook_url — retry 1 ครั้งเมื่อเจอ 429/5xx หรือ network error
    ใช้ร่วมกันทั้งการส่งแบบ JSON body (embeds ล้วน) และ multipart/form-data (แนบไฟล์)
    คืน True เมื่อสำเร็จ (HTTP 2xx), False เมื่อล้มเหลว
    """
    kwargs.setdefault("timeout", _REQUEST_TIMEOUT)
    last_error: str | None = None
    for attempt in range(2):
        try:
            resp = requests.post(webhook_url, **kwargs)
            if resp.status_code < 300:
                return True
            if resp.status_code in _RETRY_HTTP_CODES and attempt == 0:
                retry_after = _DEFAULT_RETRY_BACKOFF_SECONDS
                try:
                    retry_after = max(retry_after, float(resp.json().get("retry_after", 0)))
                except Exception:
                    pass
                time.sleep(retry_after)
                continue
            last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
            break
        except requests.exceptions.RequestException as e:
            last_error = str(e)
            if attempt == 0:
                time.sleep(_DEFAULT_RETRY_BACKOFF_SECONDS)
                continue
            break

    logger.warning("Discord webhook ส่งไม่สำเร็จ: %s", last_error)
    return False


def format_news_funnel_embeds(events: List[Dict[str, Any]], period: str = "") -> List[Dict[str, Any]]:
    """สร้าง Discord Embed ภาษาไทยจากข่าว High-Impact — สูงสุด 10 embeds (ตัดส่วนเกินทิ้ง
    ให้ caller เป็นคนแบ่ง chunk เอง ฟังก์ชันนี้ไม่ chunk ให้) description ใช้ comprehensive_summary
    """
    return [_build_embed(ev, ev.get("comprehensive_summary") or "", period) for ev in events[:_MAX_EMBEDS_PER_MESSAGE]]


def send_discord_notification(embeds: List[Dict[str, Any]], content: str = "") -> bool:
    """ส่ง HTTP POST ไป Discord Webhook ด้วย JSON body ล้วน (ไม่มีไฟล์แนบ)
    คืน True เมื่อสำเร็จ (HTTP 2xx), False เมื่อล้มเหลวหรือไม่ได้ตั้ง webhook
    """
    webhook_url = os.getenv("DISCORD_WEBHOOK_URL", "").strip()
    if not webhook_url:
        return False
    if not embeds:
        return True

    payload = {"content": content, "embeds": embeds}
    return _post_with_retry(webhook_url, json=payload)


def send_synthesized_news_discord(events: List[Dict[str, Any]], period: str = "") -> None:
    """ส่งข่าวที่สังเคราะห์เสร็จแล้วไป Discord — 1 ข้อความต่อ 1 ข่าว

    ถ้าเนื้อหาเต็ม (ev["synthesized_content"]) สั้นพอที่จะยัดใน embed เดียวได้ (≤ 3,800
    ตัวอักษร ตาม limit ของ Discord) จะส่ง embed ที่มีเนื้อหาเต็มไปเลย ไม่ต้องแนบไฟล์
    ถ้ายาวเกิน จะตัดมาเฉพาะหัวข้อ "ใจความสำคัญ" ใส่ embed แล้วแนบไฟล์ .md ฉบับเต็มที่อ่านมาจาก
    Vault จริง (ev["synthesized_note_path"]) แทน — รับประกันว่าไฟล์ใน Discord ตรงกับใน Obsidian เป๊ะ

    Failure-tolerant ต่อข่าวแต่ละชิ้น — ข่าวหนึ่งส่งไม่สำเร็จไม่กระทบข่าวอื่นในชุดเดียวกัน
    """
    if not is_discord_enabled():
        return
    webhook_url = os.getenv("DISCORD_WEBHOOK_URL", "").strip()

    for ev in events:
        macro_score = int(ev.get("macro_impact_score") or 0)
        asset_score = int(ev.get("asset_impact_score") or 0)
        title = ev.get("canonical_title") or "ข่าวไม่มีหัวข้อ"
        url = ev.get("canonical_url") or (ev.get("links") or [None])[0]

        base_payload: Dict[str, Any] = {
            "thread_name": title[:_MAX_THREAD_NAME_CHARS],
        }
        link_content = _link_content(url)
        if link_content:
            base_payload["content"] = link_content
        tag_id = _select_tag_id(max(macro_score, asset_score))
        if tag_id:
            base_payload["applied_tags"] = [tag_id]

        full_content = _strip_impact_banner(ev.get("synthesized_content") or "")

        if full_content and len(full_content) <= _MAX_INLINE_CONTENT_CHARS:
            embed = _build_embed(ev, full_content, period)
            ok = _post_with_retry(webhook_url, json={**base_payload, "embeds": [embed]})
        else:
            note_path = ev.get("synthesized_note_path")
            if not note_path:
                continue
            try:
                file_content = Path(note_path).read_bytes()
            except OSError as e:
                logger.warning("อ่านไฟล์ synthesized note ไม่สำเร็จ %s: %s", note_path, e)
                continue

            key_points = _extract_section(full_content, _KEY_POINTS_HEADER) or ev.get("comprehensive_summary") or ""
            embed = _build_embed(ev, key_points, period)
            filename = Path(note_path).name
            ok = _post_with_retry(
                webhook_url,
                data={"payload_json": json.dumps({**base_payload, "embeds": [embed]})},
                files={"file": (filename, file_content, "text/markdown")},
            )

        if not ok:
            logger.warning("ส่ง Discord สำหรับข่าว %s ไม่สำเร็จ", ev.get("event_id"))


def send_ingested_article_discord(md_content: str, note_path: str | None = None) -> None:
    """ส่งบทความที่ ingest ผ่านช่องทาง manual (วางลิงก์ในแชท หรืออนุมัติผ่าน gate ของ flow
    news_youtube) ไป Discord — ต่างจาก send_synthesized_news_discord ตรงที่ path นี้ไม่ผ่าน
    LLM Triage ของ News Funnel จึงไม่มีคะแนน macro/asset impact ให้ใช้: ใช้สีกลาง ไม่มี
    Macro/Asset Impact fields บน embed

    md_content คือ output ดิบจาก _build_article_md() (มี YAML frontmatter ครบ)
    note_path คือ path ไฟล์จริงที่ถูกบันทึกแล้ว (สำหรับแนบไฟล์กรณีเนื้อหายาวเกิน embed)

    Failure-tolerant เหมือนฟังก์ชันอื่นในโมดูลนี้ — ไม่ raise ไม่ว่าจะล้มเหลวจุดไหน
    """
    if not is_discord_enabled():
        return
    webhook_url = os.getenv("DISCORD_WEBHOOK_URL", "").strip()

    title = _frontmatter_value(md_content, "title") or "บทความไม่มีหัวข้อ"
    source_url = _frontmatter_value(md_content, "source_url")
    body = _extract_article_body(md_content)

    base_payload: Dict[str, Any] = {"thread_name": title[:_MAX_THREAD_NAME_CHARS]}
    link_content = _link_content(source_url)
    if link_content:
        base_payload["content"] = link_content

    if body and len(body) <= _MAX_INLINE_CONTENT_CHARS:
        embed = _build_simple_embed(title, body, source_url)
        ok = _post_with_retry(webhook_url, json={**base_payload, "embeds": [embed]})
    else:
        if not note_path:
            logger.warning("บทความ '%s' เนื้อหายาวเกิน embed แต่ไม่มี note_path ให้แนบไฟล์ — ข้าม", title)
            return
        try:
            file_content = Path(note_path).read_bytes()
        except OSError as e:
            logger.warning("อ่านไฟล์บทความไม่สำเร็จ %s: %s", note_path, e)
            return

        key_points = _extract_section(body, _KEY_POINTS_HEADER) or body[:500]
        embed = _build_simple_embed(title, key_points, source_url)
        filename = Path(note_path).name
        ok = _post_with_retry(
            webhook_url,
            data={"payload_json": json.dumps({**base_payload, "embeds": [embed]})},
            files={"file": (filename, file_content, "text/markdown")},
        )

    if not ok:
        logger.warning("ส่ง Discord สำหรับบทความ '%s' ไม่สำเร็จ", title)


DEFAULT_MAX_AUDIO_BYTES_LIMIT = 8 * 1024 * 1024  # 8.0 MiB
DEFAULT_SAFE_AUDIO_BYTES = int(7.5 * 1024 * 1024)  # 7.5 MiB Safe Margin target


def get_max_discord_audio_bytes() -> int:
    """อ่านขนาดไฟล์เสียงสูงสุดที่อนุญาตให้อัปโหลดเข้า Discord พร้อม Safe Margin (Default 7.5 MiB)
    Validate ค่า ณ runtime ทุกครั้งเพื่อรองรับ env dynamic และ monkeypatch ในเทสต์
    """
    raw = os.getenv("DISCORD_MAX_AUDIO_BYTES", "").strip()
    if not raw:
        return DEFAULT_SAFE_AUDIO_BYTES
    try:
        val = int(raw)
        return val if val > 0 else DEFAULT_SAFE_AUDIO_BYTES
    except (ValueError, TypeError):
        return DEFAULT_SAFE_AUDIO_BYTES


def get_notebooklm_webhook_url() -> str:
    """คืนค่า Webhook URL สำหรับห้อง NotebookLM โดยเฉพาะ หรือ fallback ไปยัง Webhook กลาง"""
    return os.getenv("DISCORD_NOTEBOOKLM_WEBHOOK_URL", "").strip() or os.getenv("DISCORD_WEBHOOK_URL", "").strip()


def is_notebooklm_discord_enabled() -> bool:
    """ตรวจว่ามีการตั้งค่า Discord Webhook สำหรับ NotebookLM หรือไม่"""
    return bool(get_notebooklm_webhook_url())


from dataclasses import dataclass

@dataclass
class DiscordAudioDeliveryResult:
    status: str  # "sent" | "skipped_disabled" | "skipped_oversize" | "failed"
    message: str = ""


def send_notebooklm_audio_discord(
    audio_path: str | Path,
    title: str,
    summary: str = "",
    source_ref: str | None = None,
) -> DiscordAudioDeliveryResult:
    """ส่งไฟล์เสียง NotebookLM Podcast ไปยัง Discord Webhook ในรูปแบบ Forum Post / Thread

    - ใช้ DISCORD_NOTEBOOKLM_WEBHOOK_URL ก่อน fallback DISCORD_WEBHOOK_URL
    - บีบอัดไฟล์อัตโนมัติหากเกิน 7.5 MiB (Safe Margin)
    - ตรวจสอบขนาดไฟล์ก่อน POST เสมอเพื่อป้องกัน HTTP 413
    - Failure-tolerant — คืน structured result (sent | skipped_disabled | skipped_oversize | failed)
    """
    webhook_url = get_notebooklm_webhook_url()
    if not webhook_url:
        logger.info("Discord Webhook URL for NotebookLM not configured, skipping notification.")
        return DiscordAudioDeliveryResult(status="skipped_disabled", message="Discord webhook not configured")

    p = Path(audio_path).resolve()
    if not p.exists() or not p.is_file():
        logger.warning("Audio file not found for Discord upload: %s", p)
        return DiscordAudioDeliveryResult(status="failed", message=f"Audio file not found: {p.name}")

    max_bytes = get_max_discord_audio_bytes()

    # Ensure compressed if larger than target max bytes
    try:
        from tools.content.notebooklm.audio_utils import compress_audio_for_discord
        p = compress_audio_for_discord(p, max_size_bytes=max_bytes)
    except Exception as e:
        logger.warning("Auto-compress before Discord upload skipped: %s", e)

    try:
        file_bytes = p.read_bytes()
    except OSError as e:
        logger.warning("Failed reading audio file for Discord: %s", e)
        return DiscordAudioDeliveryResult(status="failed", message=f"Cannot read audio file: {e}")

    # Guard: Final file size check
    if len(file_bytes) > max_bytes:
        size_mb = len(file_bytes) / (1024 * 1024)
        max_mb = max_bytes / (1024 * 1024)
        logger.warning("Audio file %s (%.2f MB) exceeds Discord limit (%.2f MB), skipping upload.", p.name, size_mb, max_mb)
        return DiscordAudioDeliveryResult(status="skipped_oversize", message=f"File size {size_mb:.2f} MB exceeds limit {max_mb:.2f} MB")

    clean_title = (title or p.stem).strip()
    thread_name = f"🎙️ {clean_title}"
    if len(thread_name) > _MAX_THREAD_NAME_CHARS:
        thread_name = thread_name[:_MAX_THREAD_NAME_CHARS - 3] + "..."

    base_payload: Dict[str, Any] = {
        "thread_name": thread_name,
    }

    tag_id = os.getenv("DISCORD_NOTEBOOKLM_TAG_ID", "").strip()
    if tag_id:
        base_payload["applied_tags"] = [tag_id]

    fields = [
        {"name": "ประเภท", "value": "Audio Overview (Podcast)", "inline": True},
        {"name": "ขนาดไฟล์", "value": f"{len(file_bytes) / (1024 * 1024):.2f} MB", "inline": True},
    ]
    if source_ref:
        # แสดงเฉพาะชื่อไฟล์หรือ relative path (ไม่ใส่ absolute path)
        fields.append({"name": "แหล่งข้อมูล", "value": str(source_ref)[:256], "inline": False})

    embed = {
        "title": f"🎙️ {clean_title}"[:256],
        "description": summary[:_MAX_INLINE_CONTENT_CHARS] if summary else "🎧 ฟังเสียงสรุปพอดแคสต์เชิงลึกจาก NotebookLM",
        "color": _COLOR_NEUTRAL,
        "fields": fields,
        "footer": {"text": "NotebookLM Podcast • Trinity Wealth Engine"},
    }

    ok = _post_with_retry(
        webhook_url,
        data={"payload_json": json.dumps({**base_payload, "embeds": [embed]})},
        files={"file": (p.name, file_bytes, "audio/mp4")},
    )
    if ok:
        logger.info("ส่ง NotebookLM Audio ไป Discord สำเร็จ: %s", clean_title)
        return DiscordAudioDeliveryResult(status="sent", message="Successfully posted to Discord")
    else:
        logger.warning("ส่ง NotebookLM Audio ไป Discord ไม่สำเร็จ: %s", clean_title)
        return DiscordAudioDeliveryResult(status="failed", message="HTTP request to Discord webhook failed")


