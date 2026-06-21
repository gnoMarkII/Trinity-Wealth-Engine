from langsmith import traceable
import json
import os
import re
import shutil
import tempfile
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import frontmatter as fm
from filelock import FileLock
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.logger import get_logger
from schemas.pkm_models import MemoryEntry

log = get_logger(__name__)

_DATE_FRONTMATTER_RE = re.compile(r'^date:\s*["\']?(\d{4}-\d{2}-\d{2})', re.MULTILINE)

from .core import _atomic_write_text, _sanitize_filename, VAULT_PATH
from .parser import _split_bullets, _parse_h3_subsections, _parse_h2_sections, _strip_frontmatter, _extract_asset_tickers, _TICKER_FRONTMATTER_RE, _VIDEO_ID_FRONTMATTER_RE, _SOURCE_URL_FRONTMATTER_RE
from .indexer import update_master_index, _index_upsert





VAULT_PATH = Path(os.getenv("OBSIDIAN_VAULT_PATH", "./memories"))
INDEX_PATH = VAULT_PATH / ".system" / "master_index.json"
INDEX_LOCK = str(INDEX_PATH) + ".lock"


@tool
@traceable(run_type="tool")
def save_memory(
    title: str,
    content: str,
    folder_path: str,
    tags: list[str],
    entity_type: str,
    aliases: list[str] | None = None,
    linked_files: list[str] | None = None,
) -> str:
    """บันทึก MemoryEntry ลง Obsidian Vault พร้อม YAML frontmatter และ Wikilinks

    Args:
        title: ชื่อ entity หรือหัวข้อ ใช้เป็นชื่อไฟล์ .md
        content: เนื้อหาหลักในรูปแบบ Markdown เกี่ยวกับ entity นี้โดยตรง
        folder_path: โฟลเดอร์ปลายทาง เช่น '30_Knowledge_Base/Stocks'
        tags: รายการ tag เช่น ['energy', 'SET100']
        entity_type: ประเภทของ entity เช่น 'Company', 'Executive', 'Macro_Event'
        aliases: ชื่อเรียกอื่นๆ เช่น ['ปตท.', 'PTT PCL']
        linked_files: ชื่อไฟล์ที่ต้องการสร้าง Wikilinks (ไม่รวม .md)
    """
    entry = MemoryEntry(
        title=title,
        content=content,
        folder_path=folder_path,
        tags=tags,
        entity_type=entity_type,
        aliases=aliases or [],
        linked_files=linked_files or [],
    )

    target_dir = VAULT_PATH / entry.folder_path
    target_dir.mkdir(parents=True, exist_ok=True)

    safe_title = _sanitize_filename(entry.title)
    file_path = target_dir / f"{safe_title}.md"

    date_str = datetime.now().strftime("%Y-%m-%d")

    body = entry.content
    if entry.linked_files:
        wikilinks = "\n".join(f"- [[{f}]]" for f in entry.linked_files)
        body += f"\n\n## Related\n{wikilinks}\n"

    if file_path.exists():
        post = fm.load(file_path)
        meta = dict(post.metadata)

        # Merge: union tags/aliases, append linked_files unique, update last_updated
        existing_tags = list(meta.get("tags") or [])
        existing_aliases = list(meta.get("aliases") or [])
        merged_tags = list(dict.fromkeys(existing_tags + entry.tags))
        merged_aliases = list(dict.fromkeys(existing_aliases + entry.aliases))

        meta["tags"] = merged_tags
        meta["aliases"] = merged_aliases
        meta["last_updated"] = date_str
        meta.setdefault("title", entry.title)
        meta.setdefault("entity_type", entry.entity_type)
        meta.setdefault("date", date_str)

        appended_body = (
            f"{post.content.rstrip()}\n\n<!-- Update -->\n\n## Update — {date_str}\n\n{body}"
        )
        new_post = fm.Post(content=appended_body)
        new_post.metadata.update(meta)
        _atomic_write_text(file_path, fm.dumps(new_post, sort_keys=False))
        _index_upsert(file_path)
        return f"เพิ่มข้อมูลสำเร็จ (append): {file_path}"

    new_post = fm.Post(content=body)
    new_post.metadata.update({
        "title": entry.title,
        "entity_type": entry.entity_type,
        "aliases": entry.aliases,
        "tags": entry.tags,
        "date": date_str,
    })
    _atomic_write_text(file_path, fm.dumps(new_post, sort_keys=False))
    _index_upsert(file_path)
    return f"บันทึกสำเร็จ (new): {file_path}"


@tool
@traceable(run_type="tool")
def write_raw_markdown(content: str, folder_path: str, filename: str) -> str:
    """บันทึกไฟล์ Markdown ดิบลง Vault โดยตรง ไม่แปลงหรือประมวลผลเนื้อหา
    ใช้สำหรับข้อมูลที่มี YAML frontmatter พร้อมแล้ว เช่น ผลลัพธ์จาก Researcher tools

    Auto-routing (deterministic จาก YAML frontmatter):
      - ถ้า path ลงท้าย 'Daily_Snapshots' → แทรก subfolder วันที่จาก `date:` field
      - ถ้า path ลงท้าย 'Stocks'           → แทรก subfolder ชื่อหุ้นจาก `ticker:` field
    เป็น single source of truth ป้องกัน LLM พิมพ์ path ผิดหรือ date/ticker คลาดเคลื่อน

    Args:
        content: เนื้อหา Markdown พร้อม frontmatter ที่ต้องการบันทึก
        folder_path: โฟลเดอร์ปลายทาง เช่น '30_Knowledge_Base/Macroeconomics/Daily_Snapshots'
                     หรือ '30_Knowledge_Base/Stocks'
        filename: ชื่อไฟล์ไม่รวมนามสกุล เช่น 'Macro_Snapshot_2025-01-15', 'TSLA_Fundamentals_2025-01-15'
    """
    resolved_path = _maybe_inject_date_subfolder(folder_path, content)
    resolved_path = _maybe_inject_ticker_subfolder(resolved_path, content)
    target_dir = VAULT_PATH / resolved_path
    target_dir.mkdir(parents=True, exist_ok=True)
    safe_name = _sanitize_filename(filename)
    safe_name = re.sub(r'[,()\[\].—–]', '', safe_name)
    safe_name = re.sub(r'_{2,}', '_', safe_name).strip('_') or "untitled"
    file_path = target_dir / f"{safe_name}.md"
    existed = file_path.exists()
    _atomic_write_text(file_path, content)
    _index_upsert(file_path)

    # Layer-1 Entity stub — auto-create {ticker}.md hub for Stocks snapshots
    if "Stocks" in resolved_path.split("/"):
        m = _TICKER_FRONTMATTER_RE.search(content)
        if m:
            _ensure_stock_entity_stub(target_dir, m.group(1).strip())

    # Auto-create Obsidian Canvas for YouTube Insights
    if re.search(r"^entity_type:\s*youtube_insight\b", content, re.MULTILINE):
        try:
            _create_youtube_canvas(file_path, content)
        except Exception as e:
            log.warning("[CANVAS FAIL] | %s: %s", file_path.name, e)

    action = "overwritten" if existed else "new"
    return f"บันทึกสำเร็จ (raw, {action}): {file_path}"


def _create_youtube_canvas(md_path: Path, content: str) -> None:
    """สร้างไฟล์ .canvas ที่ pair กับ YouTube Insight .md — เรียกอัตโนมัติจาก write_raw_markdown"""
    m_vid = _VIDEO_ID_FRONTMATTER_RE.search(content)
    m_url = _SOURCE_URL_FRONTMATTER_RE.search(content)
    video_id = m_vid.group(1).strip() if m_vid else "unknown"
    source_url = m_url.group(1).strip() if m_url else f"https://youtu.be/{video_id}"
    md_rel = str(md_path.relative_to(VAULT_PATH)).replace("\\", "/")

    body = _strip_frontmatter(content)
    sections = _parse_h2_sections(body)

    nodes: list[dict] = []
    edges: list[dict] = []
    _c = [0]

    def nid() -> str:
        _c[0] += 1
        return f"n{_c[0]:04d}"

    def edge(from_id: str, to_id: str, f_side: str = "bottom", t_side: str = "top") -> None:
        edges.append({
            "id": f"e{len(edges):04d}",
            "fromNode": from_id, "fromSide": f_side,
            "toNode": to_id, "toSide": t_side,
        })

    def txt(node_id: str, text: str, x: int, y: int, w: int, h: int, color: str = "") -> dict:
        n: dict = {"id": node_id, "type": "text", "text": text, "x": x, "y": y, "width": w, "height": h}
        if color:
            n["color"] = color
        return n

    # ── Row 0: YouTube URL + Summary .md ─────────────────────────────
    url_id = nid()
    nodes.append({"id": url_id, "type": "link", "url": source_url,
                  "x": -700, "y": -220, "width": 560, "height": 315})
    file_id = nid()
    nodes.append({"id": file_id, "type": "file", "file": md_rel,
                  "x": 0, "y": -220, "width": 480, "height": 360})
    edge(url_id, file_id, "right", "left")

    GAP, W, H = 20, 380, 180

    # ── Row 1 (y=220): ใจความสำคัญ | แนวคิดลงทุน | ตัวเลขสำคัญ ────
    row1 = [
        ("ใจความสำคัญ",            -700, 220, "3"),   # yellow
        ("แนวคิดการลงทุน",           50, 220, "4"),   # green
        ("ตัวเลขสำคัญทางเศรษฐกิจ",  800, 220, "2"),  # orange
    ]
    for sec_name, base_x, row_y, color in row1:
        text = sections.get(sec_name, "")
        if not text:
            continue
        chunks = _split_bullets(text, max_per_node=4)
        prev_id = None
        for ci, chunk in enumerate(chunks):
            node_id = nid()
            label = f"**{sec_name}**\n\n" if ci == 0 else ""
            nodes.append(txt(node_id, label + chunk, base_x + ci * (W + GAP), row_y, W, H, color))
            if ci == 0:
                edge(file_id, node_id, "bottom", "top")
            elif prev_id:
                edge(prev_id, node_id, "right", "left")
            prev_id = node_id

    # ── Row 2 (y=470): เศรษฐกิจมหภาค แยกตามประเทศ ──────────────────
    macro_text = sections.get("เศรษฐกิจมหภาค", "")
    if macro_text:
        countries = _parse_h3_subsections(macro_text)
        MW, MH, mx = 340, 160, -700
        for country_name, country_text in countries.items():
            label_name = country_name if country_name != "ทั่วไป" else "เศรษฐกิจมหภาค"
            chunks = _split_bullets(country_text, max_per_node=3)
            for ci, chunk in enumerate(chunks):
                node_id = nid()
                label = f"**{label_name}**\n\n" if ci == 0 else ""
                nodes.append(txt(node_id, label + chunk, mx, 470, MW, MH, "5"))  # cyan
                mx += MW + GAP

    # ── Row 3 (y=700): หุ้นและสินทรัพย์ (per-ticker nodes) ─────────
    assets_text = sections.get("หุ้นและสินทรัพย์", "")
    if assets_text:
        tickers = _extract_asset_tickers(assets_text)
        TW, TH, tx, ty = 280, 120, -700, 700
        for ticker, desc in tickers:
            ticker_file = f"30_Knowledge_Base/Stocks/{ticker}/{ticker}.md"
            node_id = nid()
            if (VAULT_PATH / ticker_file).exists():
                nodes.append({"id": node_id, "type": "file", "file": ticker_file,
                               "x": tx, "y": ty, "width": TW, "height": TH})
            else:
                nodes.append(txt(node_id, f"**[[{ticker}]]**\n{desc}", tx, ty, TW, TH, "6"))  # purple
            tx += TW + GAP
            if tx > 900:
                tx, ty = -700, ty + TH + GAP

    # ── Row 4 (y=900): ความเสี่ยง ────────────────────────────────────
    risk_text = sections.get("ความเสี่ยง", "")
    if risk_text:
        chunks = _split_bullets(risk_text, max_per_node=3)
        RW, RH = 380, 160
        for ci, chunk in enumerate(chunks):
            node_id = nid()
            label = "**⚠️ ความเสี่ยง**\n\n" if ci == 0 else ""
            nodes.append(txt(node_id, label + chunk, -700 + ci * (RW + GAP), 900, RW, RH, "1"))  # red

    canvas_path = md_path.with_suffix(".canvas")
    _atomic_write_text(canvas_path, json.dumps({"nodes": nodes, "edges": edges}, ensure_ascii=False, indent=2))
    log.info("[CANVAS OK] | file: %s | nodes: %d", canvas_path.name, len(nodes))


def _ensure_stock_entity_stub(target_dir: Path, ticker: str) -> None:
    """สร้าง Layer-1 Entity hub file {target_dir}/{ticker}.md ถ้ายังไม่มี
    Hub นี้รับ backlinks อัตโนมัติจากทุก snapshot/news/trade ที่ wikilink มาที่ [[ticker]]
    """
    safe_ticker = _sanitize_filename(ticker.strip().upper())
    if not safe_ticker:
        return
    stub_path = target_dir / f"{safe_ticker}.md"
    if stub_path.exists():
        return
    today = datetime.now().strftime("%Y-%m-%d")
    content = (
        "---\n"
        f"title: {safe_ticker}\n"
        "entity_type: stock_entity\n"
        f"ticker: {safe_ticker}\n"
        f"date: {today}\n"
        f"tags: [entity, stock_hub, {safe_ticker.lower()}]\n"
        "---\n\n"
        f"# {safe_ticker}\n\n"
        f"> **Entity hub** สำหรับ `{safe_ticker}` — Layer 1 ใน Graph View\n"
        f"> ไฟล์นี้รวบรวม backlinks จาก snapshots, news, trades ที่เกี่ยวข้อง\n\n"
        "## Notes\n\n"
        "*(เพิ่มบันทึกส่วนตัวที่นี่ — Obsidian จะแสดง backlinks ด้านล่างอัตโนมัติ)*\n"
    )
    _atomic_write_text(stub_path, content)
    _index_upsert(stub_path)


def _maybe_inject_date_subfolder(folder_path: str, content: str) -> str:
    """ถ้า folder_path ลงท้ายด้วย 'Daily_Snapshots' → แทรกวันที่จาก YAML frontmatter เป็น subfolder

    ตัวอย่าง:
        '30_Knowledge_Base/Macroeconomics/Daily_Snapshots'
        → '30_Knowledge_Base/Macroeconomics/Daily_Snapshots/2026-05-21'
    ใช้ค่า date จาก frontmatter เป็น single source of truth (ป้องกัน LLM พิมพ์วันที่เพี้ยน)
    fallback เป็นวันนี้ถ้าไม่มี date field
    """
    normalized = folder_path.rstrip("/")
    if not normalized.endswith("Daily_Snapshots"):
        return folder_path
    m = _DATE_FRONTMATTER_RE.search(content)
    date_str = m.group(1) if m else datetime.now().strftime("%Y-%m-%d")
    return f"{normalized}/{date_str}"


def _maybe_inject_ticker_subfolder(folder_path: str, content: str) -> str:
    """ถ้า folder_path ลงท้ายด้วย 'Stocks' → แทรกชื่อหุ้นจาก YAML `ticker:` เป็น subfolder

    ตัวอย่าง:
        '30_Knowledge_Base/Stocks' + ticker=TSLA
        → '30_Knowledge_Base/Stocks/TSLA'
    ถ้าไม่มี ticker field ใน frontmatter → ไม่ inject (เขียนลง root Stocks/)
    ticker ถูก sanitize เผื่ออักขระต้องห้าม (สำหรับหุ้นพิเศษ เช่น BRK.B, PTT.BK)
    """
    normalized = folder_path.rstrip("/")
    if not normalized.endswith("Stocks"):
        return folder_path
    m = _TICKER_FRONTMATTER_RE.search(content)
    if not m:
        return folder_path
    ticker = _sanitize_filename(m.group(1).strip().upper())
    return f"{normalized}/{ticker}" if ticker else folder_path


