from langsmith import traceable
import json
import os
import re
import shutil
import tempfile
from datetime import datetime, timezone
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

from .core import _atomic_write_text, _sanitize_filename, VAULT_PATH, get_note_lock
from .maintenance_guard import assert_write_allowed
from .composition import build_knowledge_note_writer
from .metadata import (
    _LEGACY_TYPE_MAP,
    dump_note,
    normalize_legacy_metadata,
    parse_note,
    validate_capture_note,
)
from tools.tool_errors import GOAL_VIA_BOOKKEEPER
from .parser import _split_bullets, _parse_h3_subsections, _parse_h2_sections, _strip_frontmatter, _extract_asset_tickers, _TICKER_FRONTMATTER_RE, _VIDEO_ID_FRONTMATTER_RE, _SOURCE_URL_FRONTMATTER_RE, extract_yaml_frontmatter_value, parse_company_news_items
from .indexer import update_master_index, _index_upsert
from .portable_links import render_resolved_markdown_link
from .vault_paths import VaultPaths

_CHANNEL_FRONTMATTER_RE = re.compile(r'^channel:\s*(.+)$', re.MULTILINE)





VAULT_PATH = Path(os.getenv("OBSIDIAN_VAULT_PATH", "./memories"))
INDEX_PATH = VAULT_PATH / ".system" / "master_index.json"
INDEX_LOCK = str(INDEX_PATH) + ".lock"

_PUBLISHED_ENTITY_TYPES = {
    "stock_hub",
    "equity_analysis",
    "quant_snapshot",
    "earnings_call",
    "company_news",
    "youtube_summary",
    "macro_strategy",
    "macro_snapshot",
    "briefing_book",
    "book_note",
    "concept",
}
_TICKER_ENTITY_TYPES = {"stock_hub", "equity_analysis", "quant_snapshot", "earnings_call"}
_WIKILINK_RE = re.compile(r"\[\[([^\]|#]+)(?:#[^\]|]*)?(?:\|([^\]]+))?\]\]")


def _canonical_entity_type(value: object) -> str:
    raw = str(value or "").strip().lower()
    return _LEGACY_TYPE_MAP.get(raw, raw)


def _infer_ticker(meta: dict, folder_path: str = "") -> str | None:
    explicit = meta.get("ticker")
    if explicit:
        return str(explicit).strip().upper()
    folder_parts = [part for part in str(folder_path).replace("\\", "/").split("/") if part]
    try:
        stock_index = [part.lower() for part in folder_parts].index("stocks")
        if stock_index + 1 < len(folder_parts) and folder_parts[stock_index + 1].lower() not in {"analysis", "quant", "earnings"}:
            candidate = folder_parts[stock_index + 1].strip()
            if candidate:
                return candidate.upper()
    except ValueError:
        pass
    candidates = [meta.get("title"), *(meta.get("aliases") or [])]
    for candidate in candidates:
        match = re.search(r"(?<![A-Za-z0-9])[A-Z]{1,6}(?:\.[A-Z]{1,3})?(?![A-Za-z0-9])", str(candidate or ""))
        if match:
            return match.group(0).upper()
    return None


def _portableize_wikilinks(body: str, vault_root: Path, source_path: Path) -> str:
    """Convert legacy Obsidian wikilinks to resolvable Markdown or plain text."""
    def replace(match: re.Match[str]) -> str:
        target = match.group(1).strip()
        label = (match.group(2) or target).strip()
        return render_resolved_markdown_link(vault_root, source_path, target, label=label)

    return _WIKILINK_RE.sub(replace, body or "")


def _safe_writer_filename(filename: str, *, date_value: object = None, entity_type: str = "") -> str:
    safe = _sanitize_filename(str(filename or "untitled"))
    safe = re.sub(r"[,()\[\].—–]", "", safe)
    safe = re.sub(r"_{2,}", "_", safe).strip("_") or "untitled"
    date_prefix = str(date_value or "")[:10]
    if entity_type == "company_news" and re.match(r"^\d{4}-\d{2}-\d{2}$", date_prefix) and not re.match(r"^\d{4}-\d{2}-\d{2}", safe):
        safe = f"{date_prefix} {safe}"
    return safe


def _prepare_raw_metadata(
    raw_meta: dict,
    *,
    filename: str,
    folder_path: str,
    vault_root: Path,
) -> tuple[dict, str, bool]:
    """Return normalized metadata, safe filename, and whether it is capture-only."""
    meta, _ = normalize_legacy_metadata(dict(raw_meta), producer="write_raw_markdown")
    raw_entity = str(meta.get("entity_type") or "").strip()
    entity_type = _canonical_entity_type(raw_entity)
    folder_lower = str(folder_path).replace("\\", "/").lower()

    if not raw_entity:
        if "youtube" in folder_lower and meta.get("video_id"):
            entity_type = "youtube_summary"
        elif "news" in folder_lower and (meta.get("source_url") or meta.get("publisher")):
            entity_type = "company_news"
        elif "macro" in folder_lower or "daily_snapshot" in folder_lower:
            entity_type = "macro_snapshot"
        else:
            entity_type = "capture"

    if entity_type not in _PUBLISHED_ENTITY_TYPES:
        meta.setdefault("legacy_entity_type", raw_entity)
        entity_type = "capture"

    if entity_type in _TICKER_ENTITY_TYPES:
        ticker = _infer_ticker(meta, folder_path)
        if ticker:
            meta["ticker"] = ticker
        if not ticker or (entity_type == "earnings_call" and not meta.get("period")):
            meta.setdefault("legacy_entity_type", entity_type)
            entity_type = "capture"
    if entity_type == "youtube_summary" and not meta.get("video_id"):
        meta.setdefault("legacy_entity_type", entity_type)
        entity_type = "capture"

    meta["schema_version"] = 2
    meta["entity_type"] = entity_type
    meta.setdefault("title", filename or "Untitled")
    meta.setdefault("tags", [])
    safe_name = _safe_writer_filename(
        filename or str(meta.get("title") or "untitled"),
        date_value=meta.get("date") or meta.get("published_date"),
        entity_type=entity_type,
    )

    if entity_type == "capture":
        meta.setdefault("capture_status", "pending_normalization")
        meta.setdefault("search_scope", "excluded")
        meta.setdefault("captured_at", datetime.now(timezone.utc).isoformat())
        meta.setdefault("capture_source", "write_raw_markdown")
        # Incomplete captures intentionally have no note_id/document_key. They
        # are staging inputs, not published knowledge objects.
        meta.pop("note_id", None)
        meta.pop("document_key", None)
        safe_name = _safe_writer_filename(filename or str(meta["title"]))
        return meta, safe_name, True
    return meta, safe_name, False


def _save_memory_v2(
    *,
    title: str,
    content: str,
    folder_path: str,
    tags: list[str],
    entity_type: str,
    aliases: list[str] | None,
    linked_files: list[str] | None,
    vault_root: Path,
) -> str:
    meta, _ = normalize_legacy_metadata({
        "title": title,
        "entity_type": entity_type,
        "tags": tags or [],
        "aliases": aliases or [],
        "date": datetime.now().strftime("%Y-%m-%d"),
    }, producer="save_memory")
    canonical = _canonical_entity_type(meta.get("entity_type"))
    meta["entity_type"] = canonical if canonical in _PUBLISHED_ENTITY_TYPES else "concept"
    if canonical not in _PUBLISHED_ENTITY_TYPES:
        meta["legacy_entity_type"] = entity_type
    if canonical in _TICKER_ENTITY_TYPES:
        ticker = _infer_ticker(meta, folder_path)
        if ticker:
            meta["ticker"] = ticker
        else:
            meta["legacy_entity_type"] = canonical
            meta["entity_type"] = "concept"
    meta["schema_version"] = 2
    safe_name = _safe_writer_filename(title)
    vp = VaultPaths(vault_root)
    # Structured memory entries that are not domain-routed retain their caller
    # folder; domain records are routed by the canonical VaultPaths policy.
    if meta["entity_type"] == "concept":
        target_path = vp.safe_resolve(Path(folder_path) / f"{safe_name}.md")
    else:
        target_path = vp.note_path(meta, filename=safe_name)

    existed = target_path.exists()
    existing_meta: dict = {}
    existing_body = ""
    if existed:
        existing_meta, existing_body, parse_issues = parse_note(target_path.read_text(encoding="utf-8"))
        if parse_issues:
            raise ValueError(f"Cannot append to malformed note {target_path}: {parse_issues}")
        for field in ("note_id", "document_key"):
            if existing_meta.get(field):
                meta[field] = existing_meta[field]
        meta["tags"] = list(dict.fromkeys(list(existing_meta.get("tags") or []) + list(meta.get("tags") or [])))
        meta["aliases"] = list(dict.fromkeys(list(existing_meta.get("aliases") or []) + list(meta.get("aliases") or [])))
        meta["last_updated"] = datetime.now().strftime("%Y-%m-%d")
        body = f"{existing_body.rstrip()}\n\n<!-- Update -->\n\n## Update — {meta['last_updated']}\n\n{content}"
    else:
        body = content

    body = _portableize_wikilinks(body, vault_root, target_path)
    if linked_files:
        related = "\n".join(
            "- " + render_resolved_markdown_link(
                vault_root,
                target_path,
                item,
                label=Path(str(item).replace("\\", "/")).stem,
            )
            for item in linked_files
        )
        body = f"{body.rstrip()}\n\n## Related\n{related}\n"

    committed = build_knowledge_note_writer(vault_paths=vp).write_note(
        metadata=meta,
        body=body,
        filename=safe_name,
        target_path=target_path,
    )
    _index_upsert(committed.primary_file)
    _sync_to_catalog(committed.primary_file)
    action = "append" if existed else "new"
    return f"บันทึกสำเร็จ ({action}): {committed.primary_file}"


def _write_raw_markdown_v2(
    *,
    content: str,
    folder_path: str,
    filename: str,
    vault_root: Path,
) -> str:
    meta, body, issues = parse_note(content)
    if issues:
        raise ValueError(f"Raw Markdown has invalid frontmatter: {issues}")
    if not meta:
        meta = {"title": filename, "entity_type": "capture"}
        body = content
    if str(meta.get("entity_type") or "").strip().lower() in {"goal", "financial_goal"}:
        return GOAL_VIA_BOOKKEEPER
    meta, safe_name, capture_only = _prepare_raw_metadata(
        meta,
        filename=filename,
        folder_path=folder_path,
        vault_root=vault_root,
    )
    vp = VaultPaths(vault_root)
    if capture_only:
        valid_capture, capture_issues = validate_capture_note(meta)
        if not valid_capture:
            raise ValueError(f"Invalid capture metadata: {capture_issues}")
        capture_path = build_knowledge_note_writer(vault_paths=vp).write_capture(
            meta,
            _portableize_wikilinks(body, vault_root, vp.safe_resolve(Path("00_Inbox") / f"{safe_name}.md")),
            filename=safe_name,
        )
        return f"บันทึก capture สำเร็จ: {capture_path}"

    target_path = vp.note_path(meta, filename=safe_name)
    body = _portableize_wikilinks(body, vault_root, target_path)
    committed = build_knowledge_note_writer(vault_paths=vp).write_note(
        metadata=meta,
        body=body,
        filename=safe_name,
    )
    _index_upsert(committed.primary_file)
    _sync_to_catalog(committed.primary_file)

    original_entity = _canonical_entity_type(meta.get("entity_type"))
    if original_entity == "company_news":
        try:
            parsed_data = parse_company_news_items(committed.primary_file.read_text(encoding="utf-8"))
            _atomic_write_text(committed.primary_file.with_suffix(".json"), json.dumps(parsed_data, ensure_ascii=False, indent=2))
        except Exception as exc:
            log.warning("[NEWS SIDECAR FAIL] | %s: %s", committed.primary_file.name, exc)
    if original_entity == "youtube_summary":
        _mark_youtube_digest_read(content)
    if "News" in committed.primary_file.parts:
        _mark_news_radar_read(content)
    ticker = meta.get("ticker")
    if ticker:
        _ensure_stock_entity_stub(committed.primary_file.parent, str(ticker), vault_root=vault_root)
    return f"บันทึกสำเร็จ (raw, {'overwritten' if committed.revision > 1 else 'new'}): {committed.primary_file}"


def _current_vault_path() -> Path:
    return Path(os.getenv("OBSIDIAN_VAULT_PATH", str(VAULT_PATH))).resolve()


def _sync_to_catalog(file_path: Path) -> None:
    """Safely synchronizes written note to SQLite catalog with Outbox fallback."""
    v_root = _current_vault_path()
    from tools.archivist.catalog_runtime import load_catalog_pointer, resolve_catalog_path
    try:
        cat_db = resolve_catalog_path(v_root, require_exists=True)
    except (FileNotFoundError, OSError):
        return

    # Published R5 generations are immutable read models.  A live writer
    # records the projection request in the external outbox; a catalog build
    # worker will fold it into the next generation.
    if load_catalog_pointer(v_root) is not None:
        try:
            from tools.archivist.catalog_adapter import SqliteNoteCatalogAdapter
            cat = SqliteNoteCatalogAdapter(db_path=cat_db, vault_root=v_root, read_only=True)
            rel = file_path.resolve().relative_to(v_root).as_posix()
            cat.enqueue_outbox(rel, action="upsert", error_detail="queued for next immutable catalog generation")
        except Exception as e:
            log.warning("Catalog outbox enqueue failed for %s: %s", file_path, e)
        return

    cat = None
    try:
        from tools.archivist.catalog_adapter import SqliteNoteCatalogAdapter
        cat = SqliteNoteCatalogAdapter(db_path=cat_db, vault_root=v_root)
        cat.upsert_note_from_file(file_path)
    except Exception as e:
        log.warning("Catalog sync failed for %s: %s, enqueuing outbox", file_path, e)
        if cat:
            try:
                rel = file_path.resolve().relative_to(v_root).as_posix()
                cat.enqueue_outbox(rel, action="upsert", error_detail=str(e))
            except Exception:
                pass



@tool
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

    [Usage/When to use]
    ใช้เมื่อต้องการบันทึกข้อมูลที่เป็น Entity ใหม่ หรือต้องการอัปเดต Entity เดิมที่มีอยู่แล้ว
    เช่น สร้างประวัติบริษัท, ข้อมูลบุคคล, หรือเหตุการณ์สำคัญ
    (ข้อมูลที่ต้องผ่านการจัดโครงสร้างแบบมี Field เฉพาะ และต้องการโยง Wikilink)

    [Caution]
    - ห้ามใช้เครื่องมือนี้กับเนื้อหา Markdown ดิบที่มี YAML Frontmatter สำเร็จรูปมาแล้ว (เช่น ผลลัพธ์จาก Researcher) ให้ใช้ `write_raw_markdown` แทน
    - ข้อมูลใน Entity ต้องกระชับ ไม่รวมข่าวรายวันหรือ snapshots

    Args:
        title (str): ชื่อ entity หรือหัวข้อ ใช้เป็นชื่อไฟล์ .md (เช่น 'PTT PCL')
        content (str): เนื้อหาหลักในรูปแบบ Markdown เกี่ยวกับ entity นี้โดยตรง
        folder_path (str): โฟลเดอร์ปลายทาง เช่น '30_Knowledge_Base/Stocks' หรือ '30_Knowledge_Base/People'
        tags (list[str]): รายการ tag เช่น ['energy', 'SET100']
        entity_type (str): ประเภทของ entity เช่น 'Company', 'Executive', 'Macro_Event'
        aliases (list[str], optional): ชื่อเรียกอื่นๆ เช่น ['ปตท.', 'PTT PCL']. Defaults to [].
        linked_files (list[str], optional): ชื่อไฟล์ที่ต้องการสร้าง Wikilinks เชื่อมโยงไปหา (ไม่รวม .md). Defaults to [].

    Returns:
        str: ข้อความยืนยันสถานะการบันทึก (new หรือ append) พร้อม path ของไฟล์
    """
    vault_root = _current_vault_path()
    assert_write_allowed(vault_root)
    if VaultPaths(vault_root).layout_version >= 2:
        return _save_memory_v2(
            title=title,
            content=content,
            folder_path=folder_path,
            tags=tags,
            entity_type=entity_type,
            aliases=aliases,
            linked_files=linked_files,
            vault_root=vault_root,
        )
    raise RuntimeError(
        "Legacy V1 Markdown writes are disabled; migrate the vault to layout_version=2 "
        "or use the V2 writer contract."
    )
    entry = MemoryEntry(
        title=title,
        content=content,
        folder_path=folder_path,
        tags=tags,
        entity_type=entity_type,
        aliases=aliases or [],
        linked_files=linked_files or [],
    )

    target_dir = vault_root / entry.folder_path
    target_dir.mkdir(parents=True, exist_ok=True)

    safe_title = _sanitize_filename(entry.title)
    file_path = target_dir / f"{safe_title}.md"

    date_str = datetime.now().strftime("%Y-%m-%d")

    body = entry.content
    if entry.linked_files:
        related_links = "\n".join(
            "- "
            + render_resolved_markdown_link(
                vault_root,
                file_path,
                f,
                label=Path(str(f).replace("\\", "/")).stem,
            )
            for f in entry.linked_files
        )
        body += f"\n\n## Related\n{related_links}\n"

    with get_note_lock(file_path):
        if file_path.exists():
            post = fm.loads(file_path.read_text(encoding="utf-8"))
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
            _sync_to_catalog(file_path)
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
        _sync_to_catalog(file_path)
        return f"บันทึกสำเร็จ (new): {file_path}"



@tool
def write_raw_markdown(content: str, folder_path: str, filename: str) -> str:
    """บันทึกข้อมูลผลลัพธ์สำเร็จรูปในรูปแบบ Markdown ลงใน Obsidian Vault

    [Usage/When to use]
    ใช้เมื่อต้องการบันทึกข้อมูลที่มี YAML frontmatter พร้อมแล้ว เช่น:
    - ผลลัพธ์จากการสกัดเนื้อหาด้วย ingest_article_url, ingest_youtube_transcript, ingest_pdf
    - ข้อมูล Macro Snapshot, Regional Pulse, US Sectors Pulse จาก Researcher
    - รายงาน Macro Strategy Direction จาก Strategic Allocator / Macro Intel
    - รายงานสรุปข่าวรายวัน (entity_type: article_note)
    ระบบจะทำการ Auto-route สร้าง subfolder ย่อยตามข้อมูลใน YAML อัตโนมัติ:
    - path ลงท้าย 'Daily_Snapshots' → เติม subfolder วันที่จาก `date:` field
    - path ลงท้าย 'Stocks' → เติม subfolder ชื่อหุ้นจาก `ticker:` field
    - path ลงท้าย 'YouTube_Summaries' → จัดรูปแบบชื่อไฟล์นำหน้าด้วยวันที่ [YYYY-MM-DD] อัตโนมัติ (เช่น '2026-07-22 Title.md')
    - path มีคำว่า 'News' → เติม subfolder สำนักข่าวจาก `publisher:` field

    [Caution]
    - ห้ามใช้เครื่องมือนี้ในการสร้างหรืออัปเดต Entity หลักของระบบ (เช่น ข้อมูลบริษัท, ข้อมูลบุคคล) ให้ใช้ `save_memory` แทน
    - content ต้องเป็น Markdown ที่มี YAML frontmatter (---...---) อยู่ด้านบนสุดเสมอ
    - filename ห้ามมีนามสกุล .md — ระบบเติมให้อัตโนมัติ
    - ห้ามระบุโฟลเดอร์ย่อย (เช่น วันที่ หรือ Ticker) ใน folder_path ด้วยตัวเอง เพราะระบบจะดึงจาก YAML มาเติมให้เอง

    [Folder Mapping ตาม entity_type ใน YAML]
    - macro_strategy → ให้บันทึกลงทั้ง 2 โฟลเดอร์ คือ '30_Knowledge_Base/Macroeconomics/Daily_Snapshots' และ '30_Knowledge_Base/Strategies'
    - macro_daily / us_sectors_pulse / regional_macro / economic_fundamentals 
      → folder_path='30_Knowledge_Base/Macroeconomics/Daily_Snapshots'
    - Company / Financial_Trends / Financial_Health / Stock_Momentum / Analyst_Consensus / Company_News / equity_analysis
      → folder_path='30_Knowledge_Base/Stocks'
    - youtube_insight → folder_path='30_Knowledge_Base/YouTube_Summaries'
    - article_note → folder_path='30_Knowledge_Base/News'
    - book_note → folder_path='30_Knowledge_Base/Books'
    - Strategy / Concept / macro_strategy / Macro_Strategy_Direction → folder_path='30_Knowledge_Base/Strategies'
    - goal / financial_goal → ห้ามบันทึกด้วย tool นี้ ให้หยุดและแจ้งผู้ใช้ให้ใช้คำสั่งของ Bookkeeper แทน

    Args:
        content (str): เนื้อหา Markdown พร้อม YAML frontmatter ที่ต้องการบันทึก
        folder_path (str): โฟลเดอร์ปลายทางหลัก (root path ของหมวดหมู่นั้นๆ) เช่น '30_Knowledge_Base/Macroeconomics/Daily_Snapshots' หรือ '30_Knowledge_Base/News'
        filename (str): ชื่อไฟล์ไม่รวมนามสกุล เช่น 'Macro_Snapshot_2025-01-15' หรือ 'Stock_Market_News'

    Returns:
        str: ข้อความยืนยันสถานะการบันทึกไฟล์ (เช่น raw, new หรือ raw, overwritten)
    """
    vault_path = _current_vault_path()
    assert_write_allowed(vault_path)
    if VaultPaths(vault_path).layout_version >= 2:
        return _write_raw_markdown_v2(
            content=content,
            folder_path=folder_path,
            filename=filename,
            vault_root=vault_path,
        )
    raise RuntimeError(
        "Legacy V1 Markdown writes are disabled; migrate the vault to layout_version=2 "
        "or use the capture/published V2 writer contract."
    )
    entity_type_val = extract_yaml_frontmatter_value(content, "entity_type")
    if entity_type_val in {"goal", "financial_goal"}:
        return GOAL_VIA_BOOKKEEPER

    resolved_path = _maybe_inject_ticker_subfolder(folder_path, content)
    target_dir = vault_path / resolved_path
    target_dir.mkdir(parents=True, exist_ok=True)
    safe_name = _sanitize_filename(filename)
    safe_name = re.sub(r'[,()\[\].—–]', '', safe_name)
    safe_name = re.sub(r'_{2,}', '_', safe_name).strip('_') or "untitled"

    # ครอบทั้ง youtube_insight และ article_note — เดิมมีแค่ youtube_insight ทำให้ข่าวที่ agent
    # เรียก write_raw_markdown ตรง ๆ (ไม่ผ่าน agents/news_youtube_flow.py::_save_ingested_content)
    # ไม่มี date prefix ในชื่อไฟล์เลย ต่างจาก News Funnel ที่มี {date}_{title} เสมอ — เช็คเฉพาะ
    # entity_type_val == "article_note" ตรง ๆ (ไม่ fallback ไปเช็คด้วยชื่อ folder แบบ youtube_insight)
    # เพราะโฟลเดอร์ 'News' ยังถูกใช้ generic โดยเนื้อหาที่ไม่มี entity_type อยู่ (ดู test_writer.py)
    # Format date_prefix for youtube_insight and article_note
    date_val = extract_yaml_frontmatter_value(content, "published_at") or extract_yaml_frontmatter_value(content, "date")
    if not date_val:
        m_date = _DATE_FRONTMATTER_RE.search(content)
        if m_date:
            date_val = m_date.group(1).strip()
    date_prefix = date_val[:10] if date_val and len(date_val) >= 10 else datetime.now().strftime("%Y-%m-%d")

    if (entity_type_val == "youtube_insight" or str(resolved_path).rstrip("/").endswith("YouTube_Summaries") or entity_type_val == "article_note") and not re.match(r"^\d{4}-\d{2}-\d{2}", safe_name) and not re.match(r"^YT\d+$", safe_name):
        if re.match(r"^\d{4}-\d{2}-\d{2}$", date_prefix):
            safe_name = f"{date_prefix} {safe_name}"

    # V2 routing for News and YouTube_Summaries to YYYY/MM
    vp = VaultPaths(vault_path)
    if vp.layout_version >= 2 and "Inbox" not in str(resolved_path):
        year = date_prefix[:4] if len(date_prefix) >= 4 and date_prefix[:4].isdigit() else "_undated"
        month = date_prefix[5:7] if len(date_prefix) >= 7 and date_prefix[5:7].isdigit() else ""
        sub = f"{year}/{month}" if month else year
        if entity_type_val in ("article_note", "article", "company_news") or str(resolved_path).rstrip("/").endswith("News"):
            target_dir = vault_path / "30_Knowledge_Base" / "News" / sub
            target_dir.mkdir(parents=True, exist_ok=True)
        elif entity_type_val in ("youtube_insight", "youtube_summary") or str(resolved_path).rstrip("/").endswith("YouTube_Summaries"):
            target_dir = vault_path / "30_Knowledge_Base" / "YouTube_Summaries" / sub
            target_dir.mkdir(parents=True, exist_ok=True)
        elif entity_type_val == "macro_strategy" or "Strategies" in str(resolved_path):
            target_dir = vault_path / "30_Knowledge_Base" / "Macroeconomics" / "Strategies" / sub
            target_dir.mkdir(parents=True, exist_ok=True)
        elif entity_type_val in ("macro_daily", "macro_snapshot", "us_sectors_pulse", "regional_macro", "economic_fundamentals") or "Daily_Snapshots" in str(resolved_path):
            target_dir = vault_path / "30_Knowledge_Base" / "Macroeconomics" / "Daily_Snapshots" / sub
            target_dir.mkdir(parents=True, exist_ok=True)

    file_path = target_dir / f"{safe_name}.md"
    existed = file_path.exists()
    _atomic_write_text(file_path, content)
    _index_upsert(file_path)
    _sync_to_catalog(file_path)

    # Dual-Path Archiving for Macro Strategy Direction (V1 legacy only)
    if entity_type_val == "macro_strategy":
        if VaultPaths(vault_path).layout_version < 2:
            other_path = "30_Knowledge_Base/Strategies" if "Daily_Snapshots" in str(resolved_path) else "30_Knowledge_Base/Macroeconomics/Daily_Snapshots"
            other_dir = vault_path / other_path
            other_dir.mkdir(parents=True, exist_ok=True)
            other_file = other_dir / f"{safe_name}.md"
            if other_file.resolve() != file_path.resolve():
                _atomic_write_text(other_file, content)
                _index_upsert(other_file)
                _sync_to_catalog(other_file)

    # Layer-1 Entity stub — auto-create {ticker}.md hub for Stocks snapshots
    if "Stocks" in resolved_path.split("/"):
        ticker_val = extract_yaml_frontmatter_value(content, "ticker") or extract_yaml_frontmatter_value(content, "tickers")
        if not ticker_val:
            m = _TICKER_FRONTMATTER_RE.search(content)
            if m:
                ticker_val = m.group(1).strip()
        if ticker_val:
            _ensure_stock_entity_stub(target_dir, ticker_val)

    # Auto-create Obsidian Canvas for YouTube Insights
    if entity_type_val == "youtube_insight":
        try:
            # Temporarily disabled per user request
            # _create_youtube_canvas(file_path, content)
            pass
        except Exception as e:
            log.warning("[CANVAS FAIL] | %s: %s", file_path.name, e)
        _mark_youtube_digest_read(content)

    if entity_type_val == "Company_News":
        try:
            parsed_data = parse_company_news_items(content)
            json_path = file_path.with_suffix(".json")
            _atomic_write_text(json_path, json.dumps(parsed_data, ensure_ascii=False, indent=2))
        except Exception as e:
            log.warning("[NEWS SIDECAR FAIL] | %s: %s", file_path.name, e)

    if "News" in resolved_path.split("/"):
        _mark_news_radar_read(content)

    action = "overwritten" if existed else "new"
    return f"บันทึกสำเร็จ (raw, {action}): {file_path}"


def _mark_news_radar_read(content: str) -> None:
    """ขีดฆ่าข่าวใน News-Radar-Daily หลังจากดึงข้อมูลสำเร็จ"""
    m_url = _SOURCE_URL_FRONTMATTER_RE.search(content)
    if not m_url:
        return
    url = m_url.group(1).strip()
    
    inbox_dir = VAULT_PATH / "30_Knowledge_Base/News/Inbox"
    if not inbox_dir.exists():
        return
        
    for md_file in inbox_dir.glob("News-Radar-Daily_*.md"):
        try:
            text = md_file.read_text(encoding="utf-8")
            if url in text and f"({url})~~" not in text:
                new_text = re.sub(
                    rf'\[([^\]]+)\]\({re.escape(url)}\)',
                    rf'~~[\1]({url})~~',
                    text
                )
                if new_text != text:
                    _atomic_write_text(md_file, new_text)
                    log.info(f"Marked {url} as read in {md_file.name}")
        except Exception as e:
            log.warning(f"Failed to update News-Radar-Daily: {e}")


def _mark_youtube_digest_read(content: str) -> None:
    """ขีดฆ่าลิงก์วิดีโอใน Weekly_Digest หลังจากดึงข้อมูลสำเร็จ"""
    m_url = _SOURCE_URL_FRONTMATTER_RE.search(content)
    if not m_url:
        return
    url = m_url.group(1).strip()
    
    inbox_dir = VAULT_PATH / "30_Knowledge_Base/YouTube_Summaries/Inbox"
    if not inbox_dir.exists():
        return
        
    for md_file in inbox_dir.glob("Weekly_Digest_*.md"):
        try:
            text = md_file.read_text(encoding="utf-8")
            if url in text and f"]({url})~~" not in text and f"({url})~~" not in text:
                import re
                new_text = re.sub(
                    rf'\[([^\]]+)\]\({re.escape(url)}\)',
                    rf'~~[\1]({url})~~',
                    text
                )
                if new_text != text:
                    _atomic_write_text(md_file, new_text)
                    log.info(f"Marked YouTube {url} as read in {md_file.name}")
        except Exception as e:
            log.warning(f"Failed to update Weekly_Digest: {e}")

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
                nodes.append(txt(node_id, f"**{ticker}**\n{desc}", tx, ty, TW, TH, "6"))  # purple
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


def _ensure_stock_entity_stub(target_dir: Path, ticker: str, vault_root: Path | None = None) -> None:
    """สร้าง Layer-1 Entity hub file ถ้ายังไม่มี
    Hub นี้รับความสัมพันธ์จาก snapshot/news/trade ที่อ้างถึง ticker
    """
    safe_ticker = _sanitize_filename(ticker.strip().upper())
    if not safe_ticker:
        return

    from tools.archivist.vault_paths import VaultPaths
    root = Path(vault_root).resolve() if vault_root is not None else _current_vault_path()
    vp = VaultPaths(root)
    if vp.layout_version >= 2 or "Stocks" in target_dir.parts:
        stub_path = vp.note_path({"entity_type": "stock_hub", "ticker": safe_ticker})
    else:
        stub_path = target_dir / f"{safe_ticker}.md"

    if stub_path.exists():
        return
    today = datetime.now().strftime("%Y-%m-%d")
    if vp.layout_version >= 2:
        build_knowledge_note_writer(vault_paths=vp).write_note(
            metadata={
                "schema_version": 2,
                "title": safe_ticker,
                "entity_type": "stock_hub",
                "ticker": safe_ticker,
                "date": today,
                "tags": ["entity", "stock_hub", safe_ticker.lower()],
            },
            body=(
                f"# {safe_ticker}\n\n"
                f"> **Entity hub** for `{safe_ticker}`.\n\n"
                "## Notes\n\n"
                "*(Add durable notes here; backlinks are derived from portable Markdown links.)*\n"
            ),
            filename=safe_ticker,
        )
        _index_upsert(stub_path)
        return

    stub_path.parent.mkdir(parents=True, exist_ok=True)
    entity_type_name = "stock_hub" if vp.layout_version >= 2 else "stock_entity"
    schema_ver_line = "schema_version: 2\n" if vp.layout_version >= 2 else ""
    content = (
        "---\n"
        f"{schema_ver_line}"
        f"title: {safe_ticker}\n"
        f"entity_type: {entity_type_name}\n"
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
    try:
        _atomic_write_text(stub_path, content)
        _index_upsert(stub_path)
    except Exception:
        # If stub was created concurrently by another agent, don't fail the pipeline
        if stub_path.exists():
            return
        raise




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

