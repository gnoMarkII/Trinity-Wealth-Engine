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



from .core import _atomic_write_text, read_file, VAULT_PATH, INDEX_PATH, INDEX_LOCK, _VAULT_SYSTEM_FILES, _INDEX_EXCLUDE
from .parser import _extract_asset_tickers, _strip_frontmatter






_index_cache: dict[str, list[tuple[str, str]]] = {}
_index_cache_built = False
_index_dirty = False
_LAYER1_ENTITY_TYPES = {"stock_entity"}
_LAYER1_ENTITY_TYPES = {"stock_entity"}

VAULT_PATH = Path(os.getenv("OBSIDIAN_VAULT_PATH", "./memories"))
INDEX_PATH = VAULT_PATH / ".system" / "master_index.json"
INDEX_LOCK = str(INDEX_PATH) + ".lock"


def _read_entity_type(file_path: Path) -> str:
    """ดึง entity_type จาก YAML frontmatter ของไฟล์ .md"""
    try:
        content = file_path.read_text(encoding="utf-8")
    except OSError:
        return "—"
    m = re.search(r'^entity_type:\s*(.+)$', content, re.MULTILINE)
    return m.group(1).strip() if m else "—"


def _file_folder_label(file_path: Path) -> str:
    rel = file_path.relative_to(VAULT_PATH)
    return str(rel.parent) if rel.parent != Path(".") else "Root"


def _is_indexable(file_path: Path) -> bool:
    return (
        file_path.name not in _VAULT_SYSTEM_FILES
        and not any(excl in file_path.parts for excl in _INDEX_EXCLUDE)
    )


def _build_cache_from_disk() -> None:
    """Full scan: เรียกครั้งแรกหรือเมื่อ tool update_master_index ถูกเรียก"""
    global _index_cache_built
    _index_cache.clear()
    all_files = [f for f in sorted(VAULT_PATH.rglob("*.md")) if _is_indexable(f)]
    for fp in all_files:
        _index_cache.setdefault(_file_folder_label(fp), []).append(
            (fp.stem, _read_entity_type(fp))
        )
    _index_cache_built = True


def _entity_category(folder: str) -> str:
    """แยก category จาก folder path: '30_Knowledge_Base\\Stocks\\AAPL' → 'Stocks'"""
    parts = folder.replace("\\", "/").split("/")
    for i, p in enumerate(parts):
        if p == "30_Knowledge_Base" and i + 1 < len(parts):
            return parts[i + 1]
    return "Other"


def _write_index_from_cache() -> str:
    if not _index_cache:
        return "ไม่มีไฟล์ที่จะ index ใน Vault"

    # แยก Layer-1 entities ออกจาก Layer-2 knowledge snapshots
    entities_by_category: dict[str, list[str]] = {}
    knowledge_by_folder: dict[str, list[tuple[str, str]]] = {}

    for folder, entries in _index_cache.items():
        for stem, etype in entries:
            if etype in _LAYER1_ENTITY_TYPES:
                entities_by_category.setdefault(_entity_category(folder), []).append(stem)
            else:
                knowledge_by_folder.setdefault(folder, []).append((stem, etype))

    lines = [
        "---",
        "title: Master Index",
        f"date: {datetime.now().strftime('%Y-%m-%d')}",
        "---",
        "",
        "# Master Index",
        "",
        "> ระบบ 3-Layer Graph View: **Entities** เป็น hub (Layer 1), **Knowledge** เป็น snapshot/news (Layer 2),",
        "> Portfolio (Layer 3) ดูใน [[Portfolio_Dashboard]] และ [[Trading_Journal]]",
        "",
    ]

    # Layer 1 — Entities (อยู่บนสุดเพื่อให้ scan หาเร็ว)
    if entities_by_category:
        lines += ["## 📍 Entities (Layer 1 Hubs)", ""]
        for category in sorted(entities_by_category):
            stems = sorted(entities_by_category[category])
            wikilinks = " · ".join(f"[[{s}]]" for s in stems)
            lines += [f"### {category} ({len(stems)})", "", wikilinks, ""]

    # Layer 2 — Knowledge snapshots (folder-grouped, newest first)
    if knowledge_by_folder:
        lines += ["## 📚 Knowledge (Layer 2 Snapshots)", ""]
        for folder in sorted(knowledge_by_folder):
            lines += [f"### {folder}", "", "| File | Entity Type |", "|------|-------------|"]
            # reverse sort: ถ้า filename ลงท้ายด้วยวันที่ จะได้ใหม่สุดบนสุด
            for stem, etype in sorted(knowledge_by_folder[folder], reverse=True):
                lines.append(f"| [[{stem}]] | {etype} |")
            lines.append("")

    _atomic_write_text(VAULT_PATH / "index.md", "\n".join(lines))

    entity_count = sum(len(v) for v in entities_by_category.values())
    knowledge_count = sum(len(v) for v in knowledge_by_folder.values())
    total = entity_count + knowledge_count
    return (
        f"อัปเดต index.md สำเร็จ: {total} ไฟล์ "
        f"({entity_count} entities, {knowledge_count} snapshots)"
    )


def _index_upsert(file_path: Path) -> None:
    """Incremental update cache (lazy flush) — mark dirty แทนการเขียน index.md ทุกครั้ง"""
    global _index_dirty
    if not _is_indexable(file_path):
        return

    if not _index_cache_built:
        _build_cache_from_disk()

    folder = _file_folder_label(file_path)
    entity_type = _read_entity_type(file_path)
    entries = _index_cache.setdefault(folder, [])

    for i, (stem, _) in enumerate(entries):
        if stem == file_path.stem:
            entries[i] = (file_path.stem, entity_type)
            break
    else:
        entries.append((file_path.stem, entity_type))

    _index_dirty = True


def flush_index_if_dirty() -> str | None:
    """เขียน index.md ลงดิสก์เฉพาะเมื่อ cache เปลี่ยน — เรียกหลังจบ ReAct cycle"""
    global _index_dirty
    if not _index_dirty:
        return None
    msg = _write_index_from_cache()
    _index_dirty = False
    return msg


def _rebuild_index() -> str:
    """Full rebuild — เรียกจาก tool update_master_index หรือเมื่อต้อง resync จาก disk"""
    global _index_dirty
    _build_cache_from_disk()
    msg = _write_index_from_cache()
    _index_dirty = False
    return msg


@tool
def update_master_index() -> str:
    """สร้างหรืออัปเดตไฟล์ Master Index (index.md) อัตโนมัติ

    [Usage/When to use]
    ใช้เมื่อมีการเปลี่ยนแปลงโครงสร้างไฟล์อย่างมีนัยสำคัญ (เช่น เพิ่มไฟล์หลายไฟล์พร้อมกัน, ลบไฟล์, แก้ไขชื่อไฟล์)
    - ระบบจะทำการสแกน Markdown ไฟล์ทั้งหมดใน Vault และสร้างสารบัญแยกตาม Folder Hierarchy
    - ช่วยให้ `read_file('index.md')` มองเห็นโครงสร้างล่าสุดเสมอ

    [Caution]
    - ไม่จำเป็นต้องเรียกใช้เมื่อบันทึกไฟล์แค่ไฟล์เดียวด้วย `save_memory` หรือ `write_raw_markdown` เพราะเครื่องมือเหล่านั้นมีกลไก update index ตัวเองอยู่แล้ว
    - จะใช้เวลาทำงานสักพักเนื่องจากต้องสแกนไฟล์ทั้ง Vault

    Returns:
        str: สถานะการอัปเดต Index และสถิติจำนวนไฟล์
    """
    return _rebuild_index()
