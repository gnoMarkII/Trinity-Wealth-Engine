from langsmith import traceable
import json
import os
import re
import shutil
import tempfile
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Optional, Union

import frontmatter as fm
from filelock import FileLock
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.logger import get_logger
from schemas.pkm_models import MemoryEntry

log = get_logger(__name__)



from .core import _atomic_write_text, read_file, VAULT_PATH, INDEX_PATH, INDEX_LOCK, _VAULT_SYSTEM_FILES, _INDEX_EXCLUDE
from .parser import _extract_asset_tickers, _strip_frontmatter, extract_yaml_frontmatter_value
from .portable_links import render_resolved_markdown_link






_index_cache: dict[str, list[tuple[str, str]]] = {}
_index_cache_built = False
_index_dirty = False
_index_cache_root: Optional[Path] = None
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
    val = extract_yaml_frontmatter_value(content, "entity_type")
    return val if val else "—"


def _file_folder_label(file_path: Path, vault_root: Optional[Path | str] = None) -> str:
    root = Path(vault_root) if vault_root else VAULT_PATH
    try:
        rel = file_path.resolve().relative_to(root.resolve())
        return str(rel.parent).replace("\\", "/") if rel.parent != Path(".") else "Root"
    except (ValueError, RuntimeError):
        try:
            rel = file_path.resolve().relative_to(VAULT_PATH.resolve())
            return str(rel.parent).replace("\\", "/") if rel.parent != Path(".") else "Root"
        except (ValueError, RuntimeError):
            parts = file_path.parts
            for marker in ("30_Knowledge_Base", "10_Projects", "20_Areas", "40_Archive", "00_Inbox", "01_Daily_Logs", "50_Crypto", "60_Research"):
                if marker in parts:
                    idx = parts.index(marker)
                    rel_parts = parts[idx:-1]
                    return "/".join(rel_parts) if rel_parts else marker
            return file_path.parent.name or "Root"


def _is_indexable(file_path: Path) -> bool:
    if file_path.name in _VAULT_SYSTEM_FILES:
        return False
    if any(excl in file_path.parts for excl in _INDEX_EXCLUDE):
        return False
    parts = file_path.parts
    # Exclude backup, trash, system, template, archive folders
    for p in parts:
        if p.startswith(".") or "backup" in p.lower() or p in ("40_Archive", "99_Templates", "quarantine", "outbox"):
            return False
    return True


def _build_cache_from_disk(vault_root: Optional[Path | str] = None) -> None:
    """Full scan: เรียกครั้งแรกหรือเมื่อ tool update_master_index ถูกเรียก"""
    global _index_cache_built, _index_cache_root, _index_dirty
    _index_cache.clear()
    root = (Path(vault_root) if vault_root else VAULT_PATH).resolve()
    _index_cache_root = root
    _index_dirty = False
    if not root.exists():
        _index_cache_built = True
        return
    all_files = [f for f in sorted(root.rglob("*.md")) if _is_indexable(f)]
    for fp in all_files:
        _index_cache.setdefault(_file_folder_label(fp, vault_root=root), []).append(
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


def _write_index_from_cache(vault_root: Optional[Path | str] = None) -> str:
    if not _index_cache:
        return "ไม่มีไฟล์ที่จะ index ใน Vault"

    # แยก Layer-1 entities ออกจาก Layer-2 knowledge snapshots
    entities_by_category: dict[str, list[str]] = {}
    knowledge_by_folder: dict[str, list[tuple[str, str]]] = {}

    for folder, entries in _index_cache.items():
        for stem, etype in entries:
            if etype in _LAYER1_ENTITY_TYPES or etype in ("stock_hub", "holding"):
                entities_by_category.setdefault(_entity_category(folder), []).append(stem)
            else:
                knowledge_by_folder.setdefault(folder, []).append((stem, etype))

    target_root = (Path(vault_root) if vault_root else VAULT_PATH).resolve()
    index_source = target_root / "index.md"

    lines = [
        "---",
        "schema_version: 2",
        "note_id: nav_master_index_v2",
        "document_key: navigation:v2:master",
        "entity_type: concept",
        "document_role: navigation",
        "search_scope: excluded",
        "title: Master Index",
        f"date: {datetime.now().strftime('%Y-%m-%d')}",
        "---",
        "",
        "# Master Index",
        "",
        "> ระบบ 3-Layer Graph View: **Entities** เป็น hub (Layer 1), **Knowledge** เป็น snapshot/news (Layer 2),",
        "> Portfolio (Layer 3) ดูใน Portfolio Dashboard และ Trading Journal",
        "",
    ]

    # Keep the small Layer-3 bridge explicit.  The historical master index
    # carried these relationships, and a regenerated index must not silently
    # erase them just because portfolio files are not part of the knowledge
    # category cache.
    portfolio_targets = [
        ("20_Portfolio_Management/Portfolio_Dashboard.md", "Portfolio Dashboard"),
        ("20_Portfolio_Management/Current_Holdings/Portfolios/default/Trading_Journal.md", "Trading Journal"),
        ("00_Index/Home.md", "Open V2 Home"),
    ]
    holding_root = target_root / "20_Portfolio_Management" / "Current_Holdings" / "Portfolios"
    if holding_root.is_dir():
        for holding in sorted(holding_root.glob("*/Holdings/*.md")):
            try:
                with holding.open("r", encoding="utf-8") as hf:
                    post = fm.load(hf)
                if post.metadata and post.metadata.get("status") == "archived":
                    continue
            except Exception:
                pass
            portfolio_targets.append(
                (_file_folder_label(holding, vault_root=target_root) + "/" + holding.name, holding.stem)
            )
    lines += ["## Portfolio (Layer 3)", ""]
    for target, label in portfolio_targets:
        lines.append(
            "- "
            + render_resolved_markdown_link(
                target_root,
                index_source,
                target,
                label=label,
            )
        )
    lines.append("")

    # Layer 1 — Entities
    if entities_by_category:
        lines += ["## 📍 Entities (Layer 1 Hubs)", ""]
        for category in sorted(entities_by_category):
            stems = sorted(dict.fromkeys(entities_by_category[category]))
            links: list[str] = []
            for stem in stems:
                target_hint = stem
                for folder, entries in _index_cache.items():
                    if stem in {entry_stem for entry_stem, _ in entries} and _entity_category(folder) == category:
                        target_hint = f"{folder}/{stem}" if folder != "Root" else stem
                        break
                links.append(
                    render_resolved_markdown_link(
                        target_root,
                        index_source,
                        target_hint,
                        label=stem,
                    )
                )
            rendered_links = " · ".join(links)
            lines += [f"### {category} ({len(stems)})", "", rendered_links, ""]

    # Layer 2 — Knowledge Hubs summary
    lines += ["## 📚 Knowledge Base (Layer 2 Categories)", ""]
    category_counts: dict[str, int] = {}
    for folder, entries in knowledge_by_folder.items():
        cat = _entity_category(folder)
        category_counts[cat] = category_counts.get(cat, 0) + len(entries)

    for cat in sorted(category_counts):
        lines.append(f"- **{cat}**: {category_counts[cat]:,} notes")
    lines.append("")

    # Recent / Key Knowledge Hubs (bounded to keep total chars <= 4,000)
    lines += ["### 🗂️ Major Knowledge Sections", ""]
    hub_targets = (
        ("00_Index/Stocks_Hub.md", "📈 Stocks & Equity Analysis"),
        ("00_Index/Macro_Hub.md", "🌐 Macroeconomics & Strategy"),
        ("00_Index/News_Hub.md", "📰 News & Articles"),
        ("00_Index/YouTube_Hub.md", "📺 Video Summaries"),
        ("00_Index/NotebookLM_Sources_Hub.md", "🎙️ NotebookLM Sources"),
        ("00_Index/Concepts_Hub.md", "💡 Research Concepts"),
    )
    for target, label in hub_targets:
        lines.append(
            "- "
            + render_resolved_markdown_link(
                target_root,
                index_source,
                target,
                label=label,
            )
        )
    lines.append("")

    output_text = "\n".join(lines)
    _atomic_write_text(target_root / "index.md", output_text)

    entity_count = sum(len(v) for v in entities_by_category.values())
    knowledge_count = sum(len(v) for v in knowledge_by_folder.values())
    total = entity_count + knowledge_count
    return f"อัปเดต index.md สำเร็จ: {total} ไฟล์ ({len(output_text)} ตัวอักษร)"


def _index_upsert(file_path: Path, vault_root: Optional[Path | str] = None) -> None:
    """Incremental update cache (lazy flush) — mark dirty แทนการเขียน index.md ทุกครั้ง"""
    global _index_dirty
    if not _is_indexable(file_path):
        return

    root = (Path(vault_root) if vault_root else VAULT_PATH).resolve()
    if not _index_cache_built or _index_cache_root != root:
        _build_cache_from_disk(vault_root=root)

    folder = _file_folder_label(file_path, vault_root=vault_root)
    entity_type = _read_entity_type(file_path)
    entries = _index_cache.setdefault(folder, [])

    for i, (stem, _) in enumerate(entries):
        if stem == file_path.stem:
            entries[i] = (file_path.stem, entity_type)
            break
    else:
        entries.append((file_path.stem, entity_type))

    _index_dirty = True


def flush_index_if_dirty(vault_root: Optional[Path | str] = None) -> str | None:
    """เขียน index.md ลงดิสก์เฉพาะเมื่อ cache เปลี่ยน — เรียกหลังจบ ReAct cycle"""
    global _index_dirty
    if not _index_dirty:
        return None
    msg = _write_index_from_cache(vault_root=vault_root)
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
