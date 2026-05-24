import json
import os
import re
import shutil
import tempfile
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import frontmatter as fm
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.logger import get_logger
from schemas.pkm_models import MemoryEntry

log = get_logger(__name__)


def _atomic_write_text(path: Path, content: str) -> None:
    """เขียนไฟล์แบบ atomic: temp file ใน folder เดียวกัน → os.replace()
    os.replace() เป็น atomic บนทั้ง Windows และ POSIX เมื่ออยู่ filesystem เดียวกัน
    """
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.stem}_", suffix=f"{path.suffix}.tmp", dir=str(parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

VAULT_PATH = Path(os.getenv("OBSIDIAN_VAULT_PATH", "./memories"))
CHROMA_PATH = VAULT_PATH / ".chroma_index"
_CHROMA_MTIME_FILE = VAULT_PATH / ".chroma_mtime"

_vs_cache: dict = {}  # {"cache_key": str, "vs": Chroma}

# Incremental index cache: {folder_str: [(stem, entity_type), ...]}
_index_cache: dict[str, list[tuple[str, str]]] = {}
_index_cache_built = False
_index_dirty = False  # mark เมื่อ cache เปลี่ยน — flush ผ่าน flush_index_if_dirty()

_INVALID_FILE_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_READ_FILE_LIMIT = 8000
_LINKED_CONTENT_LIMIT = 1500
_SEMANTIC_CONTENT_LIMIT = 2000
_INDEX_EXCLUDE = ("00_Inbox", "01_Daily_Logs")
_VAULT_SYSTEM_FILES = {
    "index.md",
    "Portfolio_Holdings.md",
    "Portfolio_Dashboard.md",
    "Watchlist.md",
    "Trading_Journal.md",
}


def _sanitize_filename(name: str) -> str:
    """แทนอักขระต้องห้ามบน Windows/POSIX และตัดช่องว่าง/จุดท้ายชื่อ"""
    cleaned = _INVALID_FILE_CHARS.sub("-", name).strip(" .")
    cleaned = re.sub(r'-{2,}', '-', cleaned).strip('-')
    return cleaned or "untitled"

_DEFAULT_VAULT_FOLDERS = [
    "00_Inbox",
    "01_Daily_Logs",
    "10_System_Agents",
    "20_Portfolio_Management/Current_Holdings",
    "20_Portfolio_Management/Journals_and_Reports",
    "30_Knowledge_Base/Macroeconomics",
    "30_Knowledge_Base/Macroeconomics/Daily_Snapshots",
    "30_Knowledge_Base/Stocks",
    "30_Knowledge_Base/Strategies",
    "40_Finance_and_Tax/Tax_Deductions",
    "40_Finance_and_Tax/Capital_Gains",
    "30_Knowledge_Base/YouTube_Summaries",
    "30_Knowledge_Base/Books",
    "30_Knowledge_Base/Articles",
    "99_Templates",
]


def _vault_folders() -> list[str]:
    """รวม default folders + extras จาก VAULT_EXTRA_FOLDERS env (comma-separated)
    ตัวอย่าง: VAULT_EXTRA_FOLDERS='50_Crypto,60_Research/Drafts'
    """
    extras = os.getenv("VAULT_EXTRA_FOLDERS", "").strip()
    if not extras:
        return _DEFAULT_VAULT_FOLDERS
    extra_list = [p.strip() for p in extras.split(",") if p.strip()]
    return _DEFAULT_VAULT_FOLDERS + extra_list


def init_vault_structure() -> None:
    for folder in _vault_folders():
        (VAULT_PATH / folder).mkdir(parents=True, exist_ok=True)



@lru_cache(maxsize=1)
def get_embeddings():
    log.info("กำลังโหลด embedding model สำหรับ Semantic Search (ครั้งแรกอาจใช้เวลา)")
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


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


_DATE_FRONTMATTER_RE = re.compile(r'^date:\s*["\']?(\d{4}-\d{2}-\d{2})', re.MULTILINE)
_TICKER_FRONTMATTER_RE = re.compile(r'^ticker:\s*["\']?([^\s"\'#]+)', re.MULTILINE)
_VIDEO_ID_FRONTMATTER_RE = re.compile(r'^video_id:\s*(\S+)', re.MULTILINE)
_SOURCE_URL_FRONTMATTER_RE = re.compile(r'^source_url:\s*(.+)$', re.MULTILINE)
_H2_SECTION_RE = re.compile(r'^## (.+)$', re.MULTILINE)
_H3_SECTION_RE = re.compile(r'^### (.+)$', re.MULTILINE)


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


def _strip_frontmatter(content: str) -> str:
    """ตัด YAML frontmatter (--- ... ---) ออก คืนเฉพาะ body"""
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            return parts[2].strip()
    return content.strip()


def _parse_h2_sections(body: str) -> dict[str, str]:
    """สกัด ## headers → {heading: content} dict"""
    result: dict[str, str] = {}
    matches = list(_H2_SECTION_RE.finditer(body))
    for i, m in enumerate(matches):
        heading = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(body)
        result[heading] = body[start:end].strip()
    return result


def _parse_h3_subsections(text: str) -> dict[str, str]:
    """สกัด ### sub-headers → {heading: content}, fallback 'ทั่วไป' ถ้าไม่มี sub-header"""
    result: dict[str, str] = {}
    matches = list(_H3_SECTION_RE.finditer(text))
    if not matches:
        stripped = text.strip()
        return {"ทั่วไป": stripped} if stripped else {}
    for i, m in enumerate(matches):
        heading = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if body:
            result[heading] = body
    return result


def _split_bullets(text: str, max_per_node: int = 3) -> list[str]:
    """แบ่ง text เป็น chunks ไม่เกิน max_per_node บรรทัดต่อ chunk"""
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    if not lines:
        return []
    return ["\n".join(lines[i:i + max_per_node]) for i in range(0, len(lines), max_per_node)]


def _extract_asset_tickers(text: str) -> list[tuple[str, str]]:
    """สกัด (ticker, description) จากบรรทัดที่มี [[TICKER]] wikilink"""
    result = []
    for line in text.splitlines():
        stripped = line.strip().lstrip("- *")
        m = re.search(r"\[\[([A-Z][A-Z0-9.\-]{1,9})\]\]", stripped)
        if not m:
            continue
        ticker = m.group(1)
        desc = re.sub(r"\*+", "", stripped)
        desc = re.sub(r"\[\[[^\]]+\]\]", f"#{ticker}", desc)
        desc = re.sub(r"^[#\s—\-]+", "", desc).strip()
        result.append((ticker, desc[:120]))
    return result


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


@tool
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


@tool
def read_file(filepath: str) -> str:
    """อ่านเนื้อหาไฟล์ .md จาก Vault

    Args:
        filepath: path ของไฟล์ภายใน Vault เช่น '30_Knowledge_Base/Macroeconomics/GDP.md'
    """
    file_path = VAULT_PATH / filepath
    if not file_path.exists():
        return f"ไม่พบไฟล์: {filepath}"

    content = file_path.read_text(encoding="utf-8")
    if len(content) > _READ_FILE_LIMIT:
        content = content[:_READ_FILE_LIMIT] + f"\n\n...[ตัดทอน — ไฟล์ยาว {len(content)} ตัวอักษร]"
    return f"=== {filepath} ===\n\n{content}"


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


# Layer-1 Entity types — extensible: เพิ่ม sector_entity, theme_entity ได้ในอนาคต
_LAYER1_ENTITY_TYPES = {"stock_entity"}


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
    """สแกน Vault และสร้าง/อัปเดตสารบัญ index.md อัตโนมัติ จัดกลุ่มตาม Folder
    ข้ามโฟลเดอร์ 00_Inbox และ 01_Daily_Logs รวมถึงไฟล์ระบบ index.md
    """
    return _rebuild_index()


@tool
def lint_structural_health() -> str:
    """ตรวจสุขภาพเชิงโครงสร้างของ Vault: ค้นหา Orphan files (ไม่มีไฟล์ใด Wikilink โยงมา)
    และ Empty files (ไม่มีเนื้อหา) ส่งรายงานกลับมาให้ทราบว่าไฟล์ไหนต้องจัดการ
    """
    all_files = [
        f for f in VAULT_PATH.rglob("*.md")
        if f.name not in _VAULT_SYSTEM_FILES
        and not any(excl in f.parts for excl in _INDEX_EXCLUDE)
    ]

    if not all_files:
        return "ไม่มีไฟล์ใดใน Vault ที่จะตรวจสอบ"

    stem_to_path = {f.stem: f for f in all_files}
    file_contents = {f: f.read_text(encoding="utf-8") for f in all_files}

    inbound: dict[str, set[str]] = {f.stem: set() for f in all_files}
    empty_files: list[str] = []

    for file_path in all_files:
        content = file_contents[file_path]

        for raw in re.findall(r'\[\[(.*?)\]\]', content):
            target = raw.split("|")[0].strip()
            if target in stem_to_path:
                inbound[target].add(file_path.stem)

        stripped = content.strip()
        parts = stripped.split("---", 2) if stripped.startswith("---") else ["", "", stripped]
        body = parts[2].strip() if len(parts) >= 3 else stripped
        if not body:
            empty_files.append(file_path.stem)

    orphans = sorted(s for s in stem_to_path if not inbound[s])

    lines = [
        "# Vault Health Report",
        "",
        f"**ไฟล์ทั้งหมด:** {len(all_files)}",
        f"**Orphan notes:** {len(orphans)}",
        f"**Empty files:** {len(empty_files)}",
        "",
    ]
    if orphans:
        lines += ["## Orphan Notes (ไม่มี Link เชื่อมโยง)"]
        lines += [f"- [[{s}]] ({stem_to_path[s].relative_to(VAULT_PATH).parent})" for s in orphans]
        lines.append("")
    if empty_files:
        lines += ["## Empty Files (ไม่มีเนื้อหา)"]
        lines += [f"- [[{s}]] ({stem_to_path[s].relative_to(VAULT_PATH).parent})" for s in sorted(empty_files)]
        lines.append("")
    if not orphans and not empty_files:
        lines.append("Vault อยู่ในสุขภาพดี ไม่พบปัญหาใดๆ")

    return "\n".join(lines)


@tool
def lint_semantic_conflict(target_folder_or_entity: str) -> str:
    """ดึงเนื้อหาไฟล์ใน Folder หรือ Entity ที่ระบุ เพื่อให้ LLM ตรวจหาความขัดแย้งของข้อมูล
    Python ทำหน้าที่ดึงและจำกัดขอบเขต ส่วน LLM อ่านผลลัพธ์เพื่อวิเคราะห์ความสอดคล้อง

    Args:
        target_folder_or_entity: โฟลเดอร์ (เช่น '30_Knowledge_Base/Stocks')
                                  หรือชื่อ Entity (เช่น 'PTT') ที่ต้องการตรวจสอบ
    """
    target_path = VAULT_PATH / target_folder_or_entity

    if target_path.is_dir():
        files = sorted(target_path.glob("*.md"))
        source_label = f"Folder: {target_folder_or_entity}"
    else:
        all_files = list(VAULT_PATH.rglob("*.md"))
        name_lower = target_folder_or_entity.lower()
        files = sorted(f for f in all_files if name_lower in f.stem.lower())
        source_label = f"Entity match: {target_folder_or_entity}"

    if not files:
        return f"ไม่พบไฟล์ที่ตรงกับ '{target_folder_or_entity}'"

    parts = [
        f"# Semantic Conflict Check — {source_label}",
        f"ไฟล์ที่ตรวจ: {len(files)} ไฟล์",
        "",
    ]
    for file_path in files:
        content = file_path.read_text(encoding="utf-8")
        if len(content) > _SEMANTIC_CONTENT_LIMIT:
            content = content[:_SEMANTIC_CONTENT_LIMIT] + "\n...[ตัดทอน]"
        rel = file_path.relative_to(VAULT_PATH)
        parts.append(f"## [{rel}]\n{content}\n")

    return "\n".join(parts)


def _chunk_file(file_path: Path, splitter: RecursiveCharacterTextSplitter) -> tuple[list[str], list[dict], list[str]]:
    """Chunk หนึ่งไฟล์ → คืน (texts, metas, ids) ที่พร้อม upsert เข้า Chroma"""
    content = file_path.read_text(encoding="utf-8")
    rel = str(file_path.relative_to(VAULT_PATH))
    chunks = splitter.split_text(content)
    texts = chunks
    metas = [{"source": rel} for _ in chunks]
    ids = [f"{rel}::{i}" for i in range(len(chunks))]
    return texts, metas, ids


def _load_index_state() -> dict:
    """โหลด per-file state: {rel: {mtime, chunks}}"""
    if not _CHROMA_MTIME_FILE.exists():
        return {}
    try:
        data = json.loads(_CHROMA_MTIME_FILE.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "files" in data:
            return data["files"]
    except (OSError, json.JSONDecodeError):
        pass
    return {}


def _save_index_state(files: dict) -> None:
    _atomic_write_text(
        _CHROMA_MTIME_FILE,
        json.dumps({"version": 1, "files": files}, ensure_ascii=False),
    )


def _searchable_files() -> list[Path]:
    return [
        f for f in VAULT_PATH.rglob("*.md")
        if f.name not in _VAULT_SYSTEM_FILES
        and not any(excl in f.parts for excl in _INDEX_EXCLUDE)
    ]


@tool
def search_all_memories(keyword: str) -> str:
    """ค้นหาความจำทั้งหมดใน Vault ด้วย Semantic Search (Vector RAG) แบบ Local
    เหมาะสำหรับคำถามที่ต้องการความเข้าใจความหมาย ไม่ใช่แค่ keyword ตรงๆ

    Index เป็นแบบ incremental — รีเฟรชเฉพาะไฟล์ที่ mtime เปลี่ยน, ลบไฟล์ที่หายไป
    ไม่ rebuild ทั้ง Chroma store เหมือนเวอร์ชันก่อน

    Args:
        keyword: คำถามหรือวลีที่ต้องการค้นหาตามความหมาย
    """
    md_files = _searchable_files()
    if not md_files:
        return "ยังไม่มีไฟล์ความจำใดใน Vault"

    current: dict[str, dict] = {
        str(f.relative_to(VAULT_PATH)): {"mtime": f.stat().st_mtime}
        for f in md_files
    }
    stored = _load_index_state()

    added_or_changed = [
        rel for rel, info in current.items()
        if rel not in stored or abs(stored[rel].get("mtime", 0) - info["mtime"]) > 1e-3
    ]
    removed = [rel for rel in stored if rel not in current]

    cache_valid = _vs_cache.get("cache_signature") == (len(current), tuple(sorted(current)))
    needs_update = bool(added_or_changed or removed) or not cache_valid

    if "vs" in _vs_cache and not needs_update:
        vectorstore = _vs_cache["vs"]
    else:
        try:
            vectorstore = _vs_cache.get("vs") or Chroma(
                persist_directory=str(CHROMA_PATH),
                embedding_function=get_embeddings(),
            )
        except Exception:
            # corrupted store → wipe and start fresh
            if CHROMA_PATH.exists():
                shutil.rmtree(CHROMA_PATH)
            stored = {}
            added_or_changed = list(current)
            removed = []
            try:
                vectorstore = Chroma(
                    persist_directory=str(CHROMA_PATH),
                    embedding_function=get_embeddings(),
                )
            except Exception as e:
                return f"เกิดข้อผิดพลาดในการเปิด vectorstore: {e}"

        # Delete removed + changed (changed gets re-added below)
        ids_to_delete: list[str] = []
        for rel in removed:
            n = stored.get(rel, {}).get("chunks", 0)
            ids_to_delete.extend(f"{rel}::{i}" for i in range(n))
        for rel in added_or_changed:
            n = stored.get(rel, {}).get("chunks", 0)
            if n:
                ids_to_delete.extend(f"{rel}::{i}" for i in range(n))
        if ids_to_delete:
            try:
                vectorstore.delete(ids=ids_to_delete)
            except Exception as e:
                log.warning("Chroma delete failed (continuing): %s", e)

        # Re-chunk and add changed
        if added_or_changed:
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            all_texts: list[str] = []
            all_metas: list[dict] = []
            all_ids: list[str] = []
            for rel in added_or_changed:
                fp = VAULT_PATH / rel
                if not fp.exists():
                    continue
                texts, metas, ids = _chunk_file(fp, splitter)
                all_texts.extend(texts)
                all_metas.extend(metas)
                all_ids.extend(ids)
                current[rel]["chunks"] = len(texts)
            if all_texts:
                try:
                    vectorstore.add_texts(texts=all_texts, metadatas=all_metas, ids=all_ids)
                except Exception as e:
                    return f"เกิดข้อผิดพลาดในการเพิ่ม vectorstore: {e}"

        # Preserve chunk count for unchanged files
        for rel in current:
            if "chunks" not in current[rel] and rel in stored:
                current[rel]["chunks"] = stored[rel].get("chunks", 0)

        _save_index_state(current)
        _vs_cache["vs"] = vectorstore
        _vs_cache["cache_signature"] = (len(current), tuple(sorted(current)))

    try:
        results = vectorstore.similarity_search(keyword, k=5)
    except Exception as e:
        return f"เกิดข้อผิดพลาดในการค้นหา: {e}"

    if not results:
        return f"ไม่พบความจำที่เกี่ยวข้องกับ '{keyword}'"

    parts = [f"ผลการค้นหาเชิงความหมายสำหรับ '{keyword}' ({len(results)} ผลลัพธ์):\n"]
    for i, doc in enumerate(results, 1):
        source = doc.metadata.get("source", "ไม่ทราบแหล่งที่มา")
        parts.append(f"--- ผลลัพธ์ที่ {i} | แหล่งที่มา: [{source}] ---\n{doc.page_content}\n")

    return "\n".join(parts)


def _find_file_by_name(name: str, all_files: list[Path]) -> Path | None:
    """ค้นหาไฟล์โดย stem ตรงทั้งหมดก่อน แล้ว fallback ไป partial match"""
    name_lower = name.lower()
    exact = next((f for f in all_files if f.stem.lower() == name_lower), None)
    if exact:
        return exact
    return next((f for f in all_files if name_lower in f.stem.lower()), None)


@tool
def search_graph_context(entity_name: str) -> str:
    """ค้นหาข้อมูล entity พร้อม linked entities ในรูปแบบ Graph (GraphRAG)
    ใช้เมื่อต้องการวิเคราะห์บริษัท, บุคคล, หรือเหตุการณ์แบบเจาะลึกพร้อมบริบทโดยรอบ

    Args:
        entity_name: ชื่อ entity ที่ต้องการค้นหา เช่น 'PTT', 'Somchai'
    """
    all_files = list(VAULT_PATH.rglob("*.md"))
    if not all_files:
        return "ยังไม่มีไฟล์ความจำใดใน Vault"

    # Step 1-2: หาและอ่านไฟล์หลัก
    main_file = _find_file_by_name(entity_name, all_files)
    if main_file is None:
        return f"ไม่พบไฟล์สำหรับ entity '{entity_name}' ใน Vault"

    main_content = main_file.read_text(encoding="utf-8")

    # Step 3: สกัด Wikilinks
    wikilinks = re.findall(r'\[\[(.*?)\]\]', main_content)

    output = f"--- Main Entity: {main_file.stem} ---\n{main_content}\n"

    # Step 4-5: อ่าน linked files
    if wikilinks:
        output += "\n--- Linked Connections ---\n"
        seen = set()
        for link_name in wikilinks:
            # Wikilink อาจมีรูป [[File|Alias]] — ดึงเฉพาะชื่อไฟล์
            link_target = link_name.split("|")[0].strip()
            if link_target in seen:
                continue
            seen.add(link_target)

            linked_file = _find_file_by_name(link_target, all_files)
            if linked_file:
                link_content = linked_file.read_text(encoding="utf-8")
                if len(link_content) > _LINKED_CONTENT_LIMIT:
                    link_content = link_content[:_LINKED_CONTENT_LIMIT] + "\n...[ตัดทอน]"
                output += f"\n- [{link_target}]:\n{link_content}\n"
            else:
                output += f"\n- [{link_target}]: (ไม่พบไฟล์ใน Vault)\n"

    return output
