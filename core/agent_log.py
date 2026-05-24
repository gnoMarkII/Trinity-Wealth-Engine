"""Agent activity logger — daily Markdown logs in memories/01_Daily_Logs/

จับ 2 อย่าง:
1. Agent-to-agent routing/handoffs (จาก manager_agent nodes)
2. Tool/agent warnings & errors (ผ่าน DailyMarkdownHandler ใน core.logger)

Format: Markdown ที่อ่านง่ายใน Obsidian — ไฟล์ละ 1 วัน, header pattern:
    ### [HH:MM:SS] Source → Target
    reason: ...
    "preview ของ message"
"""
import os
from datetime import datetime
from pathlib import Path
from threading import Lock

_VAULT_PATH = Path(os.getenv("OBSIDIAN_VAULT_PATH", "./memories"))
_LOG_DIR = _VAULT_PATH / "01_Daily_Logs"
_PREVIEW_LIMIT = 200
_lock = Lock()


def _today_path() -> Path:
    day = datetime.now().strftime("%Y-%m-%d")
    return _LOG_DIR / f"Agent_Log_{day}.md"


def _ensure_file(path: Path) -> None:
    """สร้างไฟล์ + YAML frontmatter ครั้งแรกของวัน — caller ต้องถือ _lock"""
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    day = path.stem.replace("Agent_Log_", "")
    header = (
        "---\n"
        f"title: Agent Log {day}\n"
        "entity_type: agent_log\n"
        f"date: {day}\n"
        "tags: [log, agents, system]\n"
        "---\n\n"
        f"# Agent Activity Log — {day}\n\n"
    )
    path.write_text(header, encoding="utf-8")


def _truncate(text: str, limit: int = _PREVIEW_LIMIT) -> str:
    """ตัดข้อความให้สั้น + ยุบ whitespace หลายตัวเป็นเดี่ยว"""
    if not text:
        return ""
    collapsed = " ".join(text.split())
    return collapsed if len(collapsed) <= limit else collapsed[:limit].rstrip() + "..."


def _write_entry(entry: str) -> None:
    path = _today_path()
    with _lock:
        _ensure_file(path)
        with path.open("a", encoding="utf-8") as f:
            f.write(entry + "\n\n")


def _label(name: str) -> str:
    """แปลง agent name เป็น display label — 'user' คงเดิม, อื่น ๆ capitalize"""
    return name if name.lower() == "user" else name.capitalize()


def log_routing(
    source: str,
    target: str,
    reason: str | None = None,
    content: str | None = None,
) -> None:
    """บันทึก agent-to-agent handoff หรือ user-to-agent

    Args:
        source: ผู้ส่ง เช่น 'user', 'manager', 'researcher'
        target: ผู้รับ เช่น 'archivist', 'bookkeeper', 'user'
        reason: เหตุผลสั้น ๆ (optional) เช่น 'structured_mutation', 'save_to_vault'
        content: เนื้อหา/instruction (จะถูก truncate เหลือ 200 chars)
    """
    time_str = datetime.now().strftime("%H:%M:%S")
    lines = [f"### [{time_str}] {_label(source)} → {_label(target)}"]
    if reason:
        lines.append(f"reason: {reason}")
    if content:
        lines.append(f'"{_truncate(content)}"')
    _write_entry("\n".join(lines))
