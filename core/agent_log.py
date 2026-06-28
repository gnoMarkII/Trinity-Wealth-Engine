"""Agent activity logger — structured Markdown logs in Obsidian

Format: Foldable Callouts representing lifecycle of a turn
"""
import os
import json
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Literal

from core.security import anonymize_pii

_VAULT_PATH = Path(os.getenv("OBSIDIAN_VAULT_PATH", "./memories"))
_LOG_DIR = _VAULT_PATH / "01_Daily_Logs"
_PREVIEW_LIMIT = 3000
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


def _write_entry(entry: str) -> None:
    path = _today_path()
    with _lock:
        _ensure_file(path)
        with path.open("a", encoding="utf-8") as f:
            f.write(entry + "\n\n")


def _format_time(elapsed_sec: float | None = None) -> str:
    time_str = datetime.now().strftime("%H:%M:%S")
    if elapsed_sec is not None:
        return f"[{time_str} | {elapsed_sec:.1f}s]"
    return f"[{time_str}]"


def _smart_format_and_truncate(text: str, limit: int = _PREVIEW_LIMIT) -> str:
    """Format as JSON if possible, otherwise text. Truncate if too long."""
    if not text:
        return ""
        
    text = text.strip()
    original_len = len(text)
    
    # Try parse JSON
    try:
        parsed = json.loads(text)
        formatted = json.dumps(parsed, ensure_ascii=False, indent=2)
        is_json = True
    except Exception:
        formatted = text
        is_json = False

    if len(formatted) > limit:
        formatted = formatted[:limit] + f"\n\n... (truncated, original length: {original_len} chars)"

    # Is it already markdown?
    if is_json:
        # Check inner backticks
        fence = "```"
        while fence in formatted:
            fence += "`"
        return f"{fence}json\n{formatted}\n{fence}"
    
    if "---" in formatted[:100] or formatted.startswith("#") or "```" in formatted or "|" in formatted:
        # Likely markdown already, no fencing to keep it rendering in Obsidian
        if formatted.endswith("chars)"):
            if formatted.count("```") % 2 != 0:
                formatted += "\n```"
        return formatted
        
    # Plain text, wrap in text fence
    fence = "```"
    while fence in formatted:
        fence += "`"
    return f"{fence}text\n{formatted}\n{fence}"


def _prefix_multiline(text: str, prefix: str = "> ") -> str:
    """Prefix every line with `> ` so it stays inside a callout."""
    if not text:
        return prefix.strip()
    return "\n".join(f"{prefix}{line}" for line in text.splitlines())


def log_turn_start(turn_id: str, user_message: str) -> None:
    """ขึ้นหัวข้อใหม่สำหรับ Turn ของ User"""
    clean_msg, _ = anonymize_pii(user_message)
    clean_msg = clean_msg.strip()
    entry = f"## 🗣️ User Request {_format_time()}\n**Turn ID:** `{turn_id}`\n\n{clean_msg}"
    _write_entry(entry)


def log_manager_plan(turn_id: str, tasks: list[dict]) -> None:
    """สรุปแผนของ Manager"""
    turn_info = f" (Turn: `{turn_id}`)" if turn_id else ""
    lines = [f"> [!abstract] 📋 Manager Plan {_format_time()}{turn_info}"]
    if not tasks:
        lines.append("> (No tasks planned)")
    else:
        for i, t in enumerate(tasks, 1):
            target = t.get("target", "Unknown").capitalize()
            instr = t.get("instruction", "").replace('\n', ' ')
            if len(instr) > 100:
                instr = instr[:97] + "..."
            lines.append(f"> - **Task {i}:** {target} - {instr}")
    
    _write_entry("\n".join(lines))


def log_worker_result(turn_id: str, worker_name: str, result: str, status: str = "success", elapsed_sec: float | None = None) -> None:
    """บันทึกผลลัพธ์ของ Worker เป็น Foldable Callout"""
    valid_statuses = {"abstract", "info", "todo", "tip", "success", "question", "warning", "failure", "danger", "bug", "example", "quote"}
    if status not in valid_statuses:
        status = "info"
        
    time_info = _format_time(elapsed_sec)
    turn_info = f" (Turn: `{turn_id}`)" if turn_id else ""
    worker = worker_name.capitalize()
    
    header = f"> [!{status}]- ⚙️ Worker: {worker} {time_info}{turn_info}"
    formatted_result = _smart_format_and_truncate(result)
    body = _prefix_multiline(formatted_result)
    
    _write_entry(f"{header}\n{body}")


def log_system_action(turn_id: str | None, action: str, details: str, status: Literal["info", "warning", "failure"] = "warning") -> None:
    """บันทึกเหตุการณ์แทรกแซงของระบบ เช่น Re-plan"""
    time_info = _format_time()
    turn_info = f" (Turn: `{turn_id}`)" if turn_id else ""
    header = f"> [!{status}] 🔄 System: {action} {time_info}{turn_info}"
    
    formatted_details = _smart_format_and_truncate(details)
    body = _prefix_multiline(formatted_details)
    
    _write_entry(f"{header}\n{body}")


def log_routing(source: str, target: str, reason: str | None = None, content: str | None = None, turn_id: str | None = None) -> None:
    """Legacy wrapper เพื่อไม่ให้โค้ดเก่าพัง"""
    t_id = turn_id or "legacy"
    details = f"Source: {source}\nTarget: {target}"
    if reason:
        details += f"\nReason: {reason}"
    if content:
        details += f"\nContent: {content}"
        
    log_system_action(t_id, "Legacy routing", details, status="info")
