"""MCP Adapter layer — จุดเดียวในระบบที่คุยกับ notebooklm-mcp server โดยตรง

หน้าที่: เปิด stdio session, เรียก tool, ดัก error, decode response เป็น dict domain-agnostic
เดียว ไม่ให้ pipeline.py ต้อง parse โครงสร้าง MCP เอง

หมายเหตุ: notebooklm-mcp ใช้ internal/undocumented NotebookLM API — โครงสร้าง response ที่แท้จริง
(โดยเฉพาะของ studio_status/download_artifact) ไม่ได้มีเอกสารทางการ โค้ดด้านล่างเขียนแบบ defensive
(รองรับได้ทั้ง structuredContent และ JSON ฝังใน text content) ตามคำเตือน Internal API Risk ในแผน
"""
import json
import shutil
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

from langsmith import traceable
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from core.logger import get_logger
from core.retry import with_retry_async
from tools.archivist.maintenance_guard import assert_write_allowed
from tools.content.notebooklm.models import PreflightError

logger = get_logger(__name__)

# Tools ที่กระทบ quota/สร้างข้อมูลจริงฝั่ง NotebookLM — ห่อ retry (exponential backoff, budget เดียวกับ
# core/retry.py) เพราะ transient network error ระหว่างเรียกพวกนี้ทำให้เสียของฟรีถ้าไม่ retry
# ส่วน studio_status/research_status เป็น polling (รอสถานะงานเสร็จ) ซึ่งเป็นคนละ concern — ไม่ห่อตรงนี้
_RETRYABLE_TOOLS = {"source_add", "studio_create", "download_artifact", "notebook_query"}


def find_notebooklm_mcp_binary() -> str | None:
    """ค้นหา path ของ notebooklm-mcp binary โดยค้นจาก PATH และ venv ของระบบ"""
    # 1. ลองค้นหาผ่าน PATH ปกติ
    found = shutil.which("notebooklm-mcp")
    if found:
        return found

    # 2. ค้นหาในโฟลเดอร์ของ Python environment ที่กำลังรันอยู่ (e.g. .venv/Scripts หรือ .venv/bin)
    venv_bin = Path(sys.executable).parent
    for name in ("notebooklm-mcp.exe", "notebooklm-mcp", "notebooklm-mcp.cmd", "notebooklm-mcp.bat"):
        candidate = venv_bin / name
        if candidate.is_file():
            return str(candidate)

    # 3. ค้นหาใน workspace .venv
    ws_bin = Path.cwd() / ".venv" / ("Scripts" if sys.platform == "win32" else "bin")
    for name in ("notebooklm-mcp.exe", "notebooklm-mcp", "notebooklm-mcp.cmd", "notebooklm-mcp.bat"):
        candidate = ws_bin / name
        if candidate.is_file():
            return str(candidate)

    return None


@asynccontextmanager
async def open_session() -> AsyncIterator[ClientSession]:
    """เปิด stdio session กับ notebooklm-mcp — หนึ่ง process ต่อหนึ่ง pipeline run เท่านั้น

    ห้ามเปิดหลาย session ซ้อนกันในเวลาเดียวกัน (ดู Account Risk warning ในแผน — server ใช้
    active default profile เดียว รันพร้อมกันหลาย process เสี่ยงสร้าง Notebook ผิดบัญชี)
    """
    cmd = find_notebooklm_mcp_binary() or "notebooklm-mcp"
    params = StdioServerParameters(command=cmd)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


def _extract_text(result: Any) -> str:
    parts = []
    for block in result.content or []:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    return "\n".join(parts)


def _decode_result(tool_name: str, result: Any) -> dict[str, Any]:
    """แปลง CallToolResult -> dict เสมอ ไม่ว่า server จะตอบมาแบบ structuredContent หรือ JSON text"""
    if result.isError:
        raise RuntimeError(f"MCP tool '{tool_name}' คืน error: {_extract_text(result)}")
    if result.structuredContent:
        return dict(result.structuredContent)
    text = _extract_text(result)
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"raw_text": text}


@traceable(run_type="tool")
async def call_tool(session: ClientSession, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """เรียก MCP tool หนึ่งตัว แล้วคืนผลเป็น dict — จุดเดียวที่ pipeline.py เรียกใช้"""
    async def _invoke() -> dict[str, Any]:
        result = await session.call_tool(tool_name, arguments)
        data = _decode_result(tool_name, result)
        # ถ้าเป็น tool ที่ retry ได้ และผลลัพธ์มี status="error" (เช่น Google API ยัง process ไม่เสร็จ)
        # ให้แปลงเป็น exception เพื่อให้ with_retry_async ดักและ retry ใหม่ตาม exponential backoff
        if tool_name in _RETRYABLE_TOOLS and data.get("status") == "error":
            raise RuntimeError(f"MCP tool '{tool_name}' คืน internal error: {data.get('error', data)}")
        return data

    if tool_name in _RETRYABLE_TOOLS:
        return await with_retry_async(_invoke)
    return await _invoke()


def check_binary_available() -> None:
    """ตรวจว่าคำสั่ง notebooklm-mcp มีอยู่จริงใน PATH หรือ venv — เช็คแบบ local ไม่ต้องเปิด session"""
    if find_notebooklm_mcp_binary() is None:
        raise PreflightError(
            "ไม่พบคำสั่ง 'notebooklm-mcp' ใน PATH — ติดตั้ง notebooklm-mcp-cli ก่อน (uv sync)"
        )


def check_output_dir_writable(output_dir: Path) -> None:
    """ตรวจว่า output directory เขียนไฟล์ได้จริง — เช็คแบบ local ไม่ต้องเปิด session"""
    assert_write_allowed(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    probe = output_dir / ".write_probe"
    try:
        probe.write_text("ok", encoding="utf-8")
    except OSError as e:
        raise PreflightError(f"เขียนไฟล์ใน output dir ไม่ได้ ({output_dir}): {e}") from e
    finally:
        probe.unlink(missing_ok=True)


@traceable(run_type="tool")
async def check_auth(session: ClientSession) -> None:
    """ตรวจ auth status ผ่าน server_info — ต้องมี session เปิดอยู่แล้ว (ใช้ session เดียวกับที่เหลือทั้ง run)"""
    info = await call_tool(session, "server_info", {})
    auth_status = info.get("auth_status")
    if auth_status != "configured":
        raise PreflightError(
            f"NotebookLM auth ยังไม่พร้อม (auth_status={auth_status!r}) — รัน `nlm login` ก่อน"
        )
