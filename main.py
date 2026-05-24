import os
import re
import time

import httpx
from dotenv import load_dotenv
from google.genai import errors as genai_errors
from langgraph.checkpoint.memory import MemorySaver
from prompt_toolkit import prompt

from core.agent_log import log_routing
from core.logger import setup_logging
from core.security import anonymize_pii
from core.utils import normalize_content

load_dotenv()
setup_logging()

_TRANSIENT_HTTP_CODES = {429, 500, 502, 503, 504}
_TRANSIENT_MSG_PATTERN = re.compile(
    r'\b(429|500|502|503|504|rate.?limit|resource.?exhausted|deadline.?exceeded|service.?unavailable)\b',
    re.IGNORECASE,
)


def _http_code(exc: Exception) -> int | None:
    """พยายามดึง HTTP status code จาก exception หลายรูปแบบ"""
    for attr in ("code", "status_code", "http_status"):
        v = getattr(exc, attr, None)
        if isinstance(v, int):
            return v
    resp = getattr(exc, "response", None)
    if resp is not None:
        return getattr(resp, "status_code", None)
    return None


def _is_transient_error(exc: Exception) -> bool:
    """ตรวจว่า error ควร retry หรือไม่ — รองรับ genai SDK, httpx, network และ LangChain-wrapped errors"""
    if isinstance(exc, (TimeoutError, ConnectionError, httpx.TimeoutException, httpx.ConnectError)):
        return True
    if isinstance(exc, (genai_errors.APIError, httpx.HTTPStatusError)):
        code = _http_code(exc)
        if code in _TRANSIENT_HTTP_CODES:
            return True
    # LangChain มักจะ wrap original error เป็น message string — fallback ตรวจจาก text
    return bool(_TRANSIENT_MSG_PATTERN.search(str(exc)))


def _label_from_route(meta: dict | None, node_name: str) -> str:
    """สร้าง display label จาก route_meta — ถ้าไม่มี ก็ใช้ node_name fallback"""
    if not meta:
        return node_name
    source = meta.get("source", node_name)
    target = meta.get("target", "")
    if target in ("", "user"):
        return source.capitalize()
    label = f"{source.capitalize()} → {target.capitalize()}"
    if target == "researcher" and meta.get("save_to_vault") is False:
        label += " (no save)"
    return label


def _display(node_name: str, state: dict) -> None:
    messages = state.get("messages") if isinstance(state, dict) else None
    if not messages:
        return
    last = messages[-1] if isinstance(messages, list) else messages
    content = normalize_content(getattr(last, "content", ""))
    if not content:
        return
    meta = state.get("route_meta") if isinstance(state, dict) else None
    label = _label_from_route(meta, node_name)
    print(f"\n[{label}]: {content}")


def main():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or api_key == "your_google_api_key_here":
        print("ERROR: กรุณาตั้งค่า GOOGLE_API_KEY ใน .env ก่อน")
        return

    from agents.manager_agent import build_graph
    from tools.archivist_tools import init_vault_structure

    init_vault_structure()

    print("=" * 60)
    print("  Investment Manager AI — Multi-Agent System")
    print("  Supervisor: The Manager | Workers: The Archivist, The Researcher, The Bookkeeper")
    print("  พิมพ์ 'quit' 'exit' หรือ 'ออก' เพื่อออกจากโปรแกรม")
    print("=" * 60)

    memory = MemorySaver()
    graph = build_graph(checkpointer=memory)
    config = {"configurable": {"thread_id": "1"}, "recursion_limit": 25}

    while True:
        try:
            user_input = prompt("\nคุณ: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nออกจากโปรแกรม...")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q", "ออก"):
            print("ออกจากโปรแกรม...")
            break

        user_input, had_pii = anonymize_pii(user_input)
        if had_pii:
            print("[Security] ตรวจพบและลบข้อมูล PII ก่อนส่งเข้าประมวลผล")

        log_routing("user", "manager", content=user_input)
        print("\n[กำลังประมวลผล...]")

        inputs = {"messages": [("user", user_input)]}

        max_retries = 3
        for attempt in range(max_retries):
            try:
                for event in graph.stream(inputs, config=config, stream_mode="updates"):
                    for node_name, state in event.items():
                        if not isinstance(state, dict) or "messages" not in state:
                            continue
                        _display(node_name, state)
                break
            except Exception as e:
                if _is_transient_error(e) and attempt < max_retries - 1:
                    print(f"\n[System]: เซิร์ฟเวอร์ AI กำลังทำงานหนัก (Attempt {attempt + 1}/{max_retries})... กำลังรอเพื่อเชื่อมต่อใหม่...")
                    time.sleep(2 ** attempt)
                    continue
                if _is_transient_error(e):
                    print(f"\n[System]: ไม่สามารถติดต่อ AI ได้ในขณะนี้ ({type(e).__name__}) โปรดลองพิมพ์คำถามของคุณใหม่อีกครั้ง")
                else:
                    print(f"\nERROR: {e}")
                break


if __name__ == "__main__":
    main()
