"""LangGraph Flow แยกอิสระสำหรับ Web UI Kanban Board (Human-in-the-Loop Approval Flow)

ไม่กระทบ manager_agent.py เดิม ครอบคลุม 3 Node:
load_pending_node -> gate_node -> synthesize_node
"""
from typing import Any, Dict, List, Literal, Optional, TypedDict

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from tools.macro.news_funnel import run_news_funnel_synthesize
from tools.macro.news_funnel_store import get_pending_high_impact_events


class NewsFunnelState(TypedDict, total=False):
    messages: list
    period: str
    candidates: List[Dict[str, Any]]
    approved_event_ids: Optional[List[str]]
    result_summary: str
    store_path: Optional[str]
    vault_root: Optional[str]


def load_pending_node(state: NewsFunnelState) -> Dict[str, Any]:
    """โหลดรายการข่าว High-Impact จาก Persistent JSON Store พร้อมส่งข้อความ AIMessage ให้แสดงใน Terminal"""
    store_path = state.get("store_path")
    candidates = get_pending_high_impact_events(store_path=store_path)

    msg_text = f"พบข่าว High-Impact รอสังเคราะห์ {len(candidates)} รายการ"
    return {
        "candidates": candidates,
        "messages": [AIMessage(content=msg_text, name="load_pending")],
    }


def gate_node(state: NewsFunnelState) -> Command[Literal["synthesize"]]:
    """หยุดชะงัก (interrupt) รอการคัดเลือกหรืออนุมัติจากผู้ใช้บนหน้า Web UI Kanban"""
    candidates = state.get("candidates", [])

    selection = interrupt({
        "type": "news_funnel_approval",
        "candidates": candidates,
    })

    if isinstance(selection, dict):
        approved_ids = selection.get("approved_event_ids")
        if approved_ids is None:
            approved_ids = []
    elif isinstance(selection, list):
        approved_ids = selection
    else:
        approved_ids = []

    return Command(
        goto="synthesize",
        update={
            "approved_event_ids": approved_ids,
        },
    )


def synthesize_node(state: NewsFunnelState) -> Dict[str, Any]:
    """สร้างโน้ตข่าวเดี่ยวตามรายการ approved_event_ids ที่ผู้ใช้อนุมัติ"""
    # period ไม่ต้อง derive ที่นี่ — run_news_funnel_synthesize resolve เองเมื่อได้ None/"auto"
    period = state.get("period")
    approved_ids = state.get("approved_event_ids")
    candidates = state.get("candidates", [])
    candidate_ids = [c.get("event_id") for c in candidates if isinstance(c, dict) and c.get("event_id")] if candidates else None
    store_path = state.get("store_path")
    vault_root = state.get("vault_root")

    result = run_news_funnel_synthesize(
        period=period,
        approved_event_ids=approved_ids,
        candidate_event_ids=candidate_ids,
        store_path=store_path,
        vault_root=vault_root,
    )

    # หมายเหตุ: node นี้ได้ approved_ids เป็น list เสมอ (gate fail-closed) ดังนั้น status
    # ที่เป็นไปได้มีแค่ no_approved_events กับ success — ไม่มี branch no_pending_events
    if result.get("status") == "no_approved_events":
        rejected_cnt = result.get("rejected_count", 0)
        if rejected_cnt > 0:
            msg_text = f"🚫 ไม่มีรายการที่ได้รับการอนุมัติ (เปลี่ยนสถานะไม่คัดเลือกเป็น Rejected {rejected_cnt} รายการ) — ไม่เขียนไฟล์ทับลง Obsidian Vault"
        else:
            msg_text = "🚫 ข้ามรอบนี้ตามคำสั่ง (0 รายการ) — ไม่สร้างโน้ตข่าวและไม่ปฏิเสธข่าวในคิว (จบ Graceful)"
    else:
        count = result.get("published_count", 0)
        rejected_cnt = result.get("rejected_count", 0)
        files = result.get("created_files", [])
        files_str = ", ".join([f.split("/")[-1].split("\\")[-1] for f in files[:3]]) + ("..." if len(files) > 3 else "")
        reject_msg = f" (และเปลี่ยนสถานะรายการที่ไม่เลือกเป็น Rejected {rejected_cnt} รายการ)" if rejected_cnt > 0 else ""
        msg_text = f"✓ เผยแพร่ข่าวเดี่ยวสำเร็จจำนวน {count} รายการ{reject_msg} -> บันทึกลง {files_str if files_str else 'Obsidian Vault'}"

    return {
        "result_summary": msg_text,
        "messages": [AIMessage(content=msg_text, name="synthesize")],
    }


def build_news_funnel_graph(checkpointer=None):
    """สร้าง LangGraph StateGraph สำหรับ News Funnel Flow"""
    builder = StateGraph(NewsFunnelState)
    builder.add_node("load_pending", load_pending_node)
    builder.add_node("gate", gate_node)
    builder.add_node("synthesize", synthesize_node)

    builder.add_edge(START, "load_pending")
    builder.add_edge("load_pending", "gate")
    builder.add_edge("synthesize", END)

    return builder.compile(checkpointer=checkpointer)
