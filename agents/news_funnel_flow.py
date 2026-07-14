"""LangGraph Flow แยกอิสระสำหรับ Web UI Kanban Board (Human-in-the-Loop Approval Flow)

ไม่กระทบ manager_agent.py เดิม ครอบคลุม 3 Node:
load_pending_node -> gate_node -> synthesize_node
"""
from datetime import datetime
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
    """สังเคราะห์ธีมเศรษฐกิจตามรายการ approved_event_ids ที่ผู้ใช้อนุมัติ"""
    period = state.get("period")
    if not period:
        period = "morning" if datetime.now().hour < 12 else "evening"
    approved_ids = state.get("approved_event_ids")
    store_path = state.get("store_path")
    vault_root = state.get("vault_root")

    result = run_news_funnel_synthesize(
        period=period,
        approved_event_ids=approved_ids,
        store_path=store_path,
        vault_root=vault_root,
    )

    if result.get("status") == "no_pending_events":
        msg_text = "ไม่มีรายการข่าวที่รอสังเคราะห์ (Zero Pending) — ไม่เขียนไฟล์ทับลง Obsidian Vault"
    elif result.get("status") == "synthesis_llm_failure":
        msg_text = "❌ การสังเคราะห์ธีมเศรษฐกิจผ่าน LLM ล้มเหลว — ไม่เขียนไฟล์และยังเก็บรายการข่าวเป็น Pending เพื่อลองใหม่ในรอบถัดไป"
    else:
        file_path = result.get("file_path", "")
        count = result.get("synthesized", 0)
        msg_text = f"✓ สังเคราะห์ธีมเศรษฐกิจสำเร็จจาก {count} รายการ -> บันทึกลง {file_path}"

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
