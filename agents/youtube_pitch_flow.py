"""StateGraph Workflow สำหรับ YouTube Content Pitching & Research-Grade Briefing Book"""
from datetime import datetime
from typing import Literal, Optional, TypedDict
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from core.logger import get_logger
from schemas.youtube_pitch_schemas import YouTubeContentPitchItem
from tools.content.youtube_pitcher import (
    fetch_news_for_pitching,
    generate_youtube_pitches,
    parse_date_filters_from_instruction,
    save_notebooklm_source,
    synthesize_notebooklm_source,
)

logger = get_logger(__name__)


class YouTubePitchState(TypedDict, total=False):
    messages: list
    instruction: str
    from_date: Optional[str]
    to_date: Optional[str]
    lookback_days: int
    news_candidates: list[dict]
    macro_baselines: str
    pitches: list[dict]
    approved_pitch_ids: list[str]
    result_summary: str


def fetch_topics_node(state: YouTubePitchState) -> dict:
    instruction = state.get("instruction", "")
    filters = parse_date_filters_from_instruction(instruction)
    from_d = filters.get("from_date") or state.get("from_date")
    to_d = filters.get("to_date") or state.get("to_date")
    lookback = filters.get("lookback_days", state.get("lookback_days", 7))

    candidates, macro_str, is_fallback = fetch_news_for_pitching(
        from_date=from_d,
        to_date=to_d,
        lookback_days=lookback,
    )

    messages = []
    if is_fallback:
        messages.append(
            AIMessage(
                content="⚠️ คำเตือน: ช่วงวันที่ระบุย้อนหลังเกิน 7 วัน (News Funnel Store ตัดข้อมูลข่าวดิบทุก 7 วัน) — ระบบได้ดึงข้อมูลจากโน้ตที่สังเคราะห์ไว้แล้วในฐานความรู้ (Layer 2) ร่วมด้วย",
                name="fetch_topics",
            )
        )
    messages.append(
        AIMessage(
            content=f"พบข่าวและบทวิเคราะห์ที่เกี่ยวข้อง {len(candidates)} รายการ (ช่วงวันที่: {from_d or 'ย้อนหลัง'} ถึง {to_d or f'{lookback} วันล่าสุด'})",
            name="fetch_topics",
        )
    )

    return {
        "from_date": from_d,
        "to_date": to_d,
        "lookback_days": lookback,
        "news_candidates": candidates,
        "macro_baselines": macro_str,
        "messages": messages,
    }


def generate_pitches_node(state: YouTubePitchState) -> dict:
    candidates = state.get("news_candidates", [])
    instruction = state.get("instruction", "")
    from_d = state.get("from_date")
    to_d = state.get("to_date")

    batch = generate_youtube_pitches(
        candidates=candidates,
        max_pitches=4,
        instruction=instruction,
        from_date=from_d,
        to_date=to_d,
    )

    pitches_dict_list = [p.model_dump() for p in batch.pitches]
    msg = f"สร้างไอเดียคลิป YouTube สำเร็จ {len(pitches_dict_list)} หัวข้อ"
    return {
        "pitches": pitches_dict_list,
        "messages": [AIMessage(content=msg, name="generate_pitches")],
    }


def gate_node(state: YouTubePitchState) -> Command[Literal["synthesize_notebooklm"]]:
    pitches = state.get("pitches", [])
    instruction = state.get("instruction", "")

    selection = interrupt({
        "type": "youtube_pitch_approval",
        "pitches": pitches,
        "instruction": instruction,
    })

    approved_ids = selection.get("approved_pitch_ids", []) if isinstance(selection, dict) else []

    return Command(
        goto="synthesize_notebooklm",
        update={
            "approved_pitch_ids": approved_ids,
        },
    )


def synthesize_notebooklm_node(state: YouTubePitchState) -> dict:
    approved_ids = set(state.get("approved_pitch_ids", []))
    pitches = state.get("pitches", [])
    candidates = state.get("news_candidates", [])
    macro_str = state.get("macro_baselines", "")

    # Zero-file protection: ถ้าไม่อนุมัติเลยแม้แต่รายการเดียว
    if not approved_ids:
        line = "ไม่มีไอเดียคลิปที่ถูกเลือก อนุมัติ 0 รายการ (ไม่สร้างไฟล์ Briefing Book)"
        return {
            "result_summary": line,
            "messages": [AIMessage(content=line, name="synthesize_notebooklm")],
        }

    messages = []
    summary_lines = []
    today_str = datetime.now().strftime("%Y-%m-%d")

    for p_dict in pitches:
        if not isinstance(p_dict, dict):
            continue
        p_id = p_dict.get("pitch_id", "")
        if p_id not in approved_ids:
            continue

        try:
            pitch_item = YouTubeContentPitchItem.model_validate(p_dict)
            main_title = pitch_item.working_titles[0] if pitch_item.working_titles else "untitled"

            # สังเคราะห์ 7 Sections
            content = synthesize_notebooklm_source(pitch_item, candidates, macro_str)
            # บันทึกลง Vault
            saved_path = save_notebooklm_source(content, main_title, date_str=today_str)

            line = f"✓ บันทึก Briefing Book สำเร็จ: {saved_path} (หัวข้อ: {main_title})"
            summary_lines.append(line)
            messages.append(AIMessage(content=line, name="synthesize_notebooklm"))
        except Exception as e:
            line = f"✗ ล้มเหลวในการสังเคราะห์ไอเดีย {p_id}: {e}"
            summary_lines.append(line)
            messages.append(AIMessage(content=line, name="synthesize_notebooklm"))

    return {
        "result_summary": "\n".join(summary_lines),
        "messages": messages,
    }


def build_youtube_pitch_graph(checkpointer=None):
    builder = StateGraph(YouTubePitchState)
    builder.add_node("fetch_topics", fetch_topics_node)
    builder.add_node("generate_pitches", generate_pitches_node)
    builder.add_node("gate", gate_node)
    builder.add_node("synthesize_notebooklm", synthesize_notebooklm_node)

    builder.add_edge(START, "fetch_topics")
    builder.add_edge("fetch_topics", "generate_pitches")
    builder.add_edge("generate_pitches", "gate")
    builder.add_edge("synthesize_notebooklm", END)

    return builder.compile(checkpointer=checkpointer)
