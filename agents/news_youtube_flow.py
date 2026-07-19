"""กราฟแยกต่างหากสำหรับ human-in-the-loop approval ของข่าว/YouTube — ไม่แตะ agents/manager_agent.py
เลย (ตั้งใจแยกให้ isolated เพื่อไม่เสี่ยงกระทบ pipeline หลักที่ทดสอบแล้วและใช้งานจริงอยู่)

Flow: fetch_news → fetch_youtube → gate (interrupt() รอ user เลือก) → ingest_node (ดึงเนื้อหาเต็ม
เฉพาะรายการที่ approve แล้วบันทึกลง Vault)

เดิม fetch news/youtube + interrupt() รวมอยู่ node เดียว (gate_node) มีปัญหา 2 อย่าง: (1) Command
update ของ node ไม่มี key "messages" เลย ทำให้ _log_manager_messages (api/jobs.py) ข้ามไปเฉยๆ —
ตลอดช่วงที่ fetch RSS จริง (หลายวินาที) ไม่มี log โผล่ให้เห็นเลยสักบรรทัด จอ approve โผล่มาแบบไม่มี
สัญญาณอะไรมาก่อน (เจอจริงจาก live test); (2) node ที่มี interrupt() ถูกเรียกซ้ำจากต้นทุกครั้งที่
resume ทำให้ fetch ต้อง idempotent โดยไม่จำเป็น — แยกเป็นคนละ node แก้ทั้งสองข้อ: แต่ละ fetch node
ส่ง messages update ทำให้เห็น progress real-time ระหว่างทาง และ resume จะ replay แค่ node gate
(มีแค่ interrupt() ล้วนๆ ไม่มี side-effect) เท่านั้น ไม่ fetch ซ้ำอีกต่อไป
"""
from datetime import datetime
from typing import Literal, Optional, TypedDict

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


class NewsYoutubeState(TypedDict, total=False):
    messages: list
    scope: str
    news_candidates: list[dict]
    youtube_candidates: list[dict]
    approved_news_links: list[str]
    approved_youtube_links: list[str]
    result_summary: str


def fetch_news_node(state: NewsYoutubeState) -> dict:
    from tools.macro.news_radar import get_news_candidates

    scope = state.get("scope", "both")
    if scope not in ("news", "both"):
        # ไม่ใส่ key "messages" ตอนข้าม — _log_manager_messages (api/jobs.py) จะไม่ log
        # อะไรเลย ทำให้ terminal ไม่โผล่ step "ดึงข่าว" ให้เห็นสำหรับการ์ดที่ scope ไม่เกี่ยวข้อง
        # (เจอจริงจาก live test: การ์ดดึงคลิป YouTube ล้วนๆ ขึ้น step "ดึงข่าว" มาด้วยแบบงงๆ)
        return {"news_candidates": []}
    candidates = get_news_candidates()
    return {
        "news_candidates": candidates,
        "messages": [AIMessage(content=f"พบข่าว {len(candidates)} รายการ", name="fetch_news")],
    }


def fetch_youtube_node(state: NewsYoutubeState) -> dict:
    from tools.knowledge.youtube_monitor import get_youtube_candidates

    scope = state.get("scope", "both")
    if scope not in ("youtube", "both"):
        # เหตุผลเดียวกับ fetch_news_node ด้านบน — การ์ดดึงข่าวล้วนๆ ไม่ควรเห็น step
        # "ดึงคลิป YouTube" โผล่มาเลย
        return {"youtube_candidates": []}
    candidates = get_youtube_candidates()
    return {
        "youtube_candidates": candidates,
        "messages": [AIMessage(content=f"พบคลิป YouTube {len(candidates)} รายการ", name="fetch_youtube")],
    }


def gate_node(state: NewsYoutubeState) -> Command[Literal["ingest"]]:
    news_candidates = state.get("news_candidates", [])
    youtube_candidates = state.get("youtube_candidates", [])

    selection = interrupt({
        "type": "news_youtube_approval",
        "news_candidates": news_candidates,
        "youtube_candidates": youtube_candidates,
    })

    approved_news = selection.get("approved_news_links", []) if isinstance(selection, dict) else []
    approved_youtube = selection.get("approved_youtube_links", []) if isinstance(selection, dict) else []

    return Command(
        goto="ingest",
        update={
            "approved_news_links": approved_news,
            "approved_youtube_links": approved_youtube,
        },
    )


def _save_ingested_content(content: str) -> str:
    from tools.archivist.core import _sanitize_filename
    from tools.archivist.parser import extract_yaml_frontmatter_value
    from tools.archivist.writer import write_raw_markdown

    entity_type = extract_yaml_frontmatter_value(content, "entity_type") or ""
    title = extract_yaml_frontmatter_value(content, "title") or datetime.now().strftime("news_youtube_%Y%m%d_%H%M%S")
    filename = _sanitize_filename(title)

    folder_map = {
        "article_note": "30_Knowledge_Base/News",
        "youtube_insight": "30_Knowledge_Base/YouTube_Summaries",
    }
    folder_path = folder_map.get(entity_type, "30_Knowledge_Base/News")
    return write_raw_markdown.invoke({"content": content, "folder_path": folder_path, "filename": filename})


def ingest_node(state: NewsYoutubeState) -> dict:
    from tools.knowledge.article import ingest_article_url
    from tools.knowledge.youtube import ingest_youtube_transcript

    # ส่ง 1 AIMessage ต่อ 1 ลิงก์ (แทนที่จะ join เป็นก้อนเดียวส่งทีตอนจบ) เพื่อให้ terminal
    # เห็น progress real-time ทีละไฟล์ระหว่างที่ fetch/save จริง (เจอจริงจาก live test: เดิม
    # เงียบสนิทตลอดจนจบ) ต้องคู่กับ api/jobs.py::_log_manager_messages ที่แก้ให้ log ทุกข้อความ
    # ใน list ไม่ใช่แค่ตัวสุดท้าย — prefix ✓/✗ ให้ scan เจอง่ายว่าอันไหนสำเร็จ/ล้มเหลว
    messages: list[AIMessage] = []
    summary_lines: list[str] = []

    def _ingest_one(kind: str, link: str, ingest_fn) -> None:
        try:
            content = ingest_fn.invoke({"url": link})
            if content.startswith("ERROR:"):
                # tool เองจับ error ไว้แล้วคืนเป็นข้อความ (ไม่ raise) — ต้องเช็คก่อนเซฟ
                # ไม่งั้นข้อความ error จะถูกบันทึกเป็นไฟล์ขยะลง Vault (เจอจริงจาก live test:
                # LLM RESOURCE_EXHAUSTED ทำให้ tool คืน "ERROR: ..." string มาแทน exception)
                line = f"✗ [{kind}] {link} → ล้มเหลว: {content}"
            else:
                save_result = _save_ingested_content(content)
                line = f"✓ [{kind}] {link} → บันทึกแล้ว: {save_result}"
        except Exception as e:
            line = f"✗ [{kind}] {link} → ล้มเหลว: {e}"
        summary_lines.append(line)
        messages.append(AIMessage(content=line, name="ingest"))

    for link in state.get("approved_news_links", []):
        _ingest_one("ข่าว", link, ingest_article_url)

    for link in state.get("approved_youtube_links", []):
        _ingest_one("YouTube", link, ingest_youtube_transcript)

    if not messages:
        line = "ไม่มีรายการที่ถูกเลือก อนุมัติ 0 รายการ"
        summary_lines.append(line)
        messages.append(AIMessage(content=line, name="ingest"))

    from tools.archivist.indexer import flush_index_if_dirty
    flush_index_if_dirty()

    return {
        "result_summary": "\n".join(summary_lines),
        "messages": messages,
    }


def build_news_youtube_graph(checkpointer=None):
    builder = StateGraph(NewsYoutubeState)
    builder.add_node("fetch_news", fetch_news_node)
    builder.add_node("fetch_youtube", fetch_youtube_node)
    builder.add_node("gate", gate_node)
    builder.add_node("ingest", ingest_node)
    builder.add_edge(START, "fetch_news")
    builder.add_edge("fetch_news", "fetch_youtube")
    builder.add_edge("fetch_youtube", "gate")
    builder.add_edge("ingest", END)
    return builder.compile(checkpointer=checkpointer)
