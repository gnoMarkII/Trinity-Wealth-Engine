import os
import time
import uuid
from functools import lru_cache
from typing import Literal, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Command
from pydantic import BaseModel, Field

from agents.archivist_agent import create_archivist
from agents.bookkeeper_agent import create_bookkeeper
from agents.researcher_agent import create_researcher
from agents.macro_analyst_agent import create_macro_analyst
from core.agent_log import log_turn_start, log_manager_plan, log_worker_result, log_system_action, log_routing
from core.llm_factory import FALLBACK_MODEL, detect_provider, get_llm
from core.logger import get_logger
from core.utils import normalize_content

log = get_logger(__name__)

# Single-tier config: ทุก agent ใช้ gemini-3.1-flash-lite-preview เป็น default
# Fallback chain (core/llm_factory.FALLBACK_MODEL) = openai/gpt-oss-120b:free (OpenRouter)
_MANAGER_MODEL = os.getenv("MANAGER_MODEL", "gemini-3.1-flash-lite-preview")
_ROUTER_MODEL = os.getenv("ROUTER_MODEL", _MANAGER_MODEL)
_ARCHIVIST_MODEL = os.getenv("ARCHIVIST_MODEL", "gemini-3.1-flash-lite-preview")
_RESEARCHER_MODEL = os.getenv("RESEARCHER_MODEL", "gemini-3.1-flash-lite-preview")
_BOOKKEEPER_MODEL = os.getenv("BOOKKEEPER_MODEL", "gemini-3.1-flash-lite-preview")
_MACRO_ANALYST_MODEL = os.getenv("MACRO_ANALYST_MODEL", "gemini-3.1-flash-lite-preview")
_ROUTER_HISTORY_LIMIT = 20
_MAX_REPLAN = 2


class RouteMeta(TypedDict, total=False):
    """Routing metadata เพิ่มใน state ของ graph — แทน string prefix แบบเดิม"""
    source: str   # "manager" | "researcher"
    target: str   # "archivist" | "researcher" | "bookkeeper" | "user"
    save_to_vault: bool
    worker_started_at: float | None


class AgentState(MessagesState):
    """MessagesState + routing metadata + pending multi-task queue + replan safety"""
    route_meta: RouteMeta
    task_queue: list[dict]
    replan_count: int
    turn_id: str


# ROUTER_PROMPT: กฎการแตกคำขอเป็น tasks + เลือก worker ของแต่ละ task + กฎเหล็ก
ROUTER_PROMPT = """คุณคือ The Manager ผู้จัดการกองทุนส่วนตัว หน้าที่: แตกคำขอของผู้ใช้ออกเป็น "รายการงาน" (tasks) แล้วส่งให้ worker ที่เหมาะสมตามลำดับ

[แตกงาน — สำคัญที่สุด]
- ผู้ใช้พิมพ์หลายคำสั่งใน turn เดียวได้ → แตกเป็นหลาย task เรียงตามลำดับที่ควรทำ (1 task = 1 worker call)
- สำหรับงานดึงข้อมูล (researcher) หากดึงข้อมูลคนละประเภทหรือคนละตลาด/พื้นที่ร่วมกัน (เช่น ทั้งดึงข้อมูลเศรษฐกิจมหภาคของไทย ของสหรัฐฯ และของภูมิภาค) → ให้แตกเป็นหลาย task แยกกันส่งให้ researcher (1 task ต่อ 1 ชิ้นงาน เช่น ดึงข้อมูลไทย, ดึงข้อมูลสหรัฐฯ, ดึงข้อมูลภูมิภาค) เพื่อให้ Researcher เรียกใช้เครื่องมือได้ครบถ้วน (ห้ามรวบเป็น task เดียว)
- คำสั่งเดียว → tasks มี 1 ชิ้น
- ตอบเองได้จากบริบทเดิม หรือคำถามทั่วไปที่ไม่ต้องดึง/แก้ข้อมูล → tasks = [] (ว่าง) แล้วใส่คำตอบใน response_text
- กฎ XOR: ถ้ามี task อย่างน้อย 1 ชิ้น → response_text ต้องว่าง (ปล่อยให้ worker เป็นคนตอบ) ห้ามใส่พร้อมกัน
  หากผู้ใช้คุยเล่น/ถามทั่วไปพ่วงท้ายคำสั่งจริง ให้รวบเข้าไปใน instruction ของ task ที่เกี่ยวข้อง หรือมองข้ามส่วนนั้น

[ลำดับงาน]
- โดยทั่วไปเรียงตามที่ผู้ใช้พิมพ์
- ถ้า turn เดียวมีทั้งงานพอร์ต (bookkeeper) และงานบันทึก/ความรู้ (archivist) → เรียง bookkeeper ก่อน archivist เสมอ

[เลือก target ของแต่ละ task ตามประเภทคำขอ]
- "researcher" → ดึงข้อมูลภายนอก: macro/sector/regional/FRED economic data, TH/US stocks (fundamentals, financials, momentum, consensus, news, ESG — Researcher จะเลือก market='TH'|'US' ตาม ticker), YouTube clip summaries, บทความจาก URL (เว็บ/สื่อการเงิน/บล็อก), ไฟล์ PDF (รายงาน/งบการเงิน/บทวิเคราะห์)
  (Researcher จะส่งผลให้ Archivist บันทึกอัตโนมัติ — ไม่ต้องส่ง archivist ซ้ำ)
- "archivist" → อ่าน/ค้นหาข้อมูลใน Vault (ข้อมูลความรู้/Entity ที่บันทึกไว้, สุขภาพ Vault, semantic search), บันทึก book note หรือ knowledge ที่ผู้ใช้พิมพ์/วางมาเองโดยตรง (entity_type: book_note)
- "macro_analyst" → วิเคราะห์สภาวะเศรษฐกิจมหภาคและการจัดสรรสินทรัพย์: คำนวณคะแนนสภาวะเศรษฐกิจของแต่ละประเทศ/ภูมิภาค ประเมินจัดทำ Matrix และรายงานสรุปสภาวะเศรษฐกิจ
- "bookkeeper" → พอร์ตการลงทุนจริง:
  - ธุรกรรม: ซื้อ/ขาย/ฝาก/ถอน (THB/USD แยก pot), dividend/interest/rental, FX update, แก้ไข holding ที่ผิด
  - สถานะ/รายงาน: NAV, P/L, holdings, เงินสด, allocation % (asset_type/currency), performance trend ย้อนหลัง
  - Watchlist: เพิ่ม/ลบ/ดู สินทรัพย์ที่จับตา (ยังไม่ซื้อ)
  - Journal: ทบทวนบันทึก mistakes / เหตุผลซื้อขายย้อนหลัง
  - เป้าหมายทางการเงิน (Goals): ตั้ง/ลบ/ดู progress เป้าหมาย เช่น NAV เป้าหมาย, เงินสดสำรอง, passive income ต่อปี
    คีย์เวิร์ด: "ตั้งเป้าหมาย", "เป้าหมายพอร์ต", "อยากมี NAV", "เงินฉุกเฉิน [ตัวเลข]", "passive income เป้า", "ดูเป้าหมาย", "progress เป้าหมาย"

[แยกให้ชัด]
- Archivist = ความรู้/Entity (เช่น "เล่าเรื่องบริษัท PTT")
- Bookkeeper = ตัวเลขพอร์ตจริง (เช่น "ซื้อ AAPL 10 หุ้น", "เงินสดเหลือเท่าไหร่")

[วิธีกรอกแต่ละ task]
- target = worker ที่รับงาน (archivist/researcher/bookkeeper)
- instruction = คำสั่งของ task นั้น กระชับ ชัดเจน
- save_to_vault = ใช้กับ researcher เท่านั้น: True (ค่าเริ่มต้น) = Archivist เซฟอัตโนมัติ,
  False = ผู้ใช้บอกชัดเจนไม่ต้องเซฟ ('ดูเฉยๆ', 'ไม่ต้องเซฟ', 'แค่อยากรู้', 'เช็คเฉยๆ')

[กฎเหล็ก]
- ห้ามนำข้อมูลดิบที่ Researcher ดึงมาสรุป/วิเคราะห์ซ้ำในคำตอบ
- ห้ามมั่วข้อมูล — ต้องดึงจากแหล่งที่เชื่อถือได้
- ห้ามตอบตัวเลขพอร์ตจากความจำเก่า — ต้องให้ Bookkeeper อ่านสถานะปัจจุบันก่อนเสมอ

[กฎ Re-plan — เมื่อเห็น [REPLAN]]
- ข้อความที่ขึ้นต้นด้วย [REPLAN] แสดงว่างานก่อนหน้าล้มเหลว
- ให้วิเคราะห์ว่า Error เกิดจากอะไร แล้ววางแผนงานใหม่ที่แก้ต้นเหตุ
- ตัวอย่าง: ถ้า Macro Analyst บอก "ไม่พบไฟล์ Global_Macro_Snapshot" → สั่ง Researcher ดึงข้อมูล Global Macro ก่อน แล้วค่อยส่ง Macro Analyst อีกครั้ง
- ห้ามทำซ้ำแผนเดิมที่ล้มเหลว — ต้องเปลี่ยนแนวทาง"""


class WorkerTask(BaseModel):
    """งานย่อย 1 ชิ้น ที่ route ไปยัง worker หนึ่งตัว"""
    target: Literal["archivist", "researcher", "bookkeeper", "macro_analyst"]
    instruction: str = Field(
        description="คำสั่งสำหรับ worker ตัวนี้ — กระชับ ชัดเจน ตัดคำนำหน้า/คำพ่วงที่ไม่เกี่ยวออก"
    )
    save_to_vault: bool = Field(
        default=True,
        description="ใช้กับ target == 'researcher' เท่านั้น — True = ส่งผลให้ Archivist บันทึก, "
                    "False = แค่ดึงข้อมูลแล้วแสดง ไม่เซฟ "
                    "(เลือก False เมื่อผู้ใช้บอกชัดเจน เช่น 'ดูเฉยๆ', 'ไม่ต้องเซฟ', 'แค่อยากรู้')",
    )


class RouterDecision(BaseModel):
    """แผนการทำงานของ turn นี้ — แตกคำขอ user เป็นรายการ task ตามลำดับ"""
    tasks: list[WorkerTask] = Field(
        default_factory=list,
        description="รายการงานเรียงตามลำดับที่ต้องทำ (1 task = 1 worker call) — "
                    "ว่าง [] เมื่อ Manager ตอบ user ได้เองโดยไม่ต้องเรียก worker",
    )
    response_text: str = Field(
        default="",
        description="คำตอบที่ส่งกลับให้ผู้ใช้โดยตรง (ภาษาไทย กระชับ) — ใช้เมื่อ tasks ว่างเท่านั้น "
                    "ไม่นำข้อมูลดิบจาก Researcher มาวิเคราะห์ซ้ำ",
    )


def _is_worker_error(messages: list) -> str | None:
    """ตรวจสอบว่า Worker ล่าสุดส่ง Error กลับมาหรือไม่

    Returns:
        ข้อความ Error ถ้าพบ, None ถ้าสำเร็จ
    """
    if not messages:
        return None
    last = messages[-1]
    if not isinstance(last, AIMessage):
        return None
    content = normalize_content(last.content).strip()
    if content.startswith("Error:"):
        return content
    return None


def _msg_role(m) -> str:
    return "human" if isinstance(m, HumanMessage) else "assistant"


def _has_researcher_frontmatter(text: str) -> bool:
    """ตรวจว่า Researcher reply เป็น Markdown ที่มี YAML frontmatter ของจริง (entity_type:)"""
    stripped = text.lstrip()
    if not stripped.startswith("---"):
        return False
    head = "\n".join(stripped.splitlines()[:30])
    return "entity_type:" in head


@lru_cache(maxsize=1)
def _get_archivist_graph():
    provider = detect_provider(_ARCHIVIST_MODEL)
    return create_archivist(
        get_llm(provider=provider, model_name=_ARCHIVIST_MODEL, use_fallback=True)
    )


@lru_cache(maxsize=1)
def _get_researcher_graph():
    provider = detect_provider(_RESEARCHER_MODEL)
    return create_researcher(
        get_llm(provider=provider, model_name=_RESEARCHER_MODEL, use_fallback=True)
    )


@lru_cache(maxsize=1)
def _get_bookkeeper_graph():
    provider = detect_provider(_BOOKKEEPER_MODEL)
    return create_bookkeeper(
        get_llm(provider=provider, model_name=_BOOKKEEPER_MODEL, use_fallback=True)
    )


@lru_cache(maxsize=1)
def _get_macro_analyst_graph():
    provider = detect_provider(_MACRO_ANALYST_MODEL)
    return create_macro_analyst(
        get_llm(provider=provider, model_name=_MACRO_ANALYST_MODEL, use_fallback=True)
    )


@lru_cache(maxsize=1)
def _get_router_model():
    # router_model ต้องใช้ with_structured_output ซึ่งไม่มีบน RunnableWithFallbacks
    # → wrap primary + fallback แยกชั้น
    provider = detect_provider(_ROUTER_MODEL)
    primary = get_llm(provider=provider, model_name=_ROUTER_MODEL)
    structured = primary.with_structured_output(RouterDecision)

    if _ROUTER_MODEL == FALLBACK_MODEL:
        return structured  # ป้องกัน wrap ตัวเองเป็น fallback

    fallback_provider = detect_provider(FALLBACK_MODEL)
    fallback = get_llm(provider=fallback_provider, model_name=FALLBACK_MODEL)
    return structured.with_fallbacks([fallback.with_structured_output(RouterDecision)])


def extract_worker_reply(messages: list) -> str:
    start_idx = 0
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], HumanMessage) and getattr(messages[i], "name", None) == "manager":
            start_idx = i
            break
    recent_msgs = messages[start_idx:]
    tool_msg = next((m for m in reversed(recent_msgs) if isinstance(m, ToolMessage)), None)
    return normalize_content(tool_msg.content if tool_msg is not None else recent_msgs[-1].content)


def _get_elapsed(meta: dict) -> float | None:
    """คำนวณ elapsed time แบบปลอดภัย ทนต่อ LangGraph checkpoint resume"""
    started_at = meta.get("worker_started_at")
    if started_at is None or not isinstance(started_at, (int, float)):
        return None
    elapsed = time.monotonic() - started_at
    return elapsed if elapsed >= 0 else None


def build_graph(checkpointer=None) -> StateGraph:
    archivist_graph = _get_archivist_graph()
    researcher_graph = _get_researcher_graph()
    bookkeeper_graph = _get_bookkeeper_graph()
    macro_analyst_graph = _get_macro_analyst_graph()
    router_model = _get_router_model()

    def supervisor_node(state: AgentState) -> Command[Literal["prepare_archivist", "researcher", "bookkeeper", "macro_analyst", "__end__"]]:
        messages = state["messages"]
        turn_id = state.get("turn_id")

        if isinstance(messages[-1], HumanMessage) and getattr(messages[-1], "name", None) != "manager":
            # turn ใหม่จาก user
            turn_id = uuid.uuid4().hex[:8]
            log_turn_start(turn_id, messages[-1].content)
            
            router_messages = [
                {"role": "system", "content": ROUTER_PROMPT},
                *[{"role": _msg_role(m), "content": normalize_content(m.content)}
                  for m in messages[-_ROUTER_HISTORY_LIMIT:] if not isinstance(m, ToolMessage)],
            ]
            decision: RouterDecision = router_model.invoke(router_messages)
            log.info("route plan: tasks=%d", len(decision.tasks))

            if not decision.tasks:
                reply = decision.response_text.strip() or "ขอโทษครับ ผมยังไม่เข้าใจคำสั่ง ลองพิมพ์ใหม่อีกครั้งได้ไหมครับ"
                log_worker_result(turn_id, "manager", reply, status="info")
                return Command(
                    goto=END,
                    update={
                        "messages": [AIMessage(content=reply)],
                        "route_meta": {"source": "manager", "target": "user"},
                        "task_queue": [],
                        "replan_count": 0,
                        "turn_id": turn_id,
                    },
                )
            log_manager_plan(turn_id, [t.model_dump() for t in decision.tasks])
            queue = [t.model_dump() for t in decision.tasks]
            messages_update = []
            replan_count_update = 0
        else:
            # วนกลับจาก worker (last != user original message) → ทำ task ที่เหลือต่อ
            error_msg = _is_worker_error(messages)
            replan_count = state.get("replan_count", 0)

            if error_msg and replan_count < _MAX_REPLAN:
                log.warning("worker error detected, re-planning (attempt %d/%d): %s",
                            replan_count + 1, _MAX_REPLAN, error_msg[:100])
                log_system_action(turn_id, "Re-plan Triggered", error_msg, status="warning")
                
                replan_hint = HumanMessage(
                    content=(
                        f"[REPLAN] งานก่อนหน้าล้มเหลว: {error_msg}\n"
                        "กรุณาวางแผนงานใหม่เพื่อแก้ปัญหา — "
                        "อาจต้องสั่ง Researcher ดึงข้อมูลที่ขาดก่อน แล้วค่อยทำงานต่อ"
                    ),
                    name="manager",
                )
                
                router_messages = [
                    {"role": "system", "content": ROUTER_PROMPT},
                    *[{"role": _msg_role(m), "content": normalize_content(m.content)}
                      for m in messages[-_ROUTER_HISTORY_LIMIT:]
                      if not isinstance(m, ToolMessage)],
                    {"role": "human", "content": replan_hint.content},
                ]
                decision: RouterDecision = router_model.invoke(router_messages)
                log.info("replan result: tasks=%d", len(decision.tasks))
                
                if not decision.tasks:
                    reply = decision.response_text.strip() or error_msg
                    log_worker_result(turn_id, "manager", reply, status="warning")
                    return Command(
                        goto=END,
                        update={
                            "messages": [replan_hint, AIMessage(content=reply)],
                            "route_meta": {"source": "manager", "target": "user"},
                            "task_queue": [],
                            "replan_count": replan_count + 1,
                            "turn_id": turn_id,
                        },
                    )
                log_manager_plan(turn_id, [t.model_dump() for t in decision.tasks])
                queue = [t.model_dump() for t in decision.tasks]
                messages_update = [replan_hint]
                replan_count_update = replan_count + 1
            elif error_msg:
                log.error("max replan reached, returning error to user: %s", error_msg[:100])
                log_system_action(turn_id, "Re-plan Exhausted", error_msg, status="failure")
                return Command(
                    goto=END,
                    update={
                        "messages": [AIMessage(content=f"ขออภัยครับ ระบบพยายามแก้ปัญหาแล้วแต่ยังไม่สำเร็จ: {error_msg}")],
                        "route_meta": {"source": "manager", "target": "user"},
                        "task_queue": [],
                        "replan_count": replan_count,
                        "turn_id": turn_id,
                    },
                )
            else:
                queue = state.get("task_queue") or []
                messages_update = []
                replan_count_update = replan_count

        if not queue:
            return Command(
                goto=END,
                update={"replan_count": 0, "turn_id": turn_id}
            )

        task, rest = queue[0], queue[1:]
        target = task["target"]
        meta: RouteMeta = {"source": "manager", "target": target}
        if target == "researcher":
            meta["save_to_vault"] = task.get("save_to_vault", True)
        
        meta["worker_started_at"] = time.monotonic()
        goto_target = "prepare_archivist" if target == "archivist" else target
        
        return Command(
            goto=goto_target,
            update={
                "messages": messages_update + [HumanMessage(content=task["instruction"], name="manager")],
                "route_meta": meta,
                "task_queue": rest,
                "replan_count": replan_count_update,
                "turn_id": turn_id,
            },
        )

    def prepare_archivist_node(state: AgentState) -> Command[Literal["archivist"]]:
        meta = state.get("route_meta") or {}
        
        if meta.get("source") == "researcher" or meta.get("source") == "macro_analyst":
            last_msg = extract_worker_reply(state["messages"])
            task = f"บันทึกข้อมูลดิบต่อไปนี้ลง Vault ทันที\n\n[ข้อมูลดิบ]\n{last_msg}"
        else:
            last_msg = normalize_content(state["messages"][-1].content) # This is the manager's instruction
            msgs = state["messages"]
            last_human_idx = next(
                (i for i in range(len(msgs) - 1, -1, -1) if isinstance(msgs[i], HumanMessage) and getattr(msgs[i], "name", None) != "manager"),
                -1,
            )
            raw_user = normalize_content(msgs[last_human_idx].content) if last_human_idx >= 0 else ""
            mid_drain = any(
                isinstance(m, AIMessage) and normalize_content(m.content).strip()
                for m in msgs[last_human_idx + 1:-1]
            )
            if not mid_drain and _has_researcher_frontmatter(raw_user) and len(raw_user) > len(last_msg) * 2:
                task = f"บันทึกข้อมูลดิบต่อไปนี้ลง Vault ทันที\n\n[ข้อมูลดิบ]\n{raw_user}"
            else:
                task = last_msg

        log.info("prepare_archivist: source=%s len=%d", meta.get("source"), len(task))
        return Command(
            goto="archivist",
            update={
                "messages": [HumanMessage(content=task, name="manager")]
            }
        )

    def post_archivist_node(state: AgentState) -> Command[Literal["supervisor"]]:
        from tools.archivist.indexer import flush_index_if_dirty
        archivist_reply = extract_worker_reply(state["messages"])
        flush_index_if_dirty()
        
        turn_id = state.get("turn_id", "unknown")
        elapsed = _get_elapsed(state.get("route_meta") or {})
        log_worker_result(turn_id, "archivist", archivist_reply, status="success", elapsed_sec=elapsed)
        
        return Command(
            goto="supervisor",
            update={
                "messages": [AIMessage(content=archivist_reply)],
                "route_meta": {"source": "archivist", "target": "user"},
                "turn_id": turn_id,
            }
        )

    def post_researcher_node(state: AgentState) -> Command[Literal["supervisor", "prepare_archivist"]]:
        meta = state.get("route_meta") or {}
        save_to_vault = meta.get("save_to_vault", True)
        researcher_reply = extract_worker_reply(state["messages"])
        turn_id = state.get("turn_id", "unknown")
        elapsed = _get_elapsed(meta)

        if not _has_researcher_frontmatter(researcher_reply):
            log_worker_result(turn_id, "researcher", researcher_reply, status="warning", elapsed_sec=elapsed)
            return Command(
                goto="supervisor", 
                update={
                    "messages": [AIMessage(content=researcher_reply)],
                    "route_meta": {"source": "researcher", "target": "user"},
                    "turn_id": turn_id,
                }
            )

        if not save_to_vault:
            log_worker_result(turn_id, "researcher", researcher_reply, status="info", elapsed_sec=elapsed)
            return Command(
                goto="supervisor", 
                update={
                    "messages": [AIMessage(content=researcher_reply)],
                    "route_meta": {"source": "researcher", "target": "user"},
                    "turn_id": turn_id,
                }
            )

        log_worker_result(turn_id, "researcher", researcher_reply, status="success", elapsed_sec=elapsed)
        return Command(
            goto="prepare_archivist", 
            update={
                "messages": [AIMessage(content=researcher_reply)],
                "route_meta": {"source": "researcher", "target": "archivist", "worker_started_at": time.monotonic()},
                "turn_id": turn_id,
            }
        )

    def post_bookkeeper_node(state: AgentState) -> Command[Literal["supervisor"]]:
        bookkeeper_reply = extract_worker_reply(state["messages"])
        turn_id = state.get("turn_id", "unknown")
        elapsed = _get_elapsed(state.get("route_meta") or {})
        
        log_worker_result(turn_id, "bookkeeper", bookkeeper_reply, status="success", elapsed_sec=elapsed)
        return Command(
            goto="supervisor", 
            update={
                "messages": [AIMessage(content=bookkeeper_reply)],
                "route_meta": {"source": "bookkeeper", "target": "user"},
                "turn_id": turn_id,
            }
        )

    def post_macro_analyst_node(state: AgentState) -> Command[Literal["supervisor", "prepare_archivist"]]:
        macro_analyst_reply = extract_worker_reply(state["messages"])
        turn_id = state.get("turn_id", "unknown")
        elapsed = _get_elapsed(state.get("route_meta") or {})
        
        if _has_researcher_frontmatter(macro_analyst_reply):
            log_worker_result(turn_id, "macro_analyst", macro_analyst_reply, status="success", elapsed_sec=elapsed)
            return Command(
                goto="prepare_archivist", 
                update={
                    "messages": [AIMessage(content=macro_analyst_reply)],
                    "route_meta": {"source": "macro_analyst", "target": "archivist", "worker_started_at": time.monotonic()},
                    "turn_id": turn_id,
                }
            )
        
        status = "failure" if macro_analyst_reply.strip().startswith("Error:") else "info"
        log_worker_result(turn_id, "macro_analyst", macro_analyst_reply, status=status, elapsed_sec=elapsed)
        return Command(
            goto="supervisor", 
            update={
                "messages": [AIMessage(content=macro_analyst_reply)],
                "route_meta": {"source": "macro_analyst", "target": "user"},
                "turn_id": turn_id,
            }
        )

    builder = StateGraph(AgentState)
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("prepare_archivist", prepare_archivist_node)
    builder.add_node("archivist", archivist_graph)
    builder.add_node("researcher", researcher_graph)
    builder.add_node("bookkeeper", bookkeeper_graph)
    builder.add_node("macro_analyst", macro_analyst_graph)
    
    builder.add_node("post_archivist", post_archivist_node)
    builder.add_node("post_researcher", post_researcher_node)
    builder.add_node("post_bookkeeper", post_bookkeeper_node)
    builder.add_node("post_macro_analyst", post_macro_analyst_node)

    builder.add_edge(START, "supervisor")
    
    builder.add_edge("archivist", "post_archivist")
    builder.add_edge("researcher", "post_researcher")
    builder.add_edge("bookkeeper", "post_bookkeeper")
    builder.add_edge("macro_analyst", "post_macro_analyst")

    return builder.compile(checkpointer=checkpointer)