import os
import re
import time
import uuid
import json
from datetime import datetime, timezone
from functools import lru_cache
from typing import Literal, TypedDict, Optional, Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables.config import RunnableConfig
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Command
from pydantic import BaseModel, Field
from application.knowledge.note_write_ports import KnowledgeNoteWritePort
from tools.archivist.composition import build_knowledge_note_writer

from agents.archivist_agent import create_archivist
from agents.bookkeeper_agent import create_bookkeeper
from agents.macro_quant_agent import create_macro_quant
from agents.macro_economist_agent import create_macro_economist
from agents.strategic_allocator import invoke_strategic_allocator
from agents.equity_quant_agent import create_equity_quant
from agents.equity_narrative_agent import create_equity_narrative
from agents.equity_synthesizer import invoke_equity_synthesizer
from schemas.macro_schemas import MacroStrategyDirection, MarketObservable
from schemas.micro_quant_schemas import QuantSignals, EquitySentimentContext, MicroQuantOutput
from tools.macro.report_formatter import format_macro_strategy_report, write_strategy_json_sidecar
from tools.macro.ingest import fetch_and_save_macro_snapshots
from tools.market.equity_report_formatter import format_equity_analysis_report
from tools.market.quant_history import save_equity_quant_snapshot
from tools.market.equity_sidecar import write_equity_sidecar
from core.agent_log import log_turn_start, log_manager_plan, log_worker_result, log_system_action, log_routing
from core.llm_factory import FALLBACK_MODEL, detect_provider, get_llm
from core.logger import get_logger
from core.model_registry import get_model_name
from core.prompt_harness import get_harness
from core.utils import normalize_content

log = get_logger(__name__)


def _msg_role(m) -> str:
    return "human" if isinstance(m, HumanMessage) else "assistant"


# Single-tier config: ทุก agent ใช้ gemini-3.1-flash-lite-preview เป็น default
# Fallback chain (core/llm_factory.FALLBACK_MODEL) = openai/gpt-oss-120b:free (OpenRouter)
# ค่า default ของแต่ละ slot มาจาก core.model_registry (single source of truth ที่ /api/debug/models
# อ่านด้วย) — ยกเว้น _ROUTER_MODEL ที่คง behavior เดิมไว้ตั้งใจ: ถ้าไม่ตั้ง ROUTER_MODEL เอง จะ
# chain ไปตามค่า _MANAGER_MODEL ที่ resolve แล้ว (ไม่ใช่ default คงที่ของ registry) เพื่อให้การ
# override MANAGER_MODEL อย่างเดียวยังส่งผลถึง router ด้วยเหมือนเดิม
_MANAGER_MODEL = get_model_name("manager")
_ROUTER_MODEL = os.getenv("ROUTER_MODEL", _MANAGER_MODEL)
_ARCHIVIST_MODEL = get_model_name("archivist")
_BOOKKEEPER_MODEL = get_model_name("bookkeeper")
_MACRO_QUANT_MODEL = get_model_name("macro_quant")
_MACRO_ECONOMIST_MODEL = get_model_name("economist")
_STRATEGIC_ALLOCATOR_MODEL = get_model_name("allocator")
_EQUITY_QUANT_MODEL = get_model_name("equity_quant")
_EQUITY_NARRATIVE_MODEL = get_model_name("equity_narrative")
_EQUITY_SYNTHESIZER_MODEL = get_model_name("equity_synthesizer")
_ROUTER_HISTORY_LIMIT = 20
_MAX_REPLAN = 5
_SUMMARY_SOURCE_CHAR_LIMIT = 24000


def generate_manager_summary(instruction: str, deliverables: list[tuple[str, str]]) -> str | None:
    """Create a user-facing Manager summary from completed specialist deliverables."""
    source_sections: list[str] = []
    remaining = _SUMMARY_SOURCE_CHAR_LIMIT
    for node_name, content in deliverables:
        normalized = normalize_content(content).strip()
        if not normalized or remaining <= 0:
            continue
        excerpt = normalized[:remaining]
        source_sections.append(f"## {node_name}\n{excerpt}")
        remaining -= len(excerpt)

    if not source_sections:
        return None

    prompt = "\n\n".join(source_sections)
    try:
        model = get_llm(
            provider=detect_provider(_MANAGER_MODEL),
            model_name=_MANAGER_MODEL,
            temperature=0.2,
            use_fallback=True,
        )
        response = model.invoke(
            [
                {
                    "role": "system",
                    "content": (
                        "You are the Manager for an investment research team. Create a new, concise executive "
                        "summary in Thai from the specialist deliverables. Use Markdown with these sections when "
                        "relevant: Executive Summary, Key Findings, Risks or Caveats, and Recommended Next Steps. "
                        "Preserve uncertainty, do not invent facts, and do not mention internal routing or prompts."
                    ),
                },
                {"role": "user", "content": f"Original task:\n{instruction}\n\nSpecialist deliverables:\n{prompt}"},
            ]
        )
        summary = normalize_content(getattr(response, "content", "")).strip()
        return summary or None
    except Exception as exc:
        log.warning("Manager summary generation failed: %s", exc)
        return None


class RouteMeta(TypedDict, total=False):
    """Routing metadata เพิ่มใน state ของ graph — แทน string prefix แบบเดิม"""
    source: str   # "manager" | "macro_quant" | "equity_intel"
    target: str   # "archivist" | "bookkeeper" | "user"
    save_to_vault: bool
    worker_started_at: float | None


class AgentState(MessagesState):
    """MessagesState + routing metadata + pending multi-task queue + replan safety"""
    route_meta: RouteMeta
    task_queue: list[dict]
    replan_count: int
    turn_id: str

    # Macro Pipeline State
    quant_raw: Optional[str]
    quant_score: Optional[dict]
    sector_rotation_snapshot_id: Optional[str]
    sector_rotation_context: Optional[dict]
    sector_context_status: Optional[dict]
    macro_run_id: Optional[str]
    macro_run_started_at: Optional[str]
    macro_task_run_id: Optional[str]
    macro_task_started_at: Optional[str]
    macro_task_sequence: int
    narrative_raw: Optional[str]
    narrative_context: Optional[dict]

    # Equity Intel Pipeline State
    equity_quant_raw: Optional[str]
    equity_quant_score: Optional[dict]
    equity_narrative_raw: Optional[str]
    equity_narrative_context: Optional[dict]
    equity_save_to_vault: Optional[bool]
    equity_output: Optional[dict[str, Any]]
    equity_news_raw: Optional[str]


# ROUTER_PROMPT ถูกย้ายไปที่ prompts/skills/manager/SKILL.md ผ่านระบบ PromptHarness


class WorkerTask(BaseModel):
    """งานย่อย 1 ชิ้น ที่ route ไปยัง worker หนึ่งตัว"""
    target: Literal["archivist", "bookkeeper", "macro_intel", "equity_intel"]
    instruction: str = Field(
        description="คำสั่งสำหรับ worker ตัวนี้ — กระชับ ชัดเจน ตัดคำนำหน้า/คำพ่วงที่ไม่เกี่ยวออก"
    )
    save_to_vault: bool = Field(
        default=True,
        description="ใช้กับ target == 'equity_intel' เท่านั้น — True = ส่งผลให้ Archivist บันทึก, "
                    "False = แค่วิเคราะห์แล้วแสดง ไม่เซฟ "
                    "(เลือก False เมื่อผู้ใช้บอกชัดเจน เช่น 'ดูเฉยๆ', 'ไม่ต้องเซฟ', 'แค่อยากรู้')",
    )

    from pydantic import model_validator
    @model_validator(mode='before')
    @classmethod
    def alias_target(cls, values):
        if isinstance(values, dict) and values.get("target") == "macro_analyst":
            values["target"] = "macro_intel"
        return values


class RouterDecision(BaseModel):
    """แผนการทำงานของ turn นี้ — แตกคำขอ user เป็นรายการ task ตามลำดับ"""
    tasks: list[WorkerTask] = Field(
        default_factory=list,
        description="รายการงานเรียงตามลำดับที่ต้องทำ (1 task = 1 worker call) — "
                    "ว่าง [] เมื่อ Manager ตอบ user ได้เองโดยไม่ต้องเรียก worker",
    )
    response_text: str = Field(
        default="",
        description="คำตอบที่ส่งกลับให้ผู้ใช้โดยตรง (ภาษาไทย กระชับ) — ใช้เมื่อ tasks ว่างเท่านั้น",
    )


def _is_worker_error(messages: list) -> str | None:
    if not messages:
        return None
    last = messages[-1]
    if not isinstance(last, AIMessage):
        return None
    content = normalize_content(last.content).strip()
    if content.startswith("Error:") or content.startswith("Error "):
        return content
    return None


def _msg_role(m) -> str:
    return "human" if isinstance(m, HumanMessage) else "assistant"


def _has_entity_frontmatter(text: str) -> bool:
    stripped = text.lstrip()
    if not stripped.startswith("---"):
        return False
    head = "\n".join(stripped.splitlines()[:30])
    return "entity_type:" in head


@lru_cache(maxsize=1)
def _get_archivist_graph():
    provider = detect_provider(_ARCHIVIST_MODEL)
    return create_archivist(get_llm(provider=provider, model_name=_ARCHIVIST_MODEL, use_fallback=True))


@lru_cache(maxsize=1)
def _get_bookkeeper_graph():
    provider = detect_provider(_BOOKKEEPER_MODEL)
    return create_bookkeeper(get_llm(provider=provider, model_name=_BOOKKEEPER_MODEL, use_fallback=True))


@lru_cache(maxsize=1)
def _get_macro_quant_graph():
    provider = detect_provider(_MACRO_QUANT_MODEL)
    return create_macro_quant(get_llm(provider=provider, model_name=_MACRO_QUANT_MODEL, use_fallback=True))

@lru_cache(maxsize=1)
def _get_macro_economist_graph():
    provider = detect_provider(_MACRO_ECONOMIST_MODEL)
    return create_macro_economist(get_llm(provider=provider, model_name=_MACRO_ECONOMIST_MODEL, use_fallback=True))


@lru_cache(maxsize=1)
def _get_equity_quant_graph():
    provider = detect_provider(_EQUITY_QUANT_MODEL)
    return create_equity_quant(get_llm(provider=provider, model_name=_EQUITY_QUANT_MODEL, use_fallback=True))


@lru_cache(maxsize=1)
def _get_equity_narrative_graph():
    provider = detect_provider(_EQUITY_NARRATIVE_MODEL)
    return create_equity_narrative(get_llm(provider=provider, model_name=_EQUITY_NARRATIVE_MODEL, use_fallback=True))


@lru_cache(maxsize=1)
def _get_router_model():
    provider = detect_provider(_ROUTER_MODEL)
    primary = get_llm(provider=provider, model_name=_ROUTER_MODEL)
    structured = primary.with_structured_output(RouterDecision)

    if _ROUTER_MODEL == FALLBACK_MODEL:
        return structured

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
    started_at = meta.get("worker_started_at")
    if started_at is None or not isinstance(started_at, (int, float)):
        return None
    elapsed = time.monotonic() - started_at
    return elapsed if elapsed >= 0 else None


def build_graph(checkpointer=None, note_writer: Optional[KnowledgeNoteWritePort] = None) -> StateGraph:
    note_writer = note_writer or build_knowledge_note_writer()
    archivist_graph = _get_archivist_graph()
    bookkeeper_graph = _get_bookkeeper_graph()
    macro_quant_graph = _get_macro_quant_graph()
    macro_economist_graph = _get_macro_economist_graph()
    equity_quant_graph = _get_equity_quant_graph()
    equity_narrative_graph = _get_equity_narrative_graph()
    router_model = _get_router_model()

    def supervisor_node(state: AgentState, config: RunnableConfig) -> Command[Literal["prepare_archivist", "bookkeeper", "macro_quant", "equity_quant", "__end__"]]:
        messages = state["messages"]
        turn_id = state.get("turn_id")
        macro_run_id = state.get("macro_run_id")
        macro_run_started_at = state.get("macro_run_started_at")
        is_new_turn = False
        macro_retry_pending = False

        if isinstance(messages[-1], HumanMessage) and getattr(messages[-1], "name", None) != "manager":
            is_new_turn = True
            turn_id = uuid.uuid4().hex[:8]
            config_meta = (config or {}).get("metadata", {}) if isinstance(config, dict) else {}
            macro_run_id = str(config_meta.get("job_id") or turn_id)
            macro_run_started_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            log_turn_start(turn_id, messages[-1].content)

            router_messages = [
                {"role": "system", "content": get_harness("manager").get_system_prompt()},
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
                        "macro_run_id": macro_run_id,
                        "macro_run_started_at": macro_run_started_at,
                        "macro_task_run_id": None,
                        "macro_task_started_at": None,
                        "macro_task_sequence": 0,
                        "sector_rotation_snapshot_id": None,
                        "sector_rotation_context": None,
                        "sector_context_status": None,
                        "quant_raw": None,
                        "quant_score": None,
                        "narrative_raw": None,
                        "narrative_context": None,
                    },
                )
            log_manager_plan(turn_id, [t.model_dump() for t in decision.tasks])
            queue = [t.model_dump() for t in decision.tasks]
            messages_update = []
            replan_count_update = 0
        else:
            error_msg = _is_worker_error(messages)
            replan_count = state.get("replan_count", 0)

            if error_msg and replan_count < _MAX_REPLAN:
                log.warning("worker error detected, re-planning (attempt %d/%d): %s", replan_count + 1, _MAX_REPLAN, error_msg[:100])
                log_system_action(turn_id, "Re-plan Triggered", error_msg, status="warning")

                replan_hint = HumanMessage(
                    content=(
                        f"[REPLAN] งานก่อนหน้าล้มเหลว: {error_msg}\n"
                        "กรุณาวางแผนงานใหม่เพื่อแก้ปัญหาโดยปรับเปลี่ยนแนวทางเดิมที่ล้มเหลว"
                    ),
                    name="manager",
                )

                router_messages = [
                    {"role": "system", "content": get_harness("manager").get_system_prompt()},
                    *[{"role": _msg_role(m), "content": normalize_content(m.content)}
                      for m in messages[-_ROUTER_HISTORY_LIMIT:]
                      if not isinstance(m, ToolMessage)],
                    {"role": "human", "content": replan_hint.content},
                ]
                decision: RouterDecision = router_model.invoke(router_messages)

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
                previous_route = state.get("route_meta") or {}
                macro_retry_pending = bool(
                    state.get("macro_task_run_id")
                    and previous_route.get("target") in {"macro_intel", "macro_economist", "strategic_allocator"}
                )
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
        if target in ("equity_intel",):
            meta["save_to_vault"] = task.get("save_to_vault", True)

        meta["worker_started_at"] = time.monotonic()

        goto_target = target
        if target == "archivist":
            goto_target = "prepare_archivist"
        elif target == "macro_intel":
            goto_target = "macro_quant"
        elif target == "equity_intel":
            goto_target = "equity_quant"

        instruction = task["instruction"]

        # equity_save_to_vault เป็น top-level state field แยกจาก route_meta โดยตั้งใจ — post_equity_quant/
        # post_equity_narrative/equity_synthesizer_node ทับ route_meta ใหม่ทั้งก้อนทุก hop ถ้าเก็บ
        # save_to_vault ไว้ใน route_meta อย่างเดียวจะหายก่อนถึง post_equity_intel_node ที่ปลายทาง
        state_update = {
            "messages": messages_update + [HumanMessage(content=instruction, name="manager")],
            "route_meta": meta,
            "task_queue": rest,
            "replan_count": replan_count_update,
            "turn_id": turn_id,
            "macro_run_id": macro_run_id or turn_id,
            "macro_run_started_at": macro_run_started_at,
        }
        if is_new_turn:
            state_update.update({
                "macro_task_run_id": None,
                "macro_task_started_at": None,
                "macro_task_sequence": 0,
                "sector_rotation_snapshot_id": None,
                "sector_rotation_context": None,
                "sector_context_status": None,
                "quant_raw": None,
                "quant_score": None,
                "narrative_raw": None,
                "narrative_context": None,
            })
        if target == "equity_intel":
            state_update["equity_save_to_vault"] = task.get("save_to_vault", True)
            state_update["equity_output"] = None
            state_update["equity_news_raw"] = None
        if target == "macro_intel":
            from application.macro.run_identity import next_macro_task_run_id
            task_run_id, next_sequence = next_macro_task_run_id(
                macro_run_id or turn_id,
                str(turn_id),
                0 if is_new_turn else int(state.get("macro_task_sequence", 0)),
                retry_run_id=(state.get("macro_task_run_id") if macro_retry_pending else None),
            )
            if macro_retry_pending and state.get("macro_task_run_id"):
                state_update.update({
                    "macro_task_run_id": task_run_id,
                    "macro_task_started_at": state.get("macro_task_started_at"),
                    "macro_task_sequence": next_sequence,
                })
            else:
                state_update.update({
                    "macro_task_sequence": next_sequence,
                    "macro_task_run_id": task_run_id,
                    "macro_task_started_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                    "sector_rotation_snapshot_id": None,
                    "sector_rotation_context": None,
                    "sector_context_status": None,
                    "quant_raw": None,
                    "quant_score": None,
                    "narrative_raw": None,
                    "narrative_context": None,
                })

        return Command(goto=goto_target, update=state_update)

    def prepare_archivist_node(state: AgentState) -> Command[Literal["archivist"]]:
        meta = state.get("route_meta") or {}
        source = meta.get("source")

        if source == "macro_intel":
            last_msg = extract_worker_reply(state["messages"])

            # write_raw_markdown ให้ Archivist LLM ตัดสินใจ filename เอง — พบว่าไม่ deterministic
            # ข้ามรอบรัน (เช่นวันเดียวกันแต่ตัดสินใจสลับ space/underscore) ทำให้เกิดไฟล์ซ้ำแทนที่จะ
            # overwrite เดิม (เจอจริงจาก live test ของ equity_intel) — ดึง title จาก YAML มาบังคับ
            # filename ให้คงที่ทุกรอบแทน ไม่ปล่อยให้ LLM เดาเอง (แก้ที่ instruction ไม่ใช่ tool เพราะ
            # write_raw_markdown ถูกทดสอบไว้แล้วว่าต้อง preserve filename ตามที่ caller ส่งมาตรงๆ
            # ทั้งแบบ space และ underscore สำหรับ entity_type อื่น เปลี่ยน tool เองจะพังเทสต์เดิม)
            from tools.archivist.parser import extract_yaml_frontmatter_value
            title = extract_yaml_frontmatter_value(last_msg, "title")
            filename_directive = (
                f"\nต้องใช้ filename='{title}' เป๊ะๆ ตามนี้เท่านั้น ห้ามดัดแปลง เปลี่ยนช่องว่างเป็นขีดล่าง "
                f"หรือเปลี่ยนรูปแบบใดๆ ทั้งสิ้น (เพื่อให้ re-run วันเดียวกัน overwrite ไฟล์เดิมแทนสร้างไฟล์ซ้ำ)"
                if title else ""
            )
            task = f"คุณต้องเรียกใช้เครื่องมือ write_raw_markdown เพื่อบันทึกข้อมูลดิบต่อไปนี้ลง Vault ทันที ห้ามตอบกลับเป็นข้อความโดยไม่เรียกใช้เครื่องมือ{filename_directive}\n\n[ข้อมูลดิบ]\n{last_msg}"
        elif source == "equity_intel":
            last_msg = extract_worker_reply(state["messages"])
            from tools.archivist.parser import extract_yaml_frontmatter_value
            report_title = extract_yaml_frontmatter_value(last_msg, "title")

            news_raw = state.get("equity_news_raw")
            valid_news = (
                isinstance(news_raw, str)
                and news_raw.lstrip().startswith("---")
                and bool(extract_yaml_frontmatter_value(news_raw, "title"))
            )

            if valid_news:
                news_title = extract_yaml_frontmatter_value(news_raw, "title")
                news_directive = (
                    f"filename='{news_title}' เป๊ะๆ ตามนี้เท่านั้น ห้ามดัดแปลง เปลี่ยนช่องว่างเป็นขีดล่าง หรือเปลี่ยนรูปแบบใดๆ ทั้งสิ้น"
                    if news_title else ""
                )
                report_directive = (
                    f"filename='{report_title}' เป๊ะๆ ตามนี้เท่านั้น ห้ามดัดแปลง เปลี่ยนช่องว่างเป็นขีดล่าง หรือเปลี่ยนรูปแบบใดๆ ทั้งสิ้น"
                    if report_title else ""
                )
                directives = (
                    f"\nคุณต้องเรียกใช้เครื่องมือ write_raw_markdown ทั้งหมด 2 ครั้งเพื่อบันทึกทั้งข่าวหุ้นและบทวิเคราะห์หุ้นดังนี้:\n"
                    f"1. บันทึกข่าวหุ้นล่าสุดด้วย {news_directive} (เพื่อให้ re-run วันเดียวกัน overwrite ไฟล์เดิมแทนสร้างไฟล์ซ้ำ)\n"
                    f"[ข้อมูลดิบ - ข่าวหุ้นล่าสุด]\n{news_raw}\n\n"
                    f"2. บันทึกบทวิเคราะห์หุ้นด้วย {report_directive} (เพื่อให้ re-run วันเดียวกัน overwrite ไฟล์เดิมแทนสร้างไฟล์ซ้ำ)\n"
                    f"[ข้อมูลดิบ - บทวิเคราะห์หุ้น]\n{last_msg}"
                )
                task = f"คุณต้องเรียกใช้เครื่องมือ write_raw_markdown ทั้งหมด 2 ครั้งเพื่อบันทึกข้อมูลดิบลง Vault ทันที ห้ามตอบกลับเป็นข้อความโดยไม่เรียกใช้เครื่องมือ{directives}"
            else:
                filename_directive = (
                    f"\nต้องใช้ filename='{report_title}' เป๊ะๆ ตามนี้เท่านั้น ห้ามดัดแปลง เปลี่ยนช่องว่างเป็นขีดล่าง "
                    f"หรือเปลี่ยนรูปแบบใดๆ ทั้งสิ้น (เพื่อให้ re-run วันเดียวกัน overwrite ไฟล์เดิมแทนสร้างไฟล์ซ้ำ)"
                    if report_title else ""
                )
                task = f"คุณต้องเรียกใช้เครื่องมือ write_raw_markdown เพื่อบันทึกข้อมูลดิบต่อไปนี้ลง Vault ทันที ห้ามตอบกลับเป็นข้อความโดยไม่เรียกใช้เครื่องมือ{filename_directive}\n\n[ข้อมูลดิบ]\n{last_msg}"
        else:
            last_msg = normalize_content(state["messages"][-1].content)
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
            if not mid_drain and _has_entity_frontmatter(raw_user) and len(raw_user) > len(last_msg) * 2:
                task = f"บันทึกข้อมูลดิบต่อไปนี้ลง Vault ทันที\n\n[ข้อมูลดิบ]\n{raw_user}"
            else:
                task = last_msg

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
        
        meta = state.get("route_meta") or {}
        if meta.get("source") == "equity_intel" and state.get("equity_save_to_vault", True):
            if not archivist_reply.lstrip().startswith("Error:"):
                equity_output_data = state.get("equity_output")
                if equity_output_data:
                    try:
                        output = MicroQuantOutput.model_validate(equity_output_data)
                        write_equity_sidecar(output)
                        log.info(f"[{turn_id}] Successfully wrote equity sidecar for {output.ticker}")
                    except Exception as e:
                        log.warning(f"[EQUITY SIDECAR WARN] Failed to write sidecar: {e}")
                else:
                    log.warning(f"[EQUITY SIDECAR WARN] equity_output missing from state, skipping sidecar write.")

        log_worker_result(turn_id, "archivist", archivist_reply, status="success", elapsed_sec=elapsed)

        return Command(
            goto="supervisor",
            update={
                "messages": [AIMessage(content=archivist_reply)],
                "route_meta": {"source": "archivist", "target": "user"},
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

    # --- Macro Pipeline Sequence ---
    def post_quant_node(state: AgentState) -> Command[Literal["macro_economist", "supervisor"]]:
        reply = extract_worker_reply(state["messages"])
        try:
            parsed = json.loads(reply)
            from schemas.macro_schemas import QuantScore
            validated = QuantScore.model_validate(parsed)
            validated_json = validated.model_dump(mode="json")
            instruction = (
                "วิเคราะห์ปัจจัยเชิงคุณภาพ (Narrative) จากข้อมูลใน Vault\n"
                f"ผลลัพธ์จาก Quant Agent (ใช้อ้างอิง): {json.dumps(validated_json, ensure_ascii=False)}"
            )
            turn_id = state.get("turn_id", "unknown")
            elapsed = _get_elapsed(state.get("route_meta") or {})
            log_worker_result(turn_id, "macro_quant", reply, status="success", elapsed_sec=elapsed)
            return Command(
                goto="macro_economist",
                update={
                    "quant_raw": reply,
                    "quant_score": validated_json,
                    "messages": [HumanMessage(content=instruction, name="manager")],
                    "route_meta": {"source": "macro_quant", "target": "macro_economist", "worker_started_at": time.monotonic()},
                }
            )
        except Exception as e:
            # If it's not JSON, it might be an error from the tool
            log_worker_result(state.get("turn_id", "unknown"), "macro_quant", f"Error: {e}\n{reply}", status="failure")
            return Command(
                goto="supervisor",
                update={"messages": [AIMessage(content=f"Error: (Quant) {str(e)} - {reply}")]}
            )

    def post_economist_node(state: AgentState) -> Command[Literal["strategic_allocator", "supervisor"]]:
        reply = extract_worker_reply(state["messages"])
        import re
        reply_clean = re.sub(r'^```(?:json)?\s*|\s*```$', '', reply.strip(), flags=re.MULTILINE)
        try:
            parsed = json.loads(reply_clean)
            from schemas.macro_schemas import NarrativeContext
            validated = NarrativeContext.model_validate(parsed)
            validated_json = validated.model_dump(mode="json")

            # Save deterministic baseline for future pivot comparisons
            from tools.macro.baselines import save_macro_baseline
            save_macro_baseline(validated_json, note_writer=note_writer)

            turn_id = state.get("turn_id", "unknown")
            elapsed = _get_elapsed(state.get("route_meta") or {})
            log_worker_result(turn_id, "macro_economist", reply, status="success", elapsed_sec=elapsed)

            return Command(
                goto="strategic_allocator",
                update={
                    "narrative_raw": reply,
                    "narrative_context": validated_json,
                    "route_meta": {"source": "macro_economist", "target": "strategic_allocator", "worker_started_at": time.monotonic()},
                }
            )
        except Exception as e:
            log_worker_result(state.get("turn_id", "unknown"), "macro_economist", f"Validation fallback: {e}\n{reply}", status="warning")
            from datetime import datetime, timezone
            fallback_dict = {
                "evaluated_at": datetime.now(timezone.utc).isoformat(),
                "dominant_themes": [],
                "market_sentiment": "neutral",
                "tail_risks": [],
                "policy_signals": [],
                "key_narratives_by_region": {},
                "sources_summary": f"Data fetch failed: {str(e)}"
            }
            from schemas.macro_schemas import NarrativeContext
            validated_fallback = NarrativeContext.model_validate(fallback_dict)
            return Command(
                goto="strategic_allocator",
                update={"narrative_raw": reply, "narrative_context": validated_fallback.model_dump(mode="json")}
            )

    def strategic_allocator_node(state: AgentState) -> Command[Literal["post_macro_intel", "supervisor"]]:
        try:
            provider = detect_provider(_STRATEGIC_ALLOCATOR_MODEL)
            model = get_llm(provider=provider, model_name=_STRATEGIC_ALLOCATOR_MODEL)

            quant_data = state.get("quant_score", {}) or {}
            observable_registry: dict[str, MarketObservable] = {}
            if isinstance(quant_data, dict):
                for raw in quant_data.get("market_observables", []) or []:
                    try:
                        obs = raw if isinstance(raw, MarketObservable) else MarketObservable.model_validate(raw)
                    except Exception:
                        continue
                    observable_registry[obs.observable_id] = obs

            valid_observables = []
            unverified_observables = []
            for obs in observable_registry.values():
                item = {
                    "observable_id": obs.observable_id,
                    "asset_bucket": obs.asset_bucket,
                    "region": obs.region,
                    "indicator": obs.indicator,
                    "value": obs.value,
                    "unit": obs.unit,
                    "observed_at": obs.observed_at,
                    "source_file": obs.source_file,
                    "provider": obs.provider,
                    "stale_reason": obs.stale_reason,
                }
                if obs.is_valid:
                    valid_observables.append(item)
                else:
                    unverified_observables.append(item)

            allocator_quant_data = dict(quant_data) if isinstance(quant_data, dict) else {}
            # Keep market-price evidence beside, never inside, macro hard-data observables.
            allocator_quant_data["sector_rotation_context"] = state.get("sector_rotation_context")
            allocator_quant_data["sector_snapshot_id"] = state.get("sector_rotation_snapshot_id")
            allocator_quant_data["sector_context_status"] = state.get("sector_context_status")
            allocator_quant_data["valid_observables"] = valid_observables
            allocator_quant_data["invalid_observables"] = unverified_observables
            allocator_quant_data["regional_data_gaps"] = allocator_quant_data.get("data_gaps", [])
            allocator_quant_data["market_observables_by_validity"] = {
                "VALID INSTITUTIONAL HARD DATA OBSERVABLES (USE FOR HIGH/MEDIUM CONFIDENCE)": valid_observables,
                "UNVERIFIED PROXIES & STALE INDICATORS (DO NOT USE FOR CONFIDENCE / LOW CONFIDENCE ONLY)": unverified_observables,
            }
            evaluated_source_files = {
                obs.source_file for obs in observable_registry.values() if obs.source_file
            }
            evaluated_date = os.getenv("EVAL_DATE") or str(allocator_quant_data.get("evaluated_at", ""))[:10]
            if not evaluated_date or len(evaluated_date) != 10:
                evaluated_date = datetime.now().strftime("%Y-%m-%d")
            evaluated_source_files.add(f"Macro_Baseline_{evaluated_date}.md")
            allocator_quant_data["evaluated_source_files"] = sorted(evaluated_source_files)

            quant_json = json.dumps(allocator_quant_data, ensure_ascii=False)
            narrative_json = json.dumps(state.get("narrative_context", {}), ensure_ascii=False)

            direction = invoke_strategic_allocator(model, quant_json, narrative_json, observable_registry=observable_registry)

            # Deep-merge Thailand Market Stance: Always preserve quantitative numbers from Terminal V2
            th_quant = (allocator_quant_data.get("regions", {}).get("Thailand", {}).get("market_stance") or {})
            th_llm = getattr(direction, "thailand_market_stance", None) or {}
            merged_stance = dict(th_quant)
            # Rationale is from LLM if provided, otherwise synthesized from quant + asset allocations
            ai_rationale = None
            if isinstance(th_llm, dict):
                ai_rationale = th_llm.get("rationale")
                if th_llm.get("observable_refs"):
                    merged_stance["observable_refs"] = th_llm["observable_refs"]
            if not ai_rationale:
                from tools.macro.terminal_observables import synthesize_thai_market_stance_narrative
                thai_assets_in_dir = [
                    a for a in getattr(direction, "asset_allocation", []) or []
                    if getattr(a, "region", "") == "Thailand" or "THB" in getattr(a, "asset_class", "")
                ]
                ai_rationale = synthesize_thai_market_stance_narrative(merged_stance, thai_assets_in_dir)
            merged_stance["rationale"] = ai_rationale
            if not merged_stance.get("observable_refs"):
                merged_stance["observable_refs"] = [
                    oid for oid in [
                        "obs_set_flow_foreign",
                        "obs_set_advance_decline_ratio",
                        "obs_set_valuation_pe",
                        "obs_gta_gold_bar_sell",
                        "obs_diff_us_th_policy_rate_bis",
                    ] if oid in observable_registry
                ]
            direction.thailand_market_stance = merged_stance

            # Guarantee SET Equities (Thailand) in asset allocation if Thai quant data exists
            has_thai_equity = any(
                a.region == "Thailand" and a.asset_bucket == "equities"
                for a in getattr(direction, "asset_allocation", []) or []
            )
            if not has_thai_equity and th_quant:
                from schemas.macro_schemas import AssetAllocationView, AssetStance
                flow_mb = th_quant.get("investor_flow", {}).get("foreign_net_mb")
                pe = th_quant.get("valuation", {}).get("pe_ratio")
                ad = th_quant.get("market_breadth", {}).get("advance_decline_ratio")

                has_eligible_data = (flow_mb is not None) or (pe is not None) or (ad is not None)
                if has_eligible_data:
                    is_bearish = (
                        (flow_mb is not None and flow_mb < 0)
                        or (pe is not None and pe > 17)
                        or (ad is not None and ad < 0.8)
                    )
                    thai_stance = AssetStance.UNDERWEIGHT if is_bearish else AssetStance.NEUTRAL
                    delta_str = "-2% vs benchmark" if thai_stance == AssetStance.UNDERWEIGHT else "0% vs benchmark"

                    support_items = []
                    obs_refs = []
                    rationale_parts = []

                    if flow_mb is not None and "obs_set_flow_foreign" in observable_registry:
                        obs_refs.append("obs_set_flow_foreign")
                        support_items.append(f"SET Foreign Investor Net Flow {flow_mb:,.2f} THB Mil")
                        if flow_mb < 0:
                            rationale_parts.append(f"กระแสเงินทุนต่างชาติขายสุทธิ ({flow_mb:,.1f} ลบ.)")
                        elif flow_mb > 0:
                            rationale_parts.append(f"กระแสเงินทุนต่างชาติซื้อสุทธิ (+{flow_mb:,.1f} ลบ.)")
                        else:
                            rationale_parts.append("กระแสเงินทุนต่างชาติทรงตัว (0.0 ลบ.)")

                    if ad is not None and "obs_set_advance_decline_ratio" in observable_registry:
                        obs_refs.append("obs_set_advance_decline_ratio")
                        support_items.append(f"SET Market Breadth A/D {ad:.2f}x")

                    if pe is not None and "obs_set_valuation_pe" in observable_registry:
                        obs_refs.append("obs_set_valuation_pe")
                        support_items.append(f"SET Index P/E {pe:.2f}")
                        rationale_parts.append(f"ระดับ Valuation P/E อยู่ที่ {pe:.1f}x")

                    if rationale_parts:
                        constructed_rationale = f"ตลาดหุ้นไทยประเมินจากหลักฐานตลาดทุน: {' และ '.join(rationale_parts)}"
                    else:
                        constructed_rationale = "ประเมินสภาวะตลาดหุ้นไทยจากข้อมูลสภาพคล่องและโครงสร้างตลาดทุน"

                    thai_eq_view = AssetAllocationView(
                        asset_class="SET Equities (Thailand)",
                        asset_bucket="equities",
                        region="Thailand",
                        stance=thai_stance,
                        rationale=merged_stance.get("rationale") or constructed_rationale,
                        confidence="medium" if len(obs_refs) >= 2 else "low",
                        supporting_data=support_items,
                        observable_refs=obs_refs,
                        source_refs=["Terminal_V2_Settrade"],
                        why_not_high="ความมั่นใจอยู่ในระดับปานกลางเนื่องจากอยู่ระหว่างรอการเชื่อมต่อ API มหภาคทางการ (NESDC/MOC)",
                        allocation_delta=delta_str,
                        benchmark_ref="SET Index",
                        time_horizon="1-3 months",
                    )
                    if not hasattr(direction, "asset_allocation") or direction.asset_allocation is None:
                        direction.asset_allocation = []
                    direction.asset_allocation.append(thai_eq_view)

            # Ensure Cash region is Global
            for a in getattr(direction, "asset_allocation", []) or []:
                if a.asset_bucket == "cash" and a.region == "US":
                    a.region = "Global"

            # Revalidate with registry to ensure all guardrails, sources, and refs are pristine
            if hasattr(direction, "revalidate_with_registry"):
                direction = direction.revalidate_with_registry(observable_registry)

            # Preserve cited sources in direction.source_files without blindly dumping all evaluated_sources
            if not direction.source_files:
                direction.source_files = [
                    obs.source_file for obs in observable_registry.values()
                    if obs.is_valid and obs.source_file
                ]

            sidecar_result = write_strategy_json_sidecar(
                direction,
                evaluated_date,
                observable_registry=observable_registry,
                report_references=(state.get("narrative_context", {}) or {}).get("report_references", []),
                regional_assessments=allocator_quant_data.get("regions"),
                evaluated_sources=sorted(list(evaluated_source_files)),
                run_id=state.get("macro_task_run_id") or state.get("macro_run_id") or state.get("turn_id"),
                job_id=state.get("macro_run_id"),
                run_started_at=state.get("macro_task_started_at") or state.get("macro_run_started_at"),
                sector_snapshot_id=state.get("sector_rotation_snapshot_id"),
                sector_analysis=(state.get("narrative_context", {}) or {}).get("sector_analysis"),
                return_canonical_payload=True,
            )
            canonical_report = (
                sidecar_result[1] if isinstance(sidecar_result, tuple)
                else {**direction.model_dump(mode="json"),
                      "sector_analysis": (state.get("narrative_context", {}) or {}).get("sector_analysis")}
            )

            direction_fields = MacroStrategyDirection.model_fields
            canonical_direction = {key: canonical_report[key] for key in direction_fields if key in canonical_report}
            direction = MacroStrategyDirection.model_validate(canonical_direction)
            sector_analysis = canonical_report.get("sector_analysis")
            report = format_macro_strategy_report(
                direction,
                sector_analysis=sector_analysis,
                strategy_report_id=canonical_report.get("strategy_report_id"),
            )

            return Command(
                goto="post_macro_intel",
                update={"messages": [AIMessage(content=report)]}
            )
        except Exception as e:
            log_worker_result(state.get("turn_id", "unknown"), "strategic_allocator", f"Error: {e}", status="failure")
            return Command(
                goto="supervisor",
                update={"messages": [AIMessage(content=f"Error: (Strategic Allocator) {str(e)}")]}
            )

    def post_macro_intel_node(state: AgentState) -> Command[Literal["supervisor", "prepare_archivist"]]:
        report = state["messages"][-1].content
        turn_id = state.get("turn_id", "unknown")
        elapsed = _get_elapsed(state.get("route_meta") or {})

        if _has_entity_frontmatter(report):
            log_worker_result(turn_id, "strategic_allocator", report, status="success", elapsed_sec=elapsed)
            return Command(
                goto="prepare_archivist",
                update={
                    "route_meta": {"source": "macro_intel", "target": "archivist", "worker_started_at": time.monotonic()}
                }
            )

        status = "failure" if report.strip().startswith("Error:") else "info"
        log_worker_result(turn_id, "strategic_allocator", report, status=status, elapsed_sec=elapsed)
        return Command(
            goto="supervisor",
            update={
                "route_meta": {"source": "macro_intel", "target": "user"}
            }
        )


    builder = StateGraph(AgentState)
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("prepare_archivist", prepare_archivist_node)

    # แต่ละ wrapper invoke sub-graph ด้วยแค่ข้อความล่าสุด (task instruction ของตัวเองที่ผู้ส่ง
    # ต้นทาง — supervisor_node/prepare_archivist_node/post_quant_node — ออกแบบให้ self-contained
    # อยู่แล้ว) แทนที่จะส่ง state ทั้งก้อนที่สะสมมาทั้งเทิร์น เดิมส่ง state เต็มทำให้ instruction
    # ของ worker ตัวก่อนหน้า (เช่น "ต้องเรียก write_raw_markdown ทันที" ที่ตั้งใจส่งให้ Archivist)
    # รั่วเข้าไปในบริบทของ worker ตัวถัดไปในเทิร์นเดียวกัน (เจอจริงจาก live test) —
    # extract_worker_reply ไม่ต้องแก้ เพราะหา HumanMessage(name="manager") ตัวล่าสุดแล้วอ่านจาก
    # ตรงนั้นอยู่แล้ว พอ input มีข้อความเดียวตั้งแต่ต้นก็ยังถูกต้อง
    def archivist_wrapper(state: AgentState, config: RunnableConfig):
        task_message = state["messages"][-1]
        result = archivist_graph.invoke({"messages": [task_message]}, config=config)
        reply = extract_worker_reply(result["messages"])
        return {"messages": [AIMessage(content=reply, name="archivist")]}

    def bookkeeper_wrapper(state: AgentState, config: RunnableConfig):
        task_message = state["messages"][-1]
        result = bookkeeper_graph.invoke({"messages": [task_message]}, config=config)
        reply = extract_worker_reply(result["messages"])
        return {"messages": [AIMessage(content=reply, name="bookkeeper")]}

    builder.add_node("archivist", archivist_wrapper)
    builder.add_node("bookkeeper", bookkeeper_wrapper)

    # Macro Intel Pipeline
    def macro_quant_wrapper(state: AgentState, config: RunnableConfig):
        fetch_and_save_macro_snapshots()
        task_message = state["messages"][-1]
        snapshot = None
        sector_context = None
        config_meta = (config or {}).get("metadata", {}) if isinstance(config, dict) else {}
        macro_run_id = state.get("macro_run_id") or config_meta.get("job_id") or state.get("turn_id")
        task_run_id = state.get("macro_task_run_id") or f"{macro_run_id}:{state.get('turn_id', 'task')}:1"
        try:
            from tools.macro.sector_rotation.bootstrap import get_sector_rotation_service
            from tools.macro.sector_rotation.domain.claims import compact_ai_context

            rotation_service = get_sector_rotation_service()
            snapshot, sector_context_status = rotation_service.pin_for_run(
                str(task_run_id),
                job_id=str(macro_run_id) if macro_run_id else None,
                task_id=str(task_run_id),
                preferred_snapshot_id=state.get("sector_rotation_snapshot_id"),
            )
            if snapshot:
                sector_context = compact_ai_context(snapshot)
                has_sector_data = any(row.status != "unavailable" for row in snapshot.rows)
                complete = snapshot.benchmark_status == "available" and all(row.status == "available" for row in snapshot.rows)
                sector_context_status = {
                    **sector_context_status,
                    "status": "available" if complete else "limited" if has_sector_data else "unavailable",
                    "reason": None if complete else "partial_sector_coverage" if has_sector_data else "sector_history_unavailable",
                }
        except Exception as exc:
            log.warning("Sector snapshot unavailable; continuing Macro analysis without sector facts: %s", exc)
            snapshot = None
            sector_context = None
            sector_context_status = {"status": "unavailable", "reason": f"snapshot_prepare_failed:{type(exc).__name__}"}
        if sector_context_status["status"] == "unavailable":
            sector_context = None
        quant_instruction = normalize_content(task_message.content)
        if sector_context:
            quant_instruction += (
                "\n\nDeterministic sector rotation evidence for this macro run (cite its values as market evidence; "
                "do not count ETFs as GDP/CPI hard-data or infer fund flows):\n"
                + json.dumps(sector_context, ensure_ascii=False)
            )
        else:
            quant_instruction += (
                "\n\nSector rotation context status: unavailable ("
                + str(sector_context_status.get("reason") or "no_valid_snapshot")
                + "). Continue the macro analysis using the other evidence and do not assert sector facts."
            )
        from tools.macro.sector_rotation.run_context import sector_run_scope
        with sector_run_scope(str(task_run_id)):
            result = macro_quant_graph.invoke({"messages": [HumanMessage(content=quant_instruction)]}, config=config)
        reply = extract_worker_reply(result["messages"])
        return {
            "messages": [AIMessage(content=reply, name="macro_quant")],
            "macro_run_id": str(macro_run_id) if macro_run_id else None,
            "macro_task_run_id": state.get("macro_task_run_id") or f"{macro_run_id}:{state.get('turn_id', 'task')}:1",
            "sector_rotation_snapshot_id": snapshot.snapshot_id if snapshot else None,
            "sector_rotation_context": sector_context,
            "sector_context_status": sector_context_status,
        }

    def macro_economist_wrapper(state: AgentState, config: RunnableConfig):
        task_message = state["messages"][-1]
        sector_snapshot = None
        sector_context = None
        sector_context_status = {"status": "unavailable", "reason": "run_binding_missing"}
        task_run_id = state.get("macro_task_run_id")
        if task_run_id:
            from tools.macro.sector_rotation.bootstrap import get_sector_rotation_service
            from tools.macro.sector_rotation.domain.claims import compact_ai_context
            try:
                sector_snapshot, binding = get_sector_rotation_service().pin_for_run(
                    str(task_run_id),
                    job_id=str(state.get("macro_run_id")) if state.get("macro_run_id") else None,
                    task_id=str(task_run_id),
                )
                if sector_snapshot is not None:
                    sector_context = compact_ai_context(sector_snapshot)
                    has_sector_data = any(row.status != "unavailable" for row in sector_snapshot.rows)
                    complete = sector_snapshot.benchmark_status == "available" and all(
                        row.status == "available" for row in sector_snapshot.rows
                    )
                    sector_context_status = {
                        "status": "available" if complete else "limited" if has_sector_data else "unavailable",
                        "reason": None if complete else "partial_sector_coverage" if has_sector_data else "sector_history_unavailable",
                        "snapshot_id": sector_snapshot.snapshot_id,
                    }
                else:
                    sector_context_status = {
                        "status": "unavailable",
                        "reason": binding.get("unavailable_reason") or binding.get("reason") or "pinned_snapshot_unavailable",
                    }
            except Exception as exc:
                log.warning("Pinned sector evidence unavailable to Macro Economist: %s", exc)
                sector_snapshot = None
                sector_context_status = {"status": "unavailable", "reason": f"pinned_snapshot_unavailable:{type(exc).__name__}"}
        result = macro_economist_graph.invoke({
            "messages": [task_message],
            "quant_score": state.get("quant_score") or {},
            "sector_rotation_context": sector_context,
            "sector_context_status": sector_context_status,
            "sector_rotation_snapshot": sector_snapshot.model_dump(mode="json") if sector_snapshot else None,
        }, config=config)
        reply = extract_worker_reply(result["messages"])
        return {
            "messages": [AIMessage(content=reply, name="macro_economist")],
            "sector_rotation_snapshot_id": sector_snapshot.snapshot_id if sector_snapshot else None,
            "sector_rotation_context": sector_context,
            "sector_context_status": sector_context_status,
        }

    builder.add_node("macro_quant", macro_quant_wrapper)
    builder.add_node("macro_economist", macro_economist_wrapper)
    builder.add_node("strategic_allocator", strategic_allocator_node)

    # --- Equity Intel Pipeline ---
    def equity_quant_wrapper(state: AgentState, config: RunnableConfig):
        task_message = state["messages"][-1]
        result = equity_quant_graph.invoke({"messages": [task_message]}, config=config)
        reply = extract_worker_reply(result["messages"])
        return {"messages": [AIMessage(content=reply, name="equity_quant")]}

    def equity_narrative_wrapper(state: AgentState, config: RunnableConfig):
        # อ่าน ticker/market/company_name จาก state ตรงๆ (ไม่ parse จากข้อความอิสระ) — equity_narrative
        # เป็น RunnableLambda ไม่ใช่ ReAct agent จึงรับ input เป็น dict ธรรมดา ไม่ใช่ {"messages": [...]}
        quant_score = state.get("equity_quant_score", {}) or {}
        result = equity_narrative_graph.invoke(
            {
                "ticker": quant_score.get("ticker"),
                "market": quant_score.get("market", "US"),
                "company_name": quant_score.get("company_name"),
            },
            config=config,
        )
        reply = extract_worker_reply(result["messages"])
        update_dict = {"messages": [AIMessage(content=reply, name="equity_narrative")]}
        news_raw = result.get("equity_news_raw")
        if news_raw:
            update_dict["equity_news_raw"] = news_raw
        return update_dict

    def post_equity_quant_node(state: AgentState) -> Command[Literal["equity_narrative", "supervisor"]]:
        reply = extract_worker_reply(state["messages"])
        try:
            validated = QuantSignals.model_validate(json.loads(reply))
            try:
                save_equity_quant_snapshot(validated, note_writer=note_writer)
            except Exception as hist_err:
                log.warning("save_equity_quant_snapshot failed (non-fatal): %s", hist_err)

            # Provisional sidecar persistence: quant signals exist on disk even before narrative LLM call
            try:
                from tools.market.equity_sidecar import write_equity_sidecar
                partial_output = MicroQuantOutput(
                    ticker=validated.ticker,
                    market=validated.market,
                    analysis_date=datetime.now().strftime("%Y-%m-%d"),
                    quant_signals=validated,
                    sentiment_context=None,
                    narrative_analysis="รอการประมวลผล Narrative จาก LLM",
                    base_case_summary="ตัวเลขรีเฟรชแล้ว กำลังสร้าง narrative",
                )
                write_equity_sidecar(partial_output)
            except Exception as sidecar_err:
                log.warning("Provisional equity sidecar write failed (non-fatal): %s", sidecar_err)

            validated_json = validated.model_dump(mode="json")
            turn_id = state.get("turn_id", "unknown")
            elapsed = _get_elapsed(state.get("route_meta") or {})
            log_worker_result(turn_id, "equity_quant", reply, status="success", elapsed_sec=elapsed)
            return Command(
                goto="equity_narrative",
                update={
                    "equity_quant_raw": reply,
                    "equity_quant_score": validated_json,
                    "route_meta": {"source": "equity_quant", "target": "equity_narrative", "worker_started_at": time.monotonic()},
                }
            )
        except Exception as e:
            log_worker_result(state.get("turn_id", "unknown"), "equity_quant", f"Error: {e}\n{reply}", status="failure")
            return Command(
                goto="supervisor",
                update={"messages": [AIMessage(content=f"Error: (Equity Quant) {str(e)} - {reply}")]}
            )

    def post_equity_narrative_node(state: AgentState) -> Command[Literal["equity_synthesizer"]]:
        reply = extract_worker_reply(state["messages"])
        import re
        reply_clean = re.sub(r'^```(?:json)?\s*|\s*```$', '', reply.strip(), flags=re.MULTILINE)
        try:
            validated_json = EquitySentimentContext.model_validate(json.loads(reply_clean)).model_dump(mode="json")
            turn_id = state.get("turn_id", "unknown")
            elapsed = _get_elapsed(state.get("route_meta") or {})
            log_worker_result(turn_id, "equity_narrative", reply, status="success", elapsed_sec=elapsed)
        except Exception as e:
            log_worker_result(state.get("turn_id", "unknown"), "equity_narrative", f"Validation fallback: {e}\n{reply}", status="warning")
            fallback_dict = {
                "evaluated_at": datetime.now(timezone.utc).isoformat(),
                "market_sentiment": "neutral",
                "key_themes": [],
                "tail_risks": [],
                "sources_summary": f"Data fetch/parse failed: {str(e)}",
            }
            validated_json = EquitySentimentContext.model_validate(fallback_dict).model_dump(mode="json")

        return Command(
            goto="equity_synthesizer",
            update={
                "equity_narrative_raw": reply,
                "equity_narrative_context": validated_json,
                "route_meta": {"source": "equity_narrative", "target": "equity_synthesizer", "worker_started_at": time.monotonic()},
            }
        )

    def equity_synthesizer_node(state: AgentState) -> Command[Literal["post_equity_intel", "supervisor"]]:
        try:
            provider = detect_provider(_EQUITY_SYNTHESIZER_MODEL)
            model = get_llm(provider=provider, model_name=_EQUITY_SYNTHESIZER_MODEL, use_fallback=True)

            quant_data = state.get("equity_quant_score", {}) or {}
            narrative_data = state.get("equity_narrative_context", {}) or {}
            quant_json = json.dumps(quant_data, ensure_ascii=False)
            narrative_json = json.dumps(narrative_data, ensure_ascii=False)

            narrative = invoke_equity_synthesizer(model, quant_json, narrative_json)

            output = MicroQuantOutput(
                ticker=quant_data["ticker"],
                market=quant_data["market"],
                analysis_date=datetime.now().strftime("%Y-%m-%d"),
                quant_signals=QuantSignals.model_validate(quant_data),
                sentiment_context=EquitySentimentContext.model_validate(narrative_data),
                narrative_analysis=narrative.narrative_analysis,
                base_case_summary=narrative.base_case_summary,
            )
            report = format_equity_analysis_report(output)

            turn_id = state.get("turn_id", "unknown")
            elapsed = _get_elapsed(state.get("route_meta") or {})
            log_worker_result(turn_id, "equity_synthesizer", report, status="success", elapsed_sec=elapsed)
            return Command(
                goto="post_equity_intel",
                update={
                    "messages": [AIMessage(content=report)],
                    "equity_output": output.model_dump(mode="json"),
                }
            )
        except Exception as e:
            log_worker_result(state.get("turn_id", "unknown"), "equity_synthesizer", f"LLM Warning/Fallback: {e}", status="warning")
            quant_data = state.get("equity_quant_score", {}) or {}
            narrative_data = state.get("equity_narrative_context", {}) or {}
            if quant_data and "ticker" in quant_data:
                # Quant succeeded -> preserve full quant signals and build partial revision
                fallback_sentiment = EquitySentimentContext(
                    evaluated_at=datetime.now(timezone.utc).isoformat(),
                    market_sentiment="neutral",
                    key_themes=[],
                    tail_risks=[],
                    sources_summary=f"LLM synthesis pending/unavailable: {str(e)}",
                )
                output = MicroQuantOutput(
                    ticker=quant_data["ticker"],
                    market=quant_data.get("market", "US"),
                    analysis_date=datetime.now().strftime("%Y-%m-%d"),
                    quant_signals=QuantSignals.model_validate(quant_data),
                    sentiment_context=EquitySentimentContext.model_validate(narrative_data) if narrative_data else fallback_sentiment,
                    narrative_analysis=f"ตัวเลขรีเฟรชแล้ว แต่ narrative ยังสร้างไม่ได้ (LLM_UNAVAILABLE: {str(e)})",
                    base_case_summary="ตัวเลขรีเฟรชแล้ว แต่ narrative ยังสร้างไม่ได้",
                )
                report = format_equity_analysis_report(output)
                return Command(
                    goto="post_equity_intel",
                    update={
                        "messages": [AIMessage(content=report)],
                        "equity_output": output.model_dump(mode="json"),
                    }
                )
            return Command(
                goto="supervisor",
                update={"messages": [AIMessage(content=f"Error: (Equity Synthesizer) {str(e)}")]}
            )

    def post_equity_intel_node(state: AgentState) -> Command[Literal["supervisor", "prepare_archivist"]]:
        report = state["messages"][-1].content
        turn_id = state.get("turn_id", "unknown")
        elapsed = _get_elapsed(state.get("route_meta") or {})
        save_to_vault = state.get("equity_save_to_vault")
        if save_to_vault is None:
            save_to_vault = True

        if _has_entity_frontmatter(report) and save_to_vault:
            log_worker_result(turn_id, "equity_synthesizer", report, status="success", elapsed_sec=elapsed)
            return Command(
                goto="prepare_archivist",
                update={
                    "route_meta": {"source": "equity_intel", "target": "archivist", "worker_started_at": time.monotonic()}
                }
            )

        status = "failure" if report.strip().startswith("Error:") else "info"
        log_worker_result(turn_id, "equity_synthesizer", report, status=status, elapsed_sec=elapsed)
        return Command(
            goto="supervisor",
            update={
                "messages": [AIMessage(content=report)],
                "route_meta": {"source": "equity_intel", "target": "user"},
                "turn_id": turn_id,
            }
        )

    builder.add_node("equity_quant", equity_quant_wrapper)
    builder.add_node("equity_narrative", equity_narrative_wrapper)
    builder.add_node("equity_synthesizer", equity_synthesizer_node)
    builder.add_node("post_equity_quant", post_equity_quant_node)
    builder.add_node("post_equity_narrative", post_equity_narrative_node)
    builder.add_node("post_equity_intel", post_equity_intel_node)

    builder.add_node("post_archivist", post_archivist_node)
    builder.add_node("post_bookkeeper", post_bookkeeper_node)

    builder.add_node("post_quant", post_quant_node)
    builder.add_node("post_economist", post_economist_node)
    builder.add_node("post_macro_intel", post_macro_intel_node)

    builder.add_edge(START, "supervisor")

    builder.add_edge("archivist", "post_archivist")
    builder.add_edge("bookkeeper", "post_bookkeeper")

    builder.add_edge("macro_quant", "post_quant")
    builder.add_edge("macro_economist", "post_economist")
    builder.add_edge("strategic_allocator", "post_macro_intel")

    builder.add_edge("equity_quant", "post_equity_quant")
    builder.add_edge("equity_narrative", "post_equity_narrative")

    return builder.compile(checkpointer=checkpointer)
