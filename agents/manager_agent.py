import os
import re
import time
import uuid
import json
from datetime import datetime
from functools import lru_cache
from typing import Literal, TypedDict, Optional

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables.config import RunnableConfig
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Command
from pydantic import BaseModel, Field

from agents.archivist_agent import create_archivist
from agents.bookkeeper_agent import create_bookkeeper
from agents.researcher_agent import create_researcher
from agents.macro_quant_agent import create_macro_quant
from agents.macro_economist_agent import create_macro_economist
from agents.strategic_allocator import invoke_strategic_allocator
from schemas.macro_schemas import MarketObservable
from tools.macro.report_formatter import format_macro_strategy_report
from core.agent_log import log_turn_start, log_manager_plan, log_worker_result, log_system_action, log_routing
from core.llm_factory import FALLBACK_MODEL, detect_provider, get_llm
from core.logger import get_logger
from core.prompt_harness import get_harness
from core.utils import normalize_content

log = get_logger(__name__)


def _msg_role(m) -> str:
    return "human" if isinstance(m, HumanMessage) else "assistant"


def _sanitize_researcher_instruction(instruction: str) -> str:
    """ตัดคำที่อาจชักจูงให้ Researcher พยายามเรียก tool บันทึกไฟล์เอง"""
    sanitized = re.sub(r'(แล้ว)?(ให้)?บันทึก(\s*(ลง|ใน|ไปที่|ไปยัง)?\s*(Vault|วอลท์|vault)?)?', '', instruction, flags=re.IGNORECASE)
    sanitized = re.sub(r'\bsave\s+(to\s+)?vault\b', '', sanitized, flags=re.IGNORECASE)
    return sanitized.strip()


# Single-tier config: ทุก agent ใช้ gemini-3.1-flash-lite-preview เป็น default
# Fallback chain (core/llm_factory.FALLBACK_MODEL) = openai/gpt-oss-120b:free (OpenRouter)
_MANAGER_MODEL = os.getenv("MANAGER_MODEL", "gemini-3.1-flash-lite-preview")
_ROUTER_MODEL = os.getenv("ROUTER_MODEL", _MANAGER_MODEL)
_ARCHIVIST_MODEL = os.getenv("ARCHIVIST_MODEL", "gemini-3.1-flash-lite-preview")
_RESEARCHER_MODEL = os.getenv("RESEARCHER_MODEL", "gemini-3.1-flash-lite-preview")
_BOOKKEEPER_MODEL = os.getenv("BOOKKEEPER_MODEL", "gemini-3.1-flash-lite-preview")
_MACRO_QUANT_MODEL = os.getenv("MACRO_QUANT_MODEL", "gemini-3.1-flash-lite-preview")
_MACRO_ECONOMIST_MODEL = os.getenv("MACRO_ECONOMIST_MODEL", "gemini-3.1-flash-lite-preview")
_STRATEGIC_ALLOCATOR_MODEL = os.getenv("STRATEGIC_ALLOCATOR_MODEL", "gemini-3.1-flash-lite-preview")
_ROUTER_HISTORY_LIMIT = 20
_MAX_REPLAN = 5


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

    # Macro Pipeline State
    quant_raw: Optional[str]
    quant_score: Optional[dict]
    narrative_raw: Optional[str]
    narrative_context: Optional[dict]


# ROUTER_PROMPT ถูกย้ายไปที่ prompts/skills/manager/SKILL.md ผ่านระบบ PromptHarness


class WorkerTask(BaseModel):
    """งานย่อย 1 ชิ้น ที่ route ไปยัง worker หนึ่งตัว"""
    target: Literal["archivist", "researcher", "bookkeeper", "macro_intel"]
    instruction: str = Field(
        description="คำสั่งสำหรับ worker ตัวนี้ — กระชับ ชัดเจน ตัดคำนำหน้า/คำพ่วงที่ไม่เกี่ยวออก"
    )
    save_to_vault: bool = Field(
        default=True,
        description="ใช้กับ target == 'researcher' เท่านั้น — True = ส่งผลให้ Archivist บันทึก, "
                    "False = แค่ดึงข้อมูลแล้วแสดง ไม่เซฟ "
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
        description="คำตอบที่ส่งกลับให้ผู้ใช้โดยตรง (ภาษาไทย กระชับ) — ใช้เมื่อ tasks ว่างเท่านั้น "
                    "ไม่นำข้อมูลดิบจาก Researcher มาวิเคราะห์ซ้ำ",
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


def _has_researcher_frontmatter(text: str) -> bool:
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
def _get_researcher_graph():
    provider = detect_provider(_RESEARCHER_MODEL)
    return create_researcher(get_llm(provider=provider, model_name=_RESEARCHER_MODEL, use_fallback=True))


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


def build_graph(checkpointer=None) -> StateGraph:
    archivist_graph = _get_archivist_graph()
    researcher_graph = _get_researcher_graph()
    bookkeeper_graph = _get_bookkeeper_graph()
    macro_quant_graph = _get_macro_quant_graph()
    macro_economist_graph = _get_macro_economist_graph()
    router_model = _get_router_model()

    def supervisor_node(state: AgentState) -> Command[Literal["prepare_archivist", "researcher", "bookkeeper", "macro_quant", "__end__"]]:
        messages = state["messages"]
        turn_id = state.get("turn_id")

        if isinstance(messages[-1], HumanMessage) and getattr(messages[-1], "name", None) != "manager":
            turn_id = uuid.uuid4().hex[:8]
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
                        "กรุณาวางแผนงานใหม่เพื่อแก้ปัญหา — "
                        "อาจต้องสั่ง Researcher ดึงข้อมูลที่ขาดก่อน แล้วค่อยทำงานต่อ"
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

        goto_target = target
        if target == "archivist":
            goto_target = "prepare_archivist"
        elif target == "macro_intel":
            goto_target = "macro_quant"

        if target == "researcher":
            instruction = _sanitize_researcher_instruction(task["instruction"])
        else:
            instruction = task["instruction"]

        return Command(
            goto=goto_target,
            update={
                "messages": messages_update + [HumanMessage(content=instruction, name="manager")],
                "route_meta": meta,
                "task_queue": rest,
                "replan_count": replan_count_update,
                "turn_id": turn_id,
            },
        )

    def prepare_archivist_node(state: AgentState) -> Command[Literal["archivist"]]:
        meta = state.get("route_meta") or {}

        if meta.get("source") in ["researcher", "macro_intel"]:
            last_msg = extract_worker_reply(state["messages"])
            task = f"คุณต้องเรียกใช้เครื่องมือ write_raw_markdown เพื่อบันทึกข้อมูลดิบต่อไปนี้ลง Vault ทันที ห้ามตอบกลับเป็นข้อความโดยไม่เรียกใช้เครื่องมือ\n\n[ข้อมูลดิบ]\n{last_msg}"
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
            if not mid_drain and _has_researcher_frontmatter(raw_user) and len(raw_user) > len(last_msg) * 2:
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

        if not _has_researcher_frontmatter(researcher_reply) or not save_to_vault:
            status = "info" if save_to_vault else "info"
            if _has_researcher_frontmatter(researcher_reply) is False and save_to_vault is True:
                status = "warning"
            log_worker_result(turn_id, "researcher", researcher_reply, status=status, elapsed_sec=elapsed)
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
            save_macro_baseline(validated_json)

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
            for source_file in sorted(evaluated_source_files):
                if source_file not in direction.source_files:
                    direction.source_files.append(source_file)
            report = format_macro_strategy_report(direction)

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

        if _has_researcher_frontmatter(report):
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
    def archivist_wrapper(state: AgentState, config: RunnableConfig):
        result = archivist_graph.invoke(state, config=config)
        reply = extract_worker_reply(result["messages"])
        return {"messages": [AIMessage(content=reply, name="archivist")]}

    def researcher_wrapper(state: AgentState, config: RunnableConfig):
        result = researcher_graph.invoke(state, config=config)
        reply = extract_worker_reply(result["messages"])
        return {"messages": [AIMessage(content=reply, name="researcher")]}

    def bookkeeper_wrapper(state: AgentState, config: RunnableConfig):
        result = bookkeeper_graph.invoke(state, config=config)
        reply = extract_worker_reply(result["messages"])
        return {"messages": [AIMessage(content=reply, name="bookkeeper")]}

    builder.add_node("archivist", archivist_wrapper)
    builder.add_node("researcher", researcher_wrapper)
    builder.add_node("bookkeeper", bookkeeper_wrapper)

    # Macro Intel Pipeline
    def macro_quant_wrapper(state: AgentState, config: RunnableConfig):
        result = macro_quant_graph.invoke(state, config=config)
        reply = extract_worker_reply(result["messages"])
        return {"messages": [AIMessage(content=reply, name="macro_quant")]}

    def macro_economist_wrapper(state: AgentState, config: RunnableConfig):
        result = macro_economist_graph.invoke(state, config=config)
        reply = extract_worker_reply(result["messages"])
        return {"messages": [AIMessage(content=reply, name="macro_economist")]}

    builder.add_node("macro_quant", macro_quant_wrapper)
    builder.add_node("macro_economist", macro_economist_wrapper)
    builder.add_node("strategic_allocator", strategic_allocator_node)

    builder.add_node("post_archivist", post_archivist_node)
    builder.add_node("post_researcher", post_researcher_node)
    builder.add_node("post_bookkeeper", post_bookkeeper_node)

    builder.add_node("post_quant", post_quant_node)
    builder.add_node("post_economist", post_economist_node)
    builder.add_node("post_macro_intel", post_macro_intel_node)

    builder.add_edge(START, "supervisor")

    builder.add_edge("archivist", "post_archivist")
    builder.add_edge("researcher", "post_researcher")
    builder.add_edge("bookkeeper", "post_bookkeeper")

    builder.add_edge("macro_quant", "post_quant")
    builder.add_edge("macro_economist", "post_economist")
    builder.add_edge("strategic_allocator", "post_macro_intel")

    return builder.compile(checkpointer=checkpointer)
