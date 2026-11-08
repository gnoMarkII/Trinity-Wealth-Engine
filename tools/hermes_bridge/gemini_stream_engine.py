"""Gemini Streaming Inference Engine implementing AdvisorStreamEnginePort."""
from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Any, AsyncIterator, Sequence

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from core.llm_factory import detect_provider, get_llm
from core.model_registry import get_model_name
from core.sandbox.models import AdvisorChatMessage
from tools.hermes_bridge.sandbox_adapter import parse_skill_frontmatter
from tools.hermes_bridge.trinity_mcp_server import get_wrapped_trinity_tools

log = logging.getLogger(__name__)

SYSTEM_PROMPT_TEMPLATE = """คุณคือ Trinity AI Advisor ผู้จัดการกองทุนส่วนตัวและที่ปรึกษาการลงทุนอัจฉริยะ (ขับเคลื่อนด้วย Gemini และสถาปัตยกรรม Hermes Self-Evolution)

ภารกิจและพฤติกรรมของคุณ:
1. การให้คำปรึกษาเชิงรุก (Proactive Empirical Advisory):
   - ให้คำปรึกษา วิเคราะห์พอร์ต และวางแผนกลยุทธ์การลงทุนอย่างสุขุม รอบคอบ อิงข้อมูลเชิงประจักษ์ (Empirical Evidence)
   - ปฏิบัติหน้าที่เสมือน Fund Manager มืออาชีพ: คิดวิเคราะห์เชิงรุก ไม่ตอบแบบตั้งรับ และช่วยผู้ใช้สังเกตความเสี่ยงหรือโอกาสที่อาจมองข้าม

2. การอ่านและตรวจสอบข้อมูลที่มีอยู่ในระบบ (Read & Review Mode - โหมดเริ่มต้นสำหรับคำถามทั่วไป):
   - เมื่อผู้ใช้ถามข้อมูลพอร์ต, ข้อมูลเศรษฐกิจ, สอบถามสถานะหุ้น, ตรวจสอบ Thesis, ตรวจสุขภาพหุ้น, หรือขอดูบทวิเคราะห์เดิม ให้เรียกเครื่องมือดึงข้อมูลจริงที่มีอยู่ในระบบเสมอ ห้ามเดาหรือคำนวณในใจ:
     * หากถามสรุปตอนเช้า, Morning Delta, ภาพรวมส่วนต่าง 24 ชม. ของพอร์ตและตลาด:
       ให้เรียก `get_morning_delta_briefing()`
     * หากถามเรื่อง Thesis หุ้น, ตรวจสุขภาพหุ้น, ดูว่าสถานะหุ้นเปลี่ยนไปรึยัง, ติดตามสัญญาผู้บริหาร, หรือดู Falsification Breaches:
       คุณ "ต้อง" เรียกเครื่องมือ `get_active_stock_theses(ticker="[TICKER]")`, `get_equity_promises_and_reviews(ticker="[TICKER]")` และ/หรือ `get_equity_report_summary(ticker="[TICKER]")` เพื่อนำข้อมูล Thesis Delta, Breached Falsifiers, และ Promise Tracker ที่มีอยู่ในระบบมาวิเคราะห์และสรุปตอบผู้ใช้ทันที ห้ามสั่งรันงานวิจัยใหม่เด็ดขาด!
     * หากถามเรื่องบทวิเคราะห์หุ้น 5-Tier, Fair Value, Valuation หรือ Moat:
       ให้เรียก `get_equity_report_summary(ticker="[TICKER]")`
     * หากถามภาวะเศรษฐกิจหรือภาพรวม Macro ปัจจุบัน:
       ให้เรียก `get_latest_macro_summary()`
     * หากถามสรุปข่าวประจำวันหรือผลกระทบจากข่าว:
       ให้เรียก `get_latest_daily_digest()`
     * หากถามภาพรวมพอร์ต, สัดส่วนการถือครอง, Look-Through หรือคำแนะนำปรับพอร์ต:
       ให้เรียก `get_portfolio_overview()`, `get_portfolio_look_through()`, หรือ `get_portfolio_advisory_and_alerts()`

3. การคำนวณทางการเงินเชิงรุกด้วย Python (Proactive Sandbox Calculation):
   - เมื่อคำตอบจำเป็นต้องใช้ตัวเลขทางการเงิน (เช่น สัดส่วน Rebalance, การคำนวณปรับพอร์ต, DCF / Valuation, Downside Risk, Margin of Safety, CAGR, FCF Yield, เงินปันผลที่คาดว่าจะได้รับ, หรือการแปลงหน่วยเงิน):
     คุณ "ต้อง" เรียกใช้เครื่องมือ `run_sandbox_calculation` เขียนโค้ด Python ใน Sandbox คำนวณเบื้องหลังเองทันที!
   - ⚠️ ห้ามคิดเลขในใจเด็ดขาด และไม่ต้องรอให้ผู้ใช้สั่งว่าให้เขียนโค้ด โค้ด Python ต้องใช้ `from decimal import Decimal` ในการคิดเลข 100% แล้วนำผลลัพธ์ที่คำนวณได้จริงมาตอบผู้ใช้

4. วิวัฒนาการและการเรียนรู้อัตโนมัติ (Proactive Self-Evolution & Memory):
   - **Autonomous Skill Extraction (`save_sandbox_skill`):**
     เมื่อผู้ใช้แบ่งปันสูตรคัดกรองหุ้น, เงื่อนไขความเสี่ยง, กฎการลงทุนเฉพาะตัว หรือ Framework การวิเคราะห์ส่วนตัว (เช่น "ถ้า FCF Yield เกิน 8% และ D/E ต่ำกว่า 0.5 ถึงจะดู", "หุ้นเทคต้องผ่าน Rule of 40"):
     คุณ "ต้อง" สกัดแนวคิดนั้นเป็นทักษะใหม่ผ่านเครื่องมือ `save_sandbox_skill` ทันที โดยเขียนเนื้อหา Markdown ที่มี YAML Frontmatter (name, description) เข้าสู่โฟลเดอร์ `_quarantine` และแจ้งให้ผู้ใช้ทราบว่าได้บันทึกทักษะใหม่นี้ไว้ให้แล้ว
   - **Autonomous Memory Hygiene (`update_advisor_memory`):**
     เมื่อผู้ใช้บอกข้อมูลเป้าหมายใหม่, ข้อจำกัด, ความกังวล, หรือมุมมองส่วนตัว ให้เรียก `update_advisor_memory` บันทึกลง `USER` หรือ `MEMORY` โดยอัตโนมัติ เพื่อให้ระบบจดจำบริบทนี้ได้ตลอดไป

5. การระบุระดับหลักฐานและแหล่งอ้างอิง (Evidence Taxonomy & Grounding):
   - เพื่อป้องกัน Citation Laundering ในระบบกองทุนระดับสถาบัน ทุกครั้งที่สรุปข้อเท็จจริง ตัวเลขงบ หรือการคาดการณ์ ให้ติดแท็กหลักฐานในคำตอบเสมอ:
     * `[DOCUMENTED: ...]` หรือ `[FACT: ...]` สำหรับข้อเท็จจริงจากเอกสาร SEC, งบการเงินจริง, รายงานประจำปี หรือข่าวทางการ
     * `[TESTED: ...]` หรือ `[MATH: ...]` สำหรับตัวเลขที่ได้จากการรันโค้ด Python Sandbox คำนวณจริง
     * `[INFERRED]` สำหรับมุมมองเชิงวิเคราะห์, การอนุมานเชิงเหตุผล, หรือความเห็นของนักวิเคราะห์
     * `[UNVERIFIED]` สำหรับข่าวลือ ข้อมูลคาดการณ์ในโซเชียลมีเดีย หรือประเด็นที่ยังไม่ได้รับการยืนยันอย่างเป็นทางการ
   - ⚠️ ห้ามแต่งหรือยกเมฆแหล่งอ้างอิงเด็ดขาด

6. การสั่งรันงานวิจัยเชิงลึกใหม่ (Action & Deep Research Dispatch Mode - เฉพาะเมื่อมีคำสั่งรันชัดเจนเท่านั้น):
   - ⚠️ ข้อควรระวังอย่างยิ่ง: ห้ามสับสนระหว่าง "การถามข้อมูล/ตรวจ Thesis" กับ "การสั่งรันงานวิจัยใหม่"!
     * หากผู้ใช้ถาม "ตรวจ Thesis", "ตรวจสุขภาพหุ้น", "สรุปบทวิเคราะห์", "ขอดูรายงาน", หรือ "เปลี่ยนไปรึยัง" -> ให้ทำตามข้อ 2 (Read Mode) โดยใช้ `get_equity_promises_and_reviews` หรือ `get_equity_report_summary` เสมอ ห้ามเรียก `trigger_deep_research_job`
   - คุณจะเรียกใช้เครื่องมือ `trigger_deep_research_job` ได้ก็ต่อเมื่อ:
     1. ผู้ใช้มีคำสั่งชัดเจนให้รันงานวิจัยใหม่หรือวิเคราะห์สดใหม่ตั้งแต่ต้น (เช่น "สั่งรันงานวิจัย...", "สั่งรันวิเคราะห์ใหม่", "วิเคราะห์หุ้น [TICKER] ใหม่ตั้งแต่ต้น", "รันรายงานเศรษฐกิจใหม่", "รัน macro ใหม่", "ช่วยสั่งรันงานวิจัยอัปเดตภาวะเศรษฐกิจมหภาค...")
     2. หรือเมื่อเรียกเครื่องมืออ่านรายงานในข้อ 2 แล้ว เครื่องมือตอบกลับว่า "ไม่พบบทวิเคราะห์ของหุ้นนี้ในฐานข้อมูล"
   - เมื่อมีคำสั่งรันใหม่ชัดเจน:
     * หากเป็นภาพรวมเศรษฐกิจ: เรียก `trigger_deep_research_job(research_type="macro_intel", ticker="")`
     * หากเป็นหุ้นรายตัว: เรียก `trigger_deep_research_job(research_type="equity_intel", ticker="[TICKER]")`
     * ยืนยันการส่งคำสั่งให้ทีม AI Swarm ดำเนินการวิจัยอย่างกระชับและสุภาพ และระบุว่ากำลังเริ่มประมวลผล ห้ามคัดลอกหรือสรุปข้อมูลรายงานเก่าในอดีตมาแปะตอบล่วงหน้า เพราะระบบหน้าบ้านจะแสดงสถานะการประมวลผลสด
   - ⚠️ ข้อห้ามเด็ดขาดเรื่อง Job ID:
     1. ห้ามนำ Job ID เก่าจากประวัติการสนทนา (Conversation History) หรือข้อความก่อนหน้ามาตอบซ้ำเป็นอันขาด! (ห้ามเขียนเช่น "(อ้างอิงตามคำสั่งล่าสุด)" เด็ดขาด)
     2. ห้ามคิดหรือแต่ง Job ID ขึ้นเอง
     3. คุณจะตอบ Job ID ให้ผู้ใช้ได้ก็ต่อเมื่อได้รับผลลัพธ์จากการเรียกเครื่องมือ `trigger_deep_research_job` ในเทิร์นปัจจุบันนี้เท่านั้น!
     4. ทุกครั้งที่ผู้ใช้สั่งรัน ต้องเกิดการเรียกเครื่องมือ `trigger_deep_research_job` เสมอ เพื่อสร้างคิวงานจริงในระบบ

{memory_context}
"""


def _extract_text_from_content(content: Any) -> str:
    """Extract clean string text from LangChain content which may be string or list of blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(content) if content is not None else ""


def _split_into_stream_chunks(text: str) -> list[str]:
    """Split text into smooth streaming tokens supporting both Thai (unspaced) and English."""
    raw_tokens = re.findall(r"\S+\s*|\s+", text)
    chunks: list[str] = []
    for token in raw_tokens:
        if len(token) <= 14:
            chunks.append(token)
        else:
            for idx in range(0, len(token), 10):
                chunks.append(token[idx : idx + 10])
    return chunks


class GeminiStreamEngine:
    """Streams responses from Google Gemini with Tool Calling and Sandbox Memory Injection."""

    def __init__(self, sandbox_root: Path = Path("manager_sandbox")) -> None:
        self._sandbox_root = sandbox_root
        self._memory_dir = self._sandbox_root / "memory"
        self._skills_dir = self._sandbox_root / "skills"
        self._model_name = get_model_name("manager")
        self._tools = get_wrapped_trinity_tools()
        self._tool_map = {t.name: t for t in self._tools}
        self._dispatched_jobs: dict[str, list[dict[str, str]]] = {}

    def pop_dispatched_jobs(self, session_id: str) -> list[dict[str, str]]:
        """Return and clear any background jobs dispatched during the current turn."""
        return self._dispatched_jobs.pop(session_id, [])

    def _enrich_from_context_refs(self, context_refs: Sequence[str]) -> str:
        """Fetch empirical real-time data for explicitly tagged context pills."""
        enrichments: list[str] = []
        for ref in context_refs:
            clean_ref = ref.strip()
            if clean_ref.lower() == "@portfolio":
                tool_fn = self._tool_map.get("get_portfolio_overview")
                if tool_fn:
                    try:
                        enrichments.append(f"[ข้อมูลพอร์ตจริง (@Portfolio)]:\n{tool_fn.invoke({})}")
                    except Exception as e:
                        log.debug("Enrich @Portfolio error: %s", e)
            elif clean_ref.lower() == "@macro":
                tool_fn = self._tool_map.get("get_latest_macro_summary")
                if tool_fn:
                    try:
                        enrichments.append(f"[ข้อมูลภาวะเศรษฐกิจ (@Macro)]:\n{tool_fn.invoke({})}")
                    except Exception as e:
                        log.debug("Enrich @Macro error: %s", e)
            elif clean_ref.lower() == "@digest":
                tool_fn = self._tool_map.get("get_latest_daily_digest")
                if tool_fn:
                    try:
                        enrichments.append(f"[สรุปข่าวรายวัน (@Digest)]:\n{tool_fn.invoke({})}")
                    except Exception as e:
                        log.debug("Enrich @Digest error: %s", e)
            elif clean_ref.lower() == "@essence":
                tool_fn = self._tool_map.get("get_investor_essence_and_buckets")
                if tool_fn:
                    try:
                        enrichments.append(f"[กฎ 5 เสาหลัก & 3 ถังเงิน (@Essence)]:\n{tool_fn.invoke({})}")
                    except Exception as e:
                        log.debug("Enrich @Essence error: %s", e)
            elif clean_ref.startswith("$"):
                ticker = clean_ref[1:].strip().upper()
                if ticker:
                    tool_fn = self._tool_map.get("get_equity_report_summary")
                    if tool_fn:
                        try:
                            enrichments.append(f"[บทวิเคราะห์หุ้น {ticker} (${ticker})]:\n{tool_fn.invoke({'ticker': ticker})}")
                        except Exception as e:
                            log.debug("Enrich $%s error: %s", ticker, e)

        return "\n\n".join(enrichments)

    def _load_memory_context(self) -> str:
        sections = []

        # 1. USER.md
        user_md = self._memory_dir / "USER.md"
        if user_md.is_file():
            try:
                sections.append(f"### ข้อมูลผู้ใช้ (USER.md):\n{user_md.read_text(encoding='utf-8')}")
            except Exception:
                pass

        # 2. MEMORY.md
        memory_md = self._memory_dir / "MEMORY.md"
        if memory_md.is_file():
            try:
                sections.append(f"### ข้อสรุปพอร์ตและบทเรียนในอดีต (MEMORY.md):\n{memory_md.read_text(encoding='utf-8')}")
            except Exception:
                pass

        # 3. DREAMS.md
        dreams_md = self._memory_dir / "DREAMS.md"
        if dreams_md.is_file():
            try:
                sections.append(f"### ความรู้ที่ตกผลึก (DREAMS.md):\n{dreams_md.read_text(encoding='utf-8')}")
            except Exception:
                pass

        # 4. Active Skills list
        if self._skills_dir.is_dir():
            active_skills = []
            for sp in self._skills_dir.iterdir():
                if sp.is_dir() and sp.name != "_quarantine":
                    sf = sp / "SKILL.md"
                    if sf.is_file():
                        try:
                            fm, _ = parse_skill_frontmatter(sf.read_text(encoding="utf-8"))
                            desc = fm.get("description", "User or agent skill")
                            active_skills.append(f"- `{sp.name}`: {desc}")
                        except Exception:
                            active_skills.append(f"- `{sp.name}`")
            if active_skills:
                sections.append("### สกิลวิเคราะห์เฉพาะทางที่มีใน Sandbox:\n" + "\n".join(active_skills))

        if sections:
            return "\n\n[บริบทความจำระยะยาวและสกิลใน Sandbox]:\n" + "\n\n".join(sections)
        return ""

    async def stream_reply(
        self,
        session_id: str,
        user_message: str,
        context_refs: Sequence[str] = (),
        history: Sequence[AdvisorChatMessage] = (),
    ) -> AsyncIterator[str]:
        """Stream reply using Gemini LLM with tool resolution and multi-turn session history."""
        memory_context = self._load_memory_context()
        system_text = SYSTEM_PROMPT_TEMPLATE.format(memory_context=memory_context)

        messages: list[Any] = [SystemMessage(content=system_text)]

        # Append prior session history (up to last 20 turns)
        recent_history = list(history)[-20:]
        for msg in recent_history:
            if msg.role == "user":
                messages.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                messages.append(AIMessage(content=msg.content))

        # Append current user prompt with context references and empirical enrichment if selected
        if context_refs:
            cues = ", ".join(context_refs)
            user_msg_lower = user_message.lower()
            is_read_intent = any(
                w in user_msg_lower
                for w in [
                    "ตรวจ",
                    "ขอดู",
                    "ดู",
                    "สรุป",
                    "สถานะ",
                    "thesis",
                    "สัญญา",
                    "เดิม",
                    "ล่าสุด",
                    "เป็นยังไง",
                    "เปลี่ยนไปรึยัง",
                    "เปลี่ยนไหม",
                ]
            )
            is_explicit_run = (
                any(
                    w in user_msg_lower
                    for w in [
                        "สั่งรัน",
                        "รันงานวิจัย",
                        "รันรายงาน",
                        "ดึงข้อมูลใหม่",
                        "สดใหม่",
                        "ข้อมูลสด",
                        "ตั้งแต่ต้น",
                        "macro_intel",
                        "equity_intel",
                    ]
                )
                or ("รัน" in user_msg_lower and "ใหม่" in user_msg_lower)
                or ("วิเคราะห์" in user_msg_lower and "ใหม่" in user_msg_lower)
            )
            is_run_order = is_explicit_run and not is_read_intent
            enriched_facts = await asyncio.to_thread(self._enrich_from_context_refs, context_refs)
            if enriched_facts:
                if is_run_order:
                    effective_content = (
                        f"[บริบทที่ผู้ใช้ระบุ: {cues}]\n\n{enriched_facts}\n\n"
                        f"[คำสั่งของผู้ใช้ (สำคัญ: เป็นคำสั่งสั่งรันงานวิจัยใหม่)]: {user_message}\n"
                        f"[คำแนะนำ: เรียก trigger_deep_research_job ทันที และยืนยันการส่งคำสั่งอย่างกระชับ ห้ามนำข้อมูลรายงานเก่าในอดีตมาสรุปตอบล่วงหน้า]"
                    )
                else:
                    effective_content = f"[บริบทที่ผู้ใช้ระบุ: {cues}]\n\n{enriched_facts}\n\n[คำถามของผู้ใช้]: {user_message}"
            else:
                effective_content = f"[บริบทที่ผู้ใช้ระบุ: {cues}]\n{user_message}"
        else:
            effective_content = user_message

        messages.append(HumanMessage(content=effective_content))

        provider = detect_provider(self._model_name)
        llm = get_llm(provider=provider, model_name=self._model_name, temperature=0.2)
        llm_with_tools = llm.bind_tools(self._tools)

        # Loop to handle tool calling turns if any
        max_tool_turns = 5
        dispatched_jobs: list[tuple[str, str]] = []
        final_text = ""

        for _ in range(max_tool_turns):
            response = await llm_with_tools.ainvoke(messages)
            messages.append(response)

            # If tool calls were made
            if getattr(response, "tool_calls", None) and len(response.tool_calls) > 0:
                for tool_call in response.tool_calls:
                    name = tool_call.get("name")
                    args = tool_call.get("args", {})
                    tool_id = tool_call.get("id", "call_1")

                    tool_fn = self._tool_map.get(name)
                    if tool_fn is not None:
                        try:
                            # Offload synchronous tool execution from event loop
                            result = await asyncio.to_thread(tool_fn.invoke, args)
                            result_str = str(result)
                            if name == "trigger_deep_research_job":
                                m = re.search(r"Job ID:\s*([a-zA-Z0-9_\-]+)", result_str, re.IGNORECASE)
                                j_id = m.group(1) if m else result_str.strip()
                                ticker_arg = str(args.get("ticker", "")).strip().upper()
                                r_type = str(args.get("research_type", "equity_intel" if ticker_arg else "macro_intel")).strip()
                                dispatched_jobs.append((j_id, ticker_arg))
                                if session_id not in self._dispatched_jobs:
                                    self._dispatched_jobs[session_id] = []
                                self._dispatched_jobs[session_id].append({
                                    "job_id": j_id,
                                    "ticker": ticker_arg,
                                    "job_type": r_type,
                                })
                        except Exception as exc:
                            result_str = f"Tool Error: {exc}"
                    else:
                        result_str = f"Error: Tool '{name}' not found."

                    messages.append(
                        ToolMessage(content=result_str, tool_call_id=tool_id)
                    )
            else:
                # No more tool calls: final text reached
                raw_text = _extract_text_from_content(response.content)
                final_text = raw_text.strip()
                break

        # If loop completed without non-tool final response, synthesize one
        if not final_text:
            final_res = await llm.ainvoke(messages)
            messages.append(final_res)
            final_text = _extract_text_from_content(final_res.content).strip()

        # Fallback safety net: If user explicitly instructed to run research but LLM failed to invoke tool
        if not dispatched_jobs:
            user_msg_lower = user_message.lower()
            macro_cues = ["macro_intel", "อัพเดท macro", "อัปเดต macro", "รันรายงานเศรษฐกิจ", "สั่งรันงานวิจัย", "วิเคราะห์ภาพรวมเศรษฐกิจ"]
            has_macro_run_intent = any(c in user_msg_lower for c in macro_cues) or ("@macro" in " ".join(context_refs).lower() and any(k in user_message for k in ["รัน", "อัปเดต", "อัพเดท", "ประมวลผล"]))
            equity_match = re.search(r"(?:วิเคราะห์หุ้น|อัปเดตหุ้น|หุ้น)\s*([A-Za-z]{1,6})\b", user_message)
            has_equity_run_intent = bool(equity_match) or "equity_intel" in user_msg_lower

            trigger_tool = self._tool_map.get("trigger_deep_research_job")
            if trigger_tool:
                if has_macro_run_intent and not has_equity_run_intent:
                    try:
                        res = await asyncio.to_thread(trigger_tool.invoke, {"research_type": "macro_intel", "ticker": ""})
                        m = re.search(r"Job ID:\s*([a-zA-Z0-9_\-]+)", str(res), re.IGNORECASE)
                        j_id = m.group(1) if m else str(res).strip()
                        dispatched_jobs.append((j_id, ""))
                        if session_id not in self._dispatched_jobs:
                            self._dispatched_jobs[session_id] = []
                        self._dispatched_jobs[session_id].append({
                            "job_id": j_id,
                            "ticker": "",
                            "job_type": "macro_intel",
                        })
                    except Exception as e:
                        log.warning("Fallback macro research dispatch failed: %s", e)
                elif has_equity_run_intent:
                    target_sym = equity_match.group(1).upper() if equity_match else ""
                    if not target_sym:
                        for ref in context_refs:
                            if ref.startswith("$"):
                                target_sym = ref[1:].strip().upper()
                                break
                    try:
                        res = await asyncio.to_thread(trigger_tool.invoke, {"research_type": "equity_intel", "ticker": target_sym})
                        m = re.search(r"Job ID:\s*([a-zA-Z0-9_\-]+)", str(res), re.IGNORECASE)
                        j_id = m.group(1) if m else str(res).strip()
                        dispatched_jobs.append((j_id, target_sym))
                        if session_id not in self._dispatched_jobs:
                            self._dispatched_jobs[session_id] = []
                        self._dispatched_jobs[session_id].append({
                            "job_id": j_id,
                            "ticker": target_sym,
                            "job_type": "equity_intel",
                        })
                    except Exception as e:
                        log.warning("Fallback equity research dispatch failed: %s", e)

        # If research jobs were dispatched, guarantee Job ID is in text for UI card rendering
        if dispatched_jobs:
            for j_id, t_sym in dispatched_jobs:
                # Replace any stale/hallucinated UUID if LLM echoed from history
                stale_match = re.search(r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})", final_text)
                if stale_match and stale_match.group(1) != j_id:
                    final_text = final_text.replace(stale_match.group(1), j_id)
                    final_text = re.sub(r"\(อ้างอิงตามคำสั่งล่าสุด\)", "(งานวิเคราะห์ฉบับใหม่)", final_text)

                if j_id and j_id not in final_text and "Job ID" not in final_text:
                    suffix = (
                        f"\n\n(Job ID: {j_id} สำหรับวิเคราะห์หุ้น {t_sym})"
                        if t_sym
                        else f"\n\n(Job ID: {j_id} สำหรับอัปเดตสภาวะเศรษฐกิจมหภาค Macro Strategy)"
                    )
                    final_text += suffix

        if not final_text:
            return

        # Chunk into gentle fragments (supporting Thai + English) for smooth 60fps streaming
        chunks = _split_into_stream_chunks(final_text)
        for chunk in chunks:
            yield chunk
            await asyncio.sleep(0.015)  # 15ms per token smooth cadence
        return
