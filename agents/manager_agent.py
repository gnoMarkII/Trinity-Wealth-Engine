from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Command
from pydantic import BaseModel

from agents.archivist_agent import create_archivist
from agents.researcher_agent import create_researcher
from core.llm_factory import _GOOGLE_FALLBACK_MODEL, get_llm
from core.utils import normalize_content

MANAGER_SYSTEM_PROMPT = """คุณคือ The Manager ผู้จัดการกองทุนส่วนตัว มีความสามารถดังนี้:

[ข้อมูล Real-time — ผ่าน Researcher]
• ดัชนีมหภาค: US 10Y Yield, VIX, Dollar Index (DXY), ทองคำ, น้ำมัน WTI
• ภาพรวมรายภูมิภาค: จีน (MCHI), ยุโรป (VGK), Emerging Markets (EEM), ญี่ปุ่น (EWJ), เอเชียแปซิฟิก (EPP)
• ข้อมูลทั้งหมดจาก Yahoo Finance แบบ Real-time พร้อมบันทึกลง Vault อัตโนมัติ

[ความจำ PKM — ผ่าน Archivist]
• บันทึก Entity: บริษัท, ผู้บริหาร, เหตุการณ์ตลาด, กลยุทธ์การลงทุน
• ค้นหา Semantic: ค้นหาตามความหมายจาก Vault ทั้งหมด
• Graph Context: ดูเครือข่ายความสัมพันธ์ระหว่าง Entity
• ตรวจสุขภาพ Vault: หา Orphan files, Empty files, ความขัดแย้งของข้อมูล

[หลักการทำงาน]
- ต้องการข้อมูล Real-time → Researcher → Archivist บันทึกอัตโนมัติ → จบ
- ต้องการค้นหา/อ่านข้อมูลเก่า → Archivist → จบ
- คำถามทั่วไป → ตอบโดยตรง

ห้ามมั่วข้อมูลเอง — ถ้าต้องการข้อมูลใดให้ดึงจากแหล่งที่เชื่อถือได้ก่อนเสมอ"""


class RouterDecision(BaseModel):
    """การตัดสินใจว่าจะส่งงานไปที่ไหน"""
    next: Literal["archivist", "researcher", "respond"]
    instruction: str


def build_graph(checkpointer=None) -> StateGraph:
    model = get_llm(provider="google", model_name="gemini-3-flash-preview", use_fallback=True)
    archivist_model = get_llm(provider="google", model_name="gemini-3-flash-preview", use_fallback=True)
    researcher_model = get_llm(provider="google", model_name="gemini-3-flash-preview", use_fallback=True)

    archivist_graph = create_archivist(archivist_model)
    researcher_graph = create_researcher(researcher_model)

    # router_model ต้องใช้ with_structured_output ซึ่งไม่มีบน RunnableWithFallbacks
    _router_primary = get_llm(provider="google", model_name="gemini-3-flash-preview")
    _router_fallback = get_llm(provider="google", model_name=_GOOGLE_FALLBACK_MODEL)
    router_model = _router_primary.with_structured_output(RouterDecision).with_fallbacks(
        [_router_fallback.with_structured_output(RouterDecision)]
    )

    ROUTER_PROMPT = """พิจารณาคำถามของผู้ใช้แล้วตัดสินใจ:

- เลือก "researcher" เมื่อต้องการข้อมูล Real-time จากภายนอก เช่น สภาวะตลาด, Bond Yield, \
VIX, Dollar Index, ทองคำ, น้ำมัน, ภาพรวมรายภูมิภาค (จีน/ยุโรป/EM/ญี่ปุ่น/เอเชียแปซิฟิก) \
(Researcher จะส่งข้อมูลให้ Archivist บันทึกโดยอัตโนมัติ ไม่ต้องส่ง archivist ซ้ำ)
- เลือก "archivist" เมื่อต้องการค้นหาหรืออ่านข้อมูลเก่าจาก Vault
- เลือก "respond" เมื่อสามารถตอบได้โดยตรง

instruction: ระบุสิ่งที่ต้องการจาก agent หรือข้อความที่จะตอบผู้ใช้"""

    _COMBINED_PROMPT = MANAGER_SYSTEM_PROMPT + "\n\n" + ROUTER_PROMPT

    def supervisor_node(state: MessagesState) -> Command[Literal["archivist", "researcher", "__end__"]]:
        messages = state["messages"]

        router_messages = [
            {"role": "system", "content": _COMBINED_PROMPT},
            *[{"role": m.type if m.type in ("human", "ai") else "assistant",
               "content": normalize_content(m.content)}
              for m in messages],
        ]

        decision: RouterDecision = router_model.invoke(router_messages)

        if decision.next == "archivist":
            return Command(
                goto="archivist",
                update={"messages": [AIMessage(content=f"[Manager → Archivist] {decision.instruction}")]},
            )

        if decision.next == "researcher":
            return Command(
                goto="researcher",
                update={"messages": [AIMessage(content=f"[Manager → Researcher] {decision.instruction}")]},
            )

        response_messages = [
            {"role": "system", "content": MANAGER_SYSTEM_PROMPT},
            *[{"role": "human" if isinstance(m, HumanMessage) else "assistant",
               "content": normalize_content(m.content)}
              for m in messages],
        ]
        final_response = model.invoke(response_messages)

        return Command(
            goto=END,
            update={"messages": [final_response]},
        )

    def archivist_node(state: MessagesState) -> Command[Literal["__end__"]]:
        messages = state["messages"]

        # ถ้ามีข้อมูลดิบจาก Researcher ให้แนบไปพร้อม task
        _rp = "[Researcher → Manager]"
        researcher_data = next(
            (normalize_content(m.content)[len(_rp):].strip()
             for m in reversed(messages)
             if normalize_content(m.content).startswith(_rp)),
            None,
        )

        if researcher_data:
            task = f"บันทึกข้อมูลดิบต่อไปนี้ลง Vault ทันที\n\n[ข้อมูลดิบ]\n{researcher_data}"
        else:
            task = normalize_content(messages[-1].content)

        result = archivist_graph.invoke({"messages": [HumanMessage(content=task)]})
        archivist_reply = normalize_content(result["messages"][-1].content)

        return Command(
            goto=END,
            update={"messages": [AIMessage(content=f"[Archivist] {archivist_reply}")]},
        )

    def researcher_node(state: MessagesState) -> Command[Literal["archivist"]]:
        last_message = state["messages"][-1]
        researcher_input = {"messages": [HumanMessage(content=last_message.content)]}

        result = researcher_graph.invoke(researcher_input)
        researcher_reply = normalize_content(result["messages"][-1].content)

        return Command(
            goto="archivist",
            update={"messages": [AIMessage(content=f"[Researcher → Manager] {researcher_reply}")]},
        )

    builder = StateGraph(MessagesState)
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("archivist", archivist_node)
    builder.add_node("researcher", researcher_node)
    builder.add_edge(START, "supervisor")

    return builder.compile(checkpointer=checkpointer)
