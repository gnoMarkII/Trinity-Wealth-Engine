from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Command
from pydantic import BaseModel

from agents.archivist_agent import create_archivist
from core.llm_factory import _GOOGLE_FALLBACK_MODEL, get_llm

MANAGER_SYSTEM_PROMPT = """คุณคือ The Manager ผู้จัดการกองทุนส่วนตัว

คุณมี Sub-agent ชื่อ Archivist คอยช่วยจำข้อมูล

เมื่อผู้ใช้ให้ข้อมูลใหม่ (เช่น กฎความเสี่ยง เป้าหมาย ข้อมูลพอร์ต) หรือเมื่อคุณต้องการ\
ข้อมูลเก่า ให้คุณส่งคำสั่ง (Delegate) ไปให้ Archivist จัดการบันทึก/ค้นหา\
ในแฟ้มของ Agent ที่เกี่ยวข้อง (เช่น Risk_Officer, Portfolio_Manager, Investment_Policy)

ห้ามมั่วข้อมูลเอง — ถ้าต้องการข้อมูลใดให้สั่ง Archivist ดึงมาก่อน

เมื่อคุณตัดสินใจจะตอบผู้ใช้โดยตรง (ไม่ต้อง delegate) ให้ตอบชัดเจนและตรงประเด็น"""


class RouterDecision(BaseModel):
    """การตัดสินใจว่าจะส่งงานไปที่ไหน"""
    next: Literal["archivist", "respond"]
    instruction: str


def build_graph(checkpointer=None) -> StateGraph:
    # model และ archivist_model: ใช้ use_fallback=True เพื่อรองรับ .invoke()/.stream() ที่ขัดข้อง
    model = get_llm(provider="google", model_name="gemini-3-flash-preview", use_fallback=True)
    archivist_model = get_llm(provider="google", model_name="gemini-3-flash-preview", use_fallback=True)

    archivist_graph = create_archivist(archivist_model)

    # router_model ต้องใช้ with_structured_output ซึ่งไม่มีบน RunnableWithFallbacks
    # จึงสร้าง fallback chain ด้วย BaseChatModel โดยตรงแล้วผูกหลัง with_structured_output
    _router_primary = get_llm(provider="google", model_name="gemini-3-flash-preview")
    _router_fallback = get_llm(provider="google", model_name=_GOOGLE_FALLBACK_MODEL)
    router_model = _router_primary.with_structured_output(RouterDecision).with_fallbacks(
        [_router_fallback.with_structured_output(RouterDecision)]
    )

    ROUTER_PROMPT = """พิจารณาการสนทนาล่าสุดและตัดสินใจว่าจะทำอะไรต่อไป:

- เลือก "archivist" เมื่อต้องการบันทึกข้อมูล, ดึงข้อมูล, หรือค้นหาข้อมูลจากระบบความจำ
- เลือก "respond" เมื่อสามารถตอบผู้ใช้ได้โดยตรงหรือทราบผลจาก Archivist แล้ว

พร้อม instruction: คำสั่งที่ชัดเจนสำหรับ Archivist (ถ้าเลือก archivist) \
หรือบทสรุปที่จะตอบ (ถ้าเลือก respond)"""

    def supervisor_node(state: MessagesState) -> Command[Literal["archivist", "__end__"]]:
        messages = state["messages"]

        router_messages = [
            {"role": "system", "content": MANAGER_SYSTEM_PROMPT + "\n\n" + ROUTER_PROMPT},
            *[{"role": m.type if m.type in ("human", "ai") else "assistant", "content": m.content}
              for m in messages],
        ]

        decision: RouterDecision = router_model.invoke(router_messages)

        if decision.next == "archivist":
            return Command(
                goto="archivist",
                update={"messages": [AIMessage(content=f"[Manager → Archivist] {decision.instruction}")]},
            )

        response_messages = [
            {"role": "system", "content": MANAGER_SYSTEM_PROMPT},
            *[{"role": "human" if isinstance(m, HumanMessage) else "assistant", "content": m.content}
              for m in messages],
        ]
        final_response = model.invoke(response_messages)

        return Command(
            goto=END,
            update={"messages": [final_response]},
        )

    def archivist_node(state: MessagesState) -> Command[Literal["supervisor"]]:
        last_message = state["messages"][-1]
        archivist_input = {"messages": [HumanMessage(content=last_message.content)]}

        result = archivist_graph.invoke(archivist_input)
        archivist_reply = result["messages"][-1].content

        return Command(
            goto="supervisor",
            update={"messages": [AIMessage(content=f"[Archivist → Manager] {archivist_reply}")]},
        )

    builder = StateGraph(MessagesState)
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("archivist", archivist_node)
    builder.add_edge(START, "supervisor")

    return builder.compile(checkpointer=checkpointer)
