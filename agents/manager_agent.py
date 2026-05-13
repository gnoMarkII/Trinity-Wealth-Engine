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

[Macro — ผ่าน Researcher]
• Macro (19 ดัชนี, 7 หมวด): Yield Curve (13W/5Y/10Y/30Y), VIX, Credit (HYG/LQD), DXY/EUR/JPY/CNY, Gold/Oil/Gas/Copper, S&P500/Nasdaq/Russell, Bitcoin
• Sector Rotation (11 GICS): XLC, XLY, XLP, XLE, XLF, XLV, XLI, XLB, XLRE, XLK, XLU — เรียงตาม % change วันนี้
• ภูมิภาค (7 ตลาด): ลาตินอเมริกา (ILF), ยุโรป (VGK), EM (EEM), ญี่ปุ่น (EWJ), อินเดีย (INDA), จีน (MCHI), เอเชียแปซิฟิก (EPP)
• Hard Data (19 ดัชนี, 6 หมวด) จาก FRED: Fed Rate/2Y Yield/Spread, CPI/PCE/Core PCE/PPI/Breakeven 5Y+10Y, BAA Credit Spread, HY Bond Spread, Unemployment/Claims, GDP/INDPRO/Retail/Housing, M2/Consumer Sentiment

[Micro — ผ่าน Researcher]
• หุ้นรายตัว (US Stocks) — ระบุ Ticker เช่น AAPL, MSFT, NVDA:
  - Fundamentals: P/E, EV/EBITDA, P/B, ROE, Profit Margin, Revenue Growth, Market Cap, Beta, Payout Ratio, ESG Score
  - Financial Trends: รายได้รวม, กำไรสุทธิ ย้อนหลัง 4 ปีงบการเงิน
  - Financial Health: Operating/Free Cash Flow, Total Cash/Debt, Debt/Equity, Current Ratio
  - Momentum: MA50, MA200, 52W High/Low, % Insider/Institution Hold, Short Ratio, Short % Float
  - Analyst Consensus: ราคาเป้าหมาย (Low/Mean/High), Recommendation, จำนวนนักวิเคราะห์
  - News: พาดหัวข่าวล่าสุด 5 ข่าว

[ความจำ PKM — ผ่าน Archivist]
• บันทึก Entity: บริษัท, ผู้บริหาร, เหตุการณ์ตลาด, กลยุทธ์การลงทุน
• ค้นหา Semantic: ค้นหาตามความหมายจาก Vault ทั้งหมด
• Graph Context: ดูเครือข่ายความสัมพันธ์ระหว่าง Entity
• ตรวจสุขภาพ Vault: หา Orphan files, Empty files, ความขัดแย้งของข้อมูล

[กฎเหล็ก]
ห้ามนำข้อมูลดิบที่ Researcher ดึงมาสรุป อธิบาย หรือวิเคราะห์ซ้ำในคำตอบเด็ดขาด
ห้ามมั่วข้อมูลเอง — ถ้าต้องการข้อมูลใดให้ดึงจากแหล่งที่เชื่อถือได้ก่อนเสมอ"""


class RouterDecision(BaseModel):
    """การตัดสินใจว่าจะส่งงานไปที่ไหน"""
    next: Literal["archivist", "researcher", "respond"]
    instruction: str


def _msg_role(m) -> str:
    return "human" if isinstance(m, HumanMessage) else "assistant"


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

- เลือก "researcher" เมื่อต้องการข้อมูลจากภายนอก เช่น สภาวะตลาด, Bond Yield, \
VIX, Dollar Index, ทองคำ, น้ำมัน, ภาพรวมรายภูมิภาค (จีน/ยุโรป/EM/ญี่ปุ่น/เอเชียแปซิฟิก), \
กลุ่มอุตสาหกรรมสหรัฐฯ, Sector, Sector Rotation, กระแสเงินไหลเข้ากลุ่มไหน, \
ตัวเลขเศรษฐกิจพื้นฐาน (เงินเฟ้อ, CPI, PCE, Core PCE, การว่างงาน, Unemployment, GDP, ดอกเบี้ย Fed, Hard Data), \
ข้อมูลหุ้นรายตัว ไม่ว่าจะเป็น Fundamentals, P/E, EV/EBITDA, ROE, Margin, Revenue Growth, Market Cap, \
แนวโน้มงบการเงิน, รายได้, กำไรย้อนหลัง, สุขภาพการเงิน, กระแสเงินสด, Free Cash Flow, \
Operating Cash Flow, หนี้สิน, Debt/Equity, Current Ratio, โมเมนตัม, MA50, MA200, \
52W High, 52W Low, Short Interest, Institution Hold, Analyst Consensus, ราคาเป้าหมาย, \
ข่าวล่าสุด, ESG Score ของ Ticker ใดก็ตาม \
(Researcher จะส่งข้อมูลให้ Archivist บันทึกโดยอัตโนมัติ ไม่ต้องส่ง archivist ซ้ำ)
- เลือก "archivist" เมื่อต้องการค้นหาหรืออ่านข้อมูลเก่าจาก Vault
- เลือก "respond" เมื่อสามารถตอบได้โดยตรง

instruction: ระบุสิ่งที่ต้องการจาก agent หรือข้อความที่จะตอบผู้ใช้"""

    _COMBINED_PROMPT = MANAGER_SYSTEM_PROMPT + "\n\n" + ROUTER_PROMPT
    _ROUTER_HISTORY_LIMIT = 20

    def supervisor_node(state: MessagesState) -> Command[Literal["archivist", "researcher", "__end__"]]:
        messages = state["messages"]

        router_messages = [
            {"role": "system", "content": _COMBINED_PROMPT},
            *[{"role": _msg_role(m), "content": normalize_content(m.content)}
              for m in messages[-_ROUTER_HISTORY_LIMIT:]],
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
            *[{"role": _msg_role(m), "content": normalize_content(m.content)}
              for m in messages[-_ROUTER_HISTORY_LIMIT:]],
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
            raw = normalize_content(messages[-1].content)
            task = raw[len("[Manager → Archivist]"):].strip() if raw.startswith("[Manager → Archivist]") else raw

        result = archivist_graph.invoke({"messages": [HumanMessage(content=task)]})
        archivist_reply = normalize_content(result["messages"][-1].content)

        return Command(
            goto=END,
            update={"messages": [AIMessage(content=f"[Archivist] {archivist_reply}")]},
        )

    def researcher_node(state: MessagesState) -> Command[Literal["archivist", "__end__"]]:
        last_message = state["messages"][-1]
        raw = normalize_content(last_message.content)
        instruction = raw[len("[Manager → Researcher]"):].strip() if raw.startswith("[Manager → Researcher]") else raw
        researcher_input = {"messages": [HumanMessage(content=instruction)]}

        result = researcher_graph.invoke(researcher_input)
        researcher_reply = normalize_content(result["messages"][-1].content)

        if researcher_reply.strip().upper().startswith("ERROR"):
            return Command(
                goto=END,
                update={"messages": [AIMessage(content=researcher_reply)]},
            )

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
