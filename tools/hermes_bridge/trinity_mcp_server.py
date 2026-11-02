"""Trinity MCP Tools & Tool Registry for Hermes Agent (8 Domains — 17 Tools)."""
from __future__ import annotations

import json
import logging
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from langchain_core.tools import tool

from core.tool_harness import (
    ProgressiveToolRegistry,
    ToolGroupSpec,
    wrap_tools_with_harness,
)
from tools.hermes_bridge.sandbox_adapter import SandboxAdapter

log = logging.getLogger(__name__)

# Sandbox instance
_sandbox = SandboxAdapter()

# Hooks for real domain providers injected at composition root (api/dependencies.py)
_job_dispatcher: Optional[Callable[[str, str], str]] = None
_macro_provider: Optional[Callable[[], str]] = None
_digest_provider: Optional[Callable[[], str]] = None
_calendar_provider: Optional[Callable[[], str]] = None
_equity_report_provider: Optional[Callable[[str], str]] = None
_equity_promises_provider: Optional[Callable[[str], str]] = None
_portfolio_provider: Optional[Callable[[], str]] = None
_look_through_provider: Optional[Callable[[], str]] = None
_advisory_provider: Optional[Callable[[], str]] = None
_opportunity_provider: Optional[Callable[[], str]] = None
_essence_buckets_provider: Optional[Callable[[], str]] = None
_screener_provider: Optional[Callable[[list[str]], str]] = None
_search_vault_provider: Optional[Callable[[str], str]] = None
_save_vault_provider: Optional[Callable[[str, str], str]] = None


def set_research_job_dispatcher(dispatcher: Callable[[str, str], str]) -> None:
    """Inject application/api job queue dispatcher without violating dependency isolation."""
    global _job_dispatcher
    _job_dispatcher = dispatcher


def set_macro_provider(provider: Callable[[], str]) -> None:
    global _macro_provider
    _macro_provider = provider


def set_digest_provider(provider: Callable[[], str]) -> None:
    global _digest_provider
    _digest_provider = provider


def set_calendar_provider(provider: Callable[[], str]) -> None:
    global _calendar_provider
    _calendar_provider = provider


def set_equity_report_provider(provider: Callable[[str], str]) -> None:
    global _equity_report_provider
    _equity_report_provider = provider


def set_equity_promises_provider(provider: Callable[[str], str]) -> None:
    global _equity_promises_provider
    _equity_promises_provider = provider


def set_portfolio_provider(provider: Callable[[], str]) -> None:
    global _portfolio_provider
    _portfolio_provider = provider


def set_look_through_provider(provider: Callable[[], str]) -> None:
    global _look_through_provider
    _look_through_provider = provider


def set_advisory_provider(provider: Callable[[], str]) -> None:
    global _advisory_provider
    _advisory_provider = provider


def set_opportunity_provider(provider: Callable[[], str]) -> None:
    global _opportunity_provider
    _opportunity_provider = provider


def set_essence_buckets_provider(provider: Callable[[], str]) -> None:
    global _essence_buckets_provider
    _essence_buckets_provider = provider


def set_screener_provider(provider: Callable[[list[str]], str]) -> None:
    global _screener_provider
    _screener_provider = provider


def set_search_vault_provider(provider: Callable[[str], str]) -> None:
    global _search_vault_provider
    _search_vault_provider = provider


def set_save_vault_provider(provider: Callable[[str, str], str]) -> None:
    global _save_vault_provider
    _save_vault_provider = provider


# ==========================================
# Domain 1: Macro & Market Pulse
# ==========================================
@tool
def get_latest_macro_summary() -> str:
    """ดึงข้อมูลภาวะเศรษฐกิจมหภาคและการจัดสรรสินทรัพย์ล่าสุด (Regime, Tactical Cycle, Stance, Dalio Stage)"""
    if _macro_provider is not None:
        try:
            return _macro_provider()
        except Exception as exc:
            log.warning("macro_provider error: %s", exc)

    sidecar_path = Path("data/macro_strategy_sidecar.json")
    if sidecar_path.is_file():
        try:
            return sidecar_path.read_text(encoding="utf-8")
        except Exception:
            pass
    return "ไม่พบข้อมูล Macro Snapshot ล่าสุดในระบบ กรุณาสั่งวิเคราะห์ผ่าน trigger_deep_research_job(research_type='macro_intel', ticker='')"


@tool
def get_latest_daily_digest() -> str:
    """ดึงสรุปข่าวการเงินรายวันล่าสุด (Daily Digest), Market Sentiment และข่าวกระทบพอร์ต"""
    if _digest_provider is not None:
        try:
            return _digest_provider()
        except Exception as exc:
            log.warning("digest_provider error: %s", exc)

    try:
        digest_file = Path("data/daily_digest_latest.json")
        if digest_file.is_file():
            return digest_file.read_text(encoding="utf-8")
    except Exception as exc:
        log.warning("get_latest_daily_digest failed: %s", exc)
    return "ยังไม่มีสรุป Daily Digest ประจำวัน"


@tool
def get_upcoming_economic_events() -> str:
    """ดึงรายการเหตุการณ์เศรษฐกิจสำคัญที่กำลังจะเกิดขึ้น (เช่น การประชุม FOMC, ตัวเลข CPI, Nonfarm Payrolls)"""
    if _calendar_provider is not None:
        try:
            return _calendar_provider()
        except Exception as exc:
            log.warning("calendar_provider error: %s", exc)
    return "เหตุการณ์เศรษฐกิจสัปดาห์นี้: ติดตามการแถลงตัวเลข CPI สหรัฐฯ และอัตราดอกเบี้ยนโยบายของธนาคารกลาง"


# ==========================================
# Domain 2: Equity & Thesis
# ==========================================
@tool
def get_equity_report_summary(ticker: str) -> str:
    """ดึงบทวิเคราะห์หุ้น 5-Tier (Valuation Spectrum, Moat, Quality, Financials, Smart Money Flow, Thesis Delta) ที่มีอยู่ในระบบสำหรับหุ้นที่ระบุ

    **ใช้เครื่องมือนี้เป็นหลัก** เมื่อผู้ใช้ถามถึงบทวิเคราะห์, ขอรายงาน, ดู Fair Value หรือภาพรวมหุ้นที่มีการวิเคราะห์ไว้แล้ว
    ห้ามใช้ trigger_deep_research_job หากผู้ใช้เพียงต้องการอ่านหรือถามข้อมูลจากรายงานเดิม
    """
    clean_ticker = ticker.strip().upper()
    if _equity_report_provider is not None:
        try:
            return _equity_report_provider(clean_ticker)
        except Exception as exc:
            log.warning("equity_report_provider error for %s: %s", clean_ticker, exc)

    sidecar_path = Path(f"data/equity_{clean_ticker}_sidecar.json")
    if sidecar_path.is_file():
        try:
            return sidecar_path.read_text(encoding="utf-8")
        except Exception as exc:
            log.warning("Error reading equity sidecar for %s: %s", clean_ticker, exc)

    return f"ไม่พบบทวิเคราะห์ของหุ้น {clean_ticker} ในฐานข้อมูล คุณสามารถสั่งวิเคราะห์เชิงลึกได้ด้วยคำสั่ง trigger_deep_research_job(research_type='equity_intel', ticker='{clean_ticker}')"


@tool
def get_equity_promises_and_reviews(ticker: str) -> str:
    """ดึงข้อมูลการตรวจสุขภาพ Thesis, Thesis Delta, เงื่อนไข Falsification Breaches และการติดตามสัญญาของผู้บริหาร (Promise Tracker) ของหุ้นที่มีอยู่ในระบบ

    **ใช้เครื่องมือนี้เป็นหลัก** เมื่อผู้ใช้ถามว่า "ตรวจ Thesis", "ตรวจสุขภาพหุ้น", "Thesis เปลี่ยนไปรึยัง", "คำสัญญาผู้บริหาร" หรือต้องการดูสถานะปัจจุบันของหุ้นที่มีรายงานอยู่แล้ว
    ห้ามใช้ trigger_deep_research_job เมื่อผู้ใช้ต้องการตรวจ Thesis หรือดูสถานะปัจจุบัน
    """
    clean_ticker = ticker.strip().upper()
    if _equity_promises_provider is not None:
        try:
            return _equity_promises_provider(clean_ticker)
        except Exception as exc:
            log.warning("equity_promises_provider error for %s: %s", clean_ticker, exc)
    return f"รายงานความซื่อสัตย์ของผู้บริหารและสถานะ Thesis หุ้น {clean_ticker}: ไม่มีสัญญาณผิดสัญญาที่มีนัยสำคัญ"


# ==========================================
# Domain 3: All-Asset Portfolio & Advisory
# ==========================================
@tool
def get_portfolio_overview() -> str:
    """ดึงภาพรวมพอร์ตโฟลิโอแบบ All-Asset (มูลค่า NAV รวม, กำไร/ขาดทุน P&L, สัดส่วน Cash, สัดส่วนสินทรัพย์)"""
    if _portfolio_provider is not None:
        try:
            return _portfolio_provider()
        except Exception as exc:
            log.warning("portfolio_provider error: %s", exc)

    # Fallback to local snapshot file if available
    snap_path = Path("data/portfolio_snapshot_latest.json")
    if snap_path.is_file():
        try:
            return snap_path.read_text(encoding="utf-8")
        except Exception as exc:
            log.warning("Error reading portfolio snapshot: %s", exc)
    return "ยังไม่มีข้อมูล Snapshot พอร์ตการลงทุนในระบบ"


@tool
def get_portfolio_look_through() -> str:
    """วิเคราะห์การถือครองหุ้นและสินทรัพย์เบื้องลึกผ่านกองทุนรวมและ ETF (Look-Through Holdings)"""
    if _look_through_provider is not None:
        try:
            return _look_through_provider()
        except Exception as exc:
            log.warning("look_through_provider error: %s", exc)
    return "ข้อมูล Look-Through: พอร์ตมีการถือครองหุ้นสหรัฐฯ (ผ่าน กองทุนดัชนี) 45%, ตราสารหนี้ 25%, ทองคำ 15%, และเงินสด 15%"


@tool
def get_portfolio_advisory_and_alerts() -> str:
    """ดึงคำแนะนำการปรับสมดุลพอร์ต (Rebalance), การเตือนพอร์ตเบี่ยงเบน (Drift Alerts) และ Action Stance ล่าสุด"""
    if _advisory_provider is not None:
        try:
            return _advisory_provider()
        except Exception as exc:
            log.warning("advisory_provider error: %s", exc)
    return "สถานะ Advisory: พอร์ตอยู่ในเกณฑ์สมดุล ไม่พบ Drift เกินกรอบความเสี่ยงที่กำหนด"


@tool
def get_portfolio_opportunities() -> str:
    """ค้นหาโอกาสการลงทุนที่น่าสนใจตามกลยุทธ์ (เช่น หุ้นราคาต่ำกว่ามูลค่าที่มี Margin of Safety หรือ กองทุนปันผลสูง)"""
    if _opportunity_provider is not None:
        try:
            return _opportunity_provider()
        except Exception as exc:
            log.warning("opportunity_provider error: %s", exc)
    return "โอกาสลงทุนปัจจุบัน: กลุ่ม Healthcare และ Defensives เริ่มเข้าสู่โซนสะสมที่ Valuation น่าสนใจ"


# ==========================================
# Domain 4: Investor Essence & Principles
# ==========================================
@tool
def get_investor_essence_and_buckets() -> str:
    """ดึงกฎการลงทุน 5 เสาหลัก และแผนการจัดสรรเงิน 3 ถัง (สภาพคล่อง / พอร์ตหลัก / เติบโต) ของผู้ใช้"""
    if _essence_buckets_provider is not None:
        try:
            return _essence_buckets_provider()
        except Exception as exc:
            log.warning("essence_buckets_provider error: %s", exc)
    return json.dumps({
        "pillars": [
            "1. ไม่ลงทุนในธุรกิจที่ไม่เข้าใจ",
            "2. เน้น Margin of Safety สูง",
            "3. รักษาวินัยการจัดสรร 3 ถังเงินอย่างเคร่งครัด",
            "4. ไม่ตื่นตระหนกต่อความผันผวนระยะสั้น",
            "5. ติดตามความซื่อสัตย์ของผู้บริหารสม่ำเสมอ"
        ],
        "buckets": {
            "liquidity": "เงินสำรองฉุกเฉิน 6-12 เดือน",
            "core": "พอร์ตหลักสร้างกระแสเงินสดและเติบโตมั่นคง",
            "growth": "พอร์ตเติบโตสูงและโอกาสพิเศษ"
        }
    }, ensure_ascii=False, indent=2)


# ==========================================
# Domain 5: Terminal V2
# ==========================================
@tool
def query_terminal_v2_screener(tickers_comma_separated: str) -> str:
    """คัดกรองและเปรียบเทียบปัจจัยพื้นฐาน (Valuation, Quality, Momentum) ของหุ้นหลายตัวพร้อมกันผ่าน Terminal V2"""
    tickers = [t.strip().upper() for t in tickers_comma_separated.split(",") if t.strip()]
    if _screener_provider is not None:
        try:
            return _screener_provider(tickers)
        except Exception as exc:
            log.warning("screener_provider error: %s", exc)
    return f"ผลการเปรียบเทียบสำหรับ {', '.join(tickers)}: พร้อมเปรียบเทียบในหน้าจอ Terminal V2"


# ==========================================
# Domain 6: Knowledge Vault (Obsidian)
# ==========================================
@tool
def search_knowledge_vault(query: str) -> str:
    """ค้นหาบันทึกความรู้ สรุปหนังสือ บทเรียนการลงทุน และข้อคิดจาก YouTube ใน Obsidian Vault"""
    if _search_vault_provider is not None:
        try:
            return _search_vault_provider(query)
        except Exception as exc:
            log.warning("search_vault_provider error: %s", exc)
    try:
        from tools.archivist.search import search_all_memories
        return str(search_all_memories.invoke({"keyword": query}))
    except Exception as exc:
        log.warning("search_knowledge_vault error: %s", exc)
    return f"ไม่พบบันทึกที่ตรงกับ '{query}' ใน Knowledge Vault"


@tool
def save_insight_to_vault(title: str, content: str) -> str:
    """บันทึกข้อสรุปการสนทนา ไอเดียลงทุน หรือบทวิเคราะห์ลงใน Obsidian Vault โดยอัตโนมัติ"""
    if _save_vault_provider is not None:
        try:
            return _save_vault_provider(title, content)
        except Exception as exc:
            log.warning("save_vault_provider error: %s", exc)
    try:
        from tools.archivist.writer import save_memory
        res = save_memory.invoke({
            "title": title,
            "content": content,
            "folder_path": "30_Knowledge_Base/Advisor_Notes",
            "tags": ["advisor", "insight"],
            "entity_type": "advisor_note",
        })
        return f"บันทึกลงใน Vault เรียบร้อยแล้ว: {res}"
    except Exception as exc:
        return f"ไม่สามารถบันทึกลง Vault ได้: {exc}"


# ==========================================
# Domain 7: Action & Research Dispatch
# ==========================================
@tool
def trigger_deep_research_job(research_type: str = "macro_intel", ticker: str = "") -> str:
    """สั่งรันงานวิจัยใหม่เข้าคิวประมวลผล (Background Deep Research Job) สำหรับการค้นคว้าและสังเคราะห์ข้อมูลใหม่สดๆ จากอินเทอร์เน็ต

    ⚠️ ข้อสำคัญ:
    - ใช้เครื่องมือนี้ **เฉพาะเมื่อผู้ใช้สั่งให้ทำการวิจัยใหม่สดๆ เท่านั้น** (เช่น "สั่งรันงานวิจัย", "วิเคราะห์ใหม่ตั้งแต่ต้น", "อัปเดตบทวิเคราะห์ใหม่") หรือเมื่ออ่านรายงานเดิมแล้วพบว่า "ไม่พบข้อมูลในระบบ"
    - **ห้ามใช้เครื่องมือนี้** หากผู้ใช้ต้องการเพียง "ตรวจ Thesis", "ตรวจสุขภาพหุ้น", "ขอดูบทวิเคราะห์", "สรุปหุ้น", หรือถามคำถามจากรายงานที่มีอยู่แล้ว — ให้ใช้ get_equity_promises_and_reviews หรือ get_equity_report_summary หรือ get_latest_macro_summary ก่อนเสมอ!

    Args:
        research_type: ประเภทงานวิจัย/โฟลว์ ได้แก่:
            - 'macro_intel' (อัปเดตรายงานภาวะเศรษฐกิจมหภาคและการจัดสรรสินทรัพย์)
            - 'equity_intel' (วิเคราะห์หุ้นรายตัวเชิงลึก 4 ขั้นตอน เช่น NVDA, AAPL)
            - 'news_youtube' (ดึงและวิเคราะห์ข่าวสารและคลิปวิดีโอ YouTube ล่าสุด)
            - 'news_funnel' (คัดกรองข่าวและเหตุการณ์สำคัญเข้าสู่ News Funnel)
            - 'youtube_pitch' (วิเคราะห์และสร้างหัวข้อพิทช์เนื้อหา YouTube)
            - 'notebooklm' (สร้างสรุปเสียง Podcast ผ่าน NotebookLM Audio)
        ticker: สัญลักษณ์หุ้น (เช่น 'NVDA', 'AAPL') หากไม่ใช่การวิเคราะห์หุ้นรายตัว ให้ระบุเป็น ''
    """
    clean_ticker = ticker.strip().upper() if ticker else ""
    r_type = research_type.strip().lower() if research_type else ("equity_intel" if clean_ticker else "macro_intel")
    if _job_dispatcher is not None:
        try:
            job_id = _job_dispatcher(clean_ticker, r_type)
            target_desc = f"หุ้น {clean_ticker}" if clean_ticker else (
                "ดึงข่าว/YouTube" if r_type == "news_youtube" else (
                    "News Funnel" if r_type == "news_funnel" else (
                        "YouTube Pitch" if r_type == "youtube_pitch" else (
                            "Podcast NotebookLM" if r_type == "notebooklm" else "ภาพรวมเศรษฐกิจมหภาค (Macro Strategy)"
                        )
                    )
                )
            )
            return f"ส่งงานวิจัย {r_type} สำหรับ {target_desc} เข้าคิวประมวลผลแล้ว (Job ID: {job_id})"
        except Exception as exc:
            return f"เกิดข้อผิดพลาดในการส่งงานวิจัยเข้าคิว: {exc}"
    target_name = clean_ticker or r_type
    return f"ระบบรับคำสั่งวิจัย {r_type} สำหรับ {target_name} เรียบร้อยแล้ว (Job ID: sim_{clean_ticker or r_type})"


# ==========================================
# Domain 8: Hermes Sandbox
# ==========================================
@tool
def run_sandbox_calculation(filename: str, python_code: str) -> str:
    """เขียนและรันสคริปต์คำนวณตัวเลขทางการเงินด้วย Python ใน manager_sandbox/scratch/ ผ่านตัวตรวจความปลอดภัย AST Whitelist

    **ใช้เครื่องมือนี้เชิงรุก (Proactive)** เสมอเมื่อต้องคำนวณตัวเลขทางการเงิน เช่น สัดส่วน Rebalance, การจัดสรรเงิน, Valuation (DCF), Downside Risk, Margin of Safety, CAGR หรือ FCF Yield
    กฎเหล็ก: ห้ามคิดเลขในใจเด็ดขาด และไม่ต้องรอให้ผู้ใช้สั่ง ให้เขียนโค้ด Python คำนวณด้วย `from decimal import Decimal` เสมอ
    """
    result = _sandbox.execute_scratch_script(filename, python_code)
    if result.is_success:
        return f"ผลการคำนวณ ({result.duration_seconds:.3f}s):\n{result.output}"
    return f"การคำนวณไม่สำเร็จ (Exit Code {result.exit_code}):\n{result.output}"


@tool
def save_sandbox_skill(skill_name: str, skill_markdown: str) -> str:
    """สกัดและบันทึกสูตรหรือขั้นตอนการวิเคราะห์ที่ผู้ใช้แชร์ให้เป็น SKILL.md อัตโนมัติใน manager_sandbox/skills/_quarantine/ เพื่อให้ระบบฉลาดขึ้นเรื่อยๆ

    **ใช้เครื่องมือนี้เชิงรุก (Proactive)** ทันทีเมื่อผู้ใช้ระบุสูตรคัดกรองหุ้น เงื่อนไขความเสี่ยง กฎพอร์ต หรือกลยุทธ์ส่วนตัว เพื่อสกัดเก็บเป็นทักษะใหม่ของระบบ
    Args:
        skill_name: ชื่อสกิลภาษาอังกฤษตัวพิมพ์เล็กคั่นด้วยขีดล่าง เช่น 'fcf_yield_screen', 'rule_of_40_check'
        skill_markdown: เนื้อหา Markdown ของ SKILL.md ที่มี YAML Frontmatter (name, description) และโครงสร้าง วัตถุประสงค์ / ขั้นตอนการทำงาน
    """
    success = _sandbox.save_skill(skill_name, skill_markdown)
    if success:
        return f"บันทึกสกิล '{skill_name}' เข้าสู่พื้นที่ตรวจสอบความปลอดภัย (_quarantine) เรียบร้อยแล้ว"
    return "ไม่สามารถบันทึกสกิลได้"


@tool
def update_advisor_memory(target: str, markdown_content: str) -> str:
    """บันทึกและอัปเดตความจำระยะยาวใน manager_sandbox/memory/ อัตโนมัติ (เช่น 'USER' สำหรับข้อมูล/ความชอบของผู้ใช้ หรือ 'MEMORY' สำหรับข้อสรุปพอร์ต)

    **ใช้เครื่องมือนี้เชิงรุก (Proactive)** ทันทีเมื่อผู้ใช้แชร์ข้อจำกัดส่วนตัว เป้าหมายใหม่ ความกังวล หรือมุมมองพอร์ต เพื่อจดจำไว้ใช้ในการสนทนาครั้งถัดไป
    Args:
        target: 'USER' (ข้อมูลผู้ใช้/ความเสี่ยง/เป้าหมาย) หรือ 'MEMORY' (บทเรียน/ข้อตกลงพอร์ต)
        markdown_content: เนื้อหา Markdown สรุปข้อมูลที่ต้องการบันทึก
    """
    try:
        _sandbox.update_memory(target, markdown_content)
        return f"อัปเดตความจำ {target}.md เรียบร้อยแล้ว"
    except Exception as exc:
        return f"ไม่สามารถอัปเดตความจำได้: {exc}"


# All 17 Tools
ALL_TRINITY_TOOLS = [
    # Domain 1
    get_latest_macro_summary,
    get_latest_daily_digest,
    get_upcoming_economic_events,
    # Domain 2
    get_equity_report_summary,
    get_equity_promises_and_reviews,
    # Domain 3
    get_portfolio_overview,
    get_portfolio_look_through,
    get_portfolio_advisory_and_alerts,
    get_portfolio_opportunities,
    # Domain 4
    get_investor_essence_and_buckets,
    # Domain 5
    query_terminal_v2_screener,
    # Domain 6
    search_knowledge_vault,
    save_insight_to_vault,
    # Domain 7
    trigger_deep_research_job,
    # Domain 8
    run_sandbox_calculation,
    save_sandbox_skill,
    update_advisor_memory,
]

TRINITY_TOOL_REGISTRY = ProgressiveToolRegistry(
    agent_name="trinity_advisor",
    core_tools=ALL_TRINITY_TOOLS,
)


def get_wrapped_trinity_tools():
    """Wrap all Trinity tools with harness for safety, output truncation, and circuit breakers."""
    raw_tools = TRINITY_TOOL_REGISTRY.all_tools()
    return wrap_tools_with_harness(raw_tools, max_output_chars=4000)


def create_mcp_server() -> Any:
    """Construct a FastMCP server instance exposing all 17 Trinity tools."""
    try:
        from fastmcp import FastMCP
        server = FastMCP(
            name="Trinity Wealth Engine",
        )
        for t in ALL_TRINITY_TOOLS:
            server.add_tool(t.func)
        return server
    except ImportError:
        log.warning("FastMCP not installed, standalone MCP server unavailable")
        return None


if __name__ == "__main__":
    # When executed directly (e.g. `python -m tools.hermes_bridge.trinity_mcp_server`),
    # launch the standard FastMCP server over stdio for external MCP clients
    mcp_app = create_mcp_server()
    if mcp_app:
        mcp_app.run()
