"""News Funnel Pipeline สำหรับสถาปัตยกรรมคัดกรองข่าวและสร้าง Obsidian Linking Key

ครอบคลุม:
1. canonicalize_ticker_names แปลงชื่อหุ้นและสัญลักษณ์ให้เป็นมาตรฐาน
2. ensure_concept_stubs_exist สร้าง stub พื้นฐานในโฟลเดอร์ 30_Knowledge_Base/Concepts/
3. run_news_funnel_ingest ดึงข่าว ทำ clustering, คัดกรองผ่าน Batch LLM และบันทึกสถานะลง JSON Store
4. run_news_funnel_synthesize สร้างโน้ตข่าวเดี่ยวสำคัญลงใน 30_Knowledge_Base/News/ พร้อมระบบ Zero-Pending Protection
"""
from datetime import datetime
import os
from pathlib import Path
import re
import shutil
from typing import Any, Dict, List, Optional
import uuid

from core.logger import get_logger
from core.nlp_utils import (
    group_similar_news,
    select_representative_news,
)
from schemas.news_funnel_schemas import (
    MacroImpactTriageResult,
    TriageBatchResult,
    strip_wikilink,
)
from tools._atomic_io import _atomic_write_to
from tools.archivist.core import _sanitize_filename
from tools.knowledge.core import _build_article_md
from tools.macro.news_funnel_store import (
    get_pending_high_impact_events,
    is_title_or_url_processed,
    load_store,
    save_triage_events,
    update_events_status,
)

logger = get_logger(__name__)


def _is_mock_mode() -> bool:
    """ตรวจสอบว่าเปิดใช้งาน MOCK_NEWS_FUNNEL_LLM สำหรับเทสต์หรือออฟไลน์หรือไม่"""
    return os.getenv("MOCK_NEWS_FUNNEL_LLM", "false").lower() == "true"


def get_synthesis_period(now: Optional[datetime] = None) -> str:
    """คืนรอบสังเคราะห์ปัจจุบัน: morning (ก่อนเที่ยง) หรือ evening — จุดเดียวที่กำหนด cutoff"""
    current = now or datetime.now()
    return "morning" if current.hour < 12 else "evening"


def _invoke_structured(schema: Any, model_env: str, prompt_lines: List[str]) -> Any:
    """Helper สำหรับสร้างและเรียกใช้ structured LLM ด้วย provider='google'"""
    from core.llm_factory import get_llm
    model_name = os.getenv(model_env, "gemini-2.5-flash")
    llm = get_llm(provider="google", model_name=model_name)
    structured_llm = llm.with_structured_output(schema)
    return structured_llm.invoke("\n".join(prompt_lines))


TICKER_ALIAS_MAP: Dict[str, str] = {
    "NVIDIA": "NVDA",
    "APPLE": "AAPL",
    "MICROSOFT": "MSFT",
    "ALPHABET": "GOOGL",
    "GOOGLE": "GOOGL",
    "AMAZON": "AMZN",
    "META": "META",
    "TESLA": "TSLA",
    "PTT": "PTT",
}


def _clean_and_truncate_summary(text: str, max_len: int = 500) -> str:
    """ทำความสะอาด HTML tags จาก summary ด้วย BeautifulSoup ก่อนทำการตัดคำ (truncate) เพื่อไม่ให้ tag ว่างหรือ HTML ขาดกลางเมื่อเจอ tag ยาว"""
    if not text:
        return ""
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(text, "html.parser")
        clean = soup.get_text(separator=" ", strip=True)
    except Exception:
        clean = re.sub(r"<[^>]+>", " ", text).strip()
        clean = re.sub(r"\s+", " ", clean)

    if len(clean) > max_len:
        return clean[:max_len].rsplit(" ", 1)[0] + "..." if " " in clean[:max_len] else clean[:max_len] + "..."
    return clean


def canonicalize_ticker_names(tickers: List[str]) -> List[str]:
    """แปลงชื่อหรือสัญลักษณ์สินทรัพย์/หุ้นให้เป็น Canonical Ticker มาตรฐาน (แบบไม่ซ้ำ)"""
    result = []
    seen = set()
    for t in tickers:
        clean = strip_wikilink(t)
        upper = clean.upper()
        canonical = TICKER_ALIAS_MAP.get(upper, clean)
        if canonical not in seen and canonical:
            seen.add(canonical)
            result.append(canonical)
    return result


def ensure_concept_stubs_exist(
    concepts: List[str],
    vault_root: Optional[str] = None,
) -> List[str]:
    """สร้าง Concept Stub ใน 30_Knowledge_Base/Concepts/ สำหรับคำที่ไม่ใช่รหัสหุ้นมาตรฐาน หากยังไม่มี"""
    root = vault_root or os.getenv("OBSIDIAN_VAULT_PATH", "./memories")
    concepts_dir = Path(root) / "30_Knowledge_Base" / "Concepts"
    concepts_dir.mkdir(parents=True, exist_ok=True)

    # Complete Legacy Cleanup: ย้าย Concept Stubs จาก News/Concepts/ ไปที่ 30_Knowledge_Base/Concepts/ พร้อมลบโฟลเดอร์ News/Concepts/ ออกทันที
    old_concepts_dir = Path(root) / "30_Knowledge_Base" / "News" / "Concepts"
    if old_concepts_dir.exists() and old_concepts_dir.is_dir():
        for old_file in old_concepts_dir.glob("*.md"):
            new_file = concepts_dir / old_file.name
            if not new_file.exists():
                try:
                    content = old_file.read_text(encoding="utf-8")
                    _atomic_write_to(new_file, content)
                except Exception as e:
                    logger.error("Failed to migrate concept stub %s: %s", old_file, e)
        shutil.rmtree(old_concepts_dir, ignore_errors=True)

    created = []
    today_str = datetime.now().strftime("%Y-%m-%d")
    for concept in concepts:
        clean = strip_wikilink(concept)
        if not clean:
            continue
        safe_name = _sanitize_filename(clean)
        file_path = concepts_dir / f"{safe_name}.md"
        if not file_path.exists():
            md_content = (
                f"---\n"
                f'title: "{clean}"\n'
                f"created: {today_str}\n"
                f"entity_type: concept_stub\n"
                f"---\n\n"
                f"# {clean}\n\n"
                f"<!-- Concept Stub created automatically by News Funnel linking key -->\n"
            )
            try:
                _atomic_write_to(file_path, md_content)
                created.append(str(file_path))
                logger.info("Created concept stub: %s", file_path)
            except Exception as e:
                logger.error("Failed to create concept stub %s: %s", file_path, e)

    return created


def _mock_or_llm_triage(candidate: Dict[str, Any]) -> MacroImpactTriageResult:
    """ประเมิน Impact Score (fallback heuristic สำหรับ deterministic unit tests)"""
    title = candidate.get("title", "")
    summary = _clean_and_truncate_summary(candidate.get("summary", ""))
    text = f"{title} {summary}".lower()

    macro_score = 5
    asset_score = 5
    tags = ["macro"]
    tickers = []
    themes = ["policy"]

    if re.search(r"\b(fed|rate|inflation|cpi|policy|treasury)\b|ดอกเบี้ย|เงินเฟ้อ", text):
        macro_score = 8
        themes.append("inflation")
        if re.search(r"\b(fed|rate)\b", text):
            themes.append("policy")
        tags.append("policy")

    if re.search(r"\b(nvda|nvidia|ai|server|chip)\b|ชิป", text):
        asset_score = 8
        tickers.extend(["NVDA"])
        themes.append("earnings")
        tags.append("tech")

    if re.search(r"\b(oil|ptt|energy|gold)\b|น้ำมัน|ทองคำ", text):
        macro_score = max(macro_score, 7)
        asset_score = max(asset_score, 7)
        if re.search(r"\bptt\b|น้ำมัน", text):
            tickers.append("PTT")
        if re.search(r"\bgold\b|ทองคำ", text):
            tickers.append("Gold")
        themes.append("geopolitics")
        tags.append("commodities")

    canonical_tickers = canonicalize_ticker_names(tickers)

    return MacroImpactTriageResult(
        macro_impact_score=macro_score,
        asset_impact_score=asset_score,
        primary_tags=tags,
        extracted_tickers=canonical_tickers,
        extracted_themes=list(set(themes)),
        triage_reasoning=f"Automated evaluation based on macro impact indicators ({macro_score}/10, {asset_score}/10)",
        thai_title=f"[TH] {title}" if title else "หัวข้อข่าวจำลอง",
        thai_summary=f"สรุปประเด็นข่าวภาษาไทย: {summary}" if summary else "สรุปเนื้อหาสำคัญของข่าวเป็นภาษาไทย...",
    )


def _llm_triage_batch(items: List[Dict[str, Any]]) -> List[MacroImpactTriageResult]:
    """ประเมิน Impact Score แบบ Batch ผ่าน Fast LLM (Gemini Flash) หรือ fallback mock"""
    if not items:
        return []
    if _is_mock_mode():
        return [_mock_or_llm_triage(item) for item in items]

    try:
        prompt_lines = [
            "Evaluate macro impact score (1-10) and asset impact score (1-10) for each news item below.",
            "IMPORTANT requirement: You MUST provide 'thai_title' (accurate headline translated into THAI language) and 'thai_summary' (comprehensive, analytical summary written in THAI language 1-2 paragraphs for Thai investors). All summary content MUST be in professional THAI language.",
            "Rubric:",
            "- Score >= 7 (HIGH IMPACT): Systemic macro shift, central bank rate decision, critical policy change, inflation surprise, systemic shock, or market-moving earnings/catalyst (e.g. NVDA, PTT).",
            "- Score < 7 (ROUTINE IMPACT): Minor commentary, routine update, or localized news without broad market impact.",
            "News items to evaluate:"
        ]
        for idx, it in enumerate(items):
            prompt_lines.append(f"{idx+1}. Title: {it.get('title', '')} | Summary: {it.get('summary', '')}")

        res = _invoke_structured(TriageBatchResult, "NEWS_FUNNEL_TRIAGE_MODEL", prompt_lines)
        if res and hasattr(res, "results") and len(res.results) == len(items):
            return res.results
        if res and hasattr(res, "results"):
            logger.error("LLM returned %d results for %d items — skipping batch", len(res.results), len(items))
        else:
            logger.error("LLM returned invalid response — skipping batch")
    except Exception as e:
        logger.error("LLM Triage batch failed: %s", e)

    # ฟังก์ชันนี้ไม่คืนคะแนน default ปลอม — คืน [] เมื่อ LLM ล้มเหลว แล้วให้ caller
    # ตัดสินใจ fallback เอง (ingest จะ tag triage_source="heuristic_fallback" ให้แยกออกจากคะแนน LLM จริง)
    return []


def run_news_funnel_ingest(
    candidates: Optional[List[Dict[str, Any]]] = None,
    store_path: Optional[str] = None,
) -> Dict[str, Any]:
    """ดึงข่าว (หรือรับจาก candidates) ทำ Clustering, คัดกรอง Batch LLM และบันทึกสถานะลง JSON Store"""
    items = candidates
    if items is None:
        try:
            from tools.macro.news_radar import get_news_candidates
            items = get_news_candidates()
        except Exception as e:
            logger.warning("Could not fetch from news_radar: %s", e)
            items = []

    if not items:
        return {"status": "success", "ingested_count": 0, "high_impact_count": 0}

    # โหลด store state ครั้งเดียวเพื่อกรองซ้ำ
    store_state = load_store(store_path=store_path)
    unprocessed = []
    for item in items:
        title = item.get("title", "").strip()
        link = item.get("link", "")
        if not title:
            continue
        if is_title_or_url_processed(title, link, store_path=store_path, store_state=store_state):
            continue
        unprocessed.append(item)

    if not unprocessed:
        return {"status": "success", "ingested_count": 0, "high_impact_count": 0}

    # Clustering ข่าวที่คล้ายกันเข้าด้วยกัน
    clusters = group_similar_news(unprocessed, threshold=0.75)
    representatives = []
    for cluster in clusters:
        rep = select_representative_news(cluster)
        all_sources = list(set(x.get("source", "RSS") for x in cluster if x.get("source")))
        all_links = list(set(x.get("link", "") for x in cluster if x.get("link")))
        rep["sources"] = all_sources or ["RSS"]
        rep["links"] = all_links
        rep["sources_count"] = len(cluster)
        representatives.append(rep)

    # ประเมินผ่าน LLM Batch
    triage_results = _llm_triage_batch(representatives)
    triage_sources = ["mock" if _is_mock_mode() else "llm"] * len(triage_results)

    # หาก len ไม่ตรงกัน หรือ LLM ล้มเหลว ให้ fallback ประเมินทีละ item เพื่อไม่ให้เกิด starvation loop จาก item เสียตัวเดียว
    # ผลจาก heuristic ถูก tag triage_source="heuristic_fallback" เพื่อให้ผู้ใช้เห็นบนการ์ด Kanban ว่าไม่ใช่คะแนน LLM จริง
    if len(triage_results) != len(representatives):
        logger.warning("LLM returned %d results for %d items — falling back to sequential triage per item to prevent starvation", len(triage_results), len(representatives))
        triage_results = []
        triage_sources = []
        for rep in representatives:
            single_res = _llm_triage_batch([rep])
            if len(single_res) == 1:
                triage_results.append(single_res[0])
                triage_sources.append("llm")
            else:
                logger.warning("Sequential triage failed for item '%s' — using heuristic fallback", rep.get("title", ""))
                triage_results.append(_mock_or_llm_triage(rep))
                triage_sources.append("heuristic_fallback")

    new_events = []
    now_iso = datetime.now().isoformat()
    for rep, triage, triage_source in zip(representatives, triage_results, triage_sources):
        title = rep.get("title", "").strip()
        link = rep.get("link", "")
        # สร้าง event_id จาก UUID4 หรือ Hash
        event_id = rep.get("event_id") or str(uuid.uuid4())
        ev = {
            "event_id": event_id,
            "canonical_title": triage.thai_title or title,
            "original_title": title,
            "comprehensive_summary": _clean_and_truncate_summary(triage.thai_summary or rep.get("summary", rep.get("freshness_reason", ""))),
            "source_count": rep.get("sources_count", 1),
            "sources": rep.get("sources", [rep.get("source", "RSS")]),
            "links": rep.get("links", [link] if link else []),
            "macro_impact_score": triage.macro_impact_score,
            "asset_impact_score": triage.asset_impact_score,
            "is_high_impact": triage.is_high_impact,
            "primary_tags": triage.primary_tags,
            "extracted_tickers": triage.extracted_tickers,
            "extracted_themes": triage.extracted_themes,
            "triage_reasoning": triage.triage_reasoning,
            "triage_source": triage_source,
            "status": "pending_synthesis",
            "ingested_at": now_iso,
        }
        new_events.append(ev)

    if new_events:
        save_triage_events(new_events, store_path=store_path)

    high_impact_count = sum(1 for e in new_events if e.get("is_high_impact"))
    return {
        "status": "success",
        "ingested_count": len(new_events),
        "high_impact_count": high_impact_count,
    }


def format_news_funnel_card_prompt(period: str, pending_items: List[Dict[str, Any]]) -> str:
    lines = [f"### 📰 รายการข่าว High-Impact ที่รอการสังเคราะห์ (รอบ {period.upper()} — {len(pending_items)} รายการ)"]
    for idx, ev in enumerate(pending_items, 1):
        title = ev.get("canonical_title", "Untitled")
        macro_score = ev.get("macro_impact_score", 0) or 0
        asset_score = ev.get("asset_impact_score", 0) or 0
        summary = ev.get("comprehensive_summary", "").strip()
        tickers = [f"[[{t}]]" for t in (ev.get("extracted_tickers") or [])]
        themes = [f"[[{th}]]" for th in (ev.get("extracted_themes") or [])]
        tags_str = " ".join(tickers + themes).strip()
        links = ev.get("links") or []
        first_link = links[0] if isinstance(links, list) and links else ""

        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append(f"#### {idx}. {title}")
        lines.append(f"- **Macro Impact:** {macro_score}/10 | **Asset Impact:** {asset_score}/10")
        if ev.get("triage_source") == "heuristic_fallback":
            lines.append("- ⚠️ **คะแนนจาก heuristic fallback (LLM triage ล้มเหลวรอบ ingest)** — โปรดตรวจสอบเนื้อหาก่อนอนุมัติ")
        if summary:
            lines.append(f"- **สรุปเนื้อหา:** {summary}")
        if tags_str:
            lines.append(f"- **แท็กที่เกี่ยวข้อง:** {tags_str}")
        if first_link:
            lines.append(f"- 🔗 [อ่านข่าวต้นฉบับ]({first_link})")
    return "\n".join(lines).strip()


def _ensure_thai_content(ev: Dict[str, Any]) -> tuple[str, str]:
    """ตรวจสอบและแปล/สังเคราะห์หัวข้อข่าวและใจความสำคัญให้เป็นภาษาไทยที่สมบูรณ์และเข้าใจง่าย"""
    title = ev.get("canonical_title", "Untitled Event")
    summary = ev.get("comprehensive_summary", "")

    if _is_mock_mode():
        return title, summary

    has_thai_summary = any('\u0e00' <= c <= '\u0e7f' for c in summary)
    has_thai_title = any('\u0e00' <= c <= '\u0e7f' for c in title)
    if has_thai_summary and has_thai_title:
        return title, summary

    try:
        from pydantic import BaseModel, Field
        class ThaiNoteSynthesis(BaseModel):
            thai_title: str = Field(description="ชื่อหัวข้อข่าวแปลและเรียบเรียงเป็นภาษาไทยที่สละสลวย กระชับ สื่อความหมายชัดเจน")
            thai_summary: str = Field(description="บทวิเคราะห์สรุปใจความสำคัญของข่าวเป็นภาษาไทย 2-3 ย่อหน้า ครอบคลุมประเด็นสำคัญว่าเกิดอะไรขึ้น ผลกระทบ และสิ่งที่นักลงทุนควรติดตาม")

        prompt_lines = [
            "Please translate, synthesize, and summarize the following financial/macro news item into professional THAI language for Thai investors.",
            f"Original Title: {title}",
            f"Original Summary/Content: {summary}",
            f"Macro Impact Score: {ev.get('macro_impact_score', 0)}/10 | Asset Impact Score: {ev.get('asset_impact_score', 0)}/10",
            "Requirements:",
            "1. 'thai_title': Clear, professional headline in THAI language.",
            "2. 'thai_summary': Comprehensive analytical summary entirely in THAI language (1-2 paragraphs). Highlight key financial/economic insights clearly.",
        ]
        res = _invoke_structured(ThaiNoteSynthesis, "NEWS_FUNNEL_SYNTHESIS_MODEL", prompt_lines)
        if res and hasattr(res, "thai_title") and hasattr(res, "thai_summary") and res.thai_title and res.thai_summary:
            return res.thai_title.strip(), res.thai_summary.strip()
    except Exception as e:
        logger.warning("LLM Thai synthesis step failed (%s), using original text", e)

    return title, summary


def _ensure_thai_content_batch(events: List[Dict[str, Any]]) -> List[tuple[str, str]]:
    """ตรวจสอบและแปล/สังเคราะห์หัวข้อข่าวและใจความสำคัญให้เป็นภาษาไทยแบบ Concurrent เพื่อลดระยะเวลาการทำงาน"""
    if not events:
        return []
    if _is_mock_mode() or len(events) == 1:
        return [_ensure_thai_content(ev) for ev in events]

    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(5, len(events))) as executor:
        return list(executor.map(_ensure_thai_content, events))


def run_news_funnel_synthesize(
    period: Optional[str] = None,
    approved_event_ids: Optional[List[str]] = None,
    candidate_event_ids: Optional[List[str]] = None,
    store_path: Optional[str] = None,
    vault_root: Optional[str] = None,
    custom_date: Optional[str] = None,
    allow_autonomous: bool = False,
) -> Dict[str, Any]:
    """สร้างโน้ตข่าวเดี่ยวสำคัญลงใน 30_Knowledge_Base/News/ พร้อม Zero-Pending Protection และ Strict HITL

    เมื่อ status = require_kanban_approval ฟังก์ชันคืน pending_events + period ให้ caller
    เป็นผู้สร้าง/อัปเดตการ์ด Kanban เอง (ผ่าน api.news_funnel_cards.upsert_news_funnel_card) —
    ชั้น tools ไม่แตะ state_db ของ Web UI
    """
    if not period or period == "auto":
        period = get_synthesis_period()
    root = vault_root or os.getenv("OBSIDIAN_VAULT_PATH", "./memories")

    # Complete Legacy Cleanup: ลบโฟลเดอร์ News/Themes ทิ้งทั้งหมดหากพบ
    old_themes_dir = Path(root) / "30_Knowledge_Base" / "News" / "Themes"
    if old_themes_dir.exists() and old_themes_dir.is_dir():
        shutil.rmtree(old_themes_dir, ignore_errors=True)

    pending = get_pending_high_impact_events(store_path=store_path)

    if approved_event_ids is None and not allow_autonomous:
        if not pending:
            logger.info("No pending events to synthesize")
            return {
                "status": "no_pending_events",
                "published_count": 0,
                "rejected_count": 0,
                "created_files": [],
                "published_events": [],
                "message": "No pending events to synthesize. Log-Only No File Overwrite.",
            }

        logger.info("Require Kanban approval for %d pending items.", len(pending))
        return {
            "status": "require_kanban_approval",
            "period": period,
            "pending_events": pending,
            "published_count": 0,
            "rejected_count": 0,
            "created_files": [],
            "published_events": [],
            "message": f"Strict HITL Enforced: {len(pending)} pending items require review on Web UI Kanban.",
        }

    rejected_event_ids = []
    if approved_event_ids is not None:
        if len(approved_event_ids) == 0:
            events_to_synthesize = []
            rejected_event_ids = []
        else:
            target_ids = set(approved_event_ids)
            events_to_synthesize = [e for e in pending if e.get("event_id") in target_ids]
            if candidate_event_ids is not None:
                snapshot_set = set(candidate_event_ids)
                rejected_event_ids = [e.get("event_id") for e in pending if e.get("event_id") in snapshot_set and e.get("event_id") not in target_ids]
            else:
                rejected_event_ids = [e.get("event_id") for e in pending if e.get("event_id") and e.get("event_id") not in target_ids]
    else:
        events_to_synthesize = pending

    # Zero-Pending Protection: หากไม่มีข่าวที่ต้องสังเคราะห์ ให้ Log-Only และไม่เขียนไฟล์ทับเด็ดขาด
    if not events_to_synthesize:
        logger.info("No pending events to synthesize (or no events approved)")
        status = "no_approved_events" if approved_event_ids is not None else "no_pending_events"
        if rejected_event_ids:
            update_events_status(rejected_ids=rejected_event_ids, store_path=store_path)
        return {
            "status": status,
            "published_count": 0,
            "rejected_count": len(rejected_event_ids),
            "created_files": [],
            "published_events": [],
            "message": "No events to synthesize. Log-Only No File Overwrite.",
        }

    date_str = custom_date or datetime.now().strftime("%Y-%m-%d")
    now_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    news_dir = Path(root) / "30_Knowledge_Base" / "News"
    news_dir.mkdir(parents=True, exist_ok=True)

    created_files = []
    published_event_ids = []
    all_extracted_concepts: set = set()

    thai_contents = _ensure_thai_content_batch(events_to_synthesize)

    for ev, (canonical_title, summary) in zip(events_to_synthesize, thai_contents):
        macro_score = ev.get("macro_impact_score", 0)
        asset_score = ev.get("asset_impact_score", 0)
        ev["canonical_title"] = canonical_title
        ev["comprehensive_summary"] = summary
        tickers = ev.get("extracted_tickers") or []
        themes = ev.get("extracted_themes") or []
        links = ev.get("links") or []
        link = links[0] if links else ""

        tickers_str = ", ".join(tickers) if tickers else "ไม่มี"
        themes_str = ", ".join(themes) if themes else "ไม่มี"

        extracted_body = (
            f"> **Macro Impact:** {macro_score}/10 | **Asset Impact:** {asset_score}/10\n\n"
            f"## ใจความสำคัญ\n"
            f"{summary}\n\n"
            f"## หุ้นและสินทรัพย์ที่เกี่ยวข้อง\n"
            f"- {tickers_str}\n\n"
            f"## ธีมเศรษฐกิจที่เกี่ยวข้อง\n"
            f"- {themes_str}"
        )

        md_content = _build_article_md(
            extracted=extracted_body,
            source_url=link,
            title=canonical_title,
            today=date_str,
            now_time=now_time,
        )

        safe_title = _sanitize_filename(canonical_title)
        out_file = news_dir / f"{date_str}_{safe_title}.md"
        counter = 2
        while out_file.exists():
            out_file = news_dir / f"{date_str}_{safe_title}_{counter}.md"
            counter += 1

        _atomic_write_to(out_file, md_content)
        created_files.append(str(out_file))
        if ev.get("event_id"):
            published_event_ids.append(ev.get("event_id"))

        # รวบรวม Wikilinks ทุกรายการ (extracted_tickers + extracted_themes) ส่งให้ ensure_concept_stubs_exist(...)
        for t in tickers:
            all_extracted_concepts.add(t)
        for th in themes:
            all_extracted_concepts.add(th)

    # ส่งให้ ensure_concept_stubs_exist สำหรับคำที่ไม่ใช่รหัสหุ้นมาตรฐาน
    if all_extracted_concepts:
        concept_candidates = []
        for concept_link in sorted(all_extracted_concepts):
            raw_name = strip_wikilink(concept_link)
            if raw_name not in TICKER_ALIAS_MAP and raw_name not in TICKER_ALIAS_MAP.values():
                concept_candidates.append(raw_name)
        ensure_concept_stubs_exist(concept_candidates, vault_root=vault_root)

    # มาร์คสถานะใน JSON Store เป็น synthesized และ rejected ใน Transaction เดียว
    update_events_status(rejected_ids=rejected_event_ids, synthesized_ids=published_event_ids, store_path=store_path)

    return {
        "status": "success",
        "published_count": len(events_to_synthesize),
        "rejected_count": len(rejected_event_ids),
        "created_files": created_files,
        "published_events": events_to_synthesize,
    }
