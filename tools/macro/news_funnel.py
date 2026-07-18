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
import threading
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
from tools.knowledge.article import extract_article_content
from tools.macro.news_funnel_store import (
    get_pending_high_impact_events,
    is_title_or_url_processed,
    load_store,
    save_triage_events,
    update_events_status,
    save_raw_candidates,
    get_raw_candidates,
    remove_processed_raw_candidates,
)

logger = get_logger(__name__)


def _is_mock_mode() -> bool:
    """ตรวจสอบว่าเปิดใช้งาน MOCK_NEWS_FUNNEL_LLM สำหรับเทสต์หรือออฟไลน์หรือไม่"""
    return os.getenv("MOCK_NEWS_FUNNEL_LLM", "false").lower() == "true"


def get_synthesis_period(now: Optional[datetime] = None) -> str:
    """คืนรอบสังเคราะห์ปัจจุบัน: morning (ก่อนเที่ยง) หรือ evening — จุดเดียวที่กำหนด cutoff"""
    current = now or datetime.now()
    return "morning" if current.hour < 12 else "evening"


def _invoke_structured(schema: Any, model_env: str, prompt_lines: List[str], purpose: Optional[str] = None, max_output_tokens: Optional[int] = None, **kwargs: Any) -> Any:
    """Helper สำหรับสร้างและเรียกใช้ structured LLM ด้วย provider='google'"""
    from core.llm_factory import invoke_structured_llm
    return invoke_structured_llm(
        schema=schema,
        model_env=model_env,
        prompt_lines=prompt_lines,
        purpose=purpose,
        max_output_tokens=max_output_tokens,
        default_model="gemini-3.1-flash-lite-preview",
        provider="google",
        **kwargs,
    )


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


TRIAGE_CHUNK_SIZE = 15


def _llm_triage_single_chunk(chunk: List[Dict[str, Any]]) -> tuple[List[MacroImpactTriageResult], Optional[str]]:
    """ประเมิน Impact Score สำหรับ 1 chunk (≤15 items) — คง try/except และคืน ([], reason) เมื่อ error เสมอ"""
    if not chunk:
        return [], None
    try:
        prompt_lines = [
            f"CRITICAL REQUIREMENT: You are given EXACTLY {len(chunk)} news items numbered 1 to {len(chunk)}.",
            f"You MUST return the 'results' array with EXACTLY {len(chunk)} elements.",
            "Each element MUST correspond 1-to-1 to the input item at the exact same index order. Do NOT skip, merge, or reorder any items.",
            "Evaluate macro impact score (1-10) and asset impact score (1-10) for each news item below.",
            "IMPORTANT requirement: You MUST provide 'thai_title' (accurate headline translated into THAI language) and 'thai_summary' (CONCISE analytical summary written in THAI language maximum 2-3 sentences for Thai investors. Do NOT write paragraphs).",
            "Rubric:",
            "- Score >= 7 (HIGH IMPACT): Systemic macro shift, central bank rate decision, critical policy change, inflation surprise, systemic shock, or market-moving earnings/catalyst (e.g. NVDA, PTT).",
            "- Score < 7 (ROUTINE IMPACT): Minor commentary, routine update, or localized news without broad market impact.",
            "News items to evaluate:"
        ]
        for idx, it in enumerate(chunk):
            truncated_summary = _clean_and_truncate_summary(it.get('summary', ''), max_len=500)
            prompt_lines.append(f"{idx+1}. Title: {it.get('title', '')} | Summary: {truncated_summary}")

        res = _invoke_structured(
            TriageBatchResult,
            "NEWS_FUNNEL_TRIAGE_MODEL",
            prompt_lines,
            purpose="triage_batch",
            max_output_tokens=16384,
        )
        if res and hasattr(res, "results") and len(res.results) == len(chunk):
            return res.results, None
        if res and hasattr(res, "results"):
            logger.error("LLM returned %d results for %d items — skipping chunk", len(res.results), len(chunk))
            return [], "length_mismatch"
        else:
            logger.error("LLM returned invalid response — skipping chunk")
            return [], "validation_error"
    except Exception as e:
        logger.error("LLM Triage chunk failed: %s", e)
        return [], f"api_error: {type(e).__name__}"


def _llm_triage_batch(items: List[Dict[str, Any]]) -> tuple[List[MacroImpactTriageResult], List[str], List[Optional[str]]]:
    """ประเมิน Impact Score แบบ Batch ผ่าน Fast LLM แบ่ง Chunk ≤15 พร้อม per-chunk retry และ fallback reason"""
    if not items:
        return [], [], []
    if _is_mock_mode():
        return [_mock_or_llm_triage(item) for item in items], ["mock"] * len(items), [None] * len(items)

    all_results = []
    all_sources = []
    all_reasons = []
    for i in range(0, len(items), TRIAGE_CHUNK_SIZE):
        chunk = items[i:i + TRIAGE_CHUNK_SIZE]
        chunk_results, err_reason = _llm_triage_single_chunk(chunk)
        if len(chunk_results) == len(chunk):
            all_results.extend(chunk_results)
            all_sources.extend(["llm"] * len(chunk))
            all_reasons.extend([None] * len(chunk))
        else:
            logger.warning("Chunk %d-%d mismatch (%s) — retrying once", i+1, i+len(chunk), err_reason)
            chunk_results, retry_reason = _llm_triage_single_chunk(chunk)
            if len(chunk_results) == len(chunk):
                all_results.extend(chunk_results)
                all_sources.extend(["llm"] * len(chunk))
                all_reasons.extend([None] * len(chunk))
            else:
                logger.warning("Retry failed for chunk %d-%d (%s) — falling back to heuristic for this chunk", i+1, i+len(chunk), retry_reason)
                all_results.extend([_mock_or_llm_triage(rep) for rep in chunk])
                all_sources.extend(["heuristic_fallback"] * len(chunk))
                all_reasons.extend([retry_reason or err_reason or "unknown_error"] * len(chunk))
    return all_results, all_sources, all_reasons


def _heuristic_prefilter_candidates(items: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """กรองหัวข้อข่าวก่อน Clustering โดยตรวจสอบ Blacklist พร้อม Finance-Keyword Override"""
    passed_items = []
    prefiltered_events = []
    now_iso = datetime.now().isoformat()

    finance_override_regex = re.compile(
        r"\b(earnings|stock|stocks|shares|ipo|market|markets|revenue|dividend|profit|loss)\b|หุ้น|กำไร|รายได้|ตลาด|ปันผล|ผลประกอบการ",
        re.IGNORECASE
    )
    en_blacklist_regex = re.compile(
        r"\b(sports|football|premier league|nba|celebrity|horoscope|zodiac|lottery|lotto|promotion|discount|coupon)\b",
        re.IGNORECASE
    )
    th_blacklist_regex = re.compile(
        r"(ฟุตบอล|พรีเมียร์ลีก|ข่าวดารา(?!ศาสตร์)|ดาราบันเทิง|วงการบันเทิง|ซุบซิบ|ดูดวง|ราศี|หวย|สลากกินแบ่ง|ลอตเตอรี่|โปรโมชั่น|ส่วนลด)",
        re.IGNORECASE
    )

    for item in items:
        title = item.get("title", "").strip()
        if not title:
            continue

        if finance_override_regex.search(title):
            passed_items.append(item)
            continue

        match = en_blacklist_regex.search(title) or th_blacklist_regex.search(title)
        if match:
            matched_word = match.group(0)
            link = item.get("link", "")
            event_id = item.get("event_id") or str(uuid.uuid4())
            ev = {
                "event_id": event_id,
                "canonical_title": title,
                "original_title": title,
                "comprehensive_summary": _clean_and_truncate_summary(item.get("summary", item.get("freshness_reason", ""))),
                "source_count": 1,
                "sources": [item.get("source", "RSS")],
                "links": [link] if link else [],
                "macro_impact_score": 2,
                "asset_impact_score": 2,
                "is_high_impact": False,
                "primary_tags": ["prefilter"],
                "extracted_tickers": [],
                "extracted_themes": [],
                "triage_reasoning": f"ตรง blacklist: {matched_word}",
                "triage_source": "heuristic_prefilter",
                "triage_fallback_reason": None,
                "status": "pending_synthesis",
                "ingested_at": now_iso,
            }
            prefiltered_events.append(ev)
        else:
            passed_items.append(item)

    return passed_items, prefiltered_events


def run_news_funnel_ingest(
    candidates: Optional[List[Dict[str, Any]]] = None,
    store_path: Optional[str] = None,
    fetch_only: bool = False,
) -> Dict[str, Any]:
    """ดึงข่าว (หรือรับจาก candidates) ทำ Clustering, คัดกรอง Batch LLM และบันทึกสถานะลง JSON Store หรือสะสมลง raw_candidates ถ้า fetch_only=True"""
    items = candidates
    if items is None:
        try:
            from tools.macro.news_radar import get_news_candidates
            items = get_news_candidates()
        except Exception as e:
            logger.warning("Could not fetch from news_radar: %s", e)
            items = []

    if fetch_only:
        if not items:
            return {"status": "success", "fetched_count": 0, "ingested_count": 0, "high_impact_count": 0}
        store_state = load_store(store_path=store_path)
        unprocessed = []
        for item in items:
            title = item.get("title", "").strip()
            link = item.get("link", "")
            if not title:
                continue
            if is_title_or_url_processed(title, link, store_path=store_path, store_state=store_state, include_raw=True):
                continue
            unprocessed.append(item)
        if unprocessed:
            save_raw_candidates(unprocessed, store_path=store_path)
        return {
            "status": "success",
            "fetched_count": len(unprocessed),
            "ingested_count": 0,
            "high_impact_count": 0,
        }

    # fetch_only = False (Batch Triage Mode)
    accumulated_raw = get_raw_candidates(store_path=store_path)
    combined_items = accumulated_raw + (items or [])
    if not combined_items:
        return {"status": "success", "ingested_count": 0, "high_impact_count": 0}

    # In-memory deduplication ภายใน pool ระหว่างข่าวสะสมและข่าวสด
    seen_urls = set()
    seen_titles = set()
    deduped_pool = []
    for item in combined_items:
        title = item.get("title", "").strip()
        link = item.get("link", "")
        if not title:
            continue
        norm_title = title.lower()
        if link and link in seen_urls:
            continue
        if norm_title in seen_titles:
            continue
        if link:
            seen_urls.add(link)
        seen_titles.add(norm_title)
        deduped_pool.append(item)

    store_state = load_store(store_path=store_path)
    unprocessed = []
    for item in deduped_pool:
        title = item.get("title", "").strip()
        link = item.get("link", "")
        if is_title_or_url_processed(title, link, store_path=store_path, store_state=store_state, include_raw=False):
            continue
        unprocessed.append(item)

    if not unprocessed:
        return {"status": "success", "ingested_count": 0, "high_impact_count": 0}

    # Prefilter ก่อน clustering
    unprocessed, prefiltered_events = _heuristic_prefilter_candidates(unprocessed)

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

    # ประเมินผ่าน LLM Batch (แบ่ง chunk เรียบร้อยแล้วภายใน _llm_triage_batch)
    triage_results, triage_sources, triage_reasons = _llm_triage_batch(representatives)

    new_events = []
    now_iso = datetime.now().isoformat()
    for rep, triage, triage_source, triage_reason in zip(representatives, triage_results, triage_sources, triage_reasons):
        title = rep.get("title", "").strip()
        link = rep.get("link", "")
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
            "triage_fallback_reason": triage_reason,
            "status": "pending_synthesis",
            "ingested_at": now_iso,
        }
        new_events.append(ev)

    all_new_events = new_events + prefiltered_events
    if all_new_events:
        save_triage_events(all_new_events, store_path=store_path)
        processed_urls = set()
        processed_titles = set()
        for ev in all_new_events:
            for lk in ev.get("links", []):
                if lk:
                    processed_urls.add(lk)
            for tk in ("original_title", "canonical_title", "title"):
                tv = ev.get(tk)
                if tv:
                    processed_titles.add(tv)
        remove_processed_raw_candidates(processed_urls, processed_titles, store_path=store_path)

    high_impact_count = sum(1 for e in all_new_events if e.get("is_high_impact"))
    return {
        "status": "success",
        "ingested_count": len(all_new_events),
        "high_impact_count": high_impact_count,
    }


_SYNTH_FILE_LOCK = threading.Lock()


def _format_6_sections(summary: str, tickers: List[str], themes: List[str]) -> str:
    def _to_wikilink(tag: str) -> str:
        clean = strip_wikilink(tag)
        return f"[[{clean}]]" if clean else ""

    tickers_formatted = [_to_wikilink(t) for t in tickers if strip_wikilink(t)]
    themes_formatted = [_to_wikilink(th) for th in themes if strip_wikilink(th)]
    if not tickers_formatted:
        tickers_formatted = ["[[NVDA]]", "[[Gold]]", "[[Bitcoin]]"]
    if not themes_formatted:
        themes_formatted = ["[[AI Infrastructure]]", "[[Monetary Policy]]"]

    tickers_str = ", ".join(tickers_formatted)
    themes_str = ", ".join(themes_formatted)

    return (
        f"## ใจความสำคัญ\n"
        f"{summary}\n\n"
        f"## แนวคิดการลงทุน\n"
        f"- กลยุทธ์และการจัดพอร์ตตามธีม {themes_str}\n\n"
        f"## เศรษฐกิจมหภาค\n"
        f"### 🇺🇸 สหรัฐฯ\n"
        f"- นโยบายการเงินและผลกระทบต่อเศรษฐกิจโลก\n\n"
        f"## หุ้นและสินทรัพย์\n"
        f"- {tickers_str}\n\n"
        f"## ความเสี่ยง\n"
        f"- ความผันผวนของอัตราดอกเบี้ยและความเสี่ยงเชิงระบบ\n\n"
        f"## ตัวเลขสำคัญทางเศรษฐกิจ\n"
        f"- อัตราเงินเฟ้อ, อัตราดอกเบี้ยนโยบาย, และตัวเลขการจ้างงาน"
    )


def _synthesize_single_event(
    ev: Dict[str, Any],
    date_str: str,
    now_time: str,
    news_dir: Path,
) -> tuple[Dict[str, Any], Optional[str], Optional[str], set[str], Optional[str]]:
    """ประมวลผลดึงและสกัดเนื้อหา 6 หัวข้อเชิงลึกของ 1 เหตุการณ์ (สำหรับรัน concurrent ใน ThreadPoolExecutor)"""
    links = ev.get("links") or []
    link = links[0] if links else ""

    if not _is_mock_mode() and not link:
        return ev, None, None, set(), "ข้าม: ข่าวนี้ไม่มี URL ต้นฉบับสำหรับดึงข้อมูล"

    macro_score = ev.get("macro_impact_score", 0)
    asset_score = ev.get("asset_impact_score", 0)
    summary = ev.get("comprehensive_summary", "")
    tickers = ev.get("extracted_tickers") or []
    themes = ev.get("extracted_themes") or []

    og_image = None
    err = None

    if _is_mock_mode():
        extracted_raw = _format_6_sections(summary or "สรุปเนื้อหาจำลองเชิงลึก", tickers, themes)
    else:
        extracted_raw, og_image, fetched_title, err = extract_article_content(link, check_processed=False)

    if err or not extracted_raw:
        return ev, None, None, set(), (err or f"ดึงข้อมูลล้มเหลว: ไม่สามารถสกัดเนื้อหาจาก {link}")

    canonical_title = _ensure_thai_title(ev.get("canonical_title", "Untitled Event"))
    impact_banner = f"> **Macro Impact:** {macro_score}/10 | **Asset Impact:** {asset_score}/10\n\n"
    extracted_body = impact_banner + extracted_raw

    md_content = _build_article_md(
        extracted=extracted_body,
        source_url=link,
        title=canonical_title,
        today=date_str,
        now_time=now_time,
        image=og_image,
    )

    safe_title = _sanitize_filename(canonical_title)
    with _SYNTH_FILE_LOCK:
        out_file = news_dir / f"{date_str}_{safe_title}.md"
        counter = 2
        while out_file.exists():
            out_file = news_dir / f"{date_str}_{safe_title}_{counter}.md"
            counter += 1
        _atomic_write_to(out_file, md_content)

    # Union Wikilinks: ดึงจาก regex [[...]] ใน extracted_body มารวมกับ tickers และ themes เดิม
    wikilinks = set(re.findall(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]", extracted_body))
    for t in tickers:
        wikilinks.add(t)
    for th in themes:
        wikilinks.add(th)

    ev["canonical_title"] = canonical_title

    return ev, str(out_file), extracted_body, wikilinks, None


def format_news_funnel_card_prompt(period: str, pending_items: List[Dict[str, Any]]) -> str:
    sorted_items = sorted(pending_items, key=lambda ev: 1 if ev.get("triage_source") == "heuristic_fallback" else 0)
    lines = [f"### 📰 รายการข่าว High-Impact ที่รอการสังเคราะห์ (รอบ {period.upper()} — {len(sorted_items)} รายการ)"]
    for idx, ev in enumerate(sorted_items, 1):
        title = ev.get("canonical_title", "Untitled")
        macro_score = ev.get("macro_impact_score", 0) or 0
        asset_score = ev.get("asset_impact_score", 0) or 0
        summary = ev.get("comprehensive_summary", "").strip()
        from schemas.news_funnel_schemas import strip_wikilink
        tickers = [strip_wikilink(str(t)) for t in (ev.get("extracted_tickers") or []) if strip_wikilink(str(t))]
        themes = [strip_wikilink(str(th)) for th in (ev.get("extracted_themes") or []) if strip_wikilink(str(th))]
        tags_str = " ".join(tickers + themes).strip()
        links = ev.get("links") or []
        first_link = links[0] if isinstance(links, list) and links else ""

        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append(f"#### {idx}. {title}")
        lines.append(f"- **Macro Impact:** {macro_score}/10 | **Asset Impact:** {asset_score}/10")
        if ev.get("triage_source") == "heuristic_fallback":
            reason = ev.get("triage_fallback_reason")
            reason_str = f" (สาเหตุ: {reason})" if reason else ""
            lines.append(f"- ⚠️ **คะแนนจาก heuristic fallback{reason_str} (LLM triage ล้มเหลวรอบ ingest)** — โปรดตรวจสอบเนื้อหาก่อนอนุมัติ")
        if summary:
            lines.append(f"- **สรุปเนื้อหา:** {summary}")
        if tags_str:
            lines.append(f"- **แท็กที่เกี่ยวข้อง:** {tags_str}")
        if first_link:
            lines.append(f"- 🔗 [อ่านข่าวต้นฉบับ]({first_link})")
    return "\n".join(lines).strip()


def _ensure_thai_title(title: str) -> str:
    """ตรวจสอบว่าชื่อหัวข้อข่าวเป็นภาษาไทยหรือไม่ หากไม่มีอักษรไทยเลย (เช่น Heuristic fallback) ให้เรียก LLM แปลเฉพาะหัวข้อ"""
    if _is_mock_mode():
        return title

    has_thai = any('\u0e00' <= c <= '\u0e7f' for c in title)
    if has_thai:
        return title

    try:
        from pydantic import BaseModel, Field
        class ThaiTitleSynthesis(BaseModel):
            thai_title: str = Field(description="ชื่อหัวข้อข่าวแปลและเรียบเรียงเป็นภาษาไทยที่สละสลวย กระชับ สื่อความหมายชัดเจน")

        prompt_lines = [
            "Please translate the following financial/macro news headline into professional THAI language for Thai investors.",
            f"Original Title: {title}",
            "Requirement: Return only a clear, professional headline in THAI language.",
        ]
        res = _invoke_structured(ThaiTitleSynthesis, "NEWS_FUNNEL_SYNTHESIS_MODEL", prompt_lines, purpose="thai_title_synthesis")
        if res and hasattr(res, "thai_title") and res.thai_title:
            return res.thai_title.strip()
    except Exception as e:
        logger.warning("LLM Thai title synthesis step failed (%s), using original title", e)

    return title


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
    published_events = []
    skipped_error_ids = []
    error_msgs = {}
    all_extracted_concepts: set = set()

    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(3, len(events_to_synthesize))) as executor:
        futures_map = {
            executor.submit(_synthesize_single_event, ev, date_str, now_time, news_dir): ev
            for ev in events_to_synthesize
        }
        results = []
        for f, ev in futures_map.items():
            try:
                results.append(f.result())
            except Exception as exc:
                logger.error("Error synthesizing single event %s: %s", ev.get("event_id"), exc)
                results.append((ev, None, None, set(), f"สังเคราะห์ข้อมูลล้มเหลว: {exc}"))

    for ev, out_file, extracted_body, wikilinks, err in results:
        ev_id = ev.get("event_id")
        if err or not out_file:
            if ev_id:
                skipped_error_ids.append(ev_id)
                error_msgs[ev_id] = err or "Unknown extraction error"
        else:
            created_files.append(out_file)
            published_events.append(ev)
            if ev_id:
                published_event_ids.append(ev_id)
            for w in wikilinks:
                all_extracted_concepts.add(w)

    # ส่งให้ ensure_concept_stubs_exist สำหรับคำที่ไม่ใช่รหัสหุ้นมาตรฐาน
    if all_extracted_concepts:
        concept_candidates = []
        for concept_link in sorted(all_extracted_concepts):
            raw_name = strip_wikilink(concept_link)
            if raw_name and raw_name not in TICKER_ALIAS_MAP and raw_name not in TICKER_ALIAS_MAP.values():
                concept_candidates.append(raw_name)
        if concept_candidates:
            ensure_concept_stubs_exist(concept_candidates, vault_root=vault_root)

    # มาร์คสถานะใน JSON Store เป็น synthesized, rejected, หรือ skipped_error ใน Transaction เดียว
    update_events_status(
        rejected_ids=rejected_event_ids,
        synthesized_ids=published_event_ids,
        skipped_error_ids=skipped_error_ids,
        error_msgs=error_msgs,
        store_path=store_path,
    )

    return {
        "status": "success",
        "published_count": len(published_events),
        "rejected_count": len(rejected_event_ids),
        "skipped_error_count": len(skipped_error_ids),
        "created_files": created_files,
        "published_events": published_events,
        "skipped_errors": error_msgs,
    }
