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
import threading
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlsplit
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
from application.knowledge.note_write_ports import KnowledgeNoteWritePort
from application.knowledge.write_context import current_note_writer
from tools._atomic_io import _atomic_write_to
from tools.archivist.core import _sanitize_filename
from tools.archivist.indexer import _index_upsert, flush_index_if_dirty
from tools.knowledge.core import _build_article_md
from tools.knowledge.article import extract_article_content
from tools.macro.news_funnel_store import (
    get_pending_high_impact_events,
    is_title_or_url_processed,
    load_store,
    save_triage_events,
    update_events_status,
    commit_event_synthesis_results,
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


def _invoke_structured(
    schema: Any,
    model_env: str,
    prompt_lines: List[str],
    purpose: Optional[str] = None,
    max_output_tokens: Optional[int] = None,
    default_model: str = "gemini-3.1-flash-lite-preview",
    **kwargs: Any,
) -> Any:
    """Helper สำหรับสร้างและเรียกใช้ structured LLM ด้วย provider='google'

    default_model เป็น named parameter จริง (ไม่ใช่แค่ hardcode inline) เพื่อให้ caller ส่ง
    ค่าจาก core.model_registry.REGISTRY[key].default เข้ามา override ได้ — ถ้า hardcode
    default_model="..." ไว้ในบรรทัด invoke_structured_llm(...) ด้านล่างเฉยๆ แล้วให้ caller ส่ง
    default_model ผ่าน **kwargs จะชนกันเป็น TypeError: got multiple values for keyword argument
    """
    from core.llm_factory import invoke_structured_llm
    return invoke_structured_llm(
        schema=schema,
        model_env=model_env,
        prompt_lines=prompt_lines,
        purpose=purpose,
        max_output_tokens=max_output_tokens,
        default_model=default_model,
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
    note_writer: Optional[KnowledgeNoteWritePort] = None,
    *,
    allow_stub_creation: bool = False,
) -> List[str]:
    """Report concept candidates; production never creates empty notes.

    The explicit opt-in exists only for legacy migration fixtures. Normal
    News Funnel execution keeps the candidates in structured
    ``related_entities`` metadata on the published note.
    """
    root = vault_root or os.getenv("OBSIDIAN_VAULT_PATH", "./memories")
    if not allow_stub_creation:
        candidates = [
            strip_wikilink(str(item)).strip()
            for item in concepts
            if strip_wikilink(str(item)).strip()
        ]
        if candidates:
            logger.info(
                "Concept admission candidates recorded without stub creation: %d (vault=%s)",
                len(candidates),
                root,
            )
        return []

    note_writer = note_writer or current_note_writer(root)
    from tools.archivist.maintenance_guard import assert_write_allowed
    assert_write_allowed(root)
    from tools.archivist.metadata import normalize_legacy_metadata, parse_note
    from tools.archivist.vault_paths import VaultPaths
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
                    old_meta, old_body, parse_issues = parse_note(content)
                    if parse_issues:
                        raise ValueError(f"malformed concept stub frontmatter: {parse_issues}")
                    old_meta, _ = normalize_legacy_metadata(old_meta, producer="news_funnel")
                    old_meta.update(
                        {
                            "schema_version": 2,
                            "title": old_meta.get("title") or old_file.stem,
                            "entity_type": "concept",
                            "search_scope": "excluded",
                        }
                    )
                    committed = note_writer.write_note(
                        metadata=old_meta,
                        body=old_body,
                        target_path=new_file,
                    )
                    _index_upsert(committed.primary_file, vault_root=vault_root)
                except Exception as e:
                    logger.error("Failed to migrate concept stub %s: %s", old_file, e)
        # This branch is explicit legacy migration only. Production callers
        # return above and never mutate or delete this folder.
        import shutil
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
            try:
                committed = note_writer.write_note(
                    metadata={
                        "schema_version": 2,
                        "title": clean,
                        "entity_type": "concept",
                        "date": today_str,
                        "tags": ["concept", "auto_stub"],
                        "created": today_str,
                    },
                    body=(
                        f"# {clean}\n\n"
                        "<!-- Concept stub created automatically by News Funnel -->\n"
                    ),
                    filename=safe_name,
                )
                _index_upsert(committed.primary_file, vault_root=vault_root)
                created.append(str(committed.primary_file))
                logger.info("Created concept stub: %s", committed.primary_file)
            except Exception as e:
                logger.error("Failed to create concept stub %s: %s", file_path, e)

    if created:
        flush_index_if_dirty(vault_root=vault_root)
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
        news_items_lines = []
        for idx, it in enumerate(chunk):
            truncated_summary = _clean_and_truncate_summary(it.get('summary', ''), max_len=500)
            news_items_lines.append(f"{idx+1}. Title: {it.get('title', '')} | Summary: {truncated_summary}")

        from core.model_registry import REGISTRY
        from core.prompt_harness import TOOLS_PROMPTS_ROOT, get_harness

        prompt_text = get_harness("news_funnel", skills_root=TOOLS_PROMPTS_ROOT).get_skill_text(
            "triage.md",
            chunk_size=str(len(chunk)),
            news_items="\n".join(news_items_lines),
        )

        slot = REGISTRY["news_triage"]
        res = _invoke_structured(
            TriageBatchResult,
            slot.env_var,
            prompt_text.split("\n"),
            purpose="triage_batch",
            max_output_tokens=16384,
            default_model=slot.default,
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


def _extract_published_at(item: Dict[str, Any], fallback_iso: Optional[str] = None) -> Optional[str]:
    pub = item.get("published_at") or item.get("published") or item.get("pubDate") or item.get("date")
    if pub:
        if hasattr(pub, "isoformat"):
            return pub.isoformat()
        return str(pub).strip()
    return fallback_iso


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
                "published_at": _extract_published_at(item, None),
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
        feed_name = rep.get("source") or rep.get("feed_name") or "RSS Feed"
        source_host = urlsplit(link).netloc.replace("www.", "") if link else "N/A"
        syndication_host = source_host
        pub = rep.get("publisher") or rep.get("author")
        publisher = pub.strip() if (pub and isinstance(pub, str) and pub.strip()) else "ไม่ยืนยัน"
        published_at = _extract_published_at(rep, None)
        has_direct_provenance = bool(link and pub and published_at)

        ev = {
            "event_id": event_id,
            "canonical_title": triage.thai_title or title,
            "original_title": title,
            "feed_name": feed_name,
            "publisher": publisher,
            "canonical_publisher": publisher if pub else None,
            "canonical_url": link or None,
            "canonical_published_at": published_at,
            "source_host": source_host,
            "syndication_host": syndication_host,
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
            "published_at": published_at,
            # The RSS record supplied a direct link, publisher, and date.  Keep
            # that tuple explicit so later synthesis can audit it rather than
            # inventing provenance from a translated title or ingestion time.
            "verification_status": "verified" if has_direct_provenance else ("partial" if pub else "unverified"),
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
    def _to_portable_reference(tag: str) -> str:
        clean = strip_wikilink(tag)
        return clean if clean else ""

    tickers_formatted = [_to_portable_reference(t) for t in tickers if strip_wikilink(t)]
    themes_formatted = [_to_portable_reference(th) for th in themes if strip_wikilink(th)]
    if not tickers_formatted:
        tickers_formatted = ["NVDA", "Gold", "Bitcoin"]
    if not themes_formatted:
        themes_formatted = ["AI Infrastructure", "Monetary Policy"]

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
    vault_root: Optional[Union[str, Path]] = None,
    existing_notes_by_event_id: Optional[Dict[str, Path]] = None,
    note_writer: Optional[KnowledgeNoteWritePort] = None,
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

    ev_id = ev.get("event_id")
    if ev_id and existing_notes_by_event_id and ev_id in existing_notes_by_event_id:
        existing_path = existing_notes_by_event_id[ev_id]
        logger.info("Pre-scan recovery: event %s already has synthesized file %s", ev_id, existing_path)
        try:
            content = existing_path.read_text(encoding="utf-8")
            from tools.archivist.parser import _strip_frontmatter, parse_frontmatter_metadata
            body = _strip_frontmatter(content)
            meta = parse_frontmatter_metadata(content)
            wikilinks = set(re.findall(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]", body))
            recovered_tickers = meta.get("tickers") or tickers
            recovered_themes = meta.get("tags") or meta.get("themes") or themes
            for t in recovered_tickers:
                wikilinks.add(t)
            for th in recovered_themes:
                wikilinks.add(th)
            ev["canonical_title"] = _ensure_thai_title(ev.get("canonical_title", "Untitled Event"))
            ev["synthesized_note_path"] = str(existing_path)
            ev["synthesized_content"] = body
            ev["extracted_tickers"] = list(recovered_tickers)
            ev["extracted_themes"] = list(recovered_themes)
            ev["thematic_tags"] = list(recovered_themes)
            if meta.get("key_metrics"):
                ev["key_metrics"] = meta.get("key_metrics")
            if meta.get("financial_contradictions"):
                ev["financial_contradictions"] = meta.get("financial_contradictions")
            if meta.get("macro_impact_score") is not None:
                ev["macro_impact_score"] = meta.get("macro_impact_score")
            if meta.get("asset_impact_score") is not None:
                ev["asset_impact_score"] = meta.get("asset_impact_score")
            ev["synthesis_completed_at"] = meta.get("synthesized_at") or meta.get("date") or now_time
            ev["synthesized_at"] = ev["synthesis_completed_at"]
            return ev, str(existing_path), body, wikilinks, None
        except Exception as exc:
            logger.warning("Could not recover existing note %s, will re-synthesize: %s", existing_path, exc)

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

    related_entities = [
        {
            "entity_type": "security",
            "key": strip_wikilink(str(ticker)),
            "relation": "mentioned",
        }
        for ticker in tickers
        if strip_wikilink(str(ticker))
    ] + [
        {
            "entity_type": "theme",
            "key": strip_wikilink(str(theme)),
            "relation": "mentioned",
        }
        for theme in themes
        if strip_wikilink(str(theme))
    ]

    published_at_val = ev.get("published_at")
    md_content = _build_article_md(
        extracted=extracted_body,
        source_url=link,
        title=canonical_title,
        today=date_str,
        now_time=now_time,
        image=og_image,
        event_id=ev_id,
        extracted_tickers=tickers,
        extracted_themes=themes,
        published_at=str(published_at_val) if published_at_val else None,
        canonical_publisher=ev.get("canonical_publisher") or ev.get("publisher"),
        canonical_url=ev.get("canonical_url") or link or None,
        verification_status=ev.get("verification_status"),
        related_entities=related_entities,
    )

    safe_title = _sanitize_filename(canonical_title)
    from tools.archivist.metadata import normalize_legacy_metadata, parse_note
    from tools.archivist.portable_links import render_resolved_markdown_link
    from tools.archivist.vault_paths import VaultPaths
    from tools.archivist.writer import _portableize_wikilinks, _sync_to_catalog

    root = Path(vault_root or news_dir.parents[1]).resolve()
    vp = VaultPaths(root)
    note_meta, note_body, parse_issues = parse_note(md_content)
    if parse_issues:
        raise ValueError(f"News article frontmatter is invalid: {parse_issues}")
    note_meta, _ = normalize_legacy_metadata(note_meta, producer="news_funnel")
    note_meta["entity_type"] = "company_news"
    note_meta["schema_version"] = 2
    note_meta["title"] = canonical_title
    filename = f"{date_str}_{safe_title}"

    with _SYNTH_FILE_LOCK:
        counter = 2
        while True:
            candidate = vp.note_path(note_meta, filename=filename)
            if not candidate.exists():
                break
            try:
                existing_meta, _, _ = parse_note(candidate.read_text(encoding="utf-8"))
            except OSError:
                existing_meta = {}
            # Reuse an existing projection only when the durable source
            # identity is actually the same. Mock/offline events often have
            # no URL; two empty URLs must not import the first note_id into a
            # different event/document_key.
            same_event = bool(
                existing_meta.get("event_id")
                and note_meta.get("event_id")
                and str(existing_meta.get("event_id")) == str(note_meta.get("event_id"))
            )
            same_source = bool(
                existing_meta.get("source_url")
                and note_meta.get("source_url")
                and str(existing_meta.get("source_url")) == str(note_meta.get("source_url"))
            )
            if same_event or same_source:
                break
            filename = f"{date_str}_{safe_title}_{counter}"
            counter += 1

        target_path = vp.note_path(note_meta, filename=filename)
        note_body = _portableize_wikilinks(note_body, root, target_path)
        committed = (note_writer or current_note_writer(root)).write_note(
            metadata=note_meta,
            body=note_body,
            filename=filename,
        )
        out_file = committed.primary_file
        _index_upsert(out_file, vault_root=str(root))
        _sync_to_catalog(out_file)


    # Union Wikilinks: ดึงจาก regex [[...]] ใน extracted_body มารวมกับ tickers และ themes เดิม
    wikilinks = set(re.findall(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]", extracted_body))
    for t in tickers:
        wikilinks.add(t)
    for th in themes:
        wikilinks.add(th)

    ev["canonical_title"] = canonical_title
    ev["synthesized_note_path"] = str(out_file)
    ev["synthesized_content"] = extracted_body
    ev["extracted_tickers"] = list(tickers)
    ev["extracted_themes"] = list(themes)
    ev["synthesis_completed_at"] = now_time

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

        from core.model_registry import REGISTRY
        from core.prompt_harness import TOOLS_PROMPTS_ROOT, get_harness

        prompt_text = get_harness("news_funnel", skills_root=TOOLS_PROMPTS_ROOT).get_skill_text(
            "thai_title.md",
            original_title=title,
        )
        slot = REGISTRY["thai_title_translation"]
        res = _invoke_structured(ThaiTitleSynthesis, slot.env_var, prompt_text.split("\n"), purpose="thai_title_synthesis", default_model=slot.default)
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
    note_writer: Optional[KnowledgeNoteWritePort] = None,
) -> Dict[str, Any]:
    """สร้างโน้ตข่าวเดี่ยวสำคัญลงใน 30_Knowledge_Base/News/ พร้อม Zero-Pending Protection และ Strict HITL

    เมื่อ status = require_kanban_approval ฟังก์ชันคืน pending_events + period ให้ caller
    เป็นผู้สร้าง/อัปเดตการ์ด Kanban เอง (ผ่าน api.news_funnel_cards.upsert_news_funnel_card) —
    ชั้น tools ไม่แตะ state_db ของ Web UI
    """
    if not period or period == "auto":
        period = get_synthesis_period()
    root = vault_root or os.getenv("OBSIDIAN_VAULT_PATH", "./memories")
    note_writer = note_writer or current_note_writer(root)
    from tools.archivist.maintenance_guard import assert_write_allowed
    assert_write_allowed(root)

    # Legacy folders are handled by an explicit migration command. A normal
    # synthesis run must never silently delete user content as a side effect.

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

    # Pre-scan Recovery Flow: สแกน news_dir หนึ่งครั้งก่อนเปิด executor เพื่อหาไฟล์ที่มี event_id ตรงกันหรือหัวข้อตรงกัน
    existing_notes_by_event_id: Dict[str, Path] = {}
    from tools.archivist.parser import parse_frontmatter_metadata
    for existing_file in news_dir.rglob("*.md"):
        if "Inbox" in existing_file.parts or "Revisions" in existing_file.parts:
            continue
        try:
            content = existing_file.read_text(encoding="utf-8")
            meta = parse_frontmatter_metadata(content)
            ev_id = meta.get("event_id")
            if ev_id:
                existing_notes_by_event_id[str(ev_id)] = existing_file
        except Exception as exc:
            logger.debug("Failed reading/parsing existing note %s during pre-scan: %s", existing_file, exc)

    created_files = []
    published_event_ids = []
    published_events = []
    skipped_error_ids = []
    error_msgs = {}
    all_extracted_concepts: set = set()

    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(3, len(events_to_synthesize))) as executor:
        futures_map = {
            executor.submit(_synthesize_single_event, ev, date_str, now_time, news_dir, vault_root, existing_notes_by_event_id, note_writer): ev
            for ev in events_to_synthesize
        }
        results = []
        for f, ev in futures_map.items():
            try:
                results.append(f.result())
            except Exception as exc:
                logger.error("Error synthesizing single event %s: %s", ev.get("event_id"), exc)
                results.append((ev, None, None, set(), f"สังเคราะห์ข้อมูลล้มเหลว: {exc}"))

    synthesis_payloads = {}
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
                synthesis_payloads[ev_id] = {
                    "synthesized_note_path": ev.get("synthesized_note_path") or str(out_file),
                    "synthesized_content": ev.get("synthesized_content") or extracted_body,
                    "extracted_tickers": ev.get("extracted_tickers", []),
                    "extracted_themes": ev.get("extracted_themes", []),
                    "thematic_tags": ev.get("thematic_tags") or ev.get("extracted_themes", []),
                    "key_metrics": ev.get("key_metrics"),
                    "financial_contradictions": ev.get("financial_contradictions"),
                    "macro_impact_score": ev.get("macro_impact_score"),
                    "asset_impact_score": ev.get("asset_impact_score"),
                    "synthesized_at": ev.get("synthesized_at") or ev.get("synthesis_completed_at") or now_time,
                    "synthesis_completed_at": ev.get("synthesis_completed_at") or now_time,
                }
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
            ensure_concept_stubs_exist(concept_candidates, vault_root=vault_root, note_writer=note_writer)

    # บันทึกสถานะใน JSON Store และ Layer 2 payloads ทั้งหมดใน Transaction เดียวภายใต้ FileLock
    commit_event_synthesis_results(
        synthesized_event_ids=published_event_ids,
        failed_event_ids=skipped_error_ids,
        error_messages=error_msgs,
        synthesis_payloads=synthesis_payloads,
        store_path=store_path,
        rejected_event_ids=rejected_event_ids,
    )

    flush_index_if_dirty(vault_root=vault_root)

    if published_events:
        try:
            from core.discord_notifier import send_synthesized_news_discord
            send_synthesized_news_discord(published_events, period=period)
        except Exception as e:
            logger.warning("ส่งข่าวสังเคราะห์ไป Discord ไม่สำเร็จ (ไม่กระทบไฟล์ที่บันทึกไปแล้ว): %s", e)

    return {
        "status": "success",
        "published_count": len(published_events),
        "rejected_count": len(rejected_event_ids),
        "skipped_error_count": len(skipped_error_ids),
        "created_files": created_files,
        "published_events": published_events,
        "skipped_errors": error_msgs,
    }
