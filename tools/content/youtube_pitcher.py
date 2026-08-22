"""เครื่องมือหลักสำหรับดักจับข่าว เสนอไอเดีย และสร้าง Research-Grade Briefing Book ให้ NotebookLM"""
from datetime import datetime, timedelta, timezone
import json
import hashlib
from filelock import FileLock
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Literal, Optional, Tuple
import uuid

from core.llm_factory import get_llm, invoke_structured_llm
from core.logger import get_logger
from core.model_registry import REGISTRY
from core.nlp_utils import _jaccard_similarity
from core.prompt_harness import TOOLS_PROMPTS_ROOT, get_harness
from core.retry import with_retry
from core.utils import normalize_content
from schemas.youtube_pitch_schemas import (
    YouTubeContentPitchBatch,
    YouTubeContentPitchItem,
    validate_generated_pitch,
)
from tools._atomic_io import _atomic_write_to
from tools.archivist.core import VAULT_PATH, _sanitize_filename
from tools.archivist.parser import extract_yaml_frontmatter_value
from tools.knowledge.search_youtube_insights import (
    extract_first_bullet_of_key_takeaways,
    extract_sections,
    get_channel_with_fallback,
)
from tools.macro.baselines import get_macro_baselines
from tools.macro.news_funnel_store import load_store

logger = get_logger(__name__)


def parse_date_filters_from_instruction(instruction: str) -> Dict[str, Any]:
    """Parse ช่วงวันที่จากคำสั่ง (Instruction Encoding) เช่น [from_date=2026-07-01, to_date=2026-07-18, lookback_days=17]"""
    result = {
        "from_date": None,
        "to_date": None,
        "lookback_days": 7,  # Default ย้อนหลัง 7 วัน
    }
    if not instruction:
        return result

    # 1. ลองหาแท็ก [from_date=..., to_date=..., lookback_days=...] หรือตัวแปรในวงเล็บ/แท็ก
    from_match = re.search(r'from_date\s*=\s*([0-9]{4}-[0-9]{2}-[0-9]{2})', instruction, re.IGNORECASE)
    if from_match:
        result["from_date"] = from_match.group(1)

    to_match = re.search(r'to_date\s*=\s*([0-9]{4}-[0-9]{2}-[0-9]{2})', instruction, re.IGNORECASE)
    if to_match:
        result["to_date"] = to_match.group(1)

    lookback_match = re.search(r'lookback_days\s*=\s*([0-9]+)', instruction, re.IGNORECASE)
    if lookback_match:
        try:
            result["lookback_days"] = int(lookback_match.group(1))
        except ValueError:
            pass
    elif not from_match and not to_match:
        # ลองหาข้อความภาษาไทย/อังกฤษ เช่น "ย้อนหลัง 14 วัน" หรือ "last 30 days"
        days_match = re.search(r'(?:ย้อนหลัง|รอบ|last|past)\s*([0-9]+)\s*(?:วัน|days)', instruction, re.IGNORECASE)
        if days_match:
            try:
                result["lookback_days"] = int(days_match.group(1))
            except ValueError:
                pass

    # หากมี from_date และ to_date ให้คำนวณ lookback_days ให้สอดคล้องกัน
    if result["from_date"]:
        try:
            f_dt = datetime.strptime(result["from_date"], "%Y-%m-%d")
            t_dt = datetime.strptime(result["to_date"], "%Y-%m-%d") if result["to_date"] else datetime.now()
            diff_days = (t_dt - f_dt).days
            if diff_days > 0:
                result["lookback_days"] = diff_days
        except Exception:
            pass

    return result


def fetch_news_for_pitching(
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    lookback_days: int = 7,
    store_path: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], str, bool]:
    """ดึงข้อมูลข่าวจาก Layer 1 (News Funnel Store) และ Layer 2 (Synthesized Notes) ตามช่วงวันที่

    Returns:
        (candidates_list, macro_baselines_str, is_layer2_fallback_triggered)
    """
    now = datetime.now()
    if to_date:
        try:
            to_dt = datetime.strptime(to_date, "%Y-%m-%d").replace(hour=23, minute=59, second=59)
        except Exception:
            to_dt = now
    else:
        to_dt = now

    if from_date:
        try:
            from_dt = datetime.strptime(from_date, "%Y-%m-%d").replace(hour=0, minute=0, second=0)
        except Exception:
            from_dt = to_dt - timedelta(days=lookback_days)
    else:
        from_dt = to_dt - timedelta(days=lookback_days)

    # เช็คว่าช่วงวันที่ย้อนหลังเกิน 7 วันหรือไม่ (Store ตัดข้อมูลทุก 7 วัน)
    days_in_past = (now - from_dt).days
    is_layer2_fallback = days_in_past > 7

    candidates: List[Dict[str, Any]] = []
    seen_urls: set = set()
    seen_titles: List[str] = []

    def _add_candidate(cand: Dict[str, Any]) -> None:
        title = cand.get("canonical_title") or cand.get("title") or ""
        if not title:
            return
        # Deduplicate by URL
        links = cand.get("links") or ([cand.get("link")] if cand.get("link") else [])
        for l in links:
            if l and l in seen_urls:
                return
        # Deduplicate by title Jaccard similarity
        norm_t = title.strip().lower()
        for pt in seen_titles:
            if pt.strip().lower() == norm_t or _jaccard_similarity(title, pt) >= 0.8:
                return

        for l in links:
            if l:
                seen_urls.add(l)
        seen_titles.append(title)
        candidates.append(cand)

    # 1. ดึงจาก Layer 1: News Funnel Store JSON
    try:
        store_state = load_store(store_path=store_path)
        for ev in store_state.get("pending_events", []):
            if not isinstance(ev, dict):
                continue
            ingested_str = ev.get("ingested_at", "")
            ev_date_dt = None
            if ingested_str:
                try:
                    ev_date_dt = datetime.fromisoformat(ingested_str.replace("Z", "+00:00")).replace(tzinfo=None)
                except Exception:
                    pass
            if ev_date_dt and (from_dt <= ev_date_dt <= to_dt):
                ev_copy = dict(ev)
                ev_copy["source_layer"] = "layer1_store"
                _add_candidate(ev_copy)
    except Exception as e:
        logger.warning("Failed loading Layer 1 candidates: %s", e)

    # 2. ดึงจาก Layer 2: Synthesized Notes (30_Knowledge_Base/News/*.md) เมื่อเป็น Fallback หรือต้องการข้อมูลเพิ่ม
    if is_layer2_fallback or len(candidates) < 10:
        try:
            news_notes_dir = Path(VAULT_PATH) / "30_Knowledge_Base" / "News"
            if news_notes_dir.exists():
                for md_file in news_notes_dir.glob("*.md"):
                    try:
                        content = md_file.read_text(encoding="utf-8")
                        # Parse Frontmatter
                        note_date_dt = None
                        date_match = re.search(r'^date:\s*([0-9]{4}-[0-9]{2}-[0-9]{2})', content, re.MULTILINE)
                        if date_match:
                            try:
                                note_date_dt = datetime.strptime(date_match.group(1), "%Y-%m-%d")
                            except Exception:
                                pass
                        if not note_date_dt:
                            # ใช้เวลาแก้ไฟล์หรือชื่อไฟล์
                            mtime = datetime.fromtimestamp(md_file.stat().st_mtime)
                            note_date_dt = mtime

                        if from_dt <= note_date_dt <= to_dt:
                            # ดึง title จาก frontmatter หรือ H1
                            title_match = re.search(r'^title:\s*(.*)', content, re.MULTILINE)
                            title = title_match.group(1).strip() if title_match else md_file.stem

                            # ดึงสรุปย่อหรือส่วนเนื้อหา
                            summary_clean = re.sub(r'---.*?---', '', content, flags=re.DOTALL).strip()
                            if len(summary_clean) > 800:
                                summary_clean = summary_clean[:800] + "..."

                            # ดึงลิงก์จากเอกสาร
                            links = re.findall(r'https?://[^\s\)\]]+', content)

                            _add_candidate({
                                "event_id": md_file.stem,
                                "canonical_title": title,
                                "comprehensive_summary": summary_clean,
                                "links": list(set(links))[:3],
                                "source_layer": "layer2_notes",
                                "ingested_at": note_date_dt.isoformat(),
                            })
                    except Exception as ex:
                        continue
        except Exception as e:
            logger.warning("Failed scanning Layer 2 notes: %s", e)

    # 2.5 ดึงจาก Layer 2: YouTube Summaries (Always Include)
    try:
        yt_summaries_dir = Path(VAULT_PATH) / "30_Knowledge_Base" / "YouTube_Summaries"
        if yt_summaries_dir.exists():
            for md_file in yt_summaries_dir.glob("*.md"):
                try:
                    content = md_file.read_text(encoding="utf-8")
                    entity_type = extract_yaml_frontmatter_value(content, "entity_type")
                    if entity_type and entity_type != "youtube_insight":
                        continue

                    date_str = extract_yaml_frontmatter_value(content, "date")
                    note_date_dt = None
                    if date_str:
                        try:
                            note_date_dt = datetime.strptime(date_str[:10], "%Y-%m-%d")
                        except Exception:
                            pass
                    if not note_date_dt:
                        mtime = datetime.fromtimestamp(md_file.stat().st_mtime)
                        note_date_dt = mtime

                    if from_dt <= note_date_dt <= to_dt:
                        channel = get_channel_with_fallback(content)
                        bullet = extract_first_bullet_of_key_takeaways(content, max_chars=90)
                        if not bullet:
                            frontmatter_title = extract_yaml_frontmatter_value(content, "title") or md_file.stem
                            bullet = frontmatter_title
                        canonical_title = f"[YouTube Guru View - {channel}] {bullet}"
                        source_url = extract_yaml_frontmatter_value(content, "source_url") or ""

                        sections = extract_sections(content, ["ใจความสำคัญ", "แนวคิดการลงทุน", "หุ้นและสินทรัพย์"])
                        summary_parts = []
                        if sections["ใจความสำคัญ"]:
                            summary_parts.append(f"[ใจความสำคัญ]\n{sections['ใจความสำคัญ']}")
                        if sections["แนวคิดการลงทุน"]:
                            summary_parts.append(f"[แนวคิดการลงทุน]\n{sections['แนวคิดการลงทุน']}")
                        if sections["หุ้นและสินทรัพย์"]:
                            summary_parts.append(f"[หุ้น/สินทรัพย์]\n{sections['หุ้นและสินทรัพย์']}")

                        combined_summary = "\n\n".join(summary_parts)
                        if len(combined_summary) > 1200:
                            combined_summary = combined_summary[:1200] + "..."

                        _add_candidate({
                            "event_id": md_file.stem,
                            "canonical_title": canonical_title,
                            "comprehensive_summary": combined_summary,
                            "links": [source_url] if source_url else [],
                            "source_layer": "layer2_youtube",
                            "ingested_at": note_date_dt.isoformat(),
                        })
                except Exception as ex:
                    continue
    except Exception as e:
        logger.warning("Failed scanning Layer 2 YouTube Summaries: %s", e)

    # 3. ดึง Macro Baselines
    macro_baselines_str = ""
    try:
        macro_baselines_str = get_macro_baselines.invoke({})
    except Exception as e:
        logger.warning("Failed fetching macro baselines: %s", e)

    return candidates, macro_baselines_str, is_layer2_fallback


def _extract_topic_terms(instruction: str) -> Tuple[List[str], str]:
    if not instruction:
        return [], ""
    cleaned = re.sub(r'lookback_days\s*=\s*[0-9]+', '', instruction, flags=re.IGNORECASE)
    cleaned = re.sub(r'from_date\s*=\s*[0-9-]+', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'to_date\s*=\s*[0-9-]+', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'ไอเดียต่อยอดจาก\s*YouTube\s*Pitch\s*\([^\)]+\):\s*', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'หาไอเดียทำคลิป(?:\s*YouTube)?(?:\s*เชิงลึก)?', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\[lookback_days=[0-9]+\]', '', cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.strip()

    terms = [w.strip() for w in re.split(r'[\s,;:\'"()]+', cleaned) if len(w.strip()) >= 2]
    stopwords = {'ของ', 'และ', 'ใน', 'ต่อ', 'ที่', 'กับ', 'หรือ', 'จาก', 'มี', 'เป็น', 'ไป', 'ได้', 'ให้', 'เรื่อง', 'เกี่ยวกับ', 'the', 'and', 'in', 'of', 'to', 'for', 'with', 'on', 'at'}
    filtered_terms = [t for t in terms if t.lower() not in stopwords]
    return filtered_terms, cleaned


def _score_candidate_relevance(candidate: Dict[str, Any], query_terms: List[str], raw_query: str) -> float:
    if not query_terms and not raw_query:
        return 0.0
    title = str(candidate.get("canonical_title") or candidate.get("title") or "").lower()
    summary = str(candidate.get("comprehensive_summary") or candidate.get("summary") or "").lower()
    tags = " ".join([str(t).lower() for t in candidate.get("tags", []) or []])
    symbols = " ".join([str(s).lower() for s in candidate.get("target_symbols", []) or []])

    score = 0.0
    full_q = raw_query.strip().lower()
    if full_q and len(full_q) >= 4:
        if full_q in title:
            score += 50.0
        elif full_q in summary:
            score += 25.0

    for term in query_terms:
        t_low = term.lower()
        if not t_low or len(t_low) < 2:
            continue
        if t_low in symbols or t_low in tags:
            score += 20.0
        if t_low in title:
            score += 15.0
        elif t_low in summary:
            score += 5.0

    return score


def _generate_pitches_internal(
    candidates: List[Dict[str, Any]],
    max_pitches: int,
    instruction: str,
    date_summary: str,
) -> YouTubeContentPitchBatch:
    """ฟังก์ชันภายในสำหรับสร้าง Pitch ผ่าน structured LLM พร้อม validation"""
    topic_terms, clean_query = _extract_topic_terms(instruction)

    # แบ่ง Quota ต่อ Layer: ข่าวทั่วไป (Layer 1/2) สูงสุด 12 รายการ + คลิปกูรู YouTube สูงสุด 8 รายการ
    news_cands = [c for c in candidates if c.get("source_layer") != "layer2_youtube"]
    yt_cands = [c for c in candidates if c.get("source_layer") == "layer2_youtube"]

    # เรียงลำดับตามความเกี่ยวข้องกับ Instruction ก่อน แล้วตามด้วยความใหม่ของข้อมูล
    news_cands.sort(
        key=lambda x: (_score_candidate_relevance(x, topic_terms, clean_query), x.get("ingested_at", "")),
        reverse=True,
    )
    yt_cands.sort(
        key=lambda x: (_score_candidate_relevance(x, topic_terms, clean_query), x.get("ingested_at", "")),
        reverse=True,
    )

    selected_candidates = news_cands[:12] + yt_cands[:8]


    cand_summary_lines = []
    for i, c in enumerate(selected_candidates, 1):
        t = c.get("canonical_title") or c.get("title") or "Untitled"
        ev_id = c.get("event_id", f"ev-{i}")
        # Dynamic Truncation: layer2_youtube ได้โควตา 550 ตัวอักษร, layer อื่น 250 ตัวอักษร
        max_chars = 550 if c.get("source_layer") == "layer2_youtube" else 250
        s = c.get("comprehensive_summary", "")[:max_chars]
        links = c.get("links") or ([c.get("link")] if c.get("link") else [])
        link_str = links[0] if links else "N/A"
        cand_summary_lines.append(f"[{ev_id}] {t}\n   สรุป: {s}\n   ลิงก์: {link_str}")

    prompt_text = get_harness("youtube_pitcher", skills_root=TOOLS_PROMPTS_ROOT).get_skill_text(
        "SKILL.md",
        instruction=instruction or "รวบรวมและนำเสนอไอเดียคลิปที่ลึกซึ้ง น่าติดตาม จากข่าวที่คัดกรอง",
        date_summary=date_summary,
        candidate_count=str(len(selected_candidates)),
        total_count=str(len(candidates)),
        max_pitches=str(max_pitches),
        candidate_list="\n".join(cand_summary_lines),
    )

    slot = REGISTRY["youtube_pitch"]
    batch = invoke_structured_llm(
        schema=YouTubeContentPitchBatch,
        model_env=slot.env_var,
        prompt_lines=prompt_text.split("\n"),
        purpose="YouTube Content Pitch Generation",
        default_model=slot.default,
        provider=os.getenv("YOUTUBE_PITCH_PROVIDER", "google"),
    )
    validate_generated_pitch(batch)

    # Server-side authoritative hydration: map primary_anchor_event_id -> candidate canonical title
    cand_by_id = {
        c.get("event_id"): (c.get("canonical_title") or c.get("title") or "")
        for c in candidates
        if c.get("event_id")
    }
    for pitch in batch.pitches:
        if pitch.primary_anchor_event_id in cand_by_id:
            pitch.primary_anchor_title = cand_by_id[pitch.primary_anchor_event_id]

    return batch


def generate_youtube_pitches(
    candidates: List[Dict[str, Any]],
    max_pitches: int = 4,
    instruction: str = "",
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
) -> YouTubeContentPitchBatch:
    """สร้าง Multi-source YouTube Pitch จากรายการข่าว พร้อม retry resiliency"""
    date_summary = f"{from_date or 'อดีต'} ถึง {to_date or 'ปัจจุบัน'}"
    if not candidates:
        return YouTubeContentPitchBatch(pitches=[], date_range_summary=date_summary, total_source_events=0)

    last_error = None
    for attempt in range(2):
        try:
            # ใช้ with_retry สำหรับ transient HTTP/network errors และ try-except สำหรับ Pydantic validation fails
            return with_retry(
                lambda: _generate_pitches_internal(candidates, max_pitches, instruction, date_summary)
            )
        except Exception as e:
            last_error = e
            logger.warning("Attempt %d generation failed: %s", attempt + 1, e)

    logger.error("Structured pitch generation failed after retries: %s. Creating heuristic fallback batch.", last_error)
    first_ev_id = candidates[0].get("event_id", "ev-1") if candidates else "ev-1"
    first_ev_title = (
        (candidates[0].get("canonical_title") or candidates[0].get("title") or "ข่าวเหตุการณ์หลัก")
        if candidates
        else "ข่าวเหตุการณ์หลัก"
    )
    # Lenient / Heuristic Fallback ในกรณีที่ LLM validation fail ซ้ำ
    fallback_item = YouTubeContentPitchItem(
        pitch_id=str(uuid.uuid4())[:8],
        working_titles=[
            f"เจาะลึก: สรุปประเด็นใหญ่จากข่าวสำคัญรอบนี้ ({len(candidates)} ข่าว)",
            "วิเคราะห์สมมติฐาน: ผลกระทบต่อตลาดทุนและโอกาสลงทุน",
            "เตือนภัยและโอกาส: เตรียมรับมือความผันผวนของเศรษฐกิจล่าสุด",
        ],
        target_audience="นักลงทุนและผู้สนใจเศรษฐกิจมหภาค",
        core_thesis=f"สรุปเหตุการณ์สำคัญจาก {len(candidates)} ข่าวเด่นที่กำลังขับเคลื่อนตลาดโลกและไทยในขณะนี้",
        primary_anchor_event_id=first_ev_id,
        primary_anchor_title=first_ev_title,
        parking_lot_ideas=[],
        key_questions_to_answer=[
            "อะไรคือสาเหตุหลักของความเคลื่อนไหวในรอบนี้?",
            "ผลกระทบที่จะส่งต่อถึงตลาดหุ้นและสินทรัพย์ต่างๆ คืออะไร?",
            "กลยุทธ์ที่เหมาะสมสำหรับนักลงทุนในระยะสั้นและระยะกลางคืออะไร?",
        ],
        research_hypotheses=[
            "สมมติฐานหลัก: หากนโยบายหรือสถานการณ์ดำเนินต่อไปตามแนวโน้มปัจจุบัน จะเกิดผลกระทบเชิงโครงสร้างต่อกลุ่มอุตสาหกรรมเป้าหมาย",
            "สมมติฐานรอง: ความผันผวนของอัตราแลกเปลี่ยนและดอกเบี้ยอาจส่งผลต่อ Valuation ของสินทรัพย์เสี่ยง",
        ],
        source_event_ids=[c.get("event_id", f"ev-{i}") for i, c in enumerate(candidates[:5], 1)],
        source_links=[(c.get("links") or [c.get("link")])[0] for c in candidates[:5] if (c.get("links") or c.get("link"))],
        source_titles=[c.get("canonical_title") or c.get("title") or "Untitled" for c in candidates[:5]],
        recommended_format="Deep Dive 15m",
        estimated_impact="ผลกระทบวงกว้างต่อความเชื่อมั่นตลาดทุนและภาพรวมเศรษฐกิจ",
    )

    return YouTubeContentPitchBatch(
        pitches=[fallback_item],
        date_range_summary=date_summary,
        total_source_events=len(candidates),
    )



def synthesize_notebooklm_source(
    pitch: YouTubeContentPitchItem,
    source_events: List[Dict[str, Any]],
    macro_baselines: str = "",
    output_mode: Literal["publishable", "unverified_draft"] = "publishable",
    override_audit: Optional[Dict[str, Any]] = None,
) -> Any:
    """สังเคราะห์เอกสาร Research-Grade & Audio-Ready Briefing Book ครบ 7 Sections ให้ NotebookLM"""
    from schemas.briefing_book_schemas import (
        PublishableBriefingResult,
        InvestigativeBriefingBookDraft,
        MacroAutopsySnapshot,
        UnverifiedBriefingDraftResult,
    )
    from tools.content.briefing_evidence import build_briefing_evidence
    from tools.content.provenance_enrichment import assess_pitch_source_readiness
    from tools.market.financial_autopsy import get_financial_autopsy

    matched_events = [ev for ev in source_events if ev.get("event_id") in pitch.source_event_ids or ev.get("canonical_title") in pitch.source_titles]
    if not matched_events:
        raise ValueError(f"No matched events found for pitch '{getattr(pitch, 'title', pitch.pitch_id)}'. Synthesis requires matched_events only.")
    target_events = matched_events

    readiness, readiness_issues, issue_codes, target_events = assess_pitch_source_readiness(
        pitch, target_events, refresh=True
    )
    if readiness != "ready":
        if output_mode == "publishable":
            raise ValueError("Briefing source preflight failed before draft generation: " + "; ".join(readiness_issues))
        else:
            from tools.content.provenance_enrichment import evaluate_unverified_draft_eligibility
            if not evaluate_unverified_draft_eligibility(issue_codes):
                raise ValueError(f"Briefing source preflight failed and issue codes {issue_codes} are not allowlisted for Unverified Draft: " + "; ".join(readiness_issues))

    mode = getattr(pitch, "investigation_mode", "mixed")

    macro_snapshot = None
    if mode in {"macro", "mixed"}:
        from tools.macro.macro_autopsy import get_typed_macro_autopsy
        from tools.content.briefing_evidence import build_macro_snapshot
        try:
            obs = get_typed_macro_autopsy(investigation_mode=mode)
            macro_snapshot = build_macro_snapshot(obs, mode)
        except Exception as e:
            macro_snapshot = MacroAutopsySnapshot(
                observations=[],
                is_complete=False,
                unavailable_reasons=[str(e)],
            )

    financial_snapshots = []
    if mode in {"stock", "mixed"}:
        assets = select_financial_autopsy_assets(pitch, target_events)
        if not assets and mode == "stock":
            raise ValueError(f"No eligible asset found for Stock Mode pitch '{getattr(pitch, 'title', pitch.pitch_id)}' before LLM invocation.")
        for asset in assets:
            try:
                res = get_financial_autopsy(asset)
                if res.status == "success" and res.snapshot:
                    financial_snapshots.append(_financial_snapshot_reference(res.snapshot))
            except Exception as e:
                logger.warning("Failed to fetch financial autopsy for %s: %s", asset.raw_symbol, e)

    bundle = build_briefing_evidence(
        pitch=pitch,
        matched_sources=target_events,
        macro_snapshot=macro_snapshot,
        financial_snapshots=financial_snapshots,
    )

    from core.providers import resolve_provider
    provider_name = resolve_provider("YOUTUBE_PITCH_MODEL", "YOUTUBE_PITCH_PROVIDER", "google")

    pitch_info = "\n".join([
        f"Working Titles: {', '.join(getattr(pitch, 'working_titles', []) or ['untitled'])}",
        f"Core Thesis: {getattr(pitch, 'core_thesis', getattr(pitch, 'core_hook', ''))}",
        f"Primary Anchor: {getattr(pitch, 'primary_anchor_title', '')} ({getattr(pitch, 'primary_anchor_event_id', '')})",
        f"Investigation Mode: {getattr(pitch, 'investigation_mode', 'mixed')}",
        f"Counter-intuitive Lead: {getattr(pitch, 'counter_intuitive_lead', '')}",
        f"Audience Takeaway: {getattr(pitch, 'audience_takeaway', '')}",
        f"Key Questions: {', '.join(getattr(pitch, 'key_questions_to_answer', []) or [])}",
        f"Research Hypotheses: {', '.join(getattr(pitch, 'research_hypotheses', []) or [])}",
    ])


    evidence_lines = []
    for s in bundle.sources:
        evidence_lines.append(f"Source [{s.source_id}]: {s.original_title} | Publisher: {s.publisher} | Date: {s.published_at or 'unverified'}")
    for e in bundle.evidence_items:
        evidence_lines.append(f"Evidence [{e.evidence_id}] (Source: {', '.join(e.source_ids)}): {e.claim}")
    evidence_bundle = "\n".join(evidence_lines)

    presentation_style = getattr(pitch, "presentation_style", "narrative")
    if presentation_style == "interview_qa":
        style_prompt = (
            "**สไตล์การนำเสนอ (Presentation Style): บทสัมภาษณ์ (Interview Q&A)**\n"
            "เนื้อหาใน Act 1, 2, และ 3 ต้องเขียนในรูปแบบบทสนทนาถาม-ตอบระหว่างพิธีกรและนักวิเคราะห์ "
            "เพื่อให้ NotebookLM สามารถเลียนแบบจังหวะการจัดรายการพอดแคสต์ได้อย่างเป็นธรรมชาติ"
        )
    else:
        style_prompt = (
            "**สไตล์การนำเสนอ (Presentation Style): บทความเชิงลึก (Narrative Deep Dive)**\n"
            "เนื้อหาใน Act 1, 2, และ 3 ต้องเขียนในรูปแบบการบรรยายเล่าเรื่องที่ลึกซึ้ง น่าติดตาม "
            "พร้อมใช้การเปรียบเปรย (Analogy) ให้เห็นภาพชัดเจน"
        )

    prompt_text = get_harness("youtube_pitcher", skills_root=TOOLS_PROMPTS_ROOT).get_skill_text(
        "briefing.md",
        pitch_info=pitch_info,
        evidence_bundle=evidence_bundle,
        style_prompt=style_prompt,
    )

    slot = REGISTRY["youtube_pitch"]
    draft = invoke_structured_llm(
        schema=InvestigativeBriefingBookDraft,
        model_env=slot.env_var,
        prompt_lines=prompt_text.split("\n"),
        purpose="Briefing Book Draft Generation",
        default_model=slot.default,
        provider=provider_name,
        max_output_tokens=16384,
    )
    draft = normalize_visual_directives(draft)
    draft = normalize_draft_evidence_references(draft, bundle)

    from tools.content.briefing_renderer import (

        render_briefing_book,
        append_data_gap_notes,
        prepend_unverified_draft_banner,
    )
    from tools.content.briefing_quality import validate_briefing_book_quality
    from tools.content.briefing_artifacts import save_briefing_artifact

    rendered_briefing = render_briefing_book(draft, bundle)
    md_content = rendered_briefing.content

    # Validation step
    report = validate_briefing_book_quality(bundle, draft, rendered_briefing)
    if output_mode != "unverified_draft":
        blockers = [i.description for i in getattr(report, "issues", []) if getattr(i, "severity", "") == "blocker"]
        if blockers or not getattr(report, "publishable", True):
            raise ValueError(f"Briefing book failed quality gate with score {report.score}: {blockers}")
    else:
        critical_unbypassable = [
            i.description for i in getattr(report, "issues", [])
            if getattr(i, "severity", "") == "blocker" and not getattr(i, "bypassable", False)
        ]
        if critical_unbypassable:
            raise ValueError(f"Briefing book (Unverified Draft) failed quality gate on UNBYPASSABLE issues: {critical_unbypassable}")

    # Surface known data-completeness gaps directly in the document — the
    # quality gate already knows about these but they used to stay sidecar-only
    # in the .quality.json, invisible to whatever reads the .md (NotebookLM included).
    numeric_warnings = [
        issue.description for issue in getattr(report, "issues", [])
        if getattr(issue, "code", "") in {
            "NUMERIC_GROUNDING_WARNING",
            "FINANCIAL_PROVIDER_UNAVAILABLE",
            "MACRO_MISSING_INFLATION",
            "MACRO_MISSING_RATES",
            "MISSING_MACRO_SNAPSHOT",
            "STALE_MACRO_SNAPSHOT"
        }
    ]
    macro_unavailable_reasons = []
    if bundle.macro_snapshot and not bundle.macro_snapshot.is_complete:
        macro_unavailable_reasons = list(bundle.macro_snapshot.unavailable_reasons)
    md_content = append_data_gap_notes(
        md_content,
        numeric_warnings=numeric_warnings,
        macro_unavailable_reasons=macro_unavailable_reasons,
    )

    if output_mode == "unverified_draft":
        draft_reason = (override_audit or {}).get("reason", "Unverified Draft fallback requested by user")
        md_content = prepend_unverified_draft_banner(
            md_content,
            reason=draft_reason,
            issues=list(getattr(pitch, "source_readiness_issues", []) or []),
        )
        # An Unverified Draft is, by definition in this system, never
        # publishable — the content-level score/blocker check above can pass
        # (e.g. a bypassable SINGLE_INDEPENDENT_SOURCE cap still leaves score
        # >= 80 with no blocker) even though the source-readiness gate is the
        # one that actually decided this pitch needed the draft path. Force
        # this here rather than trusting validate_briefing_book_quality's
        # generic score, which has no knowledge of that outer decision.
        report.publishable = False
        return UnverifiedBriefingDraftResult(
            content=md_content,
            draft=draft,
            quality_report=report,
            evidence_bundle=bundle,
            override_audit=override_audit,
        )
    return PublishableBriefingResult(
        content=md_content,
        draft=draft,
        quality_report=report,
        evidence_bundle=bundle,
    )


def normalize_visual_directives(draft: Any) -> Any:
    if not getattr(draft, "visual_directives", None):
        return draft
    acts_seen = set()
    new_directives = []

    for d in draft.visual_directives:
        act = d.act if hasattr(d, "act") else None
        if act and act not in acts_seen:
            acts_seen.add(act)

            sources = getattr(d, "sources", [])
            series = getattr(d, "series_keys", [])
            is_generic = any(str(s).lower() in {"war cost", "interest rate", "inflation", "cpi", "gdp"} for s in series)

            if "Evidence ledger" in sources or is_generic or getattr(d, "data_mode", "") == "evidence_table":
                d.data_mode = "evidence_table"
                d.series_keys = ["EVIDENCE_TABLE"]

            new_directives.append(d)
    draft.visual_directives = new_directives
    return draft


def normalize_draft_evidence_references(draft: Any, bundle: Any) -> Any:
    """Sanitize and coerce evidence IDs in draft to valid canonical EvidenceItem IDs from bundle."""
    import re
    if not draft or not bundle or not getattr(bundle, "evidence_items", None):
        return draft

    valid_eids = {e.evidence_id for e in bundle.evidence_items}
    if not valid_eids:
        return draft

    all_valid_sorted = sorted(valid_eids)

    # Build mapping for lookup
    lookup: dict = {}
    for eid in valid_eids:
        lookup[eid.upper()] = [eid]
        lookup[eid.lower()] = [eid]
        lookup[f"[{eid}]".upper()] = [eid]
        lookup[f"[{eid}]".lower()] = [eid]
        m = re.match(r"^E(\d+)$", eid, re.IGNORECASE)
        if m:
            num = int(m.group(1))
            lookup[f"E{num}".upper()] = [eid]
            lookup[f"E{num}".lower()] = [eid]
            lookup[f"[E{num}]".upper()] = [eid]
            lookup[f"[E{num}]".lower()] = [eid]

    for item in bundle.evidence_items:
        for sid in getattr(item, "source_ids", []) or []:
            lookup.setdefault(sid.upper(), []).append(item.evidence_id)
            lookup.setdefault(sid.lower(), []).append(item.evidence_id)
            lookup.setdefault(f"[{sid}]".upper(), []).append(item.evidence_id)
            lookup.setdefault(f"[{sid}]".lower(), []).append(item.evidence_id)
        if item.metric_name and ":" in item.metric_name:
            sym = item.metric_name.split(":")[0].strip()
            if sym:
                lookup.setdefault(sym.upper(), []).append(item.evidence_id)
                lookup.setdefault(sym.lower(), []).append(item.evidence_id)

    def sanitize_id_list(raw_ids: Any, fallback_ids: list) -> list:
        if isinstance(raw_ids, str):
            raw_tokens = re.findall(r"\[?E\d+\]?|\[?S[A-Za-z0-9_]+\]?|[A-Za-z0-9_-]+", raw_ids)
        elif isinstance(raw_ids, list):
            raw_tokens = []
            for item in raw_ids:
                if isinstance(item, str):
                    tokens = re.findall(r"\[?E\d+\]?|\[?S[A-Za-z0-9_]+\]?|[A-Za-z0-9_-]+", item)
                    raw_tokens.extend(tokens if tokens else [item.strip()])
                else:
                    raw_tokens.append(str(item).strip())
        else:
            raw_tokens = []

        resolved = []
        for tok in raw_tokens:
            cleaned = tok.strip()
            if not cleaned:
                continue
            if cleaned in valid_eids:
                resolved.append(cleaned)
            elif cleaned.upper() in lookup:
                resolved.extend(lookup[cleaned.upper()])
            elif cleaned in lookup:
                resolved.extend(lookup[cleaned])

        final_ids = []
        seen = set()
        for eid in resolved:
            if eid in valid_eids and eid not in seen:
                seen.add(eid)
                final_ids.append(eid)

        if not final_ids:
            return fallback_ids[:2] if fallback_ids else all_valid_sorted[:2]
        return final_ids

    for sc in getattr(draft, "causality_scenarios", []) or []:
        sc.evidence_ids = sanitize_id_list(getattr(sc, "evidence_ids", []), all_valid_sorted[:2])

    for ai in getattr(draft, "asset_impacts", []) or []:
        sym = getattr(ai, "symbol_or_name", "").strip().upper()
        sym_matched_eids = []
        for word in re.findall(r"[A-Za-z0-9]+", sym):
            if word.upper() in lookup:
                sym_matched_eids.extend(lookup[word.upper()])
        ai.evidence_ids = sanitize_id_list(getattr(ai, "evidence_ids", []), sym_matched_eids or all_valid_sorted[:2])

    for vd in getattr(draft, "visual_directives", []) or []:
        vd.evidence_ids = sanitize_id_list(getattr(vd, "evidence_ids", []), all_valid_sorted[:1])

    return draft


def select_financial_autopsy_assets(pitch: Any, events: Any) -> list:

    from tools.market.asset_resolver import resolve_asset

    potential_symbols = []

    # 1. From pitch.target_symbols if available
    if hasattr(pitch, "target_symbols") and pitch.target_symbols:
        potential_symbols.extend(pitch.target_symbols)

    # 2. From events
    for ev in events:
        if isinstance(ev, dict) and ev.get("symbols"):
            potential_symbols.extend(ev["symbols"])

    # Fallback to regex if we have absolutely nothing
    if not potential_symbols:
        import re
        text_parts = [getattr(pitch, "title", ""), getattr(pitch, "core_thesis", getattr(pitch, "core_hook", "")), *getattr(pitch, "working_titles", [])]
        for ev in events:
            text_parts.append(ev.get("canonical_title", ""))
            text_parts.append(ev.get("comprehensive_summary", ""))
        text = " ".join(text_parts)
        potential_symbols = re.findall(r'\b[A-Z][A-Z0-9]{1,5}\b', text)

    resolved = []
    seen = set()
    for sym in potential_symbols:
        if sym in seen: continue
        seen.add(sym)
        try:
            asset = resolve_asset(sym)
            if asset.eligible_for_financial_autopsy:
                resolved.append(asset)
                if len(resolved) >= 3:
                    break
        except Exception:
            pass
    return resolved

def _financial_snapshot_reference(result: Any) -> Any:
    from schemas.briefing_book_schemas import FinancialAutopsySnapshotRef, FinancialAutopsyPeriodRecord
    period_records = []
    for p in getattr(result, "periods", []) or []:
        if isinstance(p, FinancialAutopsyPeriodRecord):
            period_records.append(p)
        elif hasattr(p, "model_dump"):
            period_records.append(FinancialAutopsyPeriodRecord.model_validate(p.model_dump()))
        elif isinstance(p, dict):
            period_records.append(FinancialAutopsyPeriodRecord.model_validate(p))
        else:
            period_records.append(FinancialAutopsyPeriodRecord(
                fiscal_period_end=str(getattr(p, "fiscal_period_end", "")),
                free_cash_flow=getattr(p, "free_cash_flow", None),
                operating_cash_flow=getattr(p, "operating_cash_flow", None),
                capital_expenditure=getattr(p, "capital_expenditure", None),
                total_debt=getattr(p, "total_debt", None),
                total_revenue=getattr(p, "total_revenue", None),
                net_income=getattr(p, "net_income", None),
                dividends_paid=getattr(p, "dividends_paid", None),
                payout_ratio_pct=getattr(p, "payout_ratio_pct", None),
            ))

    return FinancialAutopsySnapshotRef(
        symbol=result.ticker,
        provider_symbol=result.provider_symbol,
        status="success",
        currency=result.currency,
        source=result.source,
        periods=period_records,
        market_cap=result.market_cap_formatted,
        revenue=result.revenue_formatted,
        net_income=result.net_income_formatted,
        fcf=result.fcf_formatted,
        total_debt=result.total_debt_formatted,
        health_notes=result.health_summary,
    )

