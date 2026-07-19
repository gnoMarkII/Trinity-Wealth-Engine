"""เครื่องมือหลักสำหรับดักจับข่าว เสนอไอเดีย และสร้าง Research-Grade Briefing Book ให้ NotebookLM"""
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple
import uuid

from core.llm_factory import get_llm, invoke_structured_llm
from core.logger import get_logger
from core.nlp_utils import _jaccard_similarity
from core.retry import with_retry
from core.utils import normalize_content
from schemas.youtube_pitch_schemas import YouTubeContentPitchBatch, YouTubeContentPitchItem
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


def _generate_pitches_internal(
    candidates: List[Dict[str, Any]],
    max_pitches: int,
    instruction: str,
    date_summary: str,
) -> YouTubeContentPitchBatch:
    """ฟังก์ชันภายในสำหรับสร้าง Pitch ผ่าน structured LLM พร้อม validation"""
    # แบ่ง Quota ต่อ Layer: ข่าวทั่วไป (Layer 1/2) สูงสุด 12 รายการ + คลิปกูรู YouTube สูงสุด 8 รายการ
    news_cands = [c for c in candidates if c.get("source_layer") != "layer2_youtube"]
    yt_cands = [c for c in candidates if c.get("source_layer") == "layer2_youtube"]
    # เรียงลำดับตามวันที่ (ใหม่ไปเก่า) หากมี ingested_at
    news_cands.sort(key=lambda x: x.get("ingested_at", ""), reverse=True)
    yt_cands.sort(key=lambda x: x.get("ingested_at", ""), reverse=True)

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

    prompt_lines = [
        "คุณคือ Chief Content Architect และ Senior Macro Analyst สำหรับช่อง YouTube การเงินและเศรษฐกิจชั้นนำ",
        f"คำสั่งพิเศษจากผู้ใช้: {instruction or 'รวบรวมและนำเสนอไอเดียคลิปที่ลึกซึ้ง น่าติดตาม จากข่าวที่คัดกรอง'}",
        f"ช่วงวันที่คัดกรอง: {date_summary}",
        f"จำนวนข้อมูลและบทวิเคราะห์ที่คัดกรองมาทั้งหมด: {len(selected_candidates)} รายการ (จากทั้งหมด {len(candidates)} รายการในคลัง)",
        "\n--- รายชื่อข่าวและบทวิเคราะห์จากกูรูที่คัดกรองมา (Candidates) ---",
        "\n".join(cand_summary_lines),
        "\n--- คำสั่งในการสร้าง Multi-source Pitch ---",
        f"1. ให้คัดเลือกหรือรวบรวมกลุ่มข่าว (Multi-source) ที่เชื่อมโยงกันเป็นเรื่องใหญ่ เพื่อนำเสนอไอเดียทำคลิป YouTube จำนวน {max_pitches} หัวข้อ",
        "2. แต่ละหัวข้อ (Item) ต้องมี 'working_titles' พอดี 3 ชื่อ ครบ 3 สไตล์: 1.คำถามเจาะลึก 2.วิเคราะห์สมมติฐาน 3.เตือนภัย/โอกาส",
        "3. ในแต่ละหัวข้อ ต้องมี 'key_questions_to_answer' อย่างน้อย 3 ข้อ และ 'research_hypotheses' อย่างน้อย 2 ข้อ เพื่อให้ NotebookLM โหมด Research ไปค้นคว้าต่อได้อย่างเข้มข้น",
        "4. กรุณาระบุ 'source_event_ids', 'source_titles', และ 'source_links' จาก Candidates ต้นทางให้ถูกต้อง",
        "5. ผลลัพธ์ทั้งหมดต้องเป็นภาษาไทยสละสลวย เป็นมืออาชีพ ดึงดูดสายตา",
    ]

    return invoke_structured_llm(
        schema=YouTubeContentPitchBatch,
        model_env="YOUTUBE_PITCH_MODEL",
        prompt_lines=prompt_lines,
        purpose="YouTube Content Pitch Generation",
        default_model="gemini-3.1-flash-lite-preview",
        provider="google",
    )


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
    # Lenient / Heuristic Fallback ในกรณีที่ LLM validation fail ซ้ำ
    fallback_item = YouTubeContentPitchItem(
        pitch_id=str(uuid.uuid4())[:8],
        working_titles=[
            f"เจาะลึก: สรุปประเด็นใหญ่จากข่าวสำคัญรอบนี้ ({len(candidates)} ข่าว)",
            "วิเคราะห์สมมติฐาน: ผลกระทบต่อตลาดทุนและโอกาสลงทุน",
            "เตือนภัยและโอกาส: เตรียมรับมือความผันผวนของเศรษฐกิจล่าสุด",
        ],
        target_audience="นักลงทุนและผู้สนใจเศรษฐกิจมหภาค",
        core_hook=f"สรุปเหตุการณ์สำคัญจาก {len(candidates)} ข่าวเด่นที่กำลังขับเคลื่อนตลาดโลกและไทยในขณะนี้",
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
) -> str:
    """สังเคราะห์เอกสาร Research-Grade & Audio-Ready Briefing Book ครบ 7 Sections ให้ NotebookLM

    ใช้ plain llm.invoke พร้อม max_output_tokens=16384 เพื่อให้เนื้อหาลึกซึ้ง ครบถ้วน ไม่ถูกตัดจบ
    """
    model_name = os.getenv("YOUTUBE_PITCH_MODEL", "gemini-3.1-flash-lite-preview")
    llm = get_llm(provider="google", model_name=model_name, max_output_tokens=16384)

    # เตรียมเนื้อหาข่าวต้นทาง
    source_details = []
    for ev in source_events:
        ev_id = ev.get("event_id", "")
        if ev_id in pitch.source_event_ids or (ev.get("canonical_title") in pitch.source_titles):
            t = ev.get("canonical_title") or ev.get("title") or ""
            s = ev.get("comprehensive_summary") or ev.get("summary") or ""
            l = (ev.get("links") or [ev.get("link")])[0] if (ev.get("links") or ev.get("link")) else "N/A"
            source_details.append(f"- **{t}** (ID: `{ev_id}`)\n  สรุป: {s}\n  อ้างอิง: {l}")

    if not source_details:
        for i, t in enumerate(pitch.source_titles):
            l = pitch.source_links[i] if i < len(pitch.source_links) else "N/A"
            source_details.append(f"- **{t}**\n  อ้างอิง: {l}")

    prompt = f"""คุณคือ Senior Research Director และ Chief Content Architect ประจำช่อง YouTube การเงินระดับโลก
หน้าที่ของคุณคือการสร้างเอกสาร **Research-Grade & Audio-Ready Briefing Book** ฉบับสมบูรณ์ สำหรับอัปโหลดเข้าสู่ระบบ **NotebookLM**
เอกสารนี้ต้องรองรับทั้ง **โหมด Research (ค้นคว้าเจาะลึก/อิง Hard Data)** และ **โหมด Audio Overview (Podcast 2 คนถกเถียงอย่างเข้มข้น)**

--- ข้อมูลไอเดียคลิป (Pitch Item) ---
Working Titles: {', '.join(pitch.working_titles)}
Target Audience: {pitch.target_audience}
Core Hook: {pitch.core_hook}
Key Questions: {', '.join(pitch.key_questions_to_answer)}
Research Hypotheses: {', '.join(pitch.research_hypotheses)}
Recommended Format: {pitch.recommended_format}
Estimated Impact: {pitch.estimated_impact}

--- ข่าวและบทวิเคราะห์ที่เกี่ยวข้อง ---
{chr(10).join(source_details)}

--- ข้อมูลเศรษฐกิจมหภาคพื้นฐาน (Macro Baselines Snapshot) ---
{macro_baselines or 'N/A'}

================================================================================
คำสั่งและข้อกำหนดในการเขียนเอกสาร Briefing Book (ต้องครบถ้วนทั้ง 7 Sections ด้านล่างนี้ ห้ามย่อหรือข้ามส่วนใดส่วนหนึ่ง):
================================================================================
เขียนเป็นภาษาไทยระดับมืออาชีพ สวยงาม จัดรูปแบบด้วย Markdown ครบทั้ง 7 Sections ดังนี้:

# 📑 สรุปผู้บริหารและแหล่งอ้างอิง (Executive Briefing & Provenance)
- สรุปภาพรวมประเด็นสำคัญใน 3-5 บรรทัด
- ตารางระบุแหล่งข้อมูลต้นทาง (สำนักข่าว, ลิงก์, วันที่, และประเมินระดับความน่าเชื่อถือ)

# 📊 ตารางแยกชั้น: ข้อเท็จจริง vs ความเห็นตลาด vs ข่าวลือ (Fact vs. Consensus vs. Speculation Matrix)
- นำเสนอเป็นตาราง Markdown 3 คอลัมน์ที่ชัดเจน:
  1. ข้อเท็จจริงและตัวเลขที่ยืนยันแล้ว (Hard Facts & Verified Metrics)
  2. ความเห็นและประมาณการของนักวิเคราะห์ตลาด (Market Consensus & Analyst Projections)
  3. ข่าวลือ สมมติฐาน และความกังวลที่ยังรอการพิสูจน์ (Speculation & Tail Risks)
*(สำคัญมากสำหรับ NotebookLM Research Mode เพื่อไม่ให้ปนข้อเท็จจริงกับความเห็น)*

# 🔗 กลไกส่งต่อผลกระทบเชิงระบบ (Structural Causality)
- อธิบายกลไกเหตุและผลเชิงโครงสร้าง (Causal Chain) จากต้นเหตุสู่ผลลัพธ์
- วิเคราะห์ผลกระทบทอดที่สอง (Second-order effects) และทอดที่สาม (Third-order effects)
- นำเสนอแผนผังลูกศร `-->` หรือผังกลไกที่ชัดเจนเข้าใจง่าย

# 🏢 สินทรัพย์และหุ้นที่เกี่ยวข้อง (Asset & Ticker Impact)
- ระบุชื่อหุ้น สินทรัพย์ สกุลเงิน หรืออุตสาหกรรมที่เกี่ยวข้อง พร้อมครอบด้วย `[[Wikilinks]]` เช่น `[[NVDA]]`, `[[SET]]`
- วิเคราะห์ทั้งโอกาสเชิงบวก (Upside Catalysts) และความเสี่ยงเชิงลบ (Downside Risks) ของแต่ละสินทรัพย์

# ⚡ ประเด็นขัดแย้งและวิวาทะ Bull vs Bear (Opposing Viewpoints & Debate Points)
- แยกมุมมองฝั่งกระทิง (Bull Case: มองบวก มองโอกาส) และฝั่งหมี (Bear Case: มองลบ เตือนความเสี่ยง) ให้ปะทะกันอย่างสมเหตุสมผล
- ระบุจุดชี้ขาด (Key Turnarounds / Trigger Points) ที่ตัดสินว่าฝั่งไหนจะเป็นฝ่ายชนะ
*(สำคัญมากสำหรับ NotebookLM Audio Overview เพื่อให้ AI Host 2 คนพูดคุยโต้เถียงกันได้อย่างสนุกและมีมิติ)*

# 🎙️ โครงเรื่องคลิปและ Talking Points
- โครงเรื่องสคริปต์คลิป YouTube แบ่งเป็น 3 Acts (Act I: The Hook & Setup, Act II: Deep Dive & Conflict, Act III: Resolution & Actionable Takeaways)
- แทรก "คำเปรียบเปรย (Analogies)" ที่ช่วยให้อธิบายเรื่องการเงินยากๆ ให้เข้าใจง่ายใน 10 วินาที

# 🔬 คำถามวิจัยขั้นสูงสำหรับ NotebookLM
- ลิสต์ชุดคำถาม 5-8 ข้อที่ลึกซึ้งและเฉียบคม พร้อมให้ผู้ใช้คัดลอก (Copy & Paste) ไปถามต่อในแชท NotebookLM Research Mode ได้ทันที

ขอให้สร้างสรรค์เนื้อหาแต่ละ Section อย่างละเอียด ลึกซึ้ง เปี่ยมด้วยคุณค่าเชิงวิเคราะห์ขั้นสูง ห้ามสรุปสั้นจนเสียรายละเอียด"""

    logger.info("Synthesizing Briefing Book with model=%s (max_output_tokens=16384)", model_name)
    response = llm.invoke(prompt)
    content = normalize_content(getattr(response, "content", str(response)))
    return content


def save_notebooklm_source(
    content: str,
    title: str,
    date_str: Optional[str] = None,
) -> str:
    """บันทึก Briefing Book ลงใน memories/30_Knowledge_Base/NotebookLM_Sources/ พร้อมจัดการชื่อไฟล์ไทยและ collision"""
    target_dir = Path(VAULT_PATH) / "30_Knowledge_Base" / "NotebookLM_Sources"
    target_dir.mkdir(parents=True, exist_ok=True)

    d_str = date_str or datetime.now().strftime("%Y-%m-%d")
    # _sanitize_filename รองรับอักษรไทยอยู่แล้ว รักษาอักษรไทยไม่ให้ถูกตัด
    safe_title = _sanitize_filename(title.strip()[:80])
    filename = f"{d_str}_{safe_title}.md"
    file_path = target_dir / filename

    counter = 2
    while file_path.exists():
        file_path = target_dir / f"{d_str}_{safe_title}_{counter}.md"
        counter += 1

    _atomic_write_to(file_path, content)
    from tools.archivist.indexer import _index_upsert, flush_index_if_dirty
    _index_upsert(file_path, vault_root=Path(VAULT_PATH))
    flush_index_if_dirty(vault_root=Path(VAULT_PATH))
    logger.info("Successfully saved NotebookLM source briefing book to %s", file_path)
    return str(file_path.resolve())
