"""News Funnel Pipeline & Display-Name Map สำหรับสถาปัตยกรรมคัดกรองข่าวและสร้าง Obsidian Linking Key

ครอบคลุม:
1. THEME_DISPLAY_NAME_MAP แปลง 7 ค่ามาตรฐาน ThemeCategory เป็นชื่อ Wikilink มาตรฐาน
2. canonicalize_ticker_names แปลงชื่อหุ้นและสัญลักษณ์ให้เป็นมาตรฐาน
3. ensure_concept_stubs_exist สร้าง stub พื้นฐานในโฟลเดอร์ 30_Knowledge_Base/Concepts/
4. run_news_funnel_ingest ดึงข่าว ทำ clustering, คัดกรองผ่าน Batch LLM และบันทึกสถานะลง JSON Store
5. run_news_funnel_synthesize สังเคราะห์ข่าวรอบ 12 ชั่วโมงผ่าน Batch LLM พร้อมระบบ Zero-Pending Protection
"""
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional
import uuid
import yaml

from core.logger import get_logger


def _is_mock_mode() -> bool:
    """ตรวจสอบว่าเปิดใช้งาน MOCK_NEWS_FUNNEL_LLM สำหรับเทสต์หรือออฟไลน์หรือไม่"""
    return os.getenv("MOCK_NEWS_FUNNEL_LLM", "false").lower() == "true"


def _invoke_structured(schema: Any, model_env: str, prompt_lines: List[str]) -> Any:
    """Helper สำหรับสร้างและเรียกใช้ structured LLM ด้วย provider='google'"""
    from core.llm_factory import get_llm
    model_name = os.getenv(model_env, "gemini-2.5-flash")
    llm = get_llm(provider="google", model_name=model_name)
    structured_llm = llm.with_structured_output(schema)
    return structured_llm.invoke("\n".join(prompt_lines))
from core.nlp_utils import (
    _jaccard_similarity,
    group_similar_news,
    select_representative_news,
)
from schemas.news_funnel_schemas import (
    DailyFunnelReport,
    MacroImpactTriageResult,
    MacroThemeDigest,
    ThemeSynthesisBatchResult,
    TriageBatchResult,
)
from tools._atomic_io import _atomic_write_to
from tools.archivist.core import _sanitize_filename
from tools.macro.news_funnel_store import (
    get_pending_high_impact_events,
    is_title_or_url_processed,
    load_store,
    mark_events_synthesized,
    save_triage_events,
)

logger = get_logger(__name__)

# 7 ค่ามาตรฐานของ ThemeCategory Enum พร้อม Display-Name Map
THEME_DISPLAY_NAME_MAP: Dict[str, str] = {
    "policy": "Monetary Policy",
    "growth": "Economic Growth & Recession",
    "inflation": "Inflation",
    "liquidity": "Fiscal Policy & Debt",
    "geopolitics": "Geopolitics & Commodities",
    "earnings": "Earnings & Corporate Profits",
    "risk_sentiment": "Market Risk Sentiment",
}

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


def strip_wikilink(text: str) -> str:
    """ลบเครื่องหมาย [[ ... ]] และ Alias [[File|Alias]] ออกจากข้อความ"""
    clean = text.strip()
    while clean.startswith("[[") and clean.endswith("]]"):
        clean = clean[2:-2].strip()
    if "|" in clean:
        clean = clean.split("|")[-1].strip()
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
    summary = candidate.get("summary", "")
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

    # ห้ามคืนค่า default ปลอมเมื่อ LLM ล้มเหลว ให้ข้าม batch นี้ไป
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

    # หาก len ไม่ตรงกัน หรือ LLM ล้มเหลว ให้ fail-closed ข้าม batch นี้ทันที (ไม่บันทึกคะแนนมั่ว ไม่กลืนข่าว)
    if len(triage_results) != len(representatives):
        logger.error("LLM returned %d results for %d items — skipping batch", len(triage_results), len(representatives))
        return {"status": "skipped_llm_failure", "ingested_count": 0, "high_impact_count": 0}

    new_events = []
    now_iso = datetime.now().isoformat()
    for i, (rep, triage) in enumerate(zip(representatives, triage_results)):
        title = rep.get("title", "").strip()
        link = rep.get("link", "")
        # สร้าง event_id จาก UUID4 หรือ Hash
        event_id = rep.get("event_id") or str(uuid.uuid4())
        ev = {
            "event_id": event_id,
            "canonical_title": title,
            "comprehensive_summary": rep.get("summary", rep.get("freshness_reason", "")),
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


def _deterministic_synthesize_themes(events: List[Dict[str, Any]]) -> List[MacroThemeDigest]:
    """Deterministic fallback สำหรับ mock mode โดยจัดกลุ่มตาม extracted_themes[0] เป็นสูงสุด 3 ธีม (ไม่ทิ้ง events)"""
    if not events:
        return []

    buckets: Dict[str, List[Dict[str, Any]]] = {}
    bucket_order: List[str] = []
    for ev in events:
        raw_themes = ev.get("extracted_themes") or []
        primary_theme = raw_themes[0] if raw_themes else "General Macro"
        display_theme = THEME_DISPLAY_NAME_MAP.get(primary_theme, primary_theme)
        if display_theme not in buckets:
            if len(bucket_order) < 3:
                buckets[display_theme] = []
                bucket_order.append(display_theme)
            else:
                # ยุบเข้า bucket สุดท้ายถ้าเกิน 3
                display_theme = bucket_order[-1]
        buckets[display_theme].append(ev)

    themes_list = []
    for b_name in bucket_order:
        b_events = buckets[b_name]
        titles = [ev.get("canonical_title", "Untitled Event") for ev in b_events]
        takeaways = [
            f"ข้อสังเกตหลัก: {ev.get('comprehensive_summary')}" if ev.get("comprehensive_summary") else f"เหตุการณ์: {ev.get('canonical_title', 'Untitled Event')}"
            for ev in b_events
        ]
        all_tickers = []
        all_themes = []
        for ev in b_events:
            all_tickers.extend(ev.get("extracted_tickers", []))
            all_themes.extend(ev.get("extracted_themes", []))

        asset_links = [f"[[{t}]]" for t in canonicalize_ticker_names(all_tickers)]
        theme_links = [f"[[{THEME_DISPLAY_NAME_MAP.get(rt, rt)}]]" for rt in list(set(all_themes)) if rt]
        if f"[[{b_name}]]" not in theme_links:
            theme_links.append(f"[[{b_name}]]")

        themes_list.append(
            MacroThemeDigest(
                theme_title=f"นัยสำคัญด้านเศรษฐกิจ: {b_name} ({', '.join(titles[:2])})",
                key_takeaways=takeaways,
                linked_assets=asset_links,
                linked_themes=theme_links,
                policy_implications="เฝ้าระวังความผันผวนของราคาสินทรัพย์ที่เชื่อมโยงในระยะสั้น",
            )
        )
    return themes_list


def _llm_synthesize_themes(
    events: List[Dict[str, Any]],
    period: str,
) -> Optional[List[MacroThemeDigest]]:
    """สังเคราะห์ Key Macro Themes สูงสุด 3 ธีมผ่าน LLM หรือ deterministic fallback สำหรับ mock mode"""
    if not events:
        return []
    if _is_mock_mode():
        return _deterministic_synthesize_themes(events)

    try:
        prompt_lines = [
            f"Synthesize the following high-impact macro news events into at most 3 Key Macro Themes for the {period} report.",
            "Group related events into cohesive themes.",
            "For each theme provide clear key_takeaways, linked_assets (tickers), linked_themes (macro themes), and practical policy_implications.",
            "Events:"
        ]
        for idx, ev in enumerate(events):
            prompt_lines.append(
                f"{idx+1}. {ev.get('canonical_title', '')} | Summary: {ev.get('comprehensive_summary', '')} | Tickers: {ev.get('extracted_tickers', [])} | Themes: {ev.get('extracted_themes', [])}"
            )

        res = _invoke_structured(ThemeSynthesisBatchResult, "NEWS_FUNNEL_SYNTHESIS_MODEL", prompt_lines)
        if res and hasattr(res, "themes") and res.themes:
            for thm in res.themes[:3]:
                clean_assets = canonicalize_ticker_names(thm.linked_assets)
                thm.linked_assets = [f"[[{t}]]" for t in clean_assets]
                thm.linked_themes = [f"[[{THEME_DISPLAY_NAME_MAP.get(strip_wikilink(t), strip_wikilink(t))}]]" for t in thm.linked_themes]
            return res.themes[:3]
    except Exception as e:
        logger.error("LLM Synthesis failed or unavailable: %s", e)

    return None


def run_news_funnel_synthesize(
    period: str,
    approved_event_ids: Optional[List[str]] = None,
    store_path: Optional[str] = None,
    vault_root: Optional[str] = None,
    custom_date: Optional[str] = None,
) -> Dict[str, Any]:
    """สังเคราะห์ข่าวรอบ 12 ชั่วโมง พร้อม Zero-Pending Protection ไม่เขียนไฟล์เปล่าทับเมื่อไม่มีข่าว"""
    pending = get_pending_high_impact_events(store_path=store_path)

    if approved_event_ids is not None:
        target_ids = set(approved_event_ids)
        events_to_synthesize = [e for e in pending if e.get("event_id") in target_ids]
        approved_by = "user_kanban_hitl"
    else:
        events_to_synthesize = pending
        approved_by = "scheduled_auto"

    # Zero-Pending Protection: หากไม่มีข่าวที่ต้องสังเคราะห์ ให้ Log-Only และไม่เขียนไฟล์ทับเด็ดขาด
    if not events_to_synthesize:
        logger.info("No pending events to synthesize")
        return {
            "status": "no_pending_events",
            "synthesized": 0,
            "message": "No pending events to synthesize. Log-Only No File Overwrite.",
        }

    date_str = custom_date or datetime.now().strftime("%Y-%m-%d")
    period_label = period.capitalize()

    themes_list = _llm_synthesize_themes(events_to_synthesize, period=period)
    if themes_list is None:
        logger.error("LLM synthesis failed. Preserving pending events and skipping file write.")
        return {
            "status": "synthesis_llm_failure",
            "synthesized": 0,
            "message": "LLM synthesis failed. No file written, pending events preserved.",
        }

    all_linked_assets: set = set()
    all_linked_themes: set = set()
    all_tags: set = {"macro", "digest", "news_funnel", "linking_key"}

    for thm in themes_list:
        all_linked_assets.update(thm.linked_assets)
        all_linked_themes.update(thm.linked_themes)

    report = DailyFunnelReport(
        report_title=f"Macro Themes Digest - {date_str} {period_label}",
        report_date=date_str,
        batch_period=f"{period.lower()}_12h",
        approved_by=approved_by,
        themes=themes_list,
        total_events_analyzed=len(events_to_synthesize),
        high_impact_event_ids=[e.get("event_id", "") for e in events_to_synthesize],
    )

    # เขียนไฟล์ลง Obsidian Vault ด้วย YAML Frontmatter ที่ถูกต้องผ่าน yaml.safe_dump
    root = vault_root or os.getenv("OBSIDIAN_VAULT_PATH", "./memories")
    out_dir = Path(root) / "30_Knowledge_Base" / "News" / "Themes"
    out_dir.mkdir(parents=True, exist_ok=True)
    base_filename = f"{date_str}_{period_label}_MacroThemes"
    out_file = out_dir / f"{base_filename}.md"
    counter = 2
    while out_file.exists():
        out_file = out_dir / f"{base_filename}_{counter}.md"
        counter += 1

    sorted_assets = sorted(all_linked_assets)
    sorted_themes = sorted(all_linked_themes)

    frontmatter_dict = {
        "title": report.report_title,
        "date": report.report_date,
        "entity_type": "news_theme_digest",
        "batch_period": report.batch_period,
        "approved_by": report.approved_by,
        "linked_assets": sorted_assets,
        "linked_themes": sorted_themes,
        "tags": sorted(all_tags),
    }

    yaml_header = yaml.safe_dump(frontmatter_dict, allow_unicode=True, sort_keys=False).strip()
    md_lines = [
        "---",
        yaml_header,
        "---\n",
        f"# 🌐 Key Macro Themes (รอบ {period_label} {date_str})\n",
        f"> **สรุป {len(report.themes)} ธีมเศรษฐกิจหลักประจำวัน** จากข่าว High-Impact ที่ผ่านการอนุมัติในรอบ 12 ชั่วโมง\n",
    ]

    for i, t in enumerate(report.themes, 1):
        md_lines.append(f"## {i}. {t.theme_title}")
        for take in t.key_takeaways:
            md_lines.append(f"- **เนื้อหาหลัก**: {take}")
        if t.linked_assets:
            md_lines.append(f"- **สินทรัพย์ที่เชื่อมโยง**: {', '.join(t.linked_assets)}")
        if t.linked_themes:
            md_lines.append(f"- **ธีมที่เกี่ยวข้อง**: {', '.join(t.linked_themes)}")
        if t.policy_implications:
            md_lines.append(f"- **นัยต่อนโยบายพอร์ต**: {t.policy_implications}")
        md_lines.append("")

    # Atomic write ลง Vault ผ่าน tools._atomic_io
    _atomic_write_to(out_file, "\n".join(md_lines))

    # สร้าง Concept Stubs สำหรับคำที่ไม่ใช่ Ticker มาตรฐาน
    concept_candidates = []
    for asset_link in sorted_assets:
        raw_name = strip_wikilink(asset_link)
        if raw_name not in TICKER_ALIAS_MAP and raw_name not in TICKER_ALIAS_MAP.values():
            concept_candidates.append(raw_name)
    ensure_concept_stubs_exist(concept_candidates, vault_root=vault_root)

    # อัปเดตสถานะใน JSON Store เป็น synthesized
    mark_events_synthesized(report.high_impact_event_ids, store_path=store_path)

    return {
        "status": "success",
        "synthesized": len(events_to_synthesize),
        "file_path": str(out_file),
        "report": report.model_dump(),
    }
