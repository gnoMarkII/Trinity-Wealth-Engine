"""Pure domain engine functions for News and YouTube Digest processing."""
from datetime import datetime, timezone
import re
from typing import Dict, List, Optional, Tuple

from core.news_digest.models import (
    FreshnessTier,
    PortfolioHoldingContext,
    PreLLMNewsCandidate,
    PreLLMYouTubeCandidate,
)


def evaluate_freshness(
    published_at_iso: Optional[str],
    now_utc: datetime,
    max_age_hours: int = 36,
) -> Tuple[int, FreshnessTier, str, bool]:
    """Evaluate article or video age with deterministic thresholds.

    Returns:
        (age_hours, freshness_tier, freshness_label, is_stale)
    """
    if not published_at_iso or not str(published_at_iso).strip():
        return (9999, "stale", "ไม่ระบุเวลาเผยแพร่", True)

    try:
        clean_str = str(published_at_iso).replace("Z", "+00:00")
        pub_dt = datetime.fromisoformat(clean_str)
        if pub_dt.tzinfo is None:
            pub_dt = pub_dt.replace(tzinfo=timezone.utc)
    except Exception:
        return (9999, "stale", "ไม่สามารถตรวจสอบเวลา", True)

    age_seconds = (now_utc - pub_dt).total_seconds()
    if age_seconds < 0:
        age_hours = 0
    else:
        age_hours = int(age_seconds / 3600)

    if age_hours <= 6:
        tier: FreshnessTier = "ultra_fresh"
        label = f"🔥 สดใหม่ ({age_hours} ชม. ที่แล้ว)" if age_hours > 0 else "🔥 สดใหม่ (เพิ่งเผยแพร่)"
        stale = False
    elif age_hours <= 24:
        tier = "fresh_today"
        label = f"📅 วันนี้ ({age_hours} ชม. ที่แล้ว)"
        stale = False
    elif age_hours <= max_age_hours:
        tier = "weekly_window"
        days_ago = max(1, age_hours // 24)
        label = f"⏳ {days_ago} วันที่แล้ว"
        stale = False
    else:
        tier = "stale"
        days_ago = max(1, age_hours // 24)
        label = f"⚠️ ข้อมูลเก่า ({days_ago} วันที่แล้ว)"
        stale = True

    return (age_hours, tier, label, stale)


def filter_fresh_news_candidates(
    raw_items: List[dict],
    now_utc: datetime,
    max_age_hours: int = 36,
) -> Tuple[List[dict], int]:
    """Filter candidate items strictly against freshness policy before triage."""
    fresh_items: List[dict] = []
    dropped_count = 0

    for item in raw_items:
        pub_raw = item.get("published_at") or item.get("published") or item.get("pubDate")
        age_h, tier, label, is_stale = evaluate_freshness(pub_raw, now_utc, max_age_hours=max_age_hours)
        if is_stale:
            dropped_count += 1
            continue

        item_copy = dict(item)
        item_copy["age_hours"] = age_h
        item_copy["freshness_tier"] = tier
        item_copy["freshness_label"] = label
        item_copy["is_stale"] = False
        fresh_items.append(item_copy)

    return fresh_items, dropped_count


from core.news_digest.entity_matcher import (
    COMMON_STOPWORDS_AND_ACRONYMS as _COMMON_ENGLISH_STOPWORDS,
    _BOUNDARY_TICKER_RE as _TICKER_BOUNDARY_PATTERN,
    _CASHTAG_RE as _CASHTAG_PATTERN,
    find_matching_tickers,
)


def match_tickers_and_portfolio(
    text: str,
    explicit_tickers: List[str],
    holdings_map: Dict[str, PortfolioHoldingContext],
) -> Tuple[List[str], List[PortfolioHoldingContext]]:
    """Deterministic ticker matcher against investor portfolio and watchlist."""
    target_symbols = set(holdings_map.keys())
    matched_set = find_matching_tickers(
        text=text,
        target_symbols=target_symbols,
        explicit_tickers=explicit_tickers,
    )

    matched_tickers = sorted(matched_set)
    matched_badges: List[PortfolioHoldingContext] = []
    for sym in matched_tickers:
        if sym in holdings_map:
            matched_badges.append(holdings_map[sym])

    return matched_tickers, matched_badges


def build_one_shot_daily_prompt(
    candidates: List[PreLLMNewsCandidate],
    youtube_docs: List[dict],
    portfolio_summary: str,
) -> str:
    """Compose concise prompt for One-Shot Daily Market Intelligence."""
    news_lines: List[str] = []
    for idx, c in enumerate(candidates, 1):
        clean_sum = (c.summary or "").strip().replace("\n", " ")[:400]
        badges_str = ", ".join(f"{b.symbol} ({b.impact_type}, weight={b.weight_pct}%)" for b in c.portfolio_badges)
        badge_text = f" | [PORTFOLIO MATCH: {badges_str}]" if badges_str else ""
        news_lines.append(
            f"{idx}. [ID: {c.candidate_id}] {c.title} ({c.freshness_label}) Source: {c.source}{badge_text}\n"
            f"   Summary: {clean_sum}"
        )

    yt_lines: List[str] = []
    for idx, y in enumerate(youtube_docs, 1):
        vid = y.get("video_id", f"yt_{idx}")
        title = y.get("title", "")
        channel = y.get("channel", "")
        snippet = (y.get("summary_snippet") or y.get("content") or "").strip().replace("\n", " ")[:800]
        yt_lines.append(
            f"{idx}. [VIDEO_ID: {vid}] {title} | Channel: {channel}\n"
            f"   Takeaways: {snippet}"
        )

    news_block = "\n".join(news_lines) if news_lines else "(ไม่มีข่าวสดใหม่ในรอบนี้)"
    yt_block = "\n".join(yt_lines) if yt_lines else "(ผู้ใช้ไม่ได้เลือกคลิป YouTube ในรอบนี้)"

    prompt = f"""คุณคือ Chief Investment Officer และ Head of Market Intelligence ประจำ Family Office
หน้าที่ของคุณคือวิเคราะห์ข่าวรายวันและคลิป YouTube ที่ผู้ใช้คัดเลือกมา โดยเชื่อมโยงเข้ากับพอร์ตการลงทุนของผู้ใช้อย่างกระชับ ชัดเจน และมีประโยชน์ในการตัดสินใจสูงสุด

=== 1. ข้อมูลพอร์ตและ WATCHLIST ของผู้ใช้ ===
{portfolio_summary}

=== 2. รายการข่าวประจำวัน (กรองความสดใหม่แล้ว) ===
{news_block}

=== 3. คลิป YOUTUBE ที่ผู้ใช้เลือกให้นำมาร่วมวิเคราะห์ ===
{yt_block}

=== กฎการวิเคราะห์ ===
1. market_mood และ transmission_chain ต้องสรุปภาพใหญ่ใน 1-2 ประโยค ชี้ให้เห็นสายพานเหตุและผล เช่น 'US CPI ต่ำคาด ➔ Bond Yield ย่อตัว ➔ หนุนหุ้นเทค'
2. items แต่ละข้อต้องระบุ source_candidate_id ให้ตรงกับ ID ของข่าวต้นทาง, เขียน thai_headline สละสลวย 1 บรรทัด, what_happened และ why_it_matters กระชับคมชัด
3. กำหนด thesis_signal ให้แม่นยำ:
   - 'thesis_breaker': ข่าวทำลายเหตุผลการลงทุน/ความเสี่ยงเชิงโครงสร้าง
   - 'buy_on_dip': ข่าวลบระยะสั้นหรือราคาลงมาใกล้จุดสะสมของหุ้น Watchlist
   - 'thesis_reinforcer': ข่าวยืนยันว่าธุรกิจเดินหน้าตามแผน แนะนำถือต่อ
   - 'noise': ข่าวเก็งกำไรระยะสั้นหรือความผันผวนรายวัน
4. ชี้เป้าทั้งผลกระทบทางตรง (direct_holding, fund_lookthrough, watchlist) และผลกระทบทางอ้อม (second_order เช่น ค่าเงิน ดอกเบี้ย หรือห่วงโซ่อุปทาน)
5. youtube_highlights สรุปแก่นคลิปและระบุว่าความเห็นในคลิปสอดคล้อง (confirmed) หรือเห็นต่าง (contrarian) จากข่าวจริงอย่างไร
"""
    return prompt.strip()


def build_one_shot_weekly_prompt(
    daily_digests: List[dict],
    recent_news: List[PreLLMNewsCandidate],
    youtube_docs: List[dict],
    calendar_events: List[dict],
    portfolio_summary: str,
) -> str:
    """Compose concise prompt for One-Shot Weekly Market & YouTube Intelligence."""
    daily_summaries = []
    for d in daily_digests:
        date_str = d.get("date", "Unknown")
        mood = d.get("market_mood", "")
        focus = d.get("top_portfolio_focus", "")
        daily_summaries.append(f"- วันที่ {date_str} [{mood}]: {focus}")

    news_lines = []
    for idx, c in enumerate(recent_news[:30], 1):
        clean_sum = (c.summary or "").strip().replace("\n", " ")[:300]
        badges_str = ", ".join(f"{b.symbol} ({b.impact_type})" for b in c.portfolio_badges)
        badge_text = f" | [PORTFOLIO MATCH: {badges_str}]" if badges_str else ""
        news_lines.append(
            f"{idx}. [ID: {c.candidate_id}] {c.title} ({c.freshness_label}) Source: {c.source}{badge_text}\n"
            f"   Summary: {clean_sum}"
        )

    yt_lines = []
    for idx, y in enumerate(youtube_docs, 1):
        vid = y.get("video_id", f"yt_{idx}")
        title = y.get("title", "")
        channel = y.get("channel", "")
        snippet = (y.get("summary_snippet") or y.get("content") or "").strip().replace("\n", " ")[:800]
        yt_lines.append(f"{idx}. [{channel}] {title} (ID: {vid}): {snippet}")

    cal_lines = []
    for ev in calendar_events[:15]:
        sym = ev.get("symbol", "")
        ev_title = ev.get("title") or ev.get("event_type") or "Corporate Action"
        ev_date = ev.get("date") or ev.get("date_str") or ""
        cal_lines.append(f"- {ev_date}: {sym} - {ev_title}")

    return f"""คุณคือ Chief Investment Officer ประจำ Family Office
หน้าที่ของคุณคือจัดทำบทสรุปยุทธศาสตร์ประจำสัปดาห์ (Weekly Intelligence Digest) จากข่าวรอบ 7 วันและบทวิเคราะห์จากคลิป YouTube

=== ข้อมูลพอร์ตโฟลิโอของผู้ใช้ ===
{portfolio_summary}

=== บทสรุปข่าวรายวันตลอดสัปดาห์ ===
{chr(10).join(daily_summaries) if daily_summaries else "(ไม่มีบันทึกรายวัน)"}

=== ข่าวและเหตุการณ์สำคัญรอบสัปดาห์ (7 วัน) ===
{chr(10).join(news_lines) if news_lines else "(ไม่มีข่าวเพิ่มเติมในรอบนี้)"}

=== คลิป YOUTUBE ที่คัดเลือกในรอบสัปดาห์ ===
{chr(10).join(yt_lines) if yt_lines else "(ไม่มีคลิป YouTube ในรอบนี้)"}

=== ปฏิทินและตัวเลขสำคัญสัปดาห์หน้า ===
{chr(10).join(cal_lines) if cal_lines else "(ไม่มีปฏิทินเร่งด่วน)"}

=== คำสั่งวิเคราะห์ ===
1. สกัด 3-5 ธีมหลัก (executive_themes) ที่ขับเคลื่อนตลาดในรอบสัปดาห์
2. สร้าง portfolio_matrix สำหรับหุ้นในพอร์ต/Watchlist ที่มีความเคลื่อนไหว ระบุสถานะ thesis_status
3. สังเคราะห์เปรียบเทียบ consensus_narratives (เรื่องที่ข่าวจริงและกูรู YouTube เห็นตรงกัน) และ contrarian_and_pitches (หุ้นเด่นหรือมุมมองสวนตลาด)
4. สรุป week_ahead_checklist สำหรับการเตรียมตัวลงทุนสัปดาห์หน้า
""".strip()
