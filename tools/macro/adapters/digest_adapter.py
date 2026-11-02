"""Concrete adapters implementing Digest application ports."""
from datetime import datetime, timezone
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple

from application.macro.digest_ports import (
    DigestCalendarPort,
    DigestCandidateFetcherPort,
    DigestLLMPort,
    DigestPortfolioReaderPort,
    DigestStoragePort,
    DigestTranscriptPort,
)
from core.logger import get_logger
from core.model_registry import REGISTRY
from core.news_digest.engine import (
    build_one_shot_daily_prompt,
    build_one_shot_weekly_prompt,
    match_tickers_and_portfolio,
)
from core.news_digest.models import (
    OneShotDailyLLMOutput,
    OneShotDigestItemLLM,
    OneShotWeeklyLLMOutput,
    OneShotYouTubeHighlightLLM,
    PortfolioHoldingContext,
    PreLLMNewsCandidate,
    PreLLMYouTubeCandidate,
    WeeklyPortfolioMatrixRowLLM,
)
from tools.knowledge.search_youtube_insights import extract_first_bullet_of_key_takeaways
from tools.knowledge.youtube_monitor import get_youtube_candidates
from tools.macro.digest_store import (
    load_digest_store,
    mark_digest_items_read,
    save_daily_digest,
    save_pre_llm_candidates,
    save_weekly_digest,
)
from tools.macro.news_radar import fetch_fresh_news_candidates

logger = get_logger(__name__)


def _is_mock_mode() -> bool:
    return os.getenv("MOCK_NEWS_FUNNEL_LLM", "false").lower() == "true" or not bool(
        os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    )


class DigestPortfolioReaderAdapter(DigestPortfolioReaderPort):
    """Reads actual portfolio, mutual fund look-through constituents, and watchlist."""

    def __init__(
        self,
        portfolio_service: Any = None,
        vault_root: Optional[str] = None,
        look_through_service: Any = None,
    ) -> None:
        self._portfolio_service = portfolio_service
        self._vault_root = Path(vault_root or os.getenv("OBSIDIAN_VAULT_PATH", "./memories")).resolve()
        self._look_through_service = look_through_service

    def get_portfolio_context(
        self, portfolio_id: str = "default"
    ) -> Tuple[List[PortfolioHoldingContext], Dict[str, PortfolioHoldingContext], str]:
        holdings_list: List[PortfolioHoldingContext] = []
        holdings_map: Dict[str, PortfolioHoldingContext] = {}

        # 1. Read holdings from portfolio service if available
        total_val = 0.0
        try:
            if self._portfolio_service:
                if hasattr(self._portfolio_service, "get_structured_portfolio_state"):
                    state = self._portfolio_service.get_structured_portfolio_state(portfolio_id)
                elif hasattr(self._portfolio_service, "get_state"):
                    state = self._portfolio_service.get_state(portfolio_id)
                else:
                    state = None
                summary = getattr(state, "summary", None)
                total_val = float(getattr(summary, "total_value_thb", 0.0)) if summary else 0.0

                for h in getattr(state, "holdings", []):
                    sym = getattr(h, "symbol", "").strip().upper().replace(".BK", "")
                    if not sym or sym in {"CASH", "CASH_THB", "CASH_USD"}:
                        continue
                    market_val = float(getattr(h, "market_value_thb", 0.0))
                    wt = round(market_val / total_val * 100.0, 2) if total_val > 0 else 0.0
                    pnl = float(getattr(h, "unrealized_pnl_percent", 0.0) or 0.0)
                    ctx = PortfolioHoldingContext(
                        symbol=sym,
                        company_name=getattr(h, "company_name", None),
                        impact_type="direct_holding",
                        weight_pct=wt,
                        unrealized_pnl_pct=pnl,
                        sector=getattr(h, "sector", None),
                    )
                    holdings_list.append(ctx)
                    holdings_map[sym] = ctx
        except Exception as e:
            logger.warning("Could not load portfolio holdings: %s", e)

        # 2. Read watchlist
        try:
            if self._portfolio_service:
                if hasattr(self._portfolio_service, "get_structured_watchlist"):
                    wl = self._portfolio_service.get_structured_watchlist(portfolio_id)
                elif hasattr(self._portfolio_service, "get_watchlist"):
                    wl = self._portfolio_service.get_watchlist(portfolio_id)
                else:
                    wl = None
                for item in getattr(wl, "items", []):
                    sym = getattr(item, "symbol", "").strip().upper().replace(".BK", "")
                    if sym and sym not in holdings_map:
                        ctx = PortfolioHoldingContext(
                            symbol=sym,
                            impact_type="watchlist",
                            target_entry_price=float(getattr(item, "target_price", 0.0) or 0.0),
                            sector=getattr(item, "sector", None),
                        )
                        holdings_list.append(ctx)
                        holdings_map[sym] = ctx
        except Exception as e:
            logger.warning("Could not load watchlist: %s", e)

        # 3. Read Mutual Fund / PVD Look-Through constituents if available
        lookthrough_items: List[PortfolioHoldingContext] = []
        try:
            if self._look_through_service and hasattr(self._look_through_service, "get_portfolio_look_through"):
                report = self._look_through_service.get_portfolio_look_through(portfolio_id)
                for h in getattr(report, "holdings_exposure", []):
                    sym = str(getattr(h, "ticker", "")).strip().upper().replace(".BK", "")
                    if not sym or sym in {"CASH", "CASH_THB", "CASH_USD"}:
                        continue
                    indirect_wt = float(getattr(h, "indirect_weight_pct", 0.0) or 0.0)
                    if indirect_wt <= 0:
                        continue
                    source_funds = [str(s[0]) for s in getattr(h, "source_funds", []) if s and len(s) > 0]

                    if sym in holdings_map:
                        existing = holdings_map[sym]
                        merged_funds = set(existing.source_fund_codes or [])
                        merged_funds.update(source_funds)
                        existing.source_fund_codes = sorted(list(merged_funds))
                        if existing.impact_type == "direct_holding":
                            existing.weight_pct = round((existing.weight_pct or 0.0) + indirect_wt, 2)
                        elif existing.impact_type == "watchlist":
                            existing.impact_type = "fund_lookthrough"
                            existing.weight_pct = round(indirect_wt, 2)
                    else:
                        ctx = PortfolioHoldingContext(
                            symbol=sym,
                            company_name=getattr(h, "company_name", None),
                            impact_type="fund_lookthrough",
                            weight_pct=round(indirect_wt, 2),
                            source_fund_codes=source_funds,
                            sector=getattr(h, "market", None),
                        )
                        holdings_list.append(ctx)
                        holdings_map[sym] = ctx
                        lookthrough_items.append(ctx)
        except Exception as e:
            logger.warning("Could not load fund look-through exposures: %s", e)

        # 4. Format concise text summary for LLM prompt
        lines = []
        direct_items = [h for h in holdings_list if h.impact_type == "direct_holding"]
        wl_items = [h for h in holdings_list if h.impact_type == "watchlist"]

        lines.append(f"• Direct Portfolio Holdings ({len(direct_items)} items):")
        for h in direct_items[:12]:
            pnl_sign = "+" if (h.unrealized_pnl_pct or 0) >= 0 else ""
            funds_str = f" (+Funds: {','.join(h.source_fund_codes[:2])})" if h.source_fund_codes else ""
            lines.append(f"  - {h.symbol}: weight={h.weight_pct}%, P&L={pnl_sign}{h.unrealized_pnl_pct}%{funds_str}")

        if lookthrough_items:
            lines.append(f"• Indirect Holdings via Mutual Funds / PVD ({len(lookthrough_items)} items):")
            for lt in lookthrough_items[:12]:
                funds_str = f" via {','.join(lt.source_fund_codes[:2])}" if lt.source_fund_codes else ""
                lines.append(f"  - {lt.symbol}: weight={lt.weight_pct}%{funds_str}")

        if wl_items:
            lines.append(f"• Watchlist ({len(wl_items)} items):")
            for w in wl_items[:8]:
                lines.append(f"  - {w.symbol}: target={w.target_entry_price}")

        summary_str = "\n".join(lines) if lines else "(ไม่มีข้อมูลพอร์ตโฟลิโอ)"
        return holdings_list, holdings_map, summary_str


class DigestCandidateFetcherAdapter(DigestCandidateFetcherPort):
    """Fetches candidates and filters by freshness and portfolio matches."""

    def __init__(self, portfolio_reader: Optional[DigestPortfolioReaderPort] = None) -> None:
        self._portfolio_reader = portfolio_reader or DigestPortfolioReaderAdapter()

    def fetch_candidates(
        self,
        portfolio_tickers: List[str],
        lookback_days: int = 7,
        max_news_age_hours: int = 36,
    ) -> Tuple[List[PreLLMNewsCandidate], List[PreLLMYouTubeCandidate], int]:
        _, holdings_map, _ = self._portfolio_reader.get_portfolio_context()

        # 1. Fetch news with dynamic ticker feeds and freshness drop
        raw_news, stale_dropped_count = fetch_fresh_news_candidates(
            portfolio_tickers=portfolio_tickers,
            max_age_hours=max_news_age_hours,
            max_items=30,
        )

        news_candidates: List[PreLLMNewsCandidate] = []
        for idx, it in enumerate(raw_news, 1):
            title = it.get("title", "")
            summary = it.get("summary", "")
            full_text = f"{title} {summary}"

            matched_tickers, badges = match_tickers_and_portfolio(
                text=full_text,
                explicit_tickers=[],
                holdings_map=holdings_map,
            )

            cid = f"news_{idx}_{abs(hash(it.get('link', '') + title)) % 1000000}"
            news_candidates.append(PreLLMNewsCandidate(
                candidate_id=cid,
                title=title,
                summary=summary,
                source=it.get("source", "RSS"),
                links=[it["link"]] if it.get("link") else [],
                sources_count=it.get("sources_count", 1),
                published_at=it.get("published_at"),
                age_hours=it.get("age_hours", 0),
                freshness_tier=it.get("freshness_tier", "fresh_today"),
                freshness_label=it.get("freshness_label", ""),
                is_stale=it.get("is_stale", False),
                matched_tickers=matched_tickers,
                portfolio_badges=badges,
            ))

        # 2. Fetch YouTube candidates
        raw_yt = get_youtube_candidates(lookback_days=lookback_days)
        yt_candidates: List[PreLLMYouTubeCandidate] = []
        for y in raw_yt:
            yt_candidates.append(PreLLMYouTubeCandidate(
                video_id=y.get("video_id") or "",
                title=y.get("title") or "",
                channel=y.get("channel") or "",
                link=y.get("link") or "",
                published_at=y.get("published_at"),
                age_hours=y.get("age_hours", 0),
                freshness_tier=y.get("freshness_tier", "fresh_today"),
                freshness_label=y.get("freshness_label", ""),
                thumbnail=y.get("thumbnail") or "",
                is_fetched=y.get("is_fetched", False),
                cached_summary_snippet=y.get("cached_summary_snippet"),
                cached_tickers=y.get("cached_tickers", []),
            ))

        return news_candidates, yt_candidates, stale_dropped_count


class DigestTranscriptAdapter(DigestTranscriptPort):
    """Loads existing YouTube insights or fetches subtitles on demand."""

    def __init__(self, vault_root: Optional[str] = None) -> None:
        self._vault_root = Path(vault_root or os.getenv("OBSIDIAN_VAULT_PATH", "./memories")).resolve()

    def resolve_youtube_document(self, video_id: str, custom_url: Optional[str] = None) -> Dict[str, Any]:
        from tools.knowledge.youtube import _extract_video_id, _get_raw_transcript, _fetch_youtube_metadata
        clean_id = _extract_video_id(custom_url or video_id) or video_id.strip()

        summaries_dir = self._vault_root / "30_Knowledge_Base" / "YouTube_Summaries"
        if summaries_dir.exists():
            for p in summaries_dir.rglob(f"*{clean_id}*.md"):
                if "Inbox" not in p.parts and "Revisions" not in p.parts:
                    try:
                        content = p.read_text(encoding="utf-8")
                        bullet = extract_first_bullet_of_key_takeaways(content, max_chars=300)
                        return {
                            "video_id": clean_id,
                            "title": p.stem,
                            "channel": "YouTube Insight",
                            "summary_snippet": bullet or content[:400],
                            "content": content[:800],
                        }
                    except Exception as e:
                        logger.warning("Could not read existing YouTube note %s: %s", p, e)

        # Fallback to fetching transcript snippet on-demand
        try:
            meta = _fetch_youtube_metadata(clean_id)
            try:
                raw_tx = _get_raw_transcript(clean_id)
            except Exception:
                raw_tx = ""
            return {
                "video_id": clean_id,
                "title": meta.original_title or f"YouTube Video {clean_id}",
                "channel": meta.channel or "YouTube Channel",
                "summary_snippet": raw_tx[:600] if raw_tx else "ไม่มีข้อความ Transcript",
                "content": raw_tx[:1000] if raw_tx else "",
            }
        except Exception as exc:
            logger.warning("Could not fetch YouTube transcript on-demand for %s: %s", clean_id, exc)
            return {
                "video_id": clean_id,
                "title": f"YouTube Video {clean_id}",
                "channel": "YouTube Channel",
                "summary_snippet": custom_url or clean_id,
                "content": "",
            }


class DigestLLMAdapter(DigestLLMPort):
    """Executes One-Shot Structured LLM synthesis with fast Gemini model."""

    def invoke_daily(
        self,
        candidates: List[PreLLMNewsCandidate],
        youtube_docs: List[Dict[str, Any]],
        portfolio_summary: str,
    ) -> OneShotDailyLLMOutput:
        if _is_mock_mode() or not candidates:
            # Deterministic mock fallback
            items = []
            for c in candidates[:5]:
                items.append(OneShotDigestItemLLM(
                    source_candidate_id=c.candidate_id,
                    thai_headline=f"[สรุป] {c.title}",
                    what_happened=f"ประเด็นสำคัญของ {c.title}",
                    why_it_matters="ส่งผลต่อบรรยากาศการลงทุนและภาพรวมกลุ่มอุตสาหกรรม",
                    thesis_signal="thesis_reinforcer" if c.portfolio_badges else "noise",
                    impact_type=c.portfolio_badges[0].impact_type if c.portfolio_badges else "macro_general",
                    affected_tickers=[b.symbol for b in c.portfolio_badges],
                    affected_sectors=[],
                    macro_impact_score=6,
                    asset_impact_score=6,
                ))
            yt_high = []
            for y in youtube_docs[:3]:
                yt_high.append(OneShotYouTubeHighlightLLM(
                    video_id=y.get("video_id", "yt_mock"),
                    thai_title=y.get("title", "YouTube Insight"),
                    channel=y.get("channel", "YouTube"),
                    key_takeaways=["วิเคราะห์ภาพรวมเศรษฐกิจ", "แนะนำกลยุทธ์การลงทุน", "คัดกรองหุ้นเด่น"],
                    alignment_with_news="confirmed",
                    alignment_note="สอดคล้องกับทิศทางข่าวประจำวัน",
                ))
            return OneShotDailyLLMOutput(
                market_mood="risk_on" if len(candidates) > 2 else "mixed",
                market_mood_summary="ตลาดได้รับแรงหนุนจากกระแสเงินทุนและความชัดเจนนโยบายเศรษฐกิจโลก",
                transmission_chain="ตัวเลขเศรษฐกิจสหรัฐฯ ทรงตัว ➔ Bond Yield ย่อตัว ➔ หนุนสินทรัพย์เสี่ยงและหุ้นเติบโต",
                top_portfolio_focus="ติดตามความต่อเนื่องของเม็ดเงินลงทุนในหุ้นเทคโนโลยีและผลประกอบการไตรมาสล่าสุด",
                items=items,
                youtube_highlights=yt_high,
            )

        prompt = build_one_shot_daily_prompt(candidates, youtube_docs, portfolio_summary)
        from core.llm_factory import invoke_structured_llm
        slot = REGISTRY["news_triage"]

        return invoke_structured_llm(
            schema=OneShotDailyLLMOutput,
            model_env=slot.env_var,
            prompt_lines=prompt.split("\n"),
            purpose="one_shot_daily_digest",
            max_output_tokens=16384,
            default_model=slot.default,
            provider="google",
        )

    def invoke_weekly(
        self,
        daily_digests: List[Dict[str, Any]],
        recent_news: List[PreLLMNewsCandidate],
        youtube_docs: List[Dict[str, Any]],
        calendar_events: List[Dict[str, Any]],
        portfolio_summary: str,
    ) -> OneShotWeeklyLLMOutput:
        if _is_mock_mode():
            yt_high = []
            for y in youtube_docs[:3]:
                yt_high.append(OneShotYouTubeHighlightLLM(
                    video_id=y.get("video_id", "yt_weekly"),
                    thai_title=y.get("title", "YouTube Weekly Insight"),
                    channel=y.get("channel", "YouTube Channel"),
                    key_takeaways=["มุมมองเชิงบวกต่อกลุ่มเติบโต", "กระจายความเสี่ยงสู่กลุ่มปันผล"],
                    alignment_with_news="confirmed",
                    alignment_note="เห็นตรงกับทิศทางตลาดตลอดสัปดาห์",
                ))
            items = []
            for idx, c in enumerate(recent_news[:5], 1):
                items.append(OneShotDigestItemLLM(
                    source_candidate_id=c.candidate_id,
                    thai_headline=f"ประเด็นสัปดาห์: {c.title[:60]}",
                    what_happened=c.summary[:120] if c.summary else "สรุปเหตุการณ์สำคัญรอบสัปดาห์",
                    why_it_matters="ส่งผลต่อการประเมินทิศทางอุตสาหกรรมและพอร์ตโฟลิโอในระยะกลาง",
                    thesis_signal="thesis_reinforcer",
                    impact_type="direct_holding" if c.portfolio_badges else "macro_general",
                    affected_tickers=c.matched_tickers,
                    affected_sectors=[],
                    macro_impact_score=6,
                    asset_impact_score=6,
                    time_horizon="short_term",
                ))
            return OneShotWeeklyLLMOutput(
                executive_themes=[
                    "การชะลอตัวของเงินเฟ้อโลกและความหวังการลดดอกเบี้ย",
                    "ดีมานด์โครงสร้างพื้นฐาน AI ยังเป็นแรงขับเคลื่อนหลัก",
                    "กระแสฟันด์โฟลว์สลับกลุ่มระหว่างตลาดพัฒนาแล้วและตลาดเกิดใหม่",
                ],
                portfolio_health_summary="พอร์ตโฟลิโอโดยรวมยังรักษาสถานะแข็งแกร่ง ไม่มีหุ้นที่ทำลาย Thesis การลงทุนหลัก",
                portfolio_matrix=[
                    WeeklyPortfolioMatrixRowLLM(
                        ticker="NVDA",
                        thesis_status="strong_hold",
                        thesis_status_label="ถือต่อตามแผน",
                        weekly_summary="ยอดสั่งซื้อชิปและงบลงทุน Data Center สนับสนุนการเติบโตต่อเนื่อง",
                    )
                ],
                consensus_narratives=[
                    "กลุ่ม AI Infrastructure ยังมีคำสั่งซื้อรองรับถึงปีหน้า",
                    "อัตราผลตอบแทนพันธบัตรผ่านจุดสูงสุดของรอบแล้ว",
                ],
                contrarian_and_pitches=[
                    "กูรูเริ่มแนะนำสลับทำกำไรหุ้นเทคบางส่วนเข้าหุ้นปันผลรับฤดูกาลปิดงบ",
                ],
                week_ahead_checklist=[
                    "ติดตามตัวเลข GDP สหรัฐฯ ไตรมาสล่าสุด",
                    "ตรวจสอบวันขึ้นเครื่องหมาย XD ของหุ้นปันผลในพอร์ต",
                ],
                items=items,
                youtube_highlights=yt_high,
            )

        prompt = build_one_shot_weekly_prompt(daily_digests, recent_news, youtube_docs, calendar_events, portfolio_summary)
        from core.llm_factory import invoke_structured_llm
        slot = REGISTRY["news_triage"]

        return invoke_structured_llm(
            schema=OneShotWeeklyLLMOutput,
            model_env=slot.env_var,
            prompt_lines=prompt.split("\n"),
            purpose="one_shot_weekly_digest",
            max_output_tokens=16384,
            default_model=slot.default,
            provider="google",
        )


class DigestStorageAdapter(DigestStoragePort):
    """Adapts calls to tools/macro/digest_store."""

    def __init__(self, store_path: Optional[str] = None, vault_root: Optional[str] = None) -> None:
        if store_path:
            self._store_path = store_path
        elif vault_root:
            self._store_path = str(Path(vault_root) / "digest_store.json")
        else:
            self._store_path = None

    def load_store(self) -> Dict[str, Any]:
        return load_digest_store(self._store_path)

    def save_candidates(
        self,
        news_candidates: List[Dict[str, Any]],
        youtube_candidates: List[Dict[str, Any]],
        stale_dropped_count: int,
        cache_bucket: str = "daily",
    ) -> None:
        save_pre_llm_candidates(
            news_candidates,
            youtube_candidates,
            stale_dropped_count,
            self._store_path,
            cache_bucket=cache_bucket,
        )

    def save_daily(self, daily_payload: Dict[str, Any]) -> None:
        save_daily_digest(daily_payload, self._store_path)

    def save_weekly(self, weekly_payload: Dict[str, Any]) -> None:
        save_weekly_digest(weekly_payload, self._store_path)

    def mark_read(self, item_ids: List[str]) -> List[str]:
        return mark_digest_items_read(item_ids, self._store_path)


class DigestCalendarAdapter(DigestCalendarPort):
    """Adapts PortfolioCalendarApplicationService to provide upcoming events."""

    def __init__(self, calendar_service: Any = None) -> None:
        self._calendar_service = calendar_service

    def get_upcoming_events(self, portfolio_id: str = "default", days_ahead: int = 14) -> List[Dict[str, Any]]:
        if not self._calendar_service or not hasattr(self._calendar_service, "get_calendar"):
            return []
        try:
            res = self._calendar_service.get_calendar(portfolio_id=portfolio_id)
            raw_events = res.get("events", []) if isinstance(res, dict) else []
            upcoming = []
            for ev in raw_events:
                days_until = ev.get("days_until")
                if days_until is not None and 0 <= days_until <= days_ahead:
                    sym = ev.get("ticker", "")
                    ev_type = ev.get("event_type", "event")
                    ev_date = ev.get("event_date", "")
                    upcoming.append({
                        "symbol": sym,
                        "ticker": sym,
                        "event_type": ev_type,
                        "date": ev_date,
                        "event_date": ev_date,
                        "days_until": days_until,
                        "title": f"{sym} ({ev_type}) ในอีก {days_until} วัน ({ev_date})",
                    })
            return upcoming
        except Exception as e:
            logger.warning("Could not load upcoming calendar events: %s", e)
            return []
