"""Application use case service for News and YouTube Digest."""
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
import uuid

from application.macro.digest_ports import (
    DigestCalendarPort,
    DigestCandidateFetcherPort,
    DigestLLMPort,
    DigestPortfolioReaderPort,
    DigestStoragePort,
    DigestTranscriptPort,
)
from core.news_digest.engine import evaluate_freshness
from core.news_digest.models import PreLLMNewsCandidate


class DigestApplicationService:
    """Coordinates News & YouTube Digest workflows under Hexagonal Architecture."""

    def __init__(
        self,
        portfolio_reader: DigestPortfolioReaderPort,
        candidate_fetcher: DigestCandidateFetcherPort,
        transcript_port: DigestTranscriptPort,
        llm_port: DigestLLMPort,
        storage_port: DigestStoragePort,
        calendar_port: Optional[DigestCalendarPort] = None,
    ) -> None:
        self._portfolio_reader = portfolio_reader
        self._candidate_fetcher = candidate_fetcher
        self._transcript_port = transcript_port
        self._llm_port = llm_port
        self._storage = storage_port
        self._calendar_port = calendar_port

    def get_candidates(
        self,
        days: int = 7,
        force_refresh: bool = False,
        max_news_age_hours: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Retrieve pre-LLM candidates for interactive selection (cost: 0 baht)."""
        actual_max_age = max_news_age_hours if max_news_age_hours is not None else (36 if days <= 2 else days * 24)
        cache_bucket = "daily" if actual_max_age <= 36 else "weekly"
        store_state = self._storage.load_store()

        last_fetched = store_state.get(f"last_{cache_bucket}_candidates_fetched_at")
        cached_news = store_state.get(f"cached_{cache_bucket}_news_candidates")
        cached_yt = store_state.get(f"cached_{cache_bucket}_youtube_candidates")
        stale_dropped = store_state.get(f"stale_dropped_count_{cache_bucket}", 0)

        # Fallback to legacy single cache if daily bucket not specifically set yet
        if not cached_news and cache_bucket == "daily":
            last_fetched = store_state.get("last_candidates_fetched_at")
            cached_news = store_state.get("cached_news_candidates")
            cached_yt = store_state.get("cached_youtube_candidates")
            stale_dropped = store_state.get("stale_dropped_count", 0)

        # Reuse cache if within 15 minutes unless force_refresh is requested
        if not force_refresh and last_fetched and cached_news:
            try:
                lf_dt = datetime.fromisoformat(str(last_fetched).replace("Z", "+00:00"))
                if (datetime.now(timezone.utc) - lf_dt).total_seconds() < 900:
                    return {
                        "last_fetched_at": last_fetched,
                        "news_candidates": cached_news,
                        "youtube_candidates": cached_yt or [],
                        "stale_dropped_count": stale_dropped,
                    }
            except Exception:
                pass

        _, holdings_map, _ = self._portfolio_reader.get_portfolio_context()
        portfolio_tickers = list(holdings_map.keys())

        news_cands, yt_cands, dropped_count = self._candidate_fetcher.fetch_candidates(
            portfolio_tickers=portfolio_tickers,
            lookback_days=days,
            max_news_age_hours=actual_max_age,
        )

        news_dicts = [c.model_dump() for c in news_cands]
        yt_dicts = [y.model_dump() for y in yt_cands]

        self._storage.save_candidates(news_dicts, yt_dicts, dropped_count, cache_bucket=cache_bucket)

        return {
            "last_fetched_at": datetime.now(timezone.utc).isoformat(),
            "news_candidates": news_dicts,
            "youtube_candidates": yt_dicts,
            "stale_dropped_count": dropped_count,
        }

    def get_daily_digest(self, target_date: Optional[str] = None) -> Dict[str, Any]:
        """Load daily digest with calculated staleness and read indicators."""
        store_state = self._storage.load_store()
        daily_history = store_state.get("daily_history", {})
        available_dates_set = set(daily_history.keys())
        latest_daily = store_state.get("latest_daily")
        if latest_daily and latest_daily.get("date"):
            available_dates_set.add(str(latest_daily["date"]))
        available_dates = sorted(available_dates_set, reverse=True)

        daily = None
        if target_date:
            daily = daily_history.get(target_date)
        if not daily:
            daily = latest_daily

        read_ids = set(store_state.get("read_item_ids", []))
        now_utc = datetime.now(timezone.utc)

        if not daily:
            return {
                "has_digest": False,
                "digest": None,
                "snapshot_age_hours": 0,
                "is_snapshot_stale": False,
                "fresh_pending_count": len(store_state.get("cached_news_candidates", [])),
                "available_dates": available_dates,
            }

        # Calculate snapshot age
        gen_at = daily.get("generated_at")
        snapshot_age_hours = 0
        is_stale = False
        if gen_at:
            try:
                gen_dt = datetime.fromisoformat(str(gen_at).replace("Z", "+00:00"))
                if gen_dt.tzinfo is None:
                    gen_dt = gen_dt.replace(tzinfo=timezone.utc)
                snapshot_age_hours = max(0, int((now_utc - gen_dt).total_seconds() / 3600))
                is_stale = snapshot_age_hours >= 8
            except Exception:
                snapshot_age_hours = 9999
                is_stale = True

        # Enrich items with read and new status
        enriched_items = []
        for it in daily.get("items", []):
            item_copy = dict(it)
            item_id = it.get("source_candidate_id") or it.get("link")
            item_copy["is_read"] = str(item_id) in read_ids
            # Items created within 6 hours of last visit or generated_at
            item_copy["is_new"] = item_copy.get("is_read") is False and snapshot_age_hours <= 6
            enriched_items.append(item_copy)

        enriched_daily = dict(daily)
        enriched_daily["items"] = enriched_items

        return {
            "has_digest": True,
            "digest": enriched_daily,
            "snapshot_age_hours": snapshot_age_hours,
            "is_snapshot_stale": is_stale,
            "fresh_pending_count": len(store_state.get("cached_news_candidates", [])),
            "available_dates": available_dates,
        }

    def synthesize_daily(
        self,
        selected_video_ids: Optional[List[str]] = None,
        excluded_news_ids: Optional[List[str]] = None,
        custom_youtube_urls: Optional[List[str]] = None,
        custom_news_items: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Execute One-Shot Daily Synthesis with user-curated inputs."""
        selected_video_ids = selected_video_ids or []
        excluded_news_ids = set(excluded_news_ids or [])
        custom_youtube_urls = custom_youtube_urls or []
        custom_news_items = custom_news_items or []

        store_state = self._storage.load_store()
        cached_news = store_state.get("cached_news_candidates", [])

        # Filter candidates based on exclusions
        active_candidates: List[PreLLMNewsCandidate] = []
        for item in cached_news:
            cand_id = item.get("candidate_id") or item.get("link")
            if cand_id not in excluded_news_ids:
                try:
                    active_candidates.append(PreLLMNewsCandidate.model_validate(item))
                except Exception:
                    pass

        # Append custom news items if any
        for idx, custom_text in enumerate(custom_news_items, 1):
            if custom_text.strip():
                active_candidates.append(PreLLMNewsCandidate(
                    candidate_id=f"custom_{idx}_{uuid.uuid4().hex[:6]}",
                    title=custom_text.strip()[:120],
                    summary=custom_text.strip(),
                    source="User Added",
                    age_hours=0,
                    freshness_tier="ultra_fresh",
                    freshness_label="🔥 สดใหม่ (ผู้ใช้ระบุ)",
                ))

        # Resolve selected YouTube documents
        yt_docs: List[Dict[str, Any]] = []
        for vid in selected_video_ids:
            doc = self._transcript_port.resolve_youtube_document(vid)
            if doc:
                yt_docs.append(doc)

        for curl in custom_youtube_urls:
            if curl.strip():
                doc = self._transcript_port.resolve_youtube_document(curl.strip(), custom_url=curl.strip())
                if doc:
                    yt_docs.append(doc)

        holding_contexts, holdings_map, portfolio_summary = self._portfolio_reader.get_portfolio_context()

        # Execute One-Shot LLM Call
        llm_output = self._llm_port.invoke_daily(
            candidates=active_candidates,
            youtube_docs=yt_docs,
            portfolio_summary=portfolio_summary,
        )

        now_time = datetime.now()
        date_str = now_time.strftime("%Y-%m-%d")
        period = "morning" if now_time.hour < 12 else "evening"
        now_iso = datetime.now(timezone.utc).isoformat()

        # Map candidate details back onto synthesized items
        candidate_map = {c.candidate_id: c for c in active_candidates}
        items_payload = []
        for item in llm_output.items:
            item_dict = item.model_dump()
            c_orig = candidate_map.get(item.source_candidate_id)
            if c_orig:
                item_dict["source"] = c_orig.source
                item_dict["link"] = c_orig.links[0] if c_orig.links else ""
                item_dict["links"] = c_orig.links
                item_dict["published_at"] = c_orig.published_at
                item_dict["age_hours"] = c_orig.age_hours
                item_dict["freshness_tier"] = c_orig.freshness_tier
                item_dict["freshness_label"] = c_orig.freshness_label
                item_dict["portfolio_badges"] = [b.model_dump() for b in c_orig.portfolio_badges]
            else:
                item_dict["source"] = "Daily Digest"
                item_dict["link"] = ""
                item_dict["links"] = []
                item_dict["published_at"] = now_iso
                item_dict["age_hours"] = 0
                item_dict["freshness_tier"] = "ultra_fresh"
                item_dict["freshness_label"] = "🔥 สดใหม่"
                item_dict["portfolio_badges"] = []
            items_payload.append(item_dict)

        yt_payload = [y.model_dump() for y in llm_output.youtube_highlights]

        daily_record = {
            "digest_id": f"daily-{date_str}-{period}",
            "date": date_str,
            "period": period,
            "generated_at": now_iso,
            "market_mood": llm_output.market_mood,
            "market_mood_summary": llm_output.market_mood_summary,
            "transmission_chain": llm_output.transmission_chain,
            "top_portfolio_focus": llm_output.top_portfolio_focus,
            "selected_video_ids": selected_video_ids,
            "items": items_payload,
            "youtube_highlights": yt_payload,
        }

        self._storage.save_daily(daily_record)
        return self.get_daily_digest(target_date=date_str)

    def get_weekly_digest(self, target_date: Optional[str] = None) -> Dict[str, Any]:
        """Load weekly digest."""
        store_state = self._storage.load_store()
        weekly_history = store_state.get("weekly_history", {})
        available_dates_set = set(weekly_history.keys())
        latest_weekly = store_state.get("latest_weekly")
        if latest_weekly:
            w_date = latest_weekly.get("to_date") or latest_weekly.get("digest_id")
            if w_date:
                available_dates_set.add(str(w_date))
        available_dates = sorted(available_dates_set, reverse=True)

        weekly = None
        if target_date:
            weekly = weekly_history.get(target_date)
        if not weekly:
            weekly = latest_weekly

        if not weekly:
            return {
                "has_digest": False,
                "digest": None,
                "available_dates": available_dates,
            }

        read_ids = set(store_state.get("read_item_ids", []))
        enriched_items = []
        for it in weekly.get("items", []):
            item_copy = dict(it)
            item_id = it.get("source_candidate_id") or it.get("link")
            item_copy["is_read"] = str(item_id) in read_ids
            item_copy["is_new"] = item_copy.get("is_read") is False
            enriched_items.append(item_copy)

        enriched_weekly = dict(weekly)
        enriched_weekly["items"] = enriched_items

        return {
            "has_digest": True,
            "digest": enriched_weekly,
            "available_dates": available_dates,
        }

    def synthesize_weekly(
        self,
        days: int = 7,
        selected_video_ids: Optional[List[str]] = None,
        excluded_news_ids: Optional[List[str]] = None,
        custom_youtube_urls: Optional[List[str]] = None,
        custom_news_items: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Execute One-Shot Weekly Synthesis with 7-day news and YouTube insights."""
        selected_video_ids = selected_video_ids or []
        excluded_news_ids = excluded_news_ids or []
        custom_youtube_urls = custom_youtube_urls or []
        custom_news_items = custom_news_items or []

        store_state = self._storage.load_store()
        daily_history = list(store_state.get("daily_history", {}).values())

        holding_contexts, holdings_map, portfolio_summary = self._portfolio_reader.get_portfolio_context()
        portfolio_tickers = list(holdings_map.keys())

        # Resolve selected YouTube documents
        yt_docs: List[Dict[str, Any]] = []
        for vid in selected_video_ids:
            doc = self._transcript_port.resolve_youtube_document(vid)
            if doc:
                yt_docs.append(doc)

        for curl in custom_youtube_urls:
            if curl.strip():
                doc = self._transcript_port.resolve_youtube_document(curl.strip(), custom_url=curl.strip())
                if doc:
                    yt_docs.append(doc)

        # Resolve 7-day news candidates (reuse cached weekly candidates if available)
        cached_weekly_news = store_state.get("cached_weekly_news_candidates")
        if cached_weekly_news:
            news_cands: List[PreLLMNewsCandidate] = []
            for item in cached_weekly_news:
                try:
                    news_cands.append(PreLLMNewsCandidate.model_validate(item))
                except Exception:
                    pass
        else:
            news_cands, _, _ = self._candidate_fetcher.fetch_candidates(
                portfolio_tickers=portfolio_tickers,
                lookback_days=days,
                max_news_age_hours=days * 24,
            )
        excluded_set = set(excluded_news_ids)
        active_news = [c for c in news_cands if (c.candidate_id not in excluded_set and (not c.links or c.links[0] not in excluded_set))]

        # Add custom news items if provided
        for cnews in custom_news_items:
            if cnews.strip():
                active_news.append(PreLLMNewsCandidate(
                    candidate_id=f"custom_news_{uuid.uuid4().hex[:8]}",
                    title=cnews.strip(),
                    summary=cnews.strip(),
                    source="ผู้ใช้ระบุเอง",
                    links=[],
                    age_hours=0,
                    freshness_tier="ultra_fresh",
                    freshness_label="ผู้ใช้ระบุเอง",
                ))

        calendar_events = self._calendar_port.get_upcoming_events(days_ahead=14) if self._calendar_port else []

        llm_output = self._llm_port.invoke_weekly(
            daily_digests=daily_history[-7:],
            recent_news=active_news,
            youtube_docs=yt_docs,
            calendar_events=calendar_events,
            portfolio_summary=portfolio_summary,
        )

        now_time = datetime.now()
        date_str = now_time.strftime("%Y-%m-%d")
        from_date_str = (now_time - timedelta(days=days)).strftime("%Y-%m-%d")
        now_iso = datetime.now(timezone.utc).isoformat()

        # Build items payload
        candidate_map = {c.candidate_id: c for c in active_news}
        items_payload = []
        for item in llm_output.items:
            item_dict = item.model_dump()
            c_orig = candidate_map.get(item.source_candidate_id)
            if c_orig:
                item_dict["source"] = c_orig.source
                item_dict["link"] = c_orig.links[0] if c_orig.links else ""
                item_dict["links"] = c_orig.links
                item_dict["published_at"] = c_orig.published_at
                item_dict["age_hours"] = c_orig.age_hours
                item_dict["freshness_tier"] = c_orig.freshness_tier
                item_dict["freshness_label"] = c_orig.freshness_label
            matched_badges = [
                holdings_map[sym].model_dump()
                for sym in item.affected_tickers
                if sym in holdings_map
            ]
            if not matched_badges and c_orig:
                matched_badges = [b.model_dump() for b in c_orig.portfolio_badges]
            item_dict["portfolio_badges"] = matched_badges
            items_payload.append(item_dict)

        weekly_record = {
            "digest_id": f"weekly-{date_str}",
            "from_date": from_date_str,
            "to_date": date_str,
            "generated_at": now_iso,
            "executive_themes": llm_output.executive_themes,
            "portfolio_health_summary": llm_output.portfolio_health_summary,
            "portfolio_matrix": [m.model_dump() for m in llm_output.portfolio_matrix],
            "consensus_narratives": llm_output.consensus_narratives,
            "contrarian_and_pitches": llm_output.contrarian_and_pitches,
            "week_ahead_checklist": llm_output.week_ahead_checklist,
            "items": items_payload,
            "youtube_highlights": [y.model_dump() for y in llm_output.youtube_highlights],
        }

        self._storage.save_weekly(weekly_record)
        return self.get_weekly_digest(target_date=date_str)

    def update_read_state(self, item_ids: List[str], mark_as_read: bool = True) -> Dict[str, Any]:
        """Update read status of digest items."""
        updated = self._storage.mark_read(item_ids)
        return {"ok": True, "read_item_ids": updated}
