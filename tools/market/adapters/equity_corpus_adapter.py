"""Equity Corpus Adapter — Reader for all retained Equity knowledge and stored market data.

Hexagonal Architecture Invariant:
Implements EquityCorpusReaderPort. Operates as a driven adapter in tools/ reading
from the Obsidian Vault, sqlite note catalog, in-memory/disk terminal cache, and portfolio state.
Does NOT invoke upstream live network providers or trigger background jobs.
"""
from __future__ import annotations

import dataclasses
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from application.equity.notebooklm_export_ports import (
    EquityCorpusReaderPort,
    EquityCorpusSnapshot,
)
from tools.market.adapters.equity_vault_query_adapter import EquityVaultQueryAdapter

logger = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _serialize_cache_object(obj: Any) -> Any:
    if obj is None:
        return None
    if dataclasses.is_dataclass(obj):
        return {k: _serialize_cache_object(v) for k, v in dataclasses.asdict(obj).items()}
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "__dict__"):
        res = {}
        for k, v in obj.__dict__.items():
            if not k.startswith("_"):
                res[k] = _serialize_cache_object(v)
        return res
    if isinstance(obj, (list, tuple)):
        return [_serialize_cache_object(item) for item in obj]
    if isinstance(obj, dict):
        return {k: _serialize_cache_object(v) for k, v in obj.items()}
    return obj


class EquityCorpusAdapter(EquityCorpusReaderPort):
    """Gathers the comprehensive Equity dataset from vault, caches, and stored state."""

    def __init__(
        self,
        vault_path: Optional[Path] = None,
        query_adapter: Optional[EquityVaultQueryAdapter] = None,
        portfolio_provider: Optional[Callable[[str], Optional[Dict[str, Any]]]] = None,
    ) -> None:
        self._vault_adapter = query_adapter or EquityVaultQueryAdapter(vault_path=vault_path)
        self._vault_path = self._vault_adapter.vault_path
        self._portfolio_provider = portfolio_provider

    @property
    def vault_path(self) -> Path:
        return self._vault_path

    def capture_snapshot(self, ticker: str, market: str = "US") -> EquityCorpusSnapshot:
        clean_ticker = (ticker or "").strip().upper()
        clean_market = (market or "US").strip().upper()
        snapshot_at = _utc_now_iso()
        warnings: List[str] = []

        # 1. Latest Analysis Sidecar & Historical Analyses
        latest_analysis = self._vault_adapter.get_detail(clean_ticker)
        historical_analyses: List[Dict[str, Any]] = []

        try:
            files = self._vault_adapter._sidecar_files(clean_ticker)
            for f in files:
                try:
                    payload = json.loads(f.read_text(encoding="utf-8"))
                    historical_analyses.append(payload)
                except Exception:
                    pass
        except Exception as exc:
            logger.warning("[EquityCorpusAdapter] Failed reading history for %s: %s", clean_ticker, exc)

        if not latest_analysis:
            warnings.append(f"No baseline analysis found in Vault for {clean_ticker}")

        # 2. Financial Statements (from disk cache or financials service)
        financials = None
        try:
            fin_cache_file = Path("data/cache") / f"financials_{clean_market}_{clean_ticker}.json"
            if fin_cache_file.is_file():
                financials = json.loads(fin_cache_file.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.debug("[EquityCorpusAdapter] No cached financials file: %s", exc)

        # Obtain shared terminal in-memory TTL cache if available
        cache = None
        try:
            from tools.market.terminal_v2.bootstrap import get_shared_market_cache

            cache = get_shared_market_cache()
        except Exception as exc:
            logger.debug("[EquityCorpusAdapter] Optional terminal cache lookup: %s", exc)

        # 3. SEC EDGAR Financials & Form 4 Insider Trades (US only)
        sec_financials = None
        sec_insider_trades = None
        if clean_market == "US":
            if cache:
                try:
                    facts_obj = cache.get(f"sec:facts:{clean_ticker}")
                    if facts_obj:
                        sec_financials = _serialize_cache_object(facts_obj)
                    insider_obj = cache.get(f"sec:insider:{clean_ticker}")
                    if insider_obj:
                        sec_insider_trades = _serialize_cache_object(insider_obj)
                except Exception as exc:
                    logger.debug("[EquityCorpusAdapter] SEC cache read error: %s", exc)

            if not sec_financials:
                try:
                    sec_fin_file = Path("data/cache") / f"sec_financials_{clean_ticker}.json"
                    if sec_fin_file.is_file():
                        sec_financials = json.loads(sec_fin_file.read_text(encoding="utf-8"))
                except Exception:
                    pass

            if not sec_insider_trades:
                try:
                    sec_insider_file = Path("data/cache") / f"sec_insider_{clean_ticker}.json"
                    if sec_insider_file.is_file():
                        sec_insider_trades = json.loads(sec_insider_file.read_text(encoding="utf-8"))
                except Exception:
                    pass

            if not sec_financials:
                warnings.append(f"SEC EDGAR financial statements not cached for {clean_ticker}")
            if not sec_insider_trades:
                warnings.append(f"SEC Form 4 insider transactions not cached for {clean_ticker}")
        else:
            warnings.append(f"SEC EDGAR filings not applicable for {clean_market} market")

        # 4. News and Curated References
        news_items: List[Dict[str, Any]] = []
        if cache:
            for cache_key in (f"news:discovery:{clean_ticker}:15", f"rss:news:{clean_ticker}"):
                try:
                    rss_snap = cache.get(cache_key)
                    if rss_snap:
                        serialized_news = _serialize_cache_object(rss_snap)
                        candidates = None
                        if isinstance(serialized_news, dict):
                            candidates = serialized_news.get("items") or serialized_news.get("candidates")
                        elif isinstance(serialized_news, (list, tuple)):
                            candidates = serialized_news
                        if isinstance(candidates, (list, tuple)) and candidates:
                            news_items.extend(list(candidates))
                            break
                except Exception as exc:
                    logger.debug("[EquityCorpusAdapter] RSS cache read error for %s: %s", cache_key, exc)

        try:
            raw_news = self._vault_adapter.get_news(clean_ticker)
            if raw_news and isinstance(raw_news.get("items"), list):
                news_items.extend(raw_news["items"])
        except Exception:
            pass

        try:
            terminal_news_file = Path("data/cache") / f"terminal_news_{clean_ticker}.json"
            if terminal_news_file.is_file():
                t_news = json.loads(terminal_news_file.read_text(encoding="utf-8"))
                if isinstance(t_news, list):
                    news_items.extend(t_news)
        except Exception:
            pass

        # 5. Options Chain, Max Pain, Consensus
        options_chain = None
        max_pain = None
        consensus = None
        if clean_market == "US":
            if cache:
                try:
                    opt_entry = cache.get(f"cboe:options:{clean_ticker}")
                    if opt_entry:
                        options_chain = _serialize_cache_object(opt_entry)
                    max_entry = cache.get(f"cboe:maxpain:{clean_ticker}")
                    if max_entry:
                        max_pain = _serialize_cache_object(max_entry)
                    cons_entry = cache.get(f"nasdaq:consensus:{clean_ticker}")
                    if cons_entry:
                        consensus = _serialize_cache_object(cons_entry)
                except Exception as exc:
                    logger.debug("[EquityCorpusAdapter] Optional terminal cache read: %s", exc)

            if not options_chain:
                warnings.append(f"Cboe Options Chain data not cached for {clean_ticker}")
            if not consensus:
                warnings.append(f"Nasdaq consensus estimates not cached for {clean_ticker}")
        else:
            warnings.append(f"Cboe Options / Nasdaq Consensus not applicable for {clean_market} market")

        # 6. Earnings Call Notes
        earnings_calls: List[Dict[str, Any]] = []
        try:
            notes_dir = self.vault_path / "30_Knowledge_Base" / "Stocks" / clean_ticker / "Earnings_Calls"
            if notes_dir.is_dir():
                for call_file in notes_dir.glob("*.md"):
                    try:
                        content = call_file.read_text(encoding="utf-8", errors="ignore")
                        earnings_calls.append({
                            "title": call_file.stem,
                            "filename": call_file.name,
                            "content": content,
                            "modified_at": datetime.fromtimestamp(
                                call_file.stat().st_mtime, tz=timezone.utc
                            ).isoformat(),
                        })
                    except Exception:
                        pass
        except Exception:
            pass

        # 7. User Notes & Ghost Notes
        user_notes: List[Dict[str, Any]] = []
        try:
            note_list = self._vault_adapter.list_notes(clean_ticker)
            if note_list and isinstance(note_list.get("items"), list):
                for item in note_list["items"]:
                    user_notes.append(item)
        except Exception:
            pass

        # 8. Portfolio Position Context
        portfolio_holding = None
        if latest_analysis and latest_analysis.get("portfolio_context"):
            portfolio_holding = latest_analysis["portfolio_context"]
        elif self._portfolio_provider:
            try:
                portfolio_holding = self._portfolio_provider(clean_ticker)
            except Exception:
                pass

        # 9. OHLCV Market Summary
        ohlcv_summary = None
        if latest_analysis and latest_analysis.get("quant_signals", {}).get("atomic_market_snapshot"):
            ohlcv_summary = latest_analysis["quant_signals"]["atomic_market_snapshot"]

        has_quant = bool(latest_analysis and latest_analysis.get("quant_signals"))
        metadata = {
            "counts": {
                "historical_analyses": len(historical_analyses),
                "earnings_calls": len(earnings_calls),
                "news_items": len(news_items),
                "user_notes": len(user_notes),
            },
            "warnings": warnings,
            "has_latest_analysis": bool(latest_analysis),
            "has_financials": bool(financials or sec_financials or has_quant),
            "has_sec_filings": bool(sec_financials),
            "has_insider_trades": bool(
                sec_insider_trades
                or (has_quant and latest_analysis["quant_signals"].get("insider_conviction"))
            ),
            "has_sec_data": bool(sec_financials or sec_insider_trades),
            "has_options_data": bool(options_chain or max_pain),
            "has_portfolio_holding": bool(portfolio_holding and portfolio_holding.get("is_held")),
        }

        return EquityCorpusSnapshot(
            ticker=clean_ticker,
            market=clean_market,
            snapshot_at=snapshot_at,
            latest_analysis=latest_analysis,
            historical_analyses=historical_analyses,
            financials=financials,
            sec_financials=sec_financials,
            sec_insider_trades=sec_insider_trades,
            earnings_calls=earnings_calls,
            news=news_items,
            options_chain=options_chain,
            max_pain=max_pain,
            consensus=consensus,
            ohlcv_summary=ohlcv_summary,
            notes=user_notes,
            portfolio_holding=portfolio_holding,
            metadata=metadata,
        )
