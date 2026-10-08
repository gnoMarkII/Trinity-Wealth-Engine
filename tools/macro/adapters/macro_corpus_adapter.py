"""Macro Corpus Adapter — Concrete reader for all retained Macro knowledge and observables.

Hexagonal Architecture Invariant:
Implements MacroCorpusReaderPort. Operates as a driven adapter in tools/ reading
from the Obsidian Vault, sqlite note catalog, in-memory terminal cache, and runtime files.
Does NOT invoke upstream network providers or trigger background jobs.
"""
from __future__ import annotations

import copy
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from application.knowledge.ports import NoteCatalogPort
from application.macro.notebooklm_export_ports import (
    MacroCorpusReaderPort,
    MacroCorpusSnapshot,
)
from tools.archivist.catalog_adapter import SqliteNoteCatalogAdapter, resolve_catalog_path
from tools.archivist.vault_paths import VaultPaths
from tools.macro.adapters.strategy_vault_adapter import IndicatorSeriesAdapter, StrategyVaultAdapter
from tools.macro.adapters.news_funnel_store_adapter import NewsFunnelStoreAdapter
from tools.macro.adapters.sector_snapshot_adapter import SectorSnapshotStore
from tools.market.terminal_v2.application.cache import ThreadSafeTTLCache
from tools.market.terminal_v2.bootstrap import get_shared_market_cache


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class MacroCorpusAdapter(MacroCorpusReaderPort):
    """Gathers the comprehensive Macro dataset from vault, catalog, and caches."""

    def __init__(
        self,
        vault_path: Optional[Path] = None,
        hard_data_path: Optional[Path] = None,
        sector_dir: Optional[Path] = None,
        shared_cache: Optional[ThreadSafeTTLCache] = None,
        catalog: Optional[NoteCatalogPort] = None,
    ) -> None:
        configured_vault = vault_path or os.getenv("OBSIDIAN_VAULT_PATH") or "./memories"
        self._vault_path = Path(configured_vault).resolve()
        self._hard_data_path = (
            Path(hard_data_path).resolve()
            if hard_data_path
            else Path("data/macro/thailand/official_hard_data.json").resolve()
        )
        self._sector_dir = (
            Path(sector_dir).resolve()
            if sector_dir
            else Path(os.getenv("SECTOR_ROTATION_RUNTIME_DIR") or "data/sector_rotation").resolve()
        )
        self._shared_cache = shared_cache or get_shared_market_cache()
        self._catalog = catalog

    @property
    def vault_path(self) -> Path:
        return self._vault_path

    def _get_catalog(self) -> Optional[NoteCatalogPort]:
        if self._catalog is not None:
            return self._catalog
        try:
            cat_path = resolve_catalog_path(self._vault_path, require_exists=True)
            if cat_path.is_file():
                self._catalog = SqliteNoteCatalogAdapter(db_path=cat_path, vault_root=self._vault_path, read_only=True)
                return self._catalog
        except Exception:
            pass
        return None

    def capture_snapshot(self) -> MacroCorpusSnapshot:
        snapshot_at = _utc_now_iso()
        warnings: List[str] = []

        # 1. Latest Strategy Report & Evidence
        strategy_adapter = StrategyVaultAdapter(self._vault_path)
        latest_report: Optional[Dict[str, Any]] = None
        strategy_report_id: Optional[str] = None
        try:
            latest_report = strategy_adapter.latest()
            strategy_report_id = str(latest_report.get("strategy_report_id") or "")
        except FileNotFoundError:
            warnings.append("No active Macro Strategy report found in vault")
        except Exception as exc:
            warnings.append(f"Failed to read latest Macro Strategy report: {exc}")

        # 2. Historical Macro Strategy Reports (retained sidecars)
        historical_reports = self._read_historical_reports(latest_report_id=strategy_report_id)

        # 3. Knowledge Catalog Notes (regional snapshots, country notes, global analysis)
        catalog_notes = self._read_catalog_macro_notes()

        # 4. Indicator Series (all series referenced in dashboard or latest report + catalog on disk)
        indicator_series = self._read_indicator_series(strategy_adapter, latest_report, historical_reports)

        # 5. Market Observables (13 Groups from in-memory cache with exact provider keys)
        market_observables = self._capture_market_observables()

        # 6. Thailand Hard Data
        thailand_hard_data = self._read_thailand_hard_data()

        # 7. Sector Rotation Snapshots
        sector_rotation = self._read_sector_rotation()

        # 8. Macro News Funnel
        news_funnel = self._read_news_funnel()

        # Metadata & Inventory summary
        metadata = {
            "snapshot_at": snapshot_at,
            "vault_path": str(self._vault_path),
            "counts": {
                "latest_report_present": latest_report is not None,
                "historical_reports": len(historical_reports),
                "catalog_notes": len(catalog_notes),
                "indicator_series": len(indicator_series),
                "market_observables_cached": sum(
                    1 for v in market_observables.values() if isinstance(v, dict) and v.get("status") == "cached"
                ),
                "market_observables_total": len(market_observables),
                "thailand_hard_data_present": thailand_hard_data is not None,
                "sector_rotation_present": sector_rotation is not None,
                "news_pending": len(news_funnel.get("pending", [])),
                "news_filtered": len(news_funnel.get("filtered", [])),
            },
            "warnings": warnings,
        }

        return MacroCorpusSnapshot(
            snapshot_at=snapshot_at,
            strategy_report_id=strategy_report_id,
            latest_report=latest_report,
            historical_reports=historical_reports,
            catalog_notes=catalog_notes,
            indicator_series=indicator_series,
            market_observables=market_observables,
            thailand_hard_data=thailand_hard_data,
            sector_rotation=sector_rotation,
            news_funnel=news_funnel,
            metadata=metadata,
        )

    def _read_historical_reports(self, latest_report_id: Optional[str]) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        seen_ids: set[str] = set()
        if latest_report_id:
            seen_ids.add(latest_report_id)

        candidate_dirs = [
            self._vault_path / "30_Knowledge_Base" / "Macroeconomics" / "Strategies",
            self._vault_path / "30_Knowledge_Base" / "Strategies",
        ]
        candidates: List[Path] = []
        for cdir in candidate_dirs:
            if cdir.exists():
                # C02: Use rglob to discover partitioned reports (e.g. Strategies/YYYY/MM/)
                candidates.extend(
                    p for p in cdir.rglob("Macro_Strategy_Direction_*.json")
                    if "Revisions" not in p.parts and ".trash" not in p.parts
                )

        def _sort_key(p: Path) -> tuple[str, str]:
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                return (str(data.get("evaluated_at", ""))[:10], p.name)
            except Exception:
                return ("", p.name)

        candidates.sort(key=_sort_key, reverse=True)
        for path in candidates:
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                rid = str(data.get("strategy_report_id") or path.stem)
                if rid in seen_ids or path.stem in seen_ids:
                    continue
                seen_ids.add(rid)
                seen_ids.add(path.stem)
                data["_file_origin"] = path.name
                data["_file_relative"] = path.relative_to(self._vault_path).as_posix()
                if not data.get("strategy_report_id"):
                    data["strategy_report_id"] = rid
                if not data.get("report_id"):
                    data["report_id"] = rid
                results.append(data)
            except Exception:
                continue
        return results

    def _read_catalog_macro_notes(self) -> List[Dict[str, Any]]:
        catalog = self._get_catalog()
        if catalog is None:
            return self._scan_vault_macro_notes_fallback()

        entity_types = ["macro_snapshot", "macro_country", "macro_global", "macro_regional", "macro_event"]
        notes: List[Dict[str, Any]] = []
        seen_paths: set[str] = set()

        for etype in entity_types:
            offset = 0
            page_size = 100
            while True:
                batch = catalog.find_notes(entity_type=etype, limit=page_size, offset=offset)
                if not batch:
                    break
                for entry in batch:
                    if entry.relative_path in seen_paths:
                        continue
                    seen_paths.add(entry.relative_path)
                    note_dict: Dict[str, Any] = {
                        "note_id": entry.note_id,
                        "title": entry.title,
                        "entity_type": entry.entity_type,
                        "relative_path": entry.relative_path,
                        "date": entry.date,
                        "content_sha256": entry.content_sha256,
                    }
                    abs_path = self._vault_path / entry.relative_path
                    if abs_path.is_file():
                        try:
                            content = abs_path.read_text(encoding="utf-8")
                            note_dict["body_snippet"] = content[:3000]
                            note_dict["full_body"] = content
                        except Exception:
                            note_dict["body_snippet"] = ""
                            note_dict["full_body"] = ""
                    notes.append(note_dict)
                if len(batch) < page_size:
                    break
                offset += len(batch)

        return notes

    def _scan_vault_macro_notes_fallback(self) -> List[Dict[str, Any]]:
        notes: List[Dict[str, Any]] = []
        macro_dir = self._vault_path / "30_Knowledge_Base" / "Macroeconomics"
        if not macro_dir.exists():
            return notes

        for md_file in macro_dir.rglob("*.md"):
            if "Revisions" in md_file.parts or ".trash" in md_file.parts:
                continue
            try:
                rel = md_file.relative_to(self._vault_path).as_posix()
                content = md_file.read_text(encoding="utf-8")
                notes.append({
                    "note_id": md_file.stem,
                    "title": md_file.stem,
                    "entity_type": "macro_note",
                    "relative_path": rel,
                    "body_snippet": content[:3000],
                    "full_body": content,
                })
            except Exception:
                continue
        return notes

    def _read_indicator_series(
        self,
        strategy_adapter: StrategyVaultAdapter,
        latest_report: Optional[Dict[str, Any]],
        historical_reports: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """C03: Read indicator series with full retained history and disk enumeration."""
        series_adapter = IndicatorSeriesAdapter(strategy_adapter)
        results: List[Dict[str, Any]] = []
        seen_keys: set[str] = set()

        # 1. Indicators referenced in latest & historical reports
        all_reports = [latest_report] if latest_report else []
        if historical_reports:
            all_reports.extend(historical_reports)

        for rep in all_reports:
            if not rep:
                continue
            indicators_list = rep.get("dashboard_indicators", [])
            if not isinstance(indicators_list, list):
                continue
            for item in indicators_list:
                if not isinstance(item, dict):
                    continue
                skey = str(item.get("series_key") or "").strip().lower()
                if not skey or skey in seen_keys:
                    continue
                seen_keys.add(skey)
                indicator_id = item.get("indicator_id") or skey
                label = item.get("label") or indicator_id
                unit = item.get("unit") or ""
                try:
                    points = series_adapter.load(skey, "1y")
                    results.append({
                        "indicator_id": indicator_id,
                        "series_key": skey,
                        "label": label,
                        "unit": unit,
                        "points": points,
                    })
                except Exception:
                    results.append({
                        "indicator_id": indicator_id,
                        "series_key": skey,
                        "label": label,
                        "unit": unit,
                        "points": [],
                        "error": "Failed to load series points",
                    })

        # 2. Disk scan of all stored series files (full retained history beyond latest dashboard)
        series_dirs = [
            self._vault_path / "30_Knowledge_Base" / "Macroeconomics" / "Indicator_Series",
            self._vault_path / "30_Knowledge_Base" / "Strategies" / "Macro_Indicator_Series",
        ]
        for sdir in series_dirs:
            if not sdir.exists():
                continue
            for sfile in sdir.glob("*.json"):
                skey = sfile.stem.strip().lower()
                if not skey or skey in seen_keys:
                    continue
                seen_keys.add(skey)
                try:
                    payload = json.loads(sfile.read_text(encoding="utf-8"))
                    if isinstance(payload, dict):
                        points = payload.get("points", [])
                        results.append({
                            "indicator_id": payload.get("indicator_id") or skey,
                            "series_key": skey,
                            "label": payload.get("label") or skey,
                            "unit": payload.get("unit") or "",
                            "points": points if isinstance(points, list) else [],
                        })
                except Exception:
                    continue

        return results

    def _capture_market_observables(self) -> Dict[str, Any]:
        """C01: Capture read-only frozen snapshots of the 13 macro observables using actual provider cache keys."""
        # Maps logical name to priority list of cache keys used by terminal_v2 providers
        observable_key_candidates = {
            "us_yield_curve": ["treasury:yield_curve:latest"],
            "ofr_financial_stress": ["ofr:fsi:latest", "ofr:financial_stress"],
            "gold_cot": ["cftc:cot:disagg:088691", "cftc:cot:metals:gold"],
            "global_policy_rates": ["bis:policy_rates:latest", "bis:policy_rates"],
            "commodity_volatility": [
                "cboe:commodity_vol:OVX",
                "cboe:commodity_vol:GVZ",
                "cboe:commodity_vol:VXSLV",
                "cboe:volatility:OVX",
            ],
            "treasury_auction_10y": [
                "treasury:auction_history:Note:10-Year:15",
                "treasury:auction:Note:10-Year",
                "treasury:auctions:10",
            ],
            "treasury_auction_13w": [
                "treasury:auction_history:Bill:13-Week:15",
                "treasury:auction:Bill:13-Week",
                "treasury:auctions:10",
            ],
            "us_national_debt": ["treasury:debt:30", "treasury:debt:limit_30"],
            "thai_investor_flow": ["settrade:flow:SET", "settrade:investor_flow:SET"],
            "thai_retail_gold": ["goldtraders:retail:quote", "goldtraders:retail_gold"],
            "thai_market_valuation": ["settrade:stats:SET", "settrade:valuation:SET"],
            "thai_market_breadth": ["settrade:breadth:SET"],
            "crypto_macro_liquidity": [
                "macro:crypto_liquidity",
                "crypto:benchmark:btc",
                "defillama:stablecoin:supply",
                "sosovalue:etf:btc",
            ],
        }

        results: Dict[str, Any] = {}
        for logical_name, candidates in observable_key_candidates.items():
            found_entries = []
            for cache_key in candidates:
                peek_res = self._shared_cache.peek(cache_key)
                if peek_res is not None:
                    found_entries.append({
                        "cache_key": cache_key,
                        "data": copy.deepcopy(peek_res["data"]),
                        "cached_at": peek_res["cached_at"],
                        "is_expired": peek_res["is_expired"],
                    })

            if not found_entries:
                results[logical_name] = {
                    "cache_key": candidates[0],
                    "status": "missing",
                    "reason": f"No cache entries found for candidate keys: {candidates}",
                }
            elif len(found_entries) == 1:
                results[logical_name] = {
                    "cache_key": found_entries[0]["cache_key"],
                    "status": "cached",
                    "data": found_entries[0]["data"],
                    "cached_at": found_entries[0]["cached_at"],
                    "is_expired": found_entries[0]["is_expired"],
                }
            else:
                # Composite / multi-key group (e.g. commodity volatility OVX/GVZ/VXSLV or crypto components)
                composite_data = {entry["cache_key"]: entry["data"] for entry in found_entries}
                latest_cached_at = max(entry["cached_at"] for entry in found_entries)
                any_expired = any(entry["is_expired"] for entry in found_entries)
                results[logical_name] = {
                    "cache_key": candidates[0],
                    "status": "cached",
                    "data": composite_data,
                    "cached_at": latest_cached_at,
                    "is_expired": any_expired,
                    "components": [e["cache_key"] for e in found_entries],
                }

        return results

    def _read_thailand_hard_data(self) -> Optional[Dict[str, Any]]:
        if not self._hard_data_path.is_file():
            return None
        try:
            return json.loads(self._hard_data_path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _read_sector_rotation(self) -> Optional[Dict[str, Any]]:
        """C08: Read state and snapshots using SectorSnapshotStore."""
        try:
            store = SectorSnapshotStore(self._sector_dir)
            state = store.read_state()
            data: Dict[str, Any] = {"state": state}

            latest_id = state.get("latest_snapshot_id")
            if latest_id:
                loaded = store.load(latest_id)
                if loaded:
                    snap_model, evidence_ref = loaded
                    data["latest_snapshot"] = snap_model.model_dump(mode="json")
                    data["evidence_ref"] = evidence_ref
                    data["snapshot"] = snap_model.model_dump(mode="json")

            # Also enumerate all retained historical snapshots
            snapshots_dir = self._sector_dir / "snapshots"
            if snapshots_dir.is_dir():
                retained_snaps = []
                for s_file in snapshots_dir.glob("*.json"):
                    try:
                        raw = json.loads(s_file.read_text(encoding="utf-8"))
                        retained_snaps.append({
                            "snapshot_id": s_file.stem,
                            "snapshot": raw.get("snapshot"),
                        })
                    except Exception:
                        continue
                data["historical_snapshots"] = retained_snaps

            return data if (data.get("snapshot") or data.get("state")) else None
        except Exception:
            return None

    def _read_news_funnel(self) -> Dict[str, List[Dict[str, Any]]]:
        try:
            adapter = NewsFunnelStoreAdapter()
            return {
                "pending": adapter.pending(),
                "filtered": adapter.filtered(),
            }
        except Exception:
            return {"pending": [], "filtered": []}
