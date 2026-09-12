"""Driven adapter for the read-only Equity Vault query port.

This module is the only owner of the Vault/Markdown/sidecar representation
used by the Equity read models.  The application service and HTTP routers deal
only in plain mappings, so the storage format can change independently.
"""
from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Optional

from application.equity.query_service import (
    EquityDataCorruptError,
    EquityNoteAccessError,
)
from core.nlp_utils import calculate_freshness
from schemas.macro_schemas import ThemeCategory
from schemas.micro_quant_schemas import MicroQuantOutput
from tools.archivist.parser import (
    extract_yaml_frontmatter_value,
    parse_company_news_items,
)

log = logging.getLogger(__name__)

try:
    from tools.archivist import core as _archivist_core

    _INITIAL_ARCHIVIST_VAULT = Path(_archivist_core.VAULT_PATH).resolve()
except Exception:  # pragma: no cover - import failure is handled on access
    _INITIAL_ARCHIVIST_VAULT = None


class EquityVaultQueryAdapter:
    """Read Equity sidecars and notes from the configured Vault."""

    def __init__(self, vault_path: Optional[Path] = None) -> None:
        self._vault_path = vault_path

    @property
    def vault_path(self) -> Path:
        if self._vault_path is not None:
            return Path(self._vault_path).resolve()
        # Kept inside the driven adapter for compatibility with callers that
        # patch the archivist configuration at runtime.
        from tools.archivist import core as archivist_core

        patched_core_path = Path(archivist_core.VAULT_PATH).resolve()
        if _INITIAL_ARCHIVIST_VAULT is None or patched_core_path != _INITIAL_ARCHIVIST_VAULT:
            return patched_core_path
        configured = os.getenv("OBSIDIAN_VAULT_PATH")
        return Path(configured).resolve() if configured else patched_core_path

    @staticmethod
    def _validate_ticker(ticker: str) -> str:
        clean = ticker.strip().upper()
        if not clean or len(clean) > 20 or not re.match(r"^[A-Z0-9.\-_]+$", clean):
            raise ValueError("Invalid ticker format")
        if ".." in clean or "/" in clean or "\\" in clean:
            raise ValueError("Path traversal not allowed")
        return clean

    @staticmethod
    def _validate_model(data: dict[str, Any], expected_ticker: str) -> Optional[MicroQuantOutput]:
        try:
            model = MicroQuantOutput.model_validate(data)
            datetime.strptime(model.analysis_date, "%Y-%m-%d")
            datetime.fromisoformat(model.quant_signals.evaluated_at.replace("Z", "+00:00"))
            datetime.fromisoformat(model.sentiment_context.evaluated_at.replace("Z", "+00:00"))
            if expected_ticker.upper() != model.ticker.upper():
                return None
            if model.ticker.upper() != model.quant_signals.ticker.upper():
                return None
            return model
        except Exception:
            return None

    @staticmethod
    def _date_key(path: Path) -> tuple[str, str]:
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            evaluated_at = payload.get("quant_signals", {}).get("evaluated_at", "")
            if evaluated_at:
                return str(evaluated_at), path.name
        except Exception:
            pass
        return path.stem.split(" ")[-1], path.name

    def _sidecar_files(self, ticker: Optional[str] = None) -> list[Path]:
        all_files: list[Path] = []

        # 1. Fast indexed path via SQLite catalog
        try:
            from tools.archivist.catalog_adapter import SqliteNoteCatalogAdapter
            from tools.archivist.catalog_runtime import resolve_catalog_path
            cat_db = resolve_catalog_path(self.vault_path, require_exists=True)
            if cat_db.exists():
                cat = SqliteNoteCatalogAdapter(db_path=cat_db, vault_root=self.vault_path)
                rel_paths = cat.get_sidecars(ticker)
                for rp in rel_paths:
                    p = self.vault_path / rp
                    if p.exists() and p.is_file():
                        all_files.append(p)
        except Exception:
            all_files.clear()

        # 2. Fallback to filesystem globbing if catalog returned no files
        if not all_files:
            if ticker:
                # V2 path: Stocks/{ticker}/Analysis/*.json
                all_files.extend(self.vault_path.glob(f"30_Knowledge_Base/Stocks/{ticker}/Analysis/* Equity Analysis *.json"))
                # V1 path: Stocks/{ticker}/*.json
                all_files.extend(self.vault_path.glob(f"30_Knowledge_Base/Stocks/{ticker}/{ticker} Equity Analysis *.json"))
                # System sidecar path (.system/sidecars/{ticker}/)
                all_files.extend(self.vault_path.glob(f".system/sidecars/{ticker}/* Equity Analysis *.json"))
            else:
                # V2 path: Stocks/*/Analysis/*.json
                all_files.extend(self.vault_path.glob("30_Knowledge_Base/Stocks/*/Analysis/* Equity Analysis *.json"))
                # V1 path: Stocks/*/*.json
                all_files.extend(self.vault_path.glob("30_Knowledge_Base/Stocks/*/* Equity Analysis *.json"))
                # System sidecar path
                all_files.extend(self.vault_path.glob(".system/sidecars/*/* Equity Analysis *.json"))

        # Deduplicate and exclude 'latest.json' pointer when date-stamped files exist
        seen: set[str] = set()
        result: list[Path] = []
        for p in all_files:
            p_res = p.resolve()
            p_str = str(p_res)
            if p_str not in seen and p.is_file():
                # Avoid duplicate latest pointer if date-specific files exist
                if p.name.endswith("latest.json") and len(all_files) > 1:
                    continue
                seen.add(p_str)
                result.append(p)
        return result

    def _latest_sidecar(
        self, files: list[Path], expected_ticker: str, *, strict: bool
    ) -> Optional[tuple[MicroQuantOutput, Path]]:
        if not files:
            return None
        if strict:
            latest_file = sorted(files, key=self._date_key, reverse=True)[0]
            try:
                with latest_file.open("r", encoding="utf-8") as handle:
                    model = self._validate_model(json.load(handle), expected_ticker)
            except Exception as exc:
                raise EquityDataCorruptError(str(latest_file)) from exc
            if model is None:
                raise EquityDataCorruptError(str(latest_file))
            return model, latest_file

        valid: list[tuple[MicroQuantOutput, Path]] = []
        for path in files:
            try:
                with path.open("r", encoding="utf-8") as handle:
                    model = self._validate_model(json.load(handle), expected_ticker)
                if model is not None:
                    valid.append((model, path))
            except Exception:
                log.warning("Skipping malformed Equity sidecar: %s", path)
        if not valid:
            return None
        valid.sort(
            key=lambda item: (
                item[0].quant_signals.evaluated_at,
                not bool(re.search(r"_[a-f0-9]{6}\.json$", item[1].name)),
                item[1].name,
            ),
            reverse=True,
        )
        return valid[0]

    @staticmethod
    def _model_detail(model: MicroQuantOutput, sidecar: Path, vault_path: Path) -> dict[str, Any]:
        relative = str(sidecar.relative_to(vault_path)).replace("\\", "/")
        quant = model.quant_signals.model_dump()
        sentiment = model.sentiment_context.model_dump()

        # Check if companion markdown report actually exists on disk
        companion_md = sidecar.with_suffix(".md")
        if not companion_md.exists():
            clean_name = re.sub(r"_[a-f0-9]{6}\.md$", ".md", companion_md.name)
            candidate = companion_md.parent / clean_name
            if candidate.exists():
                companion_md = candidate
            elif ".system" in sidecar.parts:
                kb_candidate = vault_path / "30_Knowledge_Base" / "Stocks" / model.ticker / "Analysis" / clean_name
                if kb_candidate.exists():
                    companion_md = kb_candidate

        source_file_rel = str(companion_md.relative_to(vault_path)).replace("\\", "/") if companion_md.exists() else None

        return {
            "ticker": model.ticker,
            "market": model.market,
            "company_name": model.quant_signals.company_name,
            "analysis_date": model.analysis_date,
            "evaluated_at": model.quant_signals.evaluated_at,
            "market_sentiment": model.sentiment_context.market_sentiment,
            "composite_score": model.quant_signals.composite_score,
            "data_quality_flags": getattr(model.quant_signals, "data_quality_flags", []),
            "source_file": source_file_rel,
            "sidecar_file": relative,
            "quant_signals": quant,
            "sentiment_context": sentiment,
            "narrative_analysis": model.narrative_analysis,
            "base_case_summary": model.base_case_summary,
            "generated_by": model.generated_by,
        }

    def list_latest(self) -> list[dict[str, Any]]:
        grouped: dict[str, list[Path]] = {}
        for path in self._sidecar_files():
            # In V2, parent may be 'Analysis', so ticker is parent.parent.name
            if path.parent.name.lower() == "analysis":
                ticker_candidate = path.parent.parent.name.upper()
            else:
                ticker_candidate = path.parent.name.upper()
            grouped.setdefault(ticker_candidate, []).append(path)

        result: list[dict[str, Any]] = []
        for ticker, paths in grouped.items():
            latest = self._latest_sidecar(paths, ticker, strict=False)
            if latest:
                result.append(self._model_detail(latest[0], latest[1], self.vault_path))
        result.sort(key=lambda item: (item["evaluated_at"], item["ticker"]), reverse=True)
        return result

    def get_detail(self, ticker: str) -> Optional[dict[str, Any]]:
        clean = self._validate_ticker(ticker)
        files = self._sidecar_files(clean)
        if not files:
            return None
        latest = self._latest_sidecar(files, clean, strict=True)
        if latest is None:
            return None
        return self._model_detail(latest[0], latest[1], self.vault_path)

    def get_news(self, ticker: str) -> Optional[dict[str, Any]]:
        clean = self._validate_ticker(ticker)
        stock_dir = self.vault_path / "30_Knowledge_Base" / "Stocks" / clean
        json_files = list(stock_dir.glob(f"{clean}*News*.json"))
        md_files = list(stock_dir.glob(f"{clean}*News*.md"))
        if not json_files and not md_files:
            return None

        raw_data: Optional[dict[str, Any]] = None
        if json_files:
            latest_json = max(json_files, key=lambda path: path.stat().st_mtime)
            try:
                raw_data = json.loads(latest_json.read_text(encoding="utf-8"))
            except Exception as exc:
                log.warning("Failed to read Equity news JSON %s: %s", latest_json, exc)
        if not raw_data and md_files:
            latest_md = max(md_files, key=lambda path: path.stat().st_mtime)
            try:
                raw_data = parse_company_news_items(latest_md.read_text(encoding="utf-8"))
            except Exception as exc:
                log.warning("Failed to parse Equity news Markdown %s: %s", latest_md, exc)
        if not isinstance(raw_data, dict):
            return None

        now_utc = datetime.now(timezone.utc)
        items: list[dict[str, Any]] = []
        for item in raw_data.get("items", []):
            published = item.get("published_at")
            published_dt = None
            if published:
                try:
                    published_dt = datetime.fromisoformat(str(published).replace("Z", "+00:00"))
                    if published_dt.tzinfo is None:
                        published_dt = published_dt.replace(tzinfo=timezone.utc)
                except Exception:
                    published_dt = None
            if published_dt:
                age_hours = int((now_utc - published_dt).total_seconds() / 3600)
                _, freshness_reason = calculate_freshness(age_hours, ThemeCategory.RISK_SENTIMENT)
                stale = age_hours > 48
            else:
                age_hours = int(item.get("age_hours", 9999))
                freshness_reason = item.get("freshness_reason", "Unknown age")
                stale = bool(item.get("is_stale", True))
            items.append({
                "title": item.get("title", ""),
                "source": item.get("source", "N/A"),
                "link": item.get("link", ""),
                "published_at": published,
                "age_hours": age_hours,
                "freshness_reason": freshness_reason,
                "is_stale": stale,
                "sources_count": item.get("sources_count", 1),
            })
        return {
            "ticker": raw_data.get("ticker", clean),
            "market": "TH" if raw_data.get("market", "US") == "TH" else "US",
            "last_updated": raw_data.get("last_updated"),
            "news_date": raw_data.get("date"),
            "items": items,
        }

    @staticmethod
    def _note_datetime(filename: str, mtime: float, content: str = "") -> datetime:
        match = re.search(r"20\d{2}-\d{2}-\d{2}", filename)
        if not match and content:
            match = re.search(r"20\d{2}-\d{2}-\d{2}", extract_yaml_frontmatter_value(content, "date") or "")
        if match:
            try:
                return datetime.strptime(match.group(0), "%Y-%m-%d").replace(tzinfo=timezone.utc)
            except ValueError:
                pass
        return datetime.fromtimestamp(mtime, tz=timezone.utc)

    def list_notes(self, ticker: str, days: int = 3) -> dict[str, Any]:
        clean = self._validate_ticker(ticker)
        now_utc = datetime.now(timezone.utc)
        cutoff = (now_utc - timedelta(days=days)).replace(hour=0, minute=0, second=0, microsecond=0) if days > 0 else None
        tag_pattern = re.compile(rf"(?i)(?<![A-Za-z0-9_])#{re.escape(clean)}\b")
        wikilink_pattern = re.compile(rf"(?i)\[\[(?:[^\]]+/)?{re.escape(clean)}(?:[|#][^\]]*)?\]\]")
        frontmatter_pattern = re.compile(rf"(?i)^\s*tickers?:\s*\[?.*?\b{re.escape(clean)}\b", re.MULTILINE)
        vault_name = os.getenv("OBSIDIAN_VAULT_NAME", self.vault_path.name)
        notes: list[dict[str, Any]] = []
        seen: set[str] = set()
        search_folders = [
            "News",
            "YouTube_Summaries",
            f"Earnings_Calls/{clean}",
            f"Stocks/{clean}/Earnings",
            f"Stocks/{clean}/Analysis",
        ]
        for folder_name in search_folders:
            target_dir = self.vault_path / "30_Knowledge_Base" / folder_name
            if not target_dir.exists():
                continue
            for path in target_dir.rglob("*.md"):
                if "Revisions" in path.parts:
                    continue
                relative = str(path.relative_to(self.vault_path)).replace("\\", "/")
                if relative in seen or path.name.startswith(".") or path.name == "index.md":
                    continue
                try:
                    content = path.read_text(encoding="utf-8", errors="ignore")
                    matched_by = None
                    if "Earnings_Calls" in folder_name:
                        matched_by = "earnings_call"
                    elif tag_pattern.search(content):
                        matched_by = "tag"
                    elif wikilink_pattern.search(content):
                        matched_by = "wikilink"
                    elif frontmatter_pattern.search(content):
                        matched_by = "frontmatter"
                    if not matched_by:
                        continue
                    note_dt = self._note_datetime(path.name, path.stat().st_mtime, content)
                    if cutoff is not None and note_dt < cutoff:
                        continue
                    seen.add(relative)
                    if folder_name == "YouTube_Summaries":
                        matched_by = "youtube"
                    elif "Earnings_Calls" in folder_name:
                        matched_by = "earnings_call"
                    else:
                        matched_by = "news"
                    lines = [line.strip() for line in content.splitlines() if line.strip() and not line.startswith("---")]
                    notes.append({
                        "title": path.stem,
                        "folder": str(path.parent.relative_to(self.vault_path)).replace("\\", "/"),
                        "relative_path": relative,
                        "obsidian_uri": f"obsidian://open?vault={vault_name}&file={relative}",
                        "snippet": " ".join(lines[:3])[:250],
                        "modified_at": note_dt.isoformat(),
                        "matched_by": matched_by,
                    })
                except OSError as exc:
                    log.warning("Failed to inspect Equity note %s: %s", path, exc)
        notes.sort(key=lambda item: item["modified_at"], reverse=True)
        return {"ticker": clean, "total_count": len(notes), "items": notes}

    def read_note(self, relative_path: str) -> dict[str, Any]:
        if ".." in relative_path or relative_path.startswith(("/", "\\")):
            raise EquityNoteAccessError("Invalid path format")
        root = self.vault_path
        target = (root / relative_path).resolve()
        if not target.is_relative_to(root):
            raise EquityNoteAccessError("Access denied: Outside vault boundary")
        if not target.exists() or not target.is_file():
            raise FileNotFoundError("Note file not found")
        if target.suffix != ".md":
            raise EquityNoteAccessError("Only markdown files can be read")
        content = target.read_text(encoding="utf-8")
        return {
            "title": target.stem,
            "relative_path": relative_path,
            "content": content,
            "modified_at": datetime.fromtimestamp(target.stat().st_mtime, tz=timezone.utc).isoformat(),
        }
