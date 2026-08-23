"""Common helpers, validation, and locks for equity sub-routers."""
import glob
import json
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import Optional, List, Tuple
from datetime import datetime, timezone, timedelta

from fastapi import HTTPException
from pydantic import ValidationError

from api.schemas import EquityNewsDTO, EquityNewsItemDTO
from core.nlp_utils import calculate_freshness
from schemas.macro_schemas import ThemeCategory
from schemas.micro_quant_schemas import MicroQuantOutput
from tools.archivist.core import VAULT_PATH
from tools.archivist.parser import parse_company_news_items, extract_yaml_frontmatter_value

log = logging.getLogger(__name__)


def _validate_ticker(ticker: str) -> str:
    ticker = ticker.upper()
    if not re.match(r"^[A-Z0-9.\-_]+$", ticker):
        raise HTTPException(status_code=400, detail="Invalid ticker format")
    if ".." in ticker or "/" in ticker or "\\" in ticker:
        raise HTTPException(status_code=400, detail="Path traversal not allowed")
    return ticker


def _validate_schema(data: dict, expected_ticker: str) -> MicroQuantOutput | None:
    """Deep validation of required fields for Equity Sidecar JSON using Pydantic."""
    try:
        model = MicroQuantOutput.model_validate(data)
        
        # Date formats validation
        datetime.strptime(model.analysis_date, "%Y-%m-%d")
        datetime.fromisoformat(model.quant_signals.evaluated_at.replace("Z", "+00:00"))
        datetime.fromisoformat(model.sentiment_context.evaluated_at.replace("Z", "+00:00"))
        
        if expected_ticker.upper() != model.ticker.upper():
            return None
        if model.ticker.upper() != model.quant_signals.ticker.upper():
            return None
            
        return model
    except (ValidationError, ValueError, TypeError):
        return None


def _get_equity_files(ticker: Optional[str] = None) -> list[Path]:
    """Get JSON sidecar files, optionally filtered by ticker."""
    import sys
    routes_equity = sys.modules.get("api.routes_equity")
    vault_path = getattr(routes_equity, "VAULT_PATH", VAULT_PATH) if routes_equity else VAULT_PATH

    pattern = "30_Knowledge_Base/Stocks/*/* Equity Analysis *.json"
    if ticker:
        pattern = f"30_Knowledge_Base/Stocks/{ticker}/{ticker} Equity Analysis *.json"
    
    files = list(vault_path.glob(pattern))
    return files


def _extract_date_key(file_path: Path) -> tuple[str, str]:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            ev = data.get("quant_signals", {}).get("evaluated_at", "")
            if ev:
                return (ev, file_path.name)
    except Exception:
        pass
        
    try:
        date_str = file_path.stem.split(" ")[-1]
        return (date_str, file_path.name)
    except Exception:
        return ("", file_path.name)


def _get_latest_sidecar_for_ticker(files: list[Path], expected_ticker: str, strict: bool = False) -> tuple[MicroQuantOutput, Path] | None:
    if not files:
        return None
        
    if strict:
        files_sorted = sorted(files, key=lambda f: _extract_date_key(f), reverse=True)
        latest_file = files_sorted[0]
        try:
            with open(latest_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            model = _validate_schema(data, expected_ticker)
            if model:
                return (model, latest_file)
            else:
                log.warning(f"Strict mode: latest file is invalid {latest_file}")
                return None
        except Exception:
            return None
    else:
        valid_files = []
        for f in files:
            try:
                with open(f, "r", encoding="utf-8") as file_obj:
                    data = json.load(file_obj)
                model = _validate_schema(data, expected_ticker)
                if model:
                    valid_files.append((model, f))
                else:
                    log.warning(f"Malformed or invalid schema in equity sidecar: {f}")
            except Exception:
                pass
                
        if not valid_files:
            return None
            
        valid_files.sort(key=lambda x: (x[0].quant_signals.evaluated_at, x[1].name), reverse=True)
        return valid_files[0]


def _is_agent_generated(file_path: Path, content: str) -> bool:
    name = file_path.name
    if file_path.suffix == ".json":
        return True
    
    val_entity = (extract_yaml_frontmatter_value(content, "entity_type") or "").lower().replace(" ", "_")
    val_agent = extract_yaml_frontmatter_value(content, "generated_by")
    if val_entity in ("company_news", "equity_analysis") or val_agent:
        return True

    if re.search(r"(?i)latest[_\s]news|equity[_\s]analysis", name):
        return True

    return False


def _get_equity_news_from_vault(ticker: str) -> EquityNewsDTO | None:
    import sys
    routes_equity = sys.modules.get("api.routes_equity")
    vault_path = getattr(routes_equity, "VAULT_PATH", VAULT_PATH) if routes_equity else VAULT_PATH

    json_pattern = f"30_Knowledge_Base/Stocks/{ticker}/{ticker}*News*.json"
    json_files = list(vault_path.glob(json_pattern))

    md_pattern = f"30_Knowledge_Base/Stocks/{ticker}/{ticker}*News*.md"
    md_files = list(vault_path.glob(md_pattern))

    if not json_files and not md_files:
        return None

    now_utc = datetime.now(timezone.utc)
    raw_data = None

    if json_files:
        json_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        latest_json = json_files[0]
        try:
            with open(latest_json, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
        except Exception as e:
            log.warning("Failed to read news sidecar JSON %s: %s", latest_json, e)

    if not raw_data and md_files:
        md_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        latest_md = md_files[0]
        try:
            content = latest_md.read_text(encoding="utf-8")
            raw_data = parse_company_news_items(content)
        except Exception as e:
            log.warning("Failed to fallback parse news MD %s: %s", latest_md, e)

    if not raw_data or not isinstance(raw_data, dict):
        return None

    items_dto = []
    for item in raw_data.get("items", []):
        pub_at_str = item.get("published_at")
        pub_at_dt = None
        if pub_at_str:
            try:
                pub_at_dt = datetime.fromisoformat(pub_at_str.replace("Z", "+00:00"))
                if pub_at_dt.tzinfo is None:
                    pub_at_dt = pub_at_dt.replace(tzinfo=timezone.utc)
            except Exception:
                pub_at_dt = None

        if pub_at_dt:
            age_hours = int((now_utc - pub_at_dt).total_seconds() / 3600)
            freshness_score, freshness_reason = calculate_freshness(age_hours, ThemeCategory.RISK_SENTIMENT)
            is_stale = age_hours > 48
        else:
            age_hours = item.get("age_hours", 9999)
            freshness_reason = item.get("freshness_reason", "Unknown age")
            is_stale = item.get("is_stale", True)

        items_dto.append(
            EquityNewsItemDTO(
                title=item.get("title", ""),
                source=item.get("source", "N/A"),
                link=item.get("link", ""),
                published_at=pub_at_str,
                age_hours=age_hours,
                freshness_reason=freshness_reason,
                is_stale=is_stale,
                sources_count=item.get("sources_count", 1)
            )
        )

    m_val = raw_data.get("market", "US")
    market = "TH" if m_val == "TH" else "US"

    return EquityNewsDTO(
        ticker=raw_data.get("ticker", ticker),
        market=market,
        last_updated=raw_data.get("last_updated"),
        news_date=raw_data.get("date"),
        items=items_dto
    )


_DATE_REGEX = re.compile(r"20\d{2}-\d{2}-\d{2}")


def _extract_note_datetime(filename: str, mtime: float, content: str = "") -> datetime:
    m = _DATE_REGEX.search(filename)
    if m:
        try:
            return datetime.strptime(m.group(0), "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except Exception:
            pass
    if content:
        val_date = extract_yaml_frontmatter_value(content, "date")
        if val_date:
            m_fm = _DATE_REGEX.search(str(val_date))
            if m_fm:
                try:
                    return datetime.strptime(m_fm.group(0), "%Y-%m-%d").replace(tzinfo=timezone.utc)
                except Exception:
                    pass
    return datetime.fromtimestamp(mtime, tz=timezone.utc)


_ANALYST_LOCK = threading.Lock()
_ANALYST_KEY_LOCKS: dict[str, threading.Lock] = {}
_ANALYST_BURST_FAIL_CACHE: dict[str, float] = {}
ANALYST_BURST_FAIL_TTL = 10.0
MAX_STALE_AGE_SECONDS = 7 * 24 * 3600  # 7 days


def _get_analyst_lock(symbol: str) -> threading.Lock:
    with _ANALYST_LOCK:
        if symbol not in _ANALYST_KEY_LOCKS:
            _ANALYST_KEY_LOCKS[symbol] = threading.Lock()
        return _ANALYST_KEY_LOCKS[symbol]


def positive_int_or_none(value) -> int | None:
    try:
        v = int(value)
        return v if v >= 0 else None
    except (TypeError, ValueError):
        return None
