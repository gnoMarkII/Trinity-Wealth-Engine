import glob
import json
import logging
import os
import re
import threading
import time
from pathlib import Path
from typing import List, Optional
from datetime import datetime, timezone, timedelta

import yfinance as yf
from fastapi import APIRouter, Depends, HTTPException
from pydantic import ValidationError

from api.auth import require_session
from api.schemas import (
    CorporateActionFactorDTO,
    DCFScenarioLevelDTO,
    EquityDetailDTO,
    EquityNewsDTO,
    EquityNewsItemDTO,
    EquityNoteContentDTO,
    EquityNoteItemDTO,
    EquityNotesDTO,
    EquitySentimentContextDTO,
    EquitySummaryDTO,
    InsiderFilingDTO,
    InsiderFilingsResponseDTO,
    InsiderTransactionDTO,
    ValuationTargetsDTO,
    AnalystContextDTO,
    EarningsHistoryEntryDTO,
)
from api.state_db import (
    get_connection,
    get_latest_dcf_evaluation,
    get_sec_insider_filings_and_transactions,
    record_dcf_evaluation,
    get_analyst_context_cache,
    upsert_analyst_context_cache,
)
from tools.market.asset_resolver import resolve_asset
from tools.market.calendar import get_asset_calendar
from tools.market.earnings import fetch_earnings_dates, finite_or_none
from tools.market.sec_form4_pipeline import sync_insider_filings_from_yfinance

from core.nlp_utils import calculate_freshness
from schemas.macro_schemas import ThemeCategory
from schemas.micro_quant_schemas import MicroQuantOutput
from tools.archivist.core import VAULT_PATH
from tools.archivist.parser import parse_company_news_items, extract_yaml_frontmatter_value
from tools.portfolio.journal import _JOURNAL_BLOCK_RE


log = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/equity",
    tags=["equity"],
    dependencies=[Depends(require_session)],
)

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

def _get_equity_files(ticker: str = None) -> list[Path]:
    """Get JSON sidecar files, optionally filtered by ticker."""
    pattern = "30_Knowledge_Base/Stocks/*/* Equity Analysis *.json"
    if ticker:
        pattern = f"30_Knowledge_Base/Stocks/{ticker}/{ticker} Equity Analysis *.json"
    
    # glob.glob on Vault
    files = list(VAULT_PATH.glob(pattern))
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

@router.get("/latest", response_model=List[EquitySummaryDTO])
def get_latest_equities():
    files = _get_equity_files()
    
    # Group by ticker
    ticker_files = {}
    for f in files:
        # Ticker is the folder name containing the file
        ticker = f.parent.name
        if ticker not in ticker_files:
            ticker_files[ticker] = []
        ticker_files[ticker].append(f)
            
    results = []
    for ticker, paths in ticker_files.items():
        latest = _get_latest_sidecar_for_ticker(paths, ticker, strict=False)
        if latest:
            model, file_path = latest
            rel_path = str(file_path.relative_to(VAULT_PATH)).replace("\\", "/")
            source_md = rel_path.replace(".json", ".md")
            
            summary = EquitySummaryDTO(
                ticker=model.ticker,
                market=model.market,
                company_name=model.quant_signals.company_name,
                analysis_date=model.analysis_date,
                evaluated_at=model.quant_signals.evaluated_at,
                market_sentiment=model.sentiment_context.market_sentiment,
                composite_score=model.quant_signals.composite_score,
                data_quality_flags=getattr(model.quant_signals, "data_quality_flags", []),
                source_file=source_md,
                sidecar_file=rel_path
            )
            results.append(summary)
            
    results.sort(key=lambda x: (x.evaluated_at, x.ticker), reverse=True)
    return results


def _is_agent_generated(file_path: Path, content: str) -> bool:
    name = file_path.name
    if file_path.suffix == ".json":
        return True
    
    # 1. Check Frontmatter signals (Case-insensitive)
    val_entity = (extract_yaml_frontmatter_value(content, "entity_type") or "").lower().replace(" ", "_")
    val_agent = extract_yaml_frontmatter_value(content, "generated_by")
    if val_entity in ("company_news", "equity_analysis") or val_agent:
        return True

    # 2. Check filename pattern (Supports space and underscore: Latest_News, Latest News, Equity_Analysis, Equity Analysis)
    if re.search(r"(?i)latest[_\s]news|equity[_\s]analysis", name):
        return True

    return False


@router.get("/notes/content", response_model=EquityNoteContentDTO)
def get_equity_note_content(rel_path: str):
    if ".." in rel_path or rel_path.startswith("/") or rel_path.startswith("\\"):
        raise HTTPException(status_code=400, detail="Invalid path format")

    vault_resolved = VAULT_PATH.resolve()
    target_path = (VAULT_PATH / rel_path).resolve()

    # Robust path traversal check using Path.is_relative_to()
    if not target_path.is_relative_to(vault_resolved):
        raise HTTPException(status_code=403, detail="Access denied: Outside vault boundary")
    if not target_path.exists() or not target_path.is_file():
        raise HTTPException(status_code=404, detail="Note file not found")
    if target_path.suffix != ".md":
        raise HTTPException(status_code=400, detail="Only markdown files can be read")

    try:
        content = target_path.read_text(encoding="utf-8")
        mtime = datetime.fromtimestamp(target_path.stat().st_mtime, tz=timezone.utc).isoformat()
        return EquityNoteContentDTO(
            title=target_path.stem,
            relative_path=rel_path,
            content=content,
            modified_at=mtime
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read note content: {str(e)}")


@router.get("/{ticker}", response_model=EquityDetailDTO)
def get_equity_detail(ticker: str):
    ticker = _validate_ticker(ticker)
    files = _get_equity_files(ticker)
    
    if not files:
        raise HTTPException(status_code=404, detail="Equity not found")
        
    latest = _get_latest_sidecar_for_ticker(files, ticker, strict=True)
    if not latest:
        # If strict=True and it returns None, it means the latest file was corrupted or mismatched.
        raise HTTPException(status_code=503, detail="Service Unavailable: Data corrupted or ticker mismatch")
        
    model, latest_file = latest
        
    rel_path = str(latest_file.relative_to(VAULT_PATH)).replace("\\", "/")
    source_md = rel_path.replace(".json", ".md")
    
    sentiment_ctx = EquitySentimentContextDTO(
        evaluated_at=model.sentiment_context.evaluated_at,
        market_sentiment=model.sentiment_context.market_sentiment,
        key_themes=model.sentiment_context.key_themes,
        tail_risks=model.sentiment_context.tail_risks,
        sources_summary=model.sentiment_context.sources_summary,
        report_references=model.sentiment_context.report_references
    )
    
    detail = EquityDetailDTO(
        ticker=model.ticker,
        market=model.market,
        company_name=model.quant_signals.company_name,
        analysis_date=model.analysis_date,
        evaluated_at=model.quant_signals.evaluated_at,
        market_sentiment=model.sentiment_context.market_sentiment,
        composite_score=model.quant_signals.composite_score,
        data_quality_flags=getattr(model.quant_signals, "data_quality_flags", []),
        source_file=source_md,
        sidecar_file=rel_path,
        quant_signals=model.quant_signals.model_dump(),
        sentiment_context=sentiment_ctx,
        narrative_analysis=model.narrative_analysis,
        base_case_summary=model.base_case_summary,
        generated_by=model.generated_by
    )
    
    return detail


def _get_equity_news_from_vault(ticker: str) -> EquityNewsDTO | None:
    json_pattern = f"30_Knowledge_Base/Stocks/{ticker}/{ticker}*News*.json"
    json_files = list(VAULT_PATH.glob(json_pattern))

    md_pattern = f"30_Knowledge_Base/Stocks/{ticker}/{ticker}*News*.md"
    md_files = list(VAULT_PATH.glob(md_pattern))

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


@router.get("/{ticker}/news", response_model=EquityNewsDTO)
def get_equity_news(ticker: str):
    ticker = _validate_ticker(ticker)
    news_dto = _get_equity_news_from_vault(ticker)
    if not news_dto:
        raise HTTPException(status_code=404, detail="ยังไม่มีข้อมูลข่าวสำหรับหุ้นตัวนี้ในระบบ")
    return news_dto


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


@router.get("/{ticker}/notes", response_model=EquityNotesDTO)
def get_equity_notes(ticker: str, days: int = 3):
    ticker = _validate_ticker(ticker)
    ticker_upper = ticker.upper()
    vault_name = os.getenv("OBSIDIAN_VAULT_NAME", VAULT_PATH.name)

    now_utc = datetime.now(timezone.utc)
    cutoff_dt = (now_utc - timedelta(days=days)).replace(hour=0, minute=0, second=0, microsecond=0) if days > 0 else None

    # Robust regex patterns
    tag_pattern = re.compile(rf"(?i)(?<![A-Za-z0-9_])#{re.escape(ticker_upper)}\b")
    # Matches [[AAPL]], [[AAPL|Apple]], [[AAPL#Section]], [[Stocks/AAPL]]
    wikilink_pattern = re.compile(rf"(?i)\[\[(?:[^\]]+/)?{re.escape(ticker_upper)}(?:[|#][^\]]*)?\]\]")
    frontmatter_pattern = re.compile(rf"(?i)^\s*tickers?:\s*\[?.*?\b{re.escape(ticker_upper)}\b", re.MULTILINE)

    notes: list[EquityNoteItemDTO] = []
    seen_paths = set()

    # Search explicitly in News and YouTube_Summaries folders
    target_dirs = [
        VAULT_PATH / "30_Knowledge_Base" / "News",
        VAULT_PATH / "30_Knowledge_Base" / "YouTube_Summaries",
    ]

    for target_dir in target_dirs:
        if not target_dir.exists():
            continue

        for md_file in target_dir.glob("*.md"):
            rel_path = str(md_file.relative_to(VAULT_PATH)).replace("\\", "/")
            if rel_path in seen_paths:
                continue
            if md_file.name.startswith(".") or md_file.name == "index.md":
                continue

            try:
                content = md_file.read_text(encoding="utf-8", errors="ignore")
                matched_by = None
                if tag_pattern.search(content):
                    matched_by = "tag"
                elif wikilink_pattern.search(content):
                    matched_by = "wikilink"
                elif frontmatter_pattern.search(content):
                    matched_by = "frontmatter"

                if matched_by:
                    seen_paths.add(rel_path)
                    mtime = md_file.stat().st_mtime
                    note_dt = _extract_note_datetime(md_file.name, mtime, content)
                    if cutoff_dt is not None and note_dt < cutoff_dt:
                        continue

                    folder_display = str(md_file.parent.relative_to(VAULT_PATH)).replace("\\", "/")
                    lines = [l.strip() for l in content.splitlines() if l.strip() and not l.startswith("---")]
                    snippet = " ".join(lines[:3])[:250]

                    if "YouTube_Summaries" in folder_display:
                        matched_by = "youtube"
                    elif "News" in folder_display:
                        matched_by = "news"

                    notes.append(EquityNoteItemDTO(
                        title=md_file.stem,
                        folder=folder_display,
                        relative_path=rel_path,
                        obsidian_uri=f"obsidian://open?vault={vault_name}&file={rel_path}",
                        snippet=snippet,
                        modified_at=note_dt.isoformat(),
                        matched_by=matched_by
                    ))
            except Exception as e:
                log.warning("Failed to search note file %s: %s", md_file, e)

    notes.sort(key=lambda x: x.modified_at, reverse=True)
    return EquityNotesDTO(ticker=ticker_upper, total_count=len(notes), items=notes)


# ---------------------------------------------------------------------------
# Valuation Targets Overlay (Canonical SSOT DCF Endpoint)
# ---------------------------------------------------------------------------

@router.get("/{ticker}/valuation-targets", response_model=ValuationTargetsDTO)
def get_equity_valuation_targets(ticker: str) -> ValuationTargetsDTO:
    clean_ticker = _validate_ticker(ticker)

    # 1. Query Canonical SSOT Ledger
    conn = get_connection()
    row = get_latest_dcf_evaluation(conn, clean_ticker)

    # Fallback to loading latest valid sidecar if ledger row is absent, and canonicalize it
    if row is None:
        files = _get_equity_files(clean_ticker)
        res = _get_latest_sidecar_for_ticker(files, clean_ticker, strict=True)
        if res is not None:
            model, _ = res
            if model.quant_signals and model.quant_signals.dcf_result:
                dcf = model.quant_signals.dcf_result
                eval_id = f"eval_{clean_ticker}_{model.analysis_date}_{int(datetime.now(timezone.utc).timestamp())}"
                record_dcf_evaluation(
                    conn=conn,
                    evaluation_id=eval_id,
                    ticker=clean_ticker,
                    market=model.market,
                    evaluated_at=model.quant_signals.evaluated_at or f"{model.analysis_date}T00:00:00Z",
                    scenarios={k: v.model_dump() for k, v in dcf.scenarios.items()},
                    model_version="dcf_v1.0",
                    valuation_price_basis="split_adjusted_only",
                    current_price_at_eval=getattr(model, "current_price", getattr(model.quant_signals, "current_price", None)),
                    wacc_pct=dcf.wacc_pct,
                    valuation_verdict=dcf.valuation_verdict or "unknown",
                    corporate_action_evidence=[],
                    input_snapshot={"analysis_date": model.analysis_date, "observable_refs": dcf.observable_refs},
                )
                row = get_latest_dcf_evaluation(conn, clean_ticker)


    if row is None:
        return ValuationTargetsDTO(
            evaluation_id=f"eval_empty_{clean_ticker}",
            ticker=clean_ticker,
            market="TH" if clean_ticker.endswith(".BK") else "US",
            currency="THB" if clean_ticker.endswith(".BK") else "USD",
            status="unavailable",
            evaluated_at=datetime.now(timezone.utc).isoformat(),
            as_of_label="N/A",
            comparability_status="unknown",
            comparability_reasons=["No DCF evaluation found for this ticker"],
            scenarios=[],
        )

    # Parse Row
    eval_id = row["evaluation_id"]
    market = row["market"]
    currency: Literal["USD", "THB"] = "THB" if market == "TH" else "USD"
    evaluated_at_str = row["evaluated_at"]
    model_version = row["model_version"]
    val_basis = row["valuation_price_basis"]
    current_price_eval = row["current_price_at_eval"]
    wacc_pct = row["wacc_pct"]
    valuation_verdict = row["valuation_verdict"] or "unknown"

    try:
        scenarios_dict = json.loads(row["scenarios_json"])
    except Exception:
        scenarios_dict = {}

    try:
        corp_evidence = json.loads(row["corporate_action_evidence_json"]) if row["corporate_action_evidence_json"] else []
    except Exception:
        corp_evidence = []

    try:
        input_snap = json.loads(row["input_snapshot_json"]) if row["input_snapshot_json"] else {}
    except Exception:
        input_snap = {}

    macro_refs = input_snap.get("observable_refs", [])

    # Evaluate Freshness / Staleness
    try:
        eval_dt = datetime.fromisoformat(evaluated_at_str.replace("Z", "+00:00"))
    except Exception:
        eval_dt = datetime.now(timezone.utc)

    days_elapsed = (datetime.now(timezone.utc) - eval_dt).days
    status: Literal["available", "unavailable", "stale"] = "stale" if days_elapsed > 30 else "available"
    as_of_label = f"as of {eval_dt.strftime('%Y-%m-%d')}"

    # Comparability Engine: Versioned Corporate Actions Check
    comparability_status: Literal["comparable", "not_comparable", "unknown"] = "comparable"
    comparability_reasons: list[str] = []
    corp_factors: list[CorporateActionFactorDTO] = []

    for factor in corp_evidence:
        f_type = factor.get("event_type", "split")
        f_date = factor.get("effective_date", "")
        f_ratio = factor.get("ratio")
        f_amt = factor.get("amount")
        corp_factors.append(
            CorporateActionFactorDTO(
                event_type=f_type,
                effective_date=f_date,
                ratio=f_ratio,
                amount=f_amt,
            )
        )
        if f_type == "split" and f_date > evaluated_at_str[:10]:
            comparability_status = "not_comparable"
            comparability_reasons.append(f"Unadjusted stock split ({f_ratio or 'N/A'}) occurred on {f_date} after DCF evaluation date ({evaluated_at_str[:10]})")

    # Build and validate Scenario Levels
    scenario_order_valid = True
    scenarios: list[DCFScenarioLevelDTO] = []

    base_data = scenarios_dict.get("base")
    bull_data = scenarios_dict.get("bull")
    bear_data = scenarios_dict.get("bear")

    if base_data:
        scenarios.append(
            DCFScenarioLevelDTO(
                scenario_name="base",
                label="DCF Base",
                target_price=round(float(base_data.get("target_price", 0.0)), 2),
                upside_pct=round(float(base_data.get("upside_pct", 0.0)), 2) if base_data.get("upside_pct") is not None else None,
                margin_of_safety_pct=round(float(base_data.get("margin_of_safety_pct", 0.0)), 2) if base_data.get("margin_of_safety_pct") is not None else None,
                color="emerald",
            )
        )
    if bull_data:
        scenarios.append(
            DCFScenarioLevelDTO(
                scenario_name="bull",
                label="DCF Bull",
                target_price=round(float(bull_data.get("target_price", 0.0)), 2),
                upside_pct=round(float(bull_data.get("upside_pct", 0.0)), 2) if bull_data.get("upside_pct") is not None else None,
                margin_of_safety_pct=round(float(bull_data.get("margin_of_safety_pct", 0.0)), 2) if bull_data.get("margin_of_safety_pct") is not None else None,
                color="green",
            )
        )
    if bear_data:
        scenarios.append(
            DCFScenarioLevelDTO(
                scenario_name="bear",
                label="DCF Bear",
                target_price=round(float(bear_data.get("target_price", 0.0)), 2),
                upside_pct=round(float(bear_data.get("upside_pct", 0.0)), 2) if bear_data.get("upside_pct") is not None else None,
                margin_of_safety_pct=round(float(bear_data.get("margin_of_safety_pct", 0.0)), 2) if bear_data.get("margin_of_safety_pct") is not None else None,
                color="rose",
            )
        )

    # Monotonicity check: Bear <= Base <= Bull
    if bear_data and base_data and bull_data:
        bear_p = float(bear_data.get("target_price", 0.0))
        base_p = float(base_data.get("target_price", 0.0))
        bull_p = float(bull_data.get("target_price", 0.0))
        if not (bear_p <= base_p <= bull_p):
            scenario_order_valid = False
            valuation_verdict = "unknown"
            comparability_reasons.append("DCF scenario monotonicity violated: Bear <= Base <= Bull order is inconsistent")

    return ValuationTargetsDTO(
        evaluation_id=eval_id,
        ticker=clean_ticker,
        market=market,
        currency=currency,
        chart_price_basis="provider_proportional_adj_close_ratio",
        valuation_price_basis=val_basis,
        comparability_status=comparability_status,
        comparability_reasons=comparability_reasons,
        corporate_action_factors=corp_factors,
        current_price_at_eval=current_price_eval,
        evaluated_at=evaluated_at_str,
        as_of_label=as_of_label,
        model_version=model_version,
        valuation_verdict=valuation_verdict,
        wacc_pct=wacc_pct,
        macro_observable_refs=macro_refs,
        data_quality_flags=[],
        status=status,
        scenario_order_valid=scenario_order_valid,
        scenarios=scenarios,
    )


# ---------------------------------------------------------------------------
# SEC Form 4 Insider Filings Pipeline Endpoint
# ---------------------------------------------------------------------------

@router.get("/{ticker}/insider-filings", response_model=InsiderFilingsResponseDTO)
def get_equity_insider_filings(
    ticker: str,
    range: str = "1y",
    interval: str = "1d",
) -> InsiderFilingsResponseDTO:
    clean_ticker = _validate_ticker(ticker)
    market = "TH" if clean_ticker.endswith(".BK") else "US"

    conn = get_connection()
    now = datetime.now(timezone.utc)
    range_days_map = {
        "5d": 5, "1mo": 30, "3mo": 90, "6mo": 180, "1y": 365, "2y": 730, "5y": 1825, "max": 3650
    }
    days = range_days_map.get(range, 365)
    since_date = (now - timedelta(days=days)).strftime("%Y-%m-%d")

    records = get_sec_insider_filings_and_transactions(conn, clean_ticker, since_date=since_date)
    if not records and market == "US":
        sync_insider_filings_from_yfinance(conn, clean_ticker)
        records = get_sec_insider_filings_and_transactions(conn, clean_ticker, since_date=since_date)

    filing_map: dict[str, dict] = {}

    d30_cutoff = (now - timedelta(days=30)).strftime("%Y-%m-%d")
    d90_cutoff = (now - timedelta(days=90)).strftime("%Y-%m-%d")
    d180_cutoff = (now - timedelta(days=180)).strftime("%Y-%m-%d")

    net_shares_30d = 0.0
    net_shares_90d = 0.0
    net_shares_180d = 0.0

    buyers_in_30d: set[str] = set()

    for r in records:
        acc = r["accession_number"]
        tx_date = r["transaction_date"]
        acq_disp = r["acquired_or_disposed"]
        shares = float(r["shares"])
        tx_code = r["transaction_code"]

        share_delta = shares if acq_disp == "A" else -shares
        if tx_date >= d30_cutoff:
            net_shares_30d += share_delta
            if acq_disp == "A" and r["reporting_owner_name"]:
                buyers_in_30d.add(r["reporting_owner_name"])
        if tx_date >= d90_cutoff:
            net_shares_90d += share_delta
        if tx_date >= d180_cutoff:
            net_shares_180d += share_delta

        if acc not in filing_map:
            try:
                dt = datetime.strptime(tx_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                ts = int(dt.timestamp() * 1000)
            except Exception:
                ts = int(now.timestamp() * 1000)

            filing_map[acc] = {
                "accession_number": acc,
                "issuer_cik": r["issuer_cik"],
                "ticker": clean_ticker,
                "filing_url": r["filing_url"],
                "filed_at": r["filed_at"] or tx_date,
                "timestamp": ts,
                "reporting_owner_cik": r["reporting_owner_cik"],
                "reporting_owner_name": r["reporting_owner_name"],
                "is_director": bool(r["is_director"]),
                "is_officer": bool(r["is_officer"]),
                "is_ten_percent_owner": bool(r["is_ten_percent_owner"]),
                "officer_title": r["officer_title"],
                "is_amendment": bool(r["is_amendment"]),
                "amends_accession_number": r["amends_accession_number"],
                "is_cluster_buy": False,
                "transactions": [],
            }

        filing_map[acc]["transactions"].append(
            InsiderTransactionDTO(
                transaction_id=r["transaction_id"],
                transaction_date=tx_date,
                transaction_code=tx_code,
                shares=shares,
                price_per_share=float(r["price_per_share"]),
                acquired_or_disposed="A" if acq_disp == "A" else "D",
                shares_owned_following=r["shares_owned_following"],
                ownership_nature=r["ownership_nature"],
                is_derivative=bool(r["is_derivative"]),
                normalized_weight=float(r["normalized_weight"] or 1.0),
            )
        )

    cluster_buy_signal = len(buyers_in_30d) >= 3
    filings_list: list[InsiderFilingDTO] = []
    cluster_buy_count = 0

    for f_data in filing_map.values():
        if cluster_buy_signal and f_data["reporting_owner_name"] in buyers_in_30d:
            f_data["is_cluster_buy"] = True
            cluster_buy_count += 1
        filings_list.append(InsiderFilingDTO(**f_data))

    filings_list.sort(key=lambda x: x.timestamp, reverse=True)

    return InsiderFilingsResponseDTO(
        ticker=clean_ticker,
        market=market,
        requested_range=range,
        interval=interval,
        net_shares_30d=round(net_shares_30d, 2),
        net_shares_90d=round(net_shares_90d, 2),
        net_shares_180d=round(net_shares_180d, 2),
        cluster_buy_count=cluster_buy_count,
        total_filings_count=len(filings_list),
        filings=filings_list,
    )


# ---------------------------------------------------------------------------
# Analyst Context Endpoint (Consensus Target + Earnings Date + EPS History)
# ---------------------------------------------------------------------------

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


@router.get("/{ticker}/analyst-context", response_model=AnalystContextDTO)
def get_equity_analyst_context(ticker: str) -> AnalystContextDTO:
    """ดึงข้อมูล Consensus Target Price, Next Earnings Date Countdown, และ EPS History พร้อม Cache 24h"""
    import yfinance as yf
    from zoneinfo import ZoneInfo

    clean_ticker = _validate_ticker(ticker)
    resolved = resolve_asset(clean_ticker)
    provider_symbol = resolved.provider_symbol or clean_ticker
    market: Literal["TH", "US"] = "TH" if (resolved.market == "TH" or provider_symbol.endswith(".BK")) else "US"
    currency: Literal["USD", "THB"] = "THB" if market == "TH" else "USD"
    exchange_tz = "Asia/Bangkok" if market == "TH" else "America/New_York"

    def _days_to_earnings(next_date_str: str | None) -> int | None:
        if not next_date_str:
            return None
        try:
            tz = ZoneInfo(exchange_tz)
            today = datetime.now(tz).date()
            target = datetime.strptime(next_date_str, "%Y-%m-%d").date()
            diff = (target - today).days
            return diff if diff >= 0 else None
        except Exception:
            return None

    conn = get_connection()
    now_wall = time.time()
    now_mono = time.monotonic()
    CACHE_TTL = 24 * 3600

    # 1. Check SQLite Cache
    cached = get_analyst_context_cache(conn, clean_ticker)
    if cached:
        cached_at = datetime.fromisoformat(cached["synced_at"]).timestamp()
        if (now_wall - cached_at) < CACHE_TTL:
            return AnalystContextDTO(
                **{**cached, "days_to_earnings": _days_to_earnings(cached.get("next_earnings_date"))}
            )

    # 2. Acquire per-provider-symbol single-flight lock
    clean_prov_sym = provider_symbol.strip().upper()
    key_lock = _get_analyst_lock(clean_prov_sym)

    with key_lock:
        # Double-check cache inside lock
        cached = get_analyst_context_cache(conn, clean_ticker)
        if cached:
            cached_at = datetime.fromisoformat(cached["synced_at"]).timestamp()
            if (now_wall - cached_at) < CACHE_TTL:
                return AnalystContextDTO(
                    **{**cached, "days_to_earnings": _days_to_earnings(cached.get("next_earnings_date"))}
                )

        # Check in-memory burst failure cooldown
        with _ANALYST_LOCK:
            if clean_prov_sym in _ANALYST_BURST_FAIL_CACHE:
                if (now_mono - _ANALYST_BURST_FAIL_CACHE[clean_prov_sym]) < ANALYST_BURST_FAIL_TTL:
                    if cached and ((now_wall - datetime.fromisoformat(cached["synced_at"]).timestamp()) <= MAX_STALE_AGE_SECONDS):
                        return AnalystContextDTO(
                            **{**cached, "data_status": "stale", "days_to_earnings": _days_to_earnings(cached.get("next_earnings_date"))}
                        )
                    return AnalystContextDTO(
                        ticker=clean_ticker,
                        provider_symbol=provider_symbol,
                        market=market,
                        currency=currency,
                        exchange_tz=exchange_tz,
                        target_mean=None,
                        target_high=None,
                        target_low=None,
                        num_analysts=None,
                        next_earnings_date=None,
                        days_to_earnings=None,
                        earnings_history=[],
                        source_as_of=datetime.now(ZoneInfo(exchange_tz)).isoformat(),
                        data_status="unavailable",
                        provider_tier="best_effort",
                        synced_at=datetime.now(timezone.utc).isoformat(),
                    )

        target_mean: float | None = None
        target_high: float | None = None
        target_low: float | None = None
        num_analysts: int | None = None
        next_earnings_date: str | None = None
        eps_history: list[dict] = []
        fetch_errors: list[str] = []

        # 3. Fetch Analyst Target Prices
        try:
            tk = yf.Ticker(provider_symbol)
            apt = tk.get_analyst_price_targets()
            if apt and isinstance(apt, dict):
                target_mean = finite_or_none(apt.get("mean") or apt.get("targetMeanPrice"))
                target_high = finite_or_none(apt.get("high") or apt.get("targetHighPrice"))
                target_low = finite_or_none(apt.get("low") or apt.get("targetLowPrice"))
            info = tk.info or {}
            num_analysts = positive_int_or_none(info.get("numberOfAnalystOpinions"))
            if target_mean is None and info.get("targetMeanPrice") is not None:
                target_mean = finite_or_none(info.get("targetMeanPrice"))
                target_high = finite_or_none(info.get("targetHighPrice"))
                target_low = finite_or_none(info.get("targetLowPrice"))
        except Exception as e:
            log.warning("Analyst price targets fetch failed for %s: %s", provider_symbol, e)
            fetch_errors.append("analyst_targets")

        # 4. Fetch Next Earnings Date via shared calendar helper
        try:
            cal = get_asset_calendar(provider_symbol)
            earnings_dates_raw = cal.get("Earnings Date") if isinstance(cal, dict) else None
            if isinstance(earnings_dates_raw, list) and earnings_dates_raw:
                from datetime import date as _date
                tz = ZoneInfo(exchange_tz)
                today = datetime.now(tz).date()
                candidates: list[_date] = []
                for d in earnings_dates_raw:
                    if hasattr(d, "year"):
                        candidates.append(d.date() if hasattr(d, "hour") else d)
                    elif isinstance(d, str):
                        try:
                            candidates.append(datetime.strptime(d[:10], "%Y-%m-%d").date())
                        except ValueError:
                            pass
                future = [c for c in candidates if c >= today]
                if future:
                    next_earnings_date = min(future).strftime("%Y-%m-%d")
        except Exception as e:
            log.warning("Calendar fetch failed for %s: %s", provider_symbol, e)
            fetch_errors.append("calendar")

        # 5. Fetch Historical EPS via shared earnings service
        earn_result = fetch_earnings_dates(provider_symbol, exchange_tz)
        eps_history = earn_result.rows
        if earn_result.status == "failed":
            fetch_errors.append("earnings_history")

        # Handle burst failure cooldown update if any failure occurred
        if fetch_errors:
            with _ANALYST_LOCK:
                _ANALYST_BURST_FAIL_CACHE[clean_prov_sym] = time.monotonic()

        # 6. Evaluate Stale Fallback vs Fresh Partial Evaluation
        if fetch_errors and cached:
            cached_at = datetime.fromisoformat(cached["synced_at"]).timestamp()
            if (now_wall - cached_at) <= MAX_STALE_AGE_SECONDS:
                return AnalystContextDTO(
                    **{**cached, "data_status": "stale", "days_to_earnings": _days_to_earnings(cached.get("next_earnings_date"))}
                )

        has_fresh_target = target_mean is not None
        has_fresh_calendar = next_earnings_date is not None
        has_fresh_eps = bool(eps_history)

        if not has_fresh_target and not has_fresh_calendar and not has_fresh_eps:
            data_status: Literal["ok", "partial", "stale", "unavailable"] = "unavailable"
        elif fetch_errors or not has_fresh_target or not has_fresh_calendar:
            data_status = "partial"
        else:
            data_status = "ok"

        source_as_of = datetime.now(ZoneInfo(exchange_tz)).isoformat()
        synced_at_str = datetime.now(timezone.utc).isoformat()

        # 7. Upsert to SQLite cache only on ok or partial
        if data_status in ("ok", "partial"):
            upsert_analyst_context_cache(
                conn,
                clean_ticker,
                {
                    "provider_symbol": provider_symbol,
                    "market": market,
                    "currency": currency,
                    "exchange_tz": exchange_tz,
                    "target_mean": target_mean,
                    "target_high": target_high,
                    "target_low": target_low,
                    "num_analysts": num_analysts,
                    "next_earnings_date": next_earnings_date,
                    "earnings_history": eps_history,
                    "source_as_of": source_as_of,
                    "data_status": data_status,
                    "synced_at": now_wall,
                },
            )

        return AnalystContextDTO(
            ticker=clean_ticker,
            provider_symbol=provider_symbol,
            market=market,
            currency=currency,
            exchange_tz=exchange_tz,
            target_mean=target_mean,
            target_high=target_high,
            target_low=target_low,
            num_analysts=num_analysts,
            next_earnings_date=next_earnings_date,
            days_to_earnings=_days_to_earnings(next_earnings_date),
            earnings_history=[EarningsHistoryEntryDTO(**r) for r in eps_history],
            source_as_of=source_as_of,
            data_status=data_status,
            provider_tier="best_effort",
            synced_at=synced_at_str,
        )








