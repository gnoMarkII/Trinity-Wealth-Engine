"""FastAPI Sub-router for Equity Entity details, News, and Notes."""
import os
import re
import logging
from pathlib import Path
from datetime import datetime, timezone, timedelta
from fastapi import APIRouter, Depends, HTTPException

from api.auth import require_session
from api.schemas import (
    EquityDetailDTO,
    EquitySentimentContextDTO,
    EquityNewsDTO,
    EquityNotesDTO,
    EquityNoteItemDTO,
)
from tools.archivist.core import VAULT_PATH
from api.routers.equity.common import (
    _validate_ticker,
    _get_equity_files,
    _get_latest_sidecar_for_ticker,
    _get_equity_news_from_vault,
    _extract_note_datetime,
)

log = logging.getLogger(__name__)

router = APIRouter()


@router.get("/{ticker}", response_model=EquityDetailDTO)
def get_equity_detail(ticker: str):
    import sys
    routes_equity = sys.modules.get("api.routes_equity")
    vault_path = getattr(routes_equity, "VAULT_PATH", VAULT_PATH) if routes_equity else VAULT_PATH

    ticker = _validate_ticker(ticker)
    files = _get_equity_files(ticker)
    
    if not files:
        raise HTTPException(status_code=404, detail="Equity not found")
        
    latest = _get_latest_sidecar_for_ticker(files, ticker, strict=True)
    if not latest:
        raise HTTPException(status_code=503, detail="Service Unavailable: Data corrupted or ticker mismatch")
        
    model, latest_file = latest
        
    rel_path = str(latest_file.relative_to(vault_path)).replace("\\", "/")
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


@router.get("/{ticker}/news", response_model=EquityNewsDTO)
def get_equity_news(ticker: str):
    ticker = _validate_ticker(ticker)
    news_dto = _get_equity_news_from_vault(ticker)
    if not news_dto:
        raise HTTPException(status_code=404, detail="ยังไม่มีข้อมูลข่าวสำหรับหุ้นตัวนี้ในระบบ")
    return news_dto


@router.get("/{ticker}/notes", response_model=EquityNotesDTO)
def get_equity_notes(ticker: str, days: int = 3):
    import sys
    routes_equity = sys.modules.get("api.routes_equity")
    vault_path = getattr(routes_equity, "VAULT_PATH", VAULT_PATH) if routes_equity else VAULT_PATH

    ticker = _validate_ticker(ticker)
    ticker_upper = ticker.upper()
    vault_name = os.getenv("OBSIDIAN_VAULT_NAME", vault_path.name)

    now_utc = datetime.now(timezone.utc)
    cutoff_dt = (now_utc - timedelta(days=days)).replace(hour=0, minute=0, second=0, microsecond=0) if days > 0 else None

    tag_pattern = re.compile(rf"(?i)(?<![A-Za-z0-9_])#{re.escape(ticker_upper)}\b")
    wikilink_pattern = re.compile(rf"(?i)\[\[(?:[^\]]+/)?{re.escape(ticker_upper)}(?:[|#][^\]]*)?\]\]")
    frontmatter_pattern = re.compile(rf"(?i)^\s*tickers?:\s*\[?.*?\b{re.escape(ticker_upper)}\b", re.MULTILINE)

    notes: list[EquityNoteItemDTO] = []
    seen_paths = set()

    target_dirs = [
        vault_path / "30_Knowledge_Base" / "News",
        vault_path / "30_Knowledge_Base" / "YouTube_Summaries",
    ]

    for target_dir in target_dirs:
        if not target_dir.exists():
            continue

        for md_file in target_dir.glob("*.md"):
            rel_path = str(md_file.relative_to(vault_path)).replace("\\", "/")
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

                    folder_display = str(md_file.parent.relative_to(vault_path)).replace("\\", "/")
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
