"""FastAPI Sub-router for Equity Meta, Latest Summaries, and Note Content."""
import os
import sys
from pathlib import Path
from typing import List
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException

from api.auth import require_session
from api.schemas import EquitySummaryDTO, EquityNoteContentDTO
from tools.archivist.core import VAULT_PATH
from api.routers.equity.common import (
    _get_equity_files,
    _get_latest_sidecar_for_ticker,
)

router = APIRouter()


@router.get("/latest", response_model=List[EquitySummaryDTO])
def get_latest_equities():
    import sys
    routes_equity = sys.modules.get("api.routes_equity")
    vault_path = getattr(routes_equity, "VAULT_PATH", VAULT_PATH) if routes_equity else VAULT_PATH

    files = _get_equity_files()
    
    ticker_files = {}
    for f in files:
        ticker = f.parent.name
        if ticker not in ticker_files:
            ticker_files[ticker] = []
        ticker_files[ticker].append(f)
            
    results = []
    for ticker, paths in ticker_files.items():
        latest = _get_latest_sidecar_for_ticker(paths, ticker, strict=False)
        if latest:
            model, file_path = latest
            rel_path = str(file_path.relative_to(vault_path)).replace("\\", "/")
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


@router.get("/notes/content", response_model=EquityNoteContentDTO)
def get_equity_note_content(rel_path: str):
    if ".." in rel_path or rel_path.startswith("/") or rel_path.startswith("\\"):
        raise HTTPException(status_code=400, detail="Invalid path format")

    import sys
    routes_equity = sys.modules.get("api.routes_equity")
    vault_path = getattr(routes_equity, "VAULT_PATH", VAULT_PATH) if routes_equity else VAULT_PATH

    vault_resolved = vault_path.resolve()
    target_path = (vault_path / rel_path).resolve()

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
