import json
import os
from datetime import datetime
from pathlib import Path
from schemas.micro_quant_schemas import MicroQuantOutput
from tools._atomic_io import _atomic_write_to
from langsmith import traceable

@traceable(run_type="tool")
def write_equity_sidecar(output: MicroQuantOutput) -> None:
    """
    Writes the MicroQuantOutput to a JSON sidecar file in the vault.
    Path: 30_Knowledge_Base/Stocks/<TICKER>/<TICKER> Equity Analysis <DATE>.json
    """
    ticker = output.ticker.upper()
    date_str = output.analysis_date
    
    # Validate date format (YYYY-MM-DD)
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        raise ValueError(f"Invalid analysis_date format: {date_str}. Expected YYYY-MM-DD.")
        
    # Sanitize ticker for path (basic validation)
    # The actual deep validation will be done in API routes, but we do basic safety here.
    if not ticker.isalnum() and not all(c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-_" for c in ticker):
        raise ValueError(f"Invalid ticker format for path: {ticker}")
        
    import hashlib
    vault_path = Path(os.getenv("OBSIDIAN_VAULT_PATH", "./memories"))
    sidecar_dir = vault_path / "30_Knowledge_Base" / "Stocks" / ticker
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    
    # Serialize to dict without mutating input output object
    payload = output.model_dump(mode="json")
    payload["ticker"] = ticker
    if "quant_signals" in payload and payload["quant_signals"]:
        payload["quant_signals"]["ticker"] = ticker
    
    # Serialize to JSON string
    json_data = json.dumps(payload, ensure_ascii=False, indent=2)
    content_hash = hashlib.sha256(json_data.encode("utf-8")).hexdigest()[:6]

    # Primary date path
    primary_path = sidecar_dir / f"{ticker} Equity Analysis {date_str}.json"
    _atomic_write_to(primary_path, json_data)

    # Collision-free revision path
    revision_path = sidecar_dir / f"{ticker} Equity Analysis {date_str}_{content_hash}.json"
    _atomic_write_to(revision_path, json_data)

    # Atomic latest pointer
    latest_path = sidecar_dir / f"{ticker} Equity Analysis latest.json"
    _atomic_write_to(latest_path, json_data)
