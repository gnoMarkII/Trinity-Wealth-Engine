import json
import os
from datetime import datetime
from pathlib import Path
from schemas.micro_quant_schemas import MicroQuantOutput
from tools._atomic_io import _atomic_write_to
from tools.archivist.maintenance_guard import assert_write_allowed
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
    from tools.archivist.vault_paths import VaultPaths
    vp = VaultPaths(vault_path)
    if vp.layout_version >= 2:
        sidecar_dir = vault_path / "30_Knowledge_Base" / "Stocks" / ticker / "Analysis"
    else:
        sidecar_dir = vault_path / "30_Knowledge_Base" / "Stocks" / ticker
    assert_write_allowed(sidecar_dir)
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

    # Clean system storage copy (.system/sidecars/)
    sys_dir = vault_path / ".system" / "sidecars" / ticker
    assert_write_allowed(sys_dir)
    sys_dir.mkdir(parents=True, exist_ok=True)
    sys_date_path = sys_dir / f"{ticker} Equity Analysis {date_str}.json"
    _atomic_write_to(sys_date_path, json_data)
    _atomic_write_to(sys_dir / f"{ticker} Equity Analysis latest.json", json_data)

    # Index in SQLite sidecar_catalog with fallback
    try:
        from tools.archivist.catalog_adapter import SqliteNoteCatalogAdapter
        cat = SqliteNoteCatalogAdapter(vault_root=vault_path)
        cur_time = datetime.now().timestamp()
        full_hash = hashlib.sha256(json_data.encode("utf-8")).hexdigest()
        data_len = len(json_data.encode("utf-8"))

        # Index primary (user knowledge base)
        primary_rel = primary_path.relative_to(vault_path).as_posix()
        cat.upsert_sidecar(
            ticker=ticker,
            evaluation_date=date_str,
            relative_path=primary_rel,
            storage_tier="user",
            mtime=cur_time,
            file_size=data_len,
            sha256=full_hash,
        )

        # Index system mirror
        sys_rel = sys_date_path.relative_to(vault_path).as_posix()
        cat.upsert_sidecar(
            ticker=ticker,
            evaluation_date=date_str,
            relative_path=sys_rel,
            storage_tier="system",
            mtime=cur_time,
            file_size=data_len,
            sha256=full_hash,
        )
    except Exception:
        # Sidecar storage succeeds even if catalog hook encounters temporary lock
        pass
