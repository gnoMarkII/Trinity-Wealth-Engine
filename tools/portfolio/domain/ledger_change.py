from typing import Literal, Optional, List, Dict
from pydantic import BaseModel, ConfigDict


class LedgerChange(BaseModel):
    """Encapsulates mutations to Trades_Log.csv within a Unit of Work."""
    model_config = ConfigDict(extra="allow")

    kind: Literal["append", "replace_all", "unchanged"] = "unchanged"
    row: Optional[Dict] = None          # Single row dict for 'append'
    rows: Optional[List[Dict]] = None   # Full list of row dicts for 'replace_all'
    tx_id: Optional[str] = None         # Transaction ID for idempotency checks
