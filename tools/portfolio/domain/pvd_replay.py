"""Projection of PVD snapshot rows into exact current holding values."""
import json
from datetime import date
from decimal import Decimal

from .pvd_models import PvdSnapshotReplayResult, PvdStatementMetadata


def replay_pvd_snapshot_rows(rows: list[dict]) -> PvdSnapshotReplayResult | None:
    snapshots = [row for row in rows if str(row.get("Action") or row.get("action") or "").upper() == "PVD_SNAPSHOT"]
    if not snapshots:
        return None

    def field(row, title):
        return row.get(title, row.get(title.lower()))

    snapshots.sort(key=lambda row: (str(field(row, "Timestamp") or ""), str(field(row, "Transaction_ID") or "")))
    last = snapshots[-1]
    raw_metadata = field(last, "PVD_Metadata_JSON") or field(last, "pvd_metadata_json")
    if not raw_metadata:
        raise ValueError("PVD_SNAPSHOT ledger row is missing PVD_Metadata_JSON")
    try:
        metadata_data = json.loads(raw_metadata) if isinstance(raw_metadata, str) else raw_metadata
        metadata = PvdStatementMetadata.model_validate(metadata_data)
    except Exception as exc:
        raise ValueError(f"Invalid PVD statement metadata in ledger: {exc}") from exc

    active = [
        row for row in snapshots
        if str(field(row, "Confirmation_No") or "") == str(field(last, "Confirmation_No") or "")
    ]
    latest_symbols = {str(field(row, "Symbol") or "").upper() for row in active}
    historical_symbols = {str(field(row, "Symbol") or "").upper() for row in snapshots}
    return PvdSnapshotReplayResult(
        as_of_date=metadata.as_of_date,
        statement_fingerprint=metadata.semantic_fingerprint,
        active_rows=active,
        closed_symbols=sorted(historical_symbols - latest_symbols),
        latest_metadata=metadata,
    )


def apply_pvd_snapshot_rows(holdings: list, rows: list[dict]) -> None:
    """Set the supplied portfolio Holding objects to the latest target snapshot."""
    result = replay_pvd_snapshot_rows(rows)
    if result is None:
        return

    from .pvd_models import PvdHoldingValuation
    from .models import Holding

    for row in result.active_rows:
        get = lambda name: row.get(name, row.get(name.lower()))
        symbol = str(get("Symbol") or "").upper()
        if not symbol.startswith("PVD_"):
            raise ValueError(f"Invalid PVD symbol in snapshot: {symbol}")
        try:
            valuation = PvdHoldingValuation(
                fund_code=result.latest_metadata.fund_code,
                sub_fund_code=symbol.removeprefix("PVD_"),
                units=Decimal(str(get("Units") or "0")),
                statement_nav=Decimal(str(get("Price") or "0")),
                statement_market_value=Decimal(str(get("Gross_Amount") or "0")),
                reconciliation_adjustment=Decimal(str(get("Reconciliation_Adjustment") or "0")),
                cost_basis=Decimal(str(get("Cost_THB") or "0")),
                as_of_date=result.as_of_date,
                statement_fingerprint=result.statement_fingerprint,
                archive_id=result.latest_metadata.archive_id,
            )
        except Exception as exc:
            raise ValueError(f"Invalid snapshot values for {symbol}: {exc}") from exc
        holding = next((h for h in holdings if h.symbol.upper() == symbol), None)
        target_bucket = holding.bucket_id if (holding and holding.bucket_id) else (get("Bucket_ID") or None)
        if holding is None:
            holding = Holding(symbol=symbol, asset_type="PVD", units=float(valuation.units), bucket_id=target_bucket)
            holdings.append(holding)
        holding.asset_type = "PVD"
        holding.bucket_id = target_bucket
        holding.pvd_valuation = valuation
        holding.units = float(valuation.units)
        holding.avg_cost_thb = float(valuation.cost_basis / valuation.units) if valuation.units else 0.0
        holding.current_price_thb = float(valuation.statement_nav)
        holding.status = "active" if valuation.units or valuation.statement_market_value or valuation.cost_basis else "archived"
        if holding.status == "archived" and not holding.archived_at:
            holding.archived_at = result.as_of_date.isoformat()
        elif holding.status == "active":
            holding.archived_at = None
