"""Versioned, append-only Performance Log CSV adapter."""
from __future__ import annotations

import csv
import io
import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from typing import Optional, List, Dict
from uuid import uuid4

from core.logger import get_logger
from tools._atomic_io import _atomic_write_to
from tools.archivist.maintenance_guard import assert_write_allowed
from tools.portfolio.domain.validator import validate_portfolio_id
from tools.portfolio.ports.performance_port import PerformanceRepositoryPort
from .paths import get_performance_filepath, _PERFORMANCE_LOG_HEADER
from filelock import Timeout
from .repository_adapter import _get_portfolio_lock

log = get_logger(__name__)
_MISSING_FLOW = "MISSING_FLOW_HISTORY"


def _decimal(value, *, default: Optional[Decimal] = None) -> Optional[Decimal]:
    if value is None or str(value).strip() == "":
        return default
    try:
        result = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"invalid performance decimal: {value!r}") from exc
    if not result.is_finite():
        raise ValueError("performance values must be finite decimals")
    return result


def _money(value) -> str:
    parsed = _decimal(value, default=Decimal("0"))
    return f"{parsed.quantize(Decimal('0.01')):.2f}"


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("performance timestamps must include a UTC offset")
    return parsed.astimezone(timezone.utc)


class MarkdownPerformanceAdapter(PerformanceRepositoryPort):
    """Performance snapshots are append-only revisions; legacy rows remain readable."""

    def upsert_snapshot(self, portfolio_id: str, row: Dict) -> None:
        pid = validate_portfolio_id(portfolio_id)
        with _get_portfolio_lock(pid):
            path = get_performance_filepath(pid)
            assert_write_allowed(path)
            path.parent.mkdir(parents=True, exist_ok=True)

            now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            date_value = str(row.get("Date") or now[:10])
            datetime.strptime(date_value[:10], "%Y-%m-%d")
            existing: List[Dict[str, str]] = []
            if path.exists() and path.stat().st_size:
                with path.open("r", encoding="utf-8", newline="") as handle:
                    reader = csv.DictReader(handle)
                    for raw in reader:
                        if not raw or not raw.get("Date"):
                            continue
                        item = {column: str(raw.get(column) or "") for column in _PERFORMANCE_LOG_HEADER}
                        if not item["Flow_Coverage_Status"]:
                            item["Flow_Coverage_Status"] = _MISSING_FLOW
                        existing.append(item)

            revisions = [
                int(item["Revision"])
                for item in existing
                if item["Date"][:10] == date_value[:10] and item["Revision"].isdigit()
            ]
            revision = max(revisions, default=0) + 1
            cutoff = str(row.get("Cutoff_UTC") or row.get("cutoff_utc") or now)
            recorded = str(row.get("Recorded_At_UTC") or row.get("recorded_at_utc") or now)
            cutoff = _parse_utc(cutoff).isoformat().replace("+00:00", "Z")
            recorded = _parse_utc(recorded).isoformat().replace("+00:00", "Z")

            nav = _decimal(row.get("Total_NAV_THB", row.get("Total_NAV")), default=Decimal("0"))
            net_flow = _decimal(row.get("Net_External_Flow_THB", row.get("Net_Cash_Flow")))
            coverage = str(row.get("Flow_Coverage_Status") or _MISSING_FLOW).upper()
            if coverage == "COMPLETE" and net_flow is None:
                raise ValueError("COMPLETE flow coverage requires an explicit net external flow, including zero")
            if coverage != "COMPLETE":
                net_flow = None
                if coverage == "":
                    coverage = _MISSING_FLOW

            asset_classes = row.get("Asset_Class_Values_THB")
            if isinstance(asset_classes, dict):
                asset_classes_json = json.dumps(asset_classes, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)
            elif isinstance(asset_classes, str):
                asset_classes_json = asset_classes
            else:
                asset_classes_json = ""

            event_ids = row.get("Cash_Flow_Event_IDs_JSON", row.get("cash_flow_event_ids", []))
            if isinstance(event_ids, str):
                try:
                    parsed_ids = json.loads(event_ids)
                    event_ids_json = json.dumps(parsed_ids, ensure_ascii=False, separators=(",", ":"))
                except json.JSONDecodeError as exc:
                    raise ValueError("cash flow event IDs must be valid JSON") from exc
            else:
                event_ids_json = json.dumps(list(event_ids or []), ensure_ascii=False, separators=(",", ":"))

            snapshot_id = str(row.get("Snapshot_ID") or row.get("snapshot_id") or f"PERF-{uuid4().hex}")
            source_sequence = row.get("Source_State_Sequence", row.get("source_state_sequence"))
            source_hash = row.get("Source_State_SHA256", row.get("source_state_sha256"))
            values = {
                "Date": date_value[:10],
                "Total_NAV": _money(nav),
                "Total_Cost": _money(row.get("Total_Cost")),
                "Unrealized_PnL": _money(row.get("Unrealized_PnL")),
                "Cash_Balance": _money(row.get("Cash_Balance")),
                "Realized_PnL_YTD": _money(row.get("Realized_PnL_YTD")),
                "Passive_Income_YTD": _money(row.get("Passive_Income_YTD")),
                "Asset_Class_Values_THB": asset_classes_json,
                "Snapshot_ID": snapshot_id,
                "Revision": str(revision),
                "Cutoff_UTC": cutoff,
                "Recorded_At_UTC": recorded,
                "Source_State_Sequence": str(source_sequence) if source_sequence is not None else "",
                "Source_State_SHA256": str(source_hash or ""),
                "Total_NAV_THB": _money(nav),
                "Net_External_Flow_THB": _money(net_flow) if net_flow is not None else "",
                "Flow_Coverage_Status": coverage,
                "Cash_Flow_Event_IDs_JSON": event_ids_json,
            }
            if any(item["Snapshot_ID"] == snapshot_id for item in existing):
                raise ValueError(f"performance Snapshot_ID already exists: {snapshot_id}")
            existing.append(values)

            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=_PERFORMANCE_LOG_HEADER, lineterminator="\n")
            writer.writeheader()
            writer.writerows(existing)
            _atomic_write_to(path, output.getvalue())

    def read_history(
        self,
        portfolio_id: str = "default",
        days: Optional[int] = None,
        as_of_utc: Optional[str] = None,
    ) -> List[Dict]:
        pid = validate_portfolio_id(portfolio_id)
        path = get_performance_filepath(pid)
        if not path.exists() or not path.stat().st_size:
            return []
        as_of = _parse_utc(as_of_utc) if as_of_utc else datetime.now(timezone.utc)
        cutoff_date = (as_of.date() - timedelta(days=days)).isoformat() if days else None

        def parse_rows() -> List[Dict]:
            rows: List[Dict] = []
            with path.open("r", encoding="utf-8", newline="") as handle:
                for raw in csv.DictReader(handle):
                    date_value = str(raw.get("Date") or "")
                    if not date_value:
                        continue
                    if cutoff_date and date_value[:10] < cutoff_date:
                        continue
                    item: Dict[str, object] = {
                        key: raw.get(key) if key in raw else None
                        for key in _PERFORMANCE_LOG_HEADER
                    }
                    item["Date"] = date_value[:10]
                    item["Flow_Coverage_Status"] = str(item.get("Flow_Coverage_Status") or _MISSING_FLOW).upper()
                    item["Total_NAV_THB"] = item.get("Total_NAV_THB") or item.get("Total_NAV")
                    item["Net_External_Flow_THB"] = item.get("Net_External_Flow_THB") or None
                    if item["Flow_Coverage_Status"] != "COMPLETE":
                        item["Net_External_Flow_THB"] = None
                    cutoff_value = str(item.get("Cutoff_UTC") or "").strip()
                    if cutoff_value:
                        try:
                            if _parse_utc(cutoff_value) > as_of:
                                continue
                        except ValueError:
                            item["Snapshot_Quality"] = "INVALID_CUTOFF"
                            continue
                    else:
                        item["Snapshot_Quality"] = "LEGACY_CUTOFF_UNKNOWN"
                        if date_value[:10] > as_of.date().isoformat():
                            continue
                    raw_assets = item.get("Asset_Class_Values_THB")
                    if raw_assets:
                        try:
                            item["Asset_Class_Values_THB"] = json.loads(str(raw_assets))
                        except (TypeError, ValueError):
                            item["Asset_Class_Values_THB"] = None
                    else:
                        item["Asset_Class_Values_THB"] = None
                    raw_ids = item.get("Cash_Flow_Event_IDs_JSON")
                    try:
                        item["Cash_Flow_Event_IDs_JSON"] = json.loads(str(raw_ids)) if raw_ids else []
                    except (TypeError, ValueError):
                        item["Cash_Flow_Event_IDs_JSON"] = None
                        item["Flow_Coverage_Status"] = "INVALID_FLOW_DATA"
                    rows.append(item)
            return rows

        try:
            with _get_portfolio_lock(pid):
                return parse_rows()
        except Timeout:
            log.warning("Performance read_history timed out waiting for lock on %s; falling back to atomic read", pid)
            return parse_rows()
