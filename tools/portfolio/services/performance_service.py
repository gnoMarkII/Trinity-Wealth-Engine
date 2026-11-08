"""Portfolio performance snapshots with explicit external-flow coverage."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from typing import Optional, List, Dict
from uuid import uuid4

from tools.portfolio.domain.constants import _MONEY_DP
from tools.portfolio.domain.calculations import compute_total_cost, recalc_all
from tools.portfolio.domain.validator import validate_portfolio_id
from tools.portfolio.ports.repository_port import PortfolioRepositoryPort
from tools.portfolio.ports.performance_port import PerformanceRepositoryPort
from tools.portfolio.ports.price_port import MarketPricePort


def _require_fx(state) -> float:
    fx = state.fx_rates.get("USDTHB")
    if fx is None or fx <= 0:
        raise ValueError("????? fx_rates.USDTHB ??? valid ?? portfolio")
    return fx


def _decimal(value) -> Decimal:
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"invalid performance value: {value!r}") from exc
    if not parsed.is_finite():
        raise ValueError("performance value must be finite")
    return parsed


class PortfolioPerformanceService:
    """Creates versioned snapshots and carries P&L only across covered flow windows."""

    def __init__(
        self,
        repo: PortfolioRepositoryPort,
        perf_repo: PerformanceRepositoryPort,
        price_provider: MarketPricePort,
    ) -> None:
        self.repo = repo
        self.perf_repo = perf_repo
        self.price_provider = price_provider

    def _history_as_of(self, portfolio_id: str, cutoff_utc: str) -> List[Dict]:
        try:
            return self.perf_repo.read_history(
                portfolio_id=portfolio_id, days=None, as_of_utc=cutoff_utc
            )
        except TypeError:
            return self.perf_repo.read_history(portfolio_id=portfolio_id, days=None)

    def record_performance_snapshot(self, refresh_prices: bool = True, portfolio_id: str = "default") -> str:
        try:
            pid = validate_portfolio_id(portfolio_id)
            cutoff = datetime.now(timezone.utc)
            cutoff_utc = cutoff.isoformat().replace("+00:00", "Z")
            from tools.portfolio.domain.ledger_change import LedgerChange

            with self.repo.unit_of_work(pid) as uow:
                state = uow.load_state()
                if refresh_prices and self.price_provider:
                    self.price_provider.refresh_portfolio_prices(state)
                recalc_all(state)
                uow.commit(state, LedgerChange(kind="unchanged"))

            fx = _require_fx(state)
            total_nav = _decimal(state.summary.total_value_thb)
            total_cost = _decimal(compute_total_cost(state, fx))
            unrealized = _decimal(state.summary.total_unrealized_profit)
            cash_balance = sum(
                (_decimal(h.market_value_thb or 0) for h in state.holdings if h.asset_type == "Cash"),
                Decimal("0"),
            )
            asset_class_values: Dict[str, str] = {}
            for holding in state.holdings:
                asset_class = holding.asset_type or "Unknown"
                value = _decimal(holding.market_value_thb or 0).quantize(Decimal("0.01"))
                asset_class_values[asset_class] = str(
                    Decimal(asset_class_values.get(asset_class, "0")) + value
                )

            provenance_reader = getattr(self.repo, "performance_provenance", None)
            provenance = provenance_reader(pid) if callable(provenance_reader) else {}
            current_sequence = provenance.get("source_state_sequence")
            current_hash = provenance.get("source_state_sha256")
            history = self._history_as_of(pid, cutoff_utc)
            versioned = [
                row for row in history
                if isinstance(row, dict) and row.get("Cutoff_UTC") and row.get("Snapshot_ID")
            ]
            versioned.sort(key=lambda row: (str(row.get("Cutoff_UTC")), int(row.get("Revision") or 0)))
            previous = versioned[-1] if versioned else None

            flow_coverage = "MISSING_FLOW_HISTORY"
            net_external_flow: Optional[Decimal] = None
            flow_event_ids: list[str] = []
            flow_reader = getattr(self.repo, "list_cash_flow_events", None)
            coverage_reader = getattr(self.repo, "cash_flow_coverage_between", None)
            if (
                previous is not None
                and current_sequence is not None
                and previous.get("Source_State_Sequence") not in (None, "")
                and callable(flow_reader)
                and callable(coverage_reader)
            ):
                previous_sequence = int(previous["Source_State_Sequence"])
                current_sequence = int(current_sequence)
                events = flow_reader(
                    pid, after_sequence=previous_sequence, through_sequence=current_sequence
                )
                all_flows_known = all(
                    event.get("flow_coverage_status") == "COMPLETE"
                    and event.get("amount_thb_decimal") is not None
                    and event.get("flow_kind") == "EXTERNAL_CONTRIBUTION"
                    for event in events
                )
                history_covered = coverage_reader(
                    pid, after_sequence=previous_sequence, through_sequence=current_sequence
                )
                if all_flows_known and history_covered:
                    net_external_flow = sum(
                        (_decimal(event["amount_thb_decimal"]) for event in events),
                        Decimal("0"),
                    ).quantize(Decimal("0.01"))
                    flow_coverage = "COMPLETE"
                    flow_event_ids = [str(event["event_id"]) for event in events]

            row = {
                "Date": cutoff.date().isoformat(),
                "Total_NAV": str(total_nav.quantize(Decimal("0.01"))),
                "Total_Cost": str(total_cost.quantize(Decimal("0.01"))),
                "Unrealized_PnL": str(unrealized.quantize(Decimal("0.01"))),
                "Cash_Balance": str(cash_balance.quantize(Decimal("0.01"))),
                "Realized_PnL_YTD": str(_decimal(state.summary.total_realized_profit_ytd).quantize(Decimal("0.01"))),
                "Passive_Income_YTD": str(_decimal(state.summary.passive_income_ytd).quantize(Decimal("0.01"))),
                "Asset_Class_Values_THB": asset_class_values,
                "Snapshot_ID": f"PERF-{uuid4().hex}",
                "Cutoff_UTC": cutoff_utc,
                "Recorded_At_UTC": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "Source_State_Sequence": current_sequence,
                "Source_State_SHA256": current_hash,
                "Total_NAV_THB": str(total_nav.quantize(Decimal("0.01"))),
                "Net_External_Flow_THB": str(net_external_flow) if net_external_flow is not None else None,
                "Flow_Coverage_Status": flow_coverage,
                "Cash_Flow_Event_IDs_JSON": flow_event_ids,
            }
            self.perf_repo.upsert_snapshot(pid, row)
            return (
                f"[PERF SNAPSHOT] ?????? snapshot ?????? {row['Date']} "
                f"(NAV: {row['Total_NAV_THB']} THB, flow coverage: {flow_coverage}) ??????"
            )
        except Exception as exc:
            return f"Error: {exc}"

    def read_performance_history(
        self, days: int = 30, portfolio_id: str = "default", as_of_utc: Optional[str] = None
    ) -> str:
        rows = self.get_structured_performance_history(
            days=days, portfolio_id=portfolio_id, as_of_utc=as_of_utc
        )
        return json.dumps(rows, ensure_ascii=False, indent=2)

    def get_structured_performance_history(
        self, days: Optional[int] = None, portfolio_id: str = "default",
        as_of_utc: Optional[str] = None,
    ) -> List[Dict]:
        try:
            return self.perf_repo.read_history(
                portfolio_id=portfolio_id, days=days, as_of_utc=as_of_utc
            )
        except TypeError:
            return self.perf_repo.read_history(portfolio_id=portfolio_id, days=days)
