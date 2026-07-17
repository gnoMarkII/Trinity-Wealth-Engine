"""GET /api/portfolio/latest, GET /api/macro/dashboard — อ่าน sidecar JSON ตรงๆ (Rev.3/5 ข้อ 1)

ไม่ parse markdown, ไม่ re-instantiate ผ่าน MacroStrategyDirection guardrail validator ซ้ำ —
ข้อมูลผ่าน validation ครบแล้วตอนที่ strategic_allocator_node เขียนไฟล์ (ดู tools/macro/report_formatter.py)
"""
import json
from typing import Optional

from contextlib import contextmanager
from filelock import Timeout
from pydantic import ValidationError
from fastapi import APIRouter, Depends, HTTPException

from api.auth import require_session
from api.schemas import (
    MacroIndicatorSeriesDTO,
    MacroDashboardDTO,
    PortfolioDTO,
    macro_dashboard_dto_from_raw,
    portfolio_dto_from_raw,
    ActualPortfolioStateDTO,
    BucketAllocationResponseDTO,
    BucketAllocationSummaryDTO,
    ActualWatchlistStateDTO,
    ActualGoalsResponseDTO,
    ActualGoalItemDTO,
    PerformanceSnapshotDTO,
    JournalEntryDTO,
    UpsertAllocationTargetsRequestDTO,
    AssignBucketRequestDTO,
    BatchAssignBucketRequestDTO,
    BatchRemoveHoldingsRequestDTO,
    TradeRequestDTO,
    CashFlowRequestDTO,
    IncomeRequestDTO,
    EditHoldingRequestDTO,
    UpsertWatchlistItemRequestDTO,
    UpsertGoalRequestDTO,
    AppendJournalRequestDTO,
)
from tools.archivist.core import VAULT_PATH
from tools.macro.dashboard import load_indicator_series

router = APIRouter(dependencies=[Depends(require_session)])

_STRATEGY_SUBDIR = "30_Knowledge_Base/Strategies"


def _latest_strategy_json() -> dict:
    strategy_dir = VAULT_PATH / _STRATEGY_SUBDIR
    candidates = sorted(strategy_dir.glob("Macro_Strategy_Direction_*.json"))
    if not candidates:
        raise HTTPException(
            status_code=404,
            detail="ยังไม่มีรายงาน Macro Strategy ที่มี JSON sidecar — รอรายงานถัดไปหลัง Phase 0 อัปเดต",
        )
    latest = candidates[-1]
    return json.loads(latest.read_text(encoding="utf-8"))


@router.get("/api/portfolio/latest", response_model=PortfolioDTO)
def get_latest_portfolio() -> PortfolioDTO:
    return portfolio_dto_from_raw(_latest_strategy_json())


@router.get("/api/macro/dashboard", response_model=MacroDashboardDTO)
def get_macro_dashboard() -> MacroDashboardDTO:
    return macro_dashboard_dto_from_raw(_latest_strategy_json())


@router.get("/api/macro/indicators/{indicator_id}/series", response_model=MacroIndicatorSeriesDTO)
def get_macro_indicator_series(indicator_id: str, range: str = "3m") -> MacroIndicatorSeriesDTO:
    raw = _latest_strategy_json()
    indicators = raw.get("dashboard_indicators", [])
    indicator = next(
        (
            item
            for item in indicators
            if isinstance(item, dict) and item.get("indicator_id") == indicator_id
        ),
        None,
    )
    if indicator is None:
        raise HTTPException(status_code=404, detail="Macro indicator not found in the latest report")
    if range not in {"1m", "3m", "1y"}:
        raise HTTPException(status_code=422, detail="range must be one of: 1m, 3m, 1y")

    try:
        points = load_indicator_series(VAULT_PATH, str(indicator.get("series_key", "")), range)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Macro indicator series is unavailable") from exc

    return MacroIndicatorSeriesDTO(
        indicator_id=indicator_id,
        series_key=str(indicator.get("series_key", "")),
        label=str(indicator.get("label", "")),
        unit=str(indicator.get("unit", "")),
        range=range,
        points=points,
    )

from tools.portfolio.models import _now_iso
from tools.portfolio import (
    core as portfolio_core,
    trading as portfolio_trading,
    watchlist as portfolio_watchlist,
    goals as portfolio_goals,
    performance as portfolio_perf,
    journal as portfolio_journal,
)

# ---------------------------------------------------------
# Actual Portfolio Hub Read Endpoints (Phase 1)
# ---------------------------------------------------------


@contextmanager
def handle_portfolio_exceptions(timeout_detail: str = "Portfolio lock timeout"):
    try:
        yield
    except ValidationError as exc:
        raise HTTPException(
            status_code=500, detail=f"Internal DTO validation error: {exc}"
        ) from exc
    except Timeout as exc:
        raise HTTPException(status_code=503, detail=timeout_detail) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/api/portfolio/actual/state", response_model=ActualPortfolioStateDTO)
def get_actual_portfolio_state(
    refresh_prices: bool = False, fetch_fundamentals: bool = False
) -> ActualPortfolioStateDTO:
    with handle_portfolio_exceptions(
        "Portfolio lock timeout (another operation is running)"
    ):
        state = portfolio_core.get_structured_portfolio_state(
            refresh_prices=refresh_prices, fetch_fundamentals=fetch_fundamentals
        )
        return ActualPortfolioStateDTO.model_validate(
            state.model_dump(exclude_none=True)
        )


@router.get(
    "/api/portfolio/actual/allocations", response_model=BucketAllocationResponseDTO
)
def get_actual_bucket_allocations() -> BucketAllocationResponseDTO:
    with handle_portfolio_exceptions("Allocation lock timeout"):
        summaries, warning = portfolio_core.get_structured_bucket_allocation()
        return BucketAllocationResponseDTO(
            warning=warning,
            summaries=[
                BucketAllocationSummaryDTO.model_validate(s) for s in summaries
            ],
        )


@router.get(
    "/api/portfolio/actual/watchlist", response_model=ActualWatchlistStateDTO
)
def get_actual_watchlist() -> ActualWatchlistStateDTO:
    with handle_portfolio_exceptions("Watchlist lock timeout"):
        state = portfolio_watchlist.get_structured_watchlist()
        return ActualWatchlistStateDTO.model_validate(
            state.model_dump(exclude_none=True)
        )


@router.get("/api/portfolio/actual/goals", response_model=ActualGoalsResponseDTO)
def get_actual_goals() -> ActualGoalsResponseDTO:
    with handle_portfolio_exceptions("Goals lock timeout"):
        goals = portfolio_goals.get_structured_goals()
        return ActualGoalsResponseDTO(
            n_goals=len(goals),
            goals=[ActualGoalItemDTO.model_validate(g) for g in goals],
            generated_at=_now_iso(),
        )


@router.get(
    "/api/portfolio/actual/performance", response_model=list[PerformanceSnapshotDTO]
)
def get_actual_performance(days: int = 30) -> list[PerformanceSnapshotDTO]:
    with handle_portfolio_exceptions("Performance lock timeout"):
        rows = portfolio_perf.get_structured_performance_history(days=days)
        return [PerformanceSnapshotDTO.model_validate(r) for r in rows]


@router.post(
    "/api/portfolio/actual/performance/snapshot",
    response_model=list[PerformanceSnapshotDTO],
)
def trigger_performance_snapshot(
    refresh_prices: bool = False,
) -> list[PerformanceSnapshotDTO]:
    with handle_portfolio_exceptions("Performance lock timeout"):
        portfolio_perf.record_performance_snapshot.func(refresh_prices=refresh_prices)
        rows = portfolio_perf.get_structured_performance_history(days=365)
        return [PerformanceSnapshotDTO.model_validate(r) for r in rows]


@router.get(
    "/api/portfolio/actual/journal", response_model=list[JournalEntryDTO]
)
def get_actual_journal(
    days: Optional[int] = None,
    keyword: Optional[str] = None,
    limit: int = 50,
) -> list[JournalEntryDTO]:
    with handle_portfolio_exceptions("Journal lock timeout"):
        rows = portfolio_journal.get_structured_journal(
            days=days, keyword=keyword, limit=limit
        )
        return [JournalEntryDTO.model_validate(r) for r in rows]


# ---------------------------------------------------------
# Actual Portfolio Hub Mutation Endpoints (Phase 1)
# ---------------------------------------------------------


@router.put(
    "/api/portfolio/actual/allocations/targets",
    response_model=ActualPortfolioStateDTO,
)
def upsert_allocation_targets(
    payload: UpsertAllocationTargetsRequestDTO,
) -> ActualPortfolioStateDTO:
    with handle_portfolio_exceptions():
        state = portfolio_core.structured_upsert_allocation_targets(payload.targets)
        return ActualPortfolioStateDTO.model_validate(
            state.model_dump(exclude_none=True)
        )


@router.put(
    "/api/portfolio/actual/holdings/{symbol}/bucket",
    response_model=ActualPortfolioStateDTO,
)
def assign_holding_bucket(
    symbol: str, payload: AssignBucketRequestDTO
) -> ActualPortfolioStateDTO:
    with handle_portfolio_exceptions():
        state = portfolio_core.structured_assign_holding_bucket(
            symbol, payload.bucket_id
        )
        return ActualPortfolioStateDTO.model_validate(
            state.model_dump(exclude_none=True)
        )


@router.put(
    "/api/portfolio/actual/holdings/batch-bucket",
    response_model=ActualPortfolioStateDTO,
)
def batch_assign_holding_buckets(
    payload: BatchAssignBucketRequestDTO,
) -> ActualPortfolioStateDTO:
    with handle_portfolio_exceptions():
        state = portfolio_core.structured_batch_assign_holding_buckets(
            payload.symbols, payload.bucket_id
        )
        return ActualPortfolioStateDTO.model_validate(
            state.model_dump(exclude_none=True)
        )


@router.post(
    "/api/portfolio/actual/holdings/batch-delete",
    response_model=ActualPortfolioStateDTO,
)
def batch_remove_holdings(
    payload: BatchRemoveHoldingsRequestDTO,
) -> ActualPortfolioStateDTO:
    with handle_portfolio_exceptions():
        state = portfolio_core.structured_batch_remove_holdings(payload.symbols)
        return ActualPortfolioStateDTO.model_validate(
            state.model_dump(exclude_none=True)
        )


@router.post(
    "/api/portfolio/actual/reset", response_model=ActualPortfolioStateDTO
)
def reset_portfolio_clean_slate() -> ActualPortfolioStateDTO:
    with handle_portfolio_exceptions():
        state = portfolio_core.structured_reset_clean_slate()
        return ActualPortfolioStateDTO.model_validate(
            state.model_dump(exclude_none=True)
        )


@router.post("/api/portfolio/actual/trade", response_model=ActualPortfolioStateDTO)
def execute_trade_endpoint(
    payload: TradeRequestDTO,
) -> ActualPortfolioStateDTO:
    with handle_portfolio_exceptions():
        state = portfolio_trading.structured_execute_trade(
            symbol=payload.symbol,
            asset_type=payload.asset_type,
            action=payload.action,
            units=payload.units,
            price=payload.price,
            currency=payload.currency,
            exchange_rate=payload.exchange_rate,
            date=payload.date,
            notes=payload.notes,
            bucket_id=payload.bucket_id,
        )
        return ActualPortfolioStateDTO.model_validate(
            state.model_dump(exclude_none=True)
        )


@router.post(
    "/api/portfolio/actual/cashflow", response_model=ActualPortfolioStateDTO
)
def manage_cash_flow_endpoint(
    payload: CashFlowRequestDTO,
) -> ActualPortfolioStateDTO:
    with handle_portfolio_exceptions():
        state = portfolio_trading.structured_manage_cash_flow(
            amount=payload.amount,
            action=payload.action,
            currency=payload.currency,
            exchange_rate=payload.exchange_rate,
            date=payload.date,
            notes=payload.notes,
        )
        return ActualPortfolioStateDTO.model_validate(
            state.model_dump(exclude_none=True)
        )


@router.post(
    "/api/portfolio/actual/income", response_model=ActualPortfolioStateDTO
)
def record_income_endpoint(
    payload: IncomeRequestDTO,
) -> ActualPortfolioStateDTO:
    with handle_portfolio_exceptions():
        state = portfolio_trading.structured_record_income(
            income_type=payload.income_type,
            amount_thb=payload.amount_thb,
            source_symbol=payload.source_symbol,
            date=payload.date,
            notes=payload.notes,
        )
        return ActualPortfolioStateDTO.model_validate(
            state.model_dump(exclude_none=True)
        )


@router.put(
    "/api/portfolio/actual/holdings/{symbol}/edit",
    response_model=ActualPortfolioStateDTO,
)
def edit_holding_endpoint(
    symbol: str, payload: EditHoldingRequestDTO
) -> ActualPortfolioStateDTO:
    with handle_portfolio_exceptions():
        state = portfolio_trading.structured_edit_holding(
            symbol=symbol,
            units=payload.units,
            avg_cost=payload.avg_cost,
            accumulated_dividend_thb=payload.accumulated_dividend_thb,
            asset_type=payload.asset_type,
            reason=payload.reason,
            bucket_id=payload.bucket_id,
        )
        return ActualPortfolioStateDTO.model_validate(
            state.model_dump(exclude_none=True)
        )


@router.delete(
    "/api/portfolio/actual/holdings/{symbol}",
    response_model=ActualPortfolioStateDTO,
)
def remove_holding_endpoint(symbol: str) -> ActualPortfolioStateDTO:
    with handle_portfolio_exceptions():
        state = portfolio_trading.structured_remove_holding(symbol)
        return ActualPortfolioStateDTO.model_validate(
            state.model_dump(exclude_none=True)
        )


@router.put(
    "/api/portfolio/actual/watchlist/{symbol}",
    response_model=ActualWatchlistStateDTO,
)
def upsert_watchlist_item_endpoint(
    symbol: str, payload: UpsertWatchlistItemRequestDTO
) -> ActualWatchlistStateDTO:
    with handle_portfolio_exceptions("Watchlist lock timeout"):
        state = portfolio_watchlist.structured_upsert_watchlist_item(
            symbol=symbol,
            asset_type=payload.asset_type,
            target_price=payload.target_price,
            notes=payload.notes,
        )
        return ActualWatchlistStateDTO.model_validate(
            state.model_dump(exclude_none=True)
        )


@router.delete(
    "/api/portfolio/actual/watchlist/{symbol}",
    response_model=ActualWatchlistStateDTO,
)
def remove_watchlist_item_endpoint(symbol: str) -> ActualWatchlistStateDTO:
    with handle_portfolio_exceptions("Watchlist lock timeout"):
        state = portfolio_watchlist.structured_remove_watchlist_item(symbol)
        return ActualWatchlistStateDTO.model_validate(
            state.model_dump(exclude_none=True)
        )


@router.put(
    "/api/portfolio/actual/goals/{name}", response_model=ActualGoalsResponseDTO
)
def upsert_goal_endpoint(
    name: str, payload: UpsertGoalRequestDTO
) -> ActualGoalsResponseDTO:
    with handle_portfolio_exceptions("Goals lock timeout"):
        goals = portfolio_goals.structured_upsert_goal(
            name=name,
            goal_type=payload.goal_type,
            target_amount_thb=payload.target_amount_thb,
            deadline=payload.deadline,
            years_from_now=payload.years_from_now,
            notes=payload.notes,
        )
        return ActualGoalsResponseDTO(
            n_goals=len(goals),
            goals=[ActualGoalItemDTO.model_validate(g) for g in goals],
            generated_at=_now_iso(),
        )


@router.delete(
    "/api/portfolio/actual/goals/{name}", response_model=ActualGoalsResponseDTO
)
def remove_goal_endpoint(name: str) -> ActualGoalsResponseDTO:
    with handle_portfolio_exceptions("Goals lock timeout"):
        goals = portfolio_goals.structured_remove_goal(name)
        return ActualGoalsResponseDTO(
            n_goals=len(goals),
            goals=[ActualGoalItemDTO.model_validate(g) for g in goals],
            generated_at=_now_iso(),
        )


@router.post(
    "/api/portfolio/actual/journal", response_model=list[JournalEntryDTO]
)
def append_journal_endpoint(
    payload: AppendJournalRequestDTO,
) -> list[JournalEntryDTO]:
    with handle_portfolio_exceptions("Journal lock timeout"):
        rows = portfolio_journal.structured_append_journal(payload.entry)
        return [JournalEntryDTO.model_validate(r) for r in rows]
