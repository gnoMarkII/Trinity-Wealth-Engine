"""FastAPI Sub-router for Watchlist, Goals, Journal, and Performance."""
from datetime import datetime, timezone
from typing import Optional, List
from fastapi import APIRouter, Depends

from api.auth import require_session
from api.dependencies import get_portfolio_service
from tools.portfolio.service import PortfolioService
from api.schemas import (
    ActualWatchlistStateDTO,
    UpsertWatchlistItemRequestDTO,
    ActualGoalsResponseDTO,
    ActualGoalItemDTO,
    UpsertGoalRequestDTO,
    JournalEntryDTO,
    AppendJournalRequestDTO,
    PerformanceSnapshotDTO,
)
from api.routers.portfolio.common import handle_portfolio_exceptions

router = APIRouter(dependencies=[Depends(require_session)])


# ---------------------------------------------------------
# Watchlist Endpoints
# ---------------------------------------------------------

@router.get("/api/portfolio/actual/watchlist", response_model=ActualWatchlistStateDTO)
def get_actual_watchlist(
    portfolio_id: str = "default",
    service: PortfolioService = Depends(get_portfolio_service),
) -> ActualWatchlistStateDTO:
    with handle_portfolio_exceptions("Watchlist lock timeout"):
        state = service.get_structured_watchlist(portfolio_id=portfolio_id)
        return ActualWatchlistStateDTO.model_validate(state.model_dump(exclude_none=True))


@router.put("/api/portfolio/actual/watchlist/{symbol}", response_model=ActualWatchlistStateDTO)
def upsert_watchlist_item_endpoint(
    symbol: str,
    payload: UpsertWatchlistItemRequestDTO,
    portfolio_id: str = "default",
    service: PortfolioService = Depends(get_portfolio_service),
) -> ActualWatchlistStateDTO:
    with handle_portfolio_exceptions("Watchlist lock timeout"):
        state = service.structured_upsert_watchlist_item(
            symbol=symbol,
            asset_type=payload.asset_type,
            target_price=payload.target_price,
            notes=payload.notes,
            portfolio_id=portfolio_id,
        )
        return ActualWatchlistStateDTO.model_validate(state.model_dump(exclude_none=True))


@router.delete("/api/portfolio/actual/watchlist/{symbol}", response_model=ActualWatchlistStateDTO)
def remove_watchlist_item_endpoint(
    symbol: str,
    portfolio_id: str = "default",
    service: PortfolioService = Depends(get_portfolio_service),
) -> ActualWatchlistStateDTO:
    with handle_portfolio_exceptions("Watchlist lock timeout"):
        state = service.structured_remove_watchlist_item(symbol, portfolio_id=portfolio_id)
        return ActualWatchlistStateDTO.model_validate(state.model_dump(exclude_none=True))


# ---------------------------------------------------------
# Goals Endpoints
# ---------------------------------------------------------

@router.get("/api/portfolio/actual/goals", response_model=ActualGoalsResponseDTO)
def get_actual_goals(
    portfolio_id: Optional[str] = None,
    service: PortfolioService = Depends(get_portfolio_service),
) -> ActualGoalsResponseDTO:
    with handle_portfolio_exceptions("Goals lock timeout"):
        goals = service.get_structured_goals(portfolio_id=portfolio_id)
        return ActualGoalsResponseDTO(
            n_goals=len(goals),
            goals=[ActualGoalItemDTO.model_validate(g) for g in goals],
            generated_at=datetime.now(timezone.utc).isoformat(),
        )


@router.put("/api/portfolio/actual/goals/{name}", response_model=ActualGoalsResponseDTO)
def upsert_goal_endpoint(
    name: str,
    payload: UpsertGoalRequestDTO,
    service: PortfolioService = Depends(get_portfolio_service),
) -> ActualGoalsResponseDTO:
    with handle_portfolio_exceptions("Goals lock timeout"):
        goals = service.structured_upsert_goal(
            name=name,
            goal_type=payload.goal_type,
            target_amount_thb=payload.target_amount_thb,
            deadline=payload.deadline,
            years_from_now=payload.years_from_now,
            notes=payload.notes,
            portfolio_id=getattr(payload, "portfolio_id", "default") or "default",
            bucket_id=getattr(payload, "bucket_id", None),
        )
        return ActualGoalsResponseDTO(
            n_goals=len(goals),
            goals=[ActualGoalItemDTO.model_validate(g) for g in goals],
            generated_at=datetime.now(timezone.utc).isoformat(),
        )


@router.delete("/api/portfolio/actual/goals/{name}", response_model=ActualGoalsResponseDTO)
def remove_goal_endpoint(
    name: str,
    portfolio_id: Optional[str] = None,
    service: PortfolioService = Depends(get_portfolio_service),
) -> ActualGoalsResponseDTO:
    with handle_portfolio_exceptions("Goals lock timeout"):
        goals = service.structured_remove_goal(name, portfolio_id=portfolio_id or "default")
        return ActualGoalsResponseDTO(
            n_goals=len(goals),
            goals=[ActualGoalItemDTO.model_validate(g) for g in goals],
            generated_at=datetime.now(timezone.utc).isoformat(),
        )


# ---------------------------------------------------------
# Journal Endpoints
# ---------------------------------------------------------

@router.get("/api/portfolio/actual/journal", response_model=list[JournalEntryDTO])
def get_actual_journal(
    days: Optional[int] = None,
    keyword: Optional[str] = None,
    limit: int = 50,
    portfolio_id: str = "default",
    service: PortfolioService = Depends(get_portfolio_service),
) -> list[JournalEntryDTO]:
    with handle_portfolio_exceptions("Journal lock timeout"):
        rows = service.get_structured_journal(
            days=days, keyword=keyword, limit=limit, portfolio_id=portfolio_id
        )
        return [JournalEntryDTO.model_validate(r) for r in rows]


@router.post("/api/portfolio/actual/journal", response_model=list[JournalEntryDTO])
def append_journal_endpoint(
    payload: AppendJournalRequestDTO,
    portfolio_id: str = "default",
    service: PortfolioService = Depends(get_portfolio_service),
) -> list[JournalEntryDTO]:
    with handle_portfolio_exceptions("Journal lock timeout"):
        rows = service.structured_append_journal(payload.entry, portfolio_id=portfolio_id)
        return [JournalEntryDTO.model_validate(r) for r in rows]


# ---------------------------------------------------------
# Performance Endpoints
# ---------------------------------------------------------

@router.get("/api/portfolio/actual/performance", response_model=list[PerformanceSnapshotDTO])
def get_actual_performance(
    days: int = 30,
    portfolio_id: str = "default",
    service: PortfolioService = Depends(get_portfolio_service),
) -> list[PerformanceSnapshotDTO]:
    with handle_portfolio_exceptions("Performance lock timeout"):
        rows = service.get_structured_performance_history(days=days, portfolio_id=portfolio_id)
        return [PerformanceSnapshotDTO.model_validate(r) for r in rows]


@router.post("/api/portfolio/actual/performance/snapshot", response_model=list[PerformanceSnapshotDTO])
def trigger_performance_snapshot(
    refresh_prices: bool = False,
    portfolio_id: str = "default",
    service: PortfolioService = Depends(get_portfolio_service),
) -> list[PerformanceSnapshotDTO]:
    with handle_portfolio_exceptions("Performance lock timeout"):
        service.record_performance_snapshot(refresh_prices=refresh_prices, portfolio_id=portfolio_id)
        rows = service.get_structured_performance_history(days=365, portfolio_id=portfolio_id)
        return [PerformanceSnapshotDTO.model_validate(r) for r in rows]
