"""FastAPI Sub-router for Portfolio CRUD, State, Buckets, and Allocation Targets."""
from fastapi import APIRouter, Depends

from api.auth import require_session
from api.dependencies import get_portfolio_service
from tools.portfolio.service import PortfolioService
from api.schemas import (
    PortfolioMetaDTO,
    CreatePortfolioRequestDTO,
    RenamePortfolioRequestDTO,
    ActualPortfolioStateDTO,
    BucketAllocationResponseDTO,
    BucketAllocationSummaryDTO,
    UpsertAllocationTargetsRequestDTO,
    AssignBucketRequestDTO,
    BatchAssignBucketRequestDTO,
    BatchRemoveHoldingsRequestDTO,
)
from api.routers.portfolio.common import handle_portfolio_exceptions

router = APIRouter(dependencies=[Depends(require_session)])


@router.get("/api/portfolio/list", response_model=list[PortfolioMetaDTO])
def list_portfolios_endpoint(
    service: PortfolioService = Depends(get_portfolio_service),
) -> list[PortfolioMetaDTO]:
    with handle_portfolio_exceptions():
        items = service.list_portfolios()
        return [PortfolioMetaDTO.model_validate(i.model_dump()) for i in items]


@router.post("/api/portfolio/create", response_model=PortfolioMetaDTO)
def create_portfolio_endpoint(
    payload: CreatePortfolioRequestDTO,
    service: PortfolioService = Depends(get_portfolio_service),
) -> PortfolioMetaDTO:
    with handle_portfolio_exceptions():
        item = service.create_portfolio(name=payload.name, portfolio_id=payload.portfolio_id)
        return PortfolioMetaDTO.model_validate(item.model_dump())


@router.delete("/api/portfolio/{portfolio_id}")
def delete_portfolio_endpoint(
    portfolio_id: str,
    service: PortfolioService = Depends(get_portfolio_service),
):
    with handle_portfolio_exceptions():
        service.delete_portfolio(portfolio_id=portfolio_id)
        return {"status": "success", "deleted_portfolio_id": portfolio_id}


@router.put("/api/portfolio/{portfolio_id}/rename", response_model=PortfolioMetaDTO)
def rename_portfolio_endpoint(
    portfolio_id: str,
    payload: RenamePortfolioRequestDTO,
    service: PortfolioService = Depends(get_portfolio_service),
) -> PortfolioMetaDTO:
    with handle_portfolio_exceptions():
        item = service.update_portfolio_name(portfolio_id=portfolio_id, name=payload.name)
        return PortfolioMetaDTO.model_validate(item.model_dump())


@router.get("/api/portfolio/actual/state", response_model=ActualPortfolioStateDTO)
def get_actual_portfolio_state(
    refresh_prices: bool = False,
    fetch_fundamentals: bool = False,
    portfolio_id: str = "default",
    service: PortfolioService = Depends(get_portfolio_service),
) -> ActualPortfolioStateDTO:
    with handle_portfolio_exceptions("Portfolio lock timeout (another operation is running)"):
        state = service.get_structured_portfolio_state(portfolio_id=portfolio_id)
        needs_fundamentals = fetch_fundamentals or any(
            h.fundamentals_updated_at is None for h in state.holdings if h.asset_type != "Cash"
        )
        if refresh_prices or needs_fundamentals:
            state = service.get_structured_portfolio_state(
                refresh_prices=refresh_prices,
                fetch_fundamentals=needs_fundamentals,
                portfolio_id=portfolio_id,
            )
        return ActualPortfolioStateDTO.model_validate(state.model_dump(exclude_none=True))


@router.get("/api/portfolio/actual/allocations", response_model=BucketAllocationResponseDTO)
def get_actual_bucket_allocations(
    portfolio_id: str = "default",
    service: PortfolioService = Depends(get_portfolio_service),
) -> BucketAllocationResponseDTO:
    with handle_portfolio_exceptions("Allocation lock timeout"):
        summaries, warning = service.get_structured_bucket_allocation(portfolio_id=portfolio_id)
        return BucketAllocationResponseDTO(
            warning=warning,
            summaries=[BucketAllocationSummaryDTO.model_validate(s) for s in summaries],
        )


@router.put("/api/portfolio/actual/allocations/targets", response_model=ActualPortfolioStateDTO)
def upsert_allocation_targets(
    payload: UpsertAllocationTargetsRequestDTO,
    portfolio_id: str = "default",
    service: PortfolioService = Depends(get_portfolio_service),
) -> ActualPortfolioStateDTO:
    with handle_portfolio_exceptions():
        state = service.structured_upsert_allocation_targets(payload.targets, portfolio_id=portfolio_id)
        return ActualPortfolioStateDTO.model_validate(state.model_dump(exclude_none=True))


@router.put("/api/portfolio/actual/holdings/{symbol}/bucket", response_model=ActualPortfolioStateDTO)
def assign_holding_bucket(
    symbol: str,
    payload: AssignBucketRequestDTO,
    portfolio_id: str = "default",
    service: PortfolioService = Depends(get_portfolio_service),
) -> ActualPortfolioStateDTO:
    with handle_portfolio_exceptions():
        state = service.structured_assign_holding_bucket(symbol, payload.bucket_id, portfolio_id=portfolio_id)
        return ActualPortfolioStateDTO.model_validate(state.model_dump(exclude_none=True))


@router.put("/api/portfolio/actual/holdings/batch-bucket", response_model=ActualPortfolioStateDTO)
def batch_assign_holding_buckets(
    payload: BatchAssignBucketRequestDTO,
    portfolio_id: str = "default",
    service: PortfolioService = Depends(get_portfolio_service),
) -> ActualPortfolioStateDTO:
    with handle_portfolio_exceptions():
        state = service.structured_batch_assign_holding_buckets(payload.symbols, payload.bucket_id, portfolio_id=portfolio_id)
        return ActualPortfolioStateDTO.model_validate(state.model_dump(exclude_none=True))


@router.post("/api/portfolio/actual/holdings/batch-delete", response_model=ActualPortfolioStateDTO)
def batch_remove_holdings(
    payload: BatchRemoveHoldingsRequestDTO,
    portfolio_id: str = "default",
    service: PortfolioService = Depends(get_portfolio_service),
) -> ActualPortfolioStateDTO:
    with handle_portfolio_exceptions():
        state = service.structured_batch_remove_holdings(payload.symbols, portfolio_id=portfolio_id)
        return ActualPortfolioStateDTO.model_validate(state.model_dump(exclude_none=True))


@router.post("/api/portfolio/actual/reset", response_model=ActualPortfolioStateDTO)
def reset_portfolio_clean_slate(
    portfolio_id: str = "default",
    service: PortfolioService = Depends(get_portfolio_service),
) -> ActualPortfolioStateDTO:
    with handle_portfolio_exceptions():
        state = service.structured_reset_clean_slate(portfolio_id=portfolio_id)
        return ActualPortfolioStateDTO.model_validate(state.model_dump(exclude_none=True))
