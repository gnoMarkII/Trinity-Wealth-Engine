"""Inbound HTTP adapter for Equity Event Study & Historical Scenario Analysis."""
import logging
from typing import List
from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import get_event_study_service
from api.schemas.event_study import (
    AverageTrajectoryPointDTO,
    EventInstanceDTO,
    EventPathPointDTO,
    EventPresetItemDTO,
    EventStudyGuardrailWarningDTO,
    EventStudyRequestDTO,
    EventStudyResponseDTO,
    HorizonStatsDTO,
)
from application.equity.event_study_service import EventStudyApplicationService
from application.equity.validation import validate_ticker

log = logging.getLogger(__name__)

router = APIRouter()


@router.get("/event-study/presets", response_model=List[EventPresetItemDTO])
def get_event_study_presets(
    service: EventStudyApplicationService = Depends(get_event_study_service),
) -> List[EventPresetItemDTO]:
    """Lists all available historical event study presets."""
    raw_presets = service.get_available_presets()
    return [EventPresetItemDTO(**p) for p in raw_presets]


@router.post("/{ticker}/event-study", response_model=EventStudyResponseDTO)
def execute_event_study(
    ticker: str,
    payload: EventStudyRequestDTO,
    service: EventStudyApplicationService = Depends(get_event_study_service),
) -> EventStudyResponseDTO:
    """Executes a quantitative Event Study for a stock across specified event dates."""
    try:
        clean_ticker = validate_ticker(ticker)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        report = service.run_event_study(
            ticker=clean_ticker,
            preset_id=payload.preset_id,
            custom_dates=payload.custom_dates,
            custom_event_name=payload.custom_event_name,
            horizons=payload.horizons,
        )
    except Exception as exc:
        log.exception("Event study execution failed for %s: %s", clean_ticker, exc)
        raise HTTPException(status_code=500, detail=f"Failed to execute event study: {exc}") from exc

    # Map domain models to DTOs
    events_dto: List[EventInstanceDTO] = []
    for ev in report.events:
        trajectory_dto = [
            EventPathPointDTO(
                day_offset=pt.day_offset,
                date=pt.date,
                close=pt.close,
                cumulative_return_pct=pt.cumulative_return_pct,
            )
            for pt in ev.daily_trajectory
        ]
        events_dto.append(
            EventInstanceDTO(
                event_id=ev.event_id,
                event_date=ev.event_date,
                entry_date=ev.entry_date,
                entry_price=ev.entry_price,
                description=ev.description,
                forward_returns={str(k): v for k, v in ev.forward_returns.items()},
                daily_trajectory=trajectory_dto,
                is_win_by_horizon={str(k): v for k, v in ev.is_win_by_horizon.items()},
            )
        )

    horizon_stats_dto: dict[str, HorizonStatsDTO] = {}
    for h, st in report.horizon_stats.items():
        horizon_stats_dto[str(h)] = HorizonStatsDTO(
            horizon_days=st.horizon_days,
            sample_size=st.sample_size,
            hit_rate_pct=st.hit_rate_pct,
            avg_return_pct=st.avg_return_pct,
            median_return_pct=st.median_return_pct,
            best_return_pct=st.best_return_pct,
            worst_return_pct=st.worst_return_pct,
            avg_win_pct=st.avg_win_pct,
            avg_loss_pct=st.avg_loss_pct,
            risk_reward_ratio=st.risk_reward_ratio,
            win_count=st.win_count,
            loss_count=st.loss_count,
        )

    avg_traj_dto = [
        AverageTrajectoryPointDTO(
            day_offset=pt.day_offset,
            avg_return_pct=pt.avg_return_pct,
            sample_count=pt.sample_count,
        )
        for pt in report.average_trajectory
    ]

    warnings_dto = [
        EventStudyGuardrailWarningDTO(code=w.code, message=w.message)
        for w in report.warnings
    ]

    return EventStudyResponseDTO(
        ticker=report.ticker,
        event_name=report.event_name,
        horizons=report.horizons,
        events=events_dto,
        horizon_stats=horizon_stats_dto,
        average_trajectory=avg_traj_dto,
        warnings=warnings_dto,
    )


__all__ = ["router", "get_event_study_presets", "execute_event_study"]
