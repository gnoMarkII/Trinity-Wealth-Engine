"""Pure domain calculation engine for Event Study & Historical Scenario Analysis.

Implements institutional quantitative event study methodology:
- Forward returns calculation at 5, 10, 20, 60 trading days post-event
- Entry at next trading day Open (Open[t+1]) to eliminate look-ahead bias
- Daily cumulative performance trajectory for multi-line spaghetti charts
- Robust statistics: Hit Rate (Win %), Median, Best/Worst, Avg Win/Loss, Risk-Reward
- Statistical sample size guardrails (N < 5 warning)
"""
import bisect
import statistics
from typing import Dict, List, Optional, Tuple, Union

from .models import (
    AverageTrajectoryPoint,
    DailyPriceBar,
    EventInstanceResult,
    EventPathPoint,
    EventStudyGuardrailWarning,
    EventStudyReport,
    HorizonSummaryStats,
)


class EventStudyEngine:
    """Deterministic, pure-domain calculation engine for Event Studies."""

    DEFAULT_HORIZONS = [5, 10, 20, 60]
    MIN_SAMPLE_THRESHOLD = 5

    @classmethod
    def evaluate(
        cls,
        ticker: str,
        event_name: str,
        bars: List[DailyPriceBar],
        event_items: List[Union[str, Tuple[str, Optional[str]]]],
        horizons: Optional[List[int]] = None,
    ) -> EventStudyReport:
        """Evaluates an event study for a given ticker and set of historical event dates."""
        target_horizons = sorted(horizons or cls.DEFAULT_HORIZONS)
        max_horizon = max(target_horizons) if target_horizons else 60

        # Sort bars chronologically
        sorted_bars = sorted(bars, key=lambda b: b.date)
        bar_dates = [b.date for b in sorted_bars]
        warnings: List[EventStudyGuardrailWarning] = []

        if not sorted_bars:
            warnings.append(
                EventStudyGuardrailWarning(
                    code="NO_DATA_AVAILABLE",
                    message="No historical price bars available for analysis.",
                )
            )
            return EventStudyReport(
                ticker=ticker,
                event_name=event_name,
                horizons=target_horizons,
                events=[],
                horizon_stats={
                    h: HorizonSummaryStats(
                        horizon_days=h,
                        sample_size=0,
                        hit_rate_pct=0.0,
                        avg_return_pct=0.0,
                        median_return_pct=0.0,
                    )
                    for h in target_horizons
                },
                average_trajectory=[],
                warnings=warnings,
            )

        event_results: List[EventInstanceResult] = []

        for idx, item in enumerate(event_items, start=1):
            if isinstance(item, tuple):
                raw_date, description = item[0], item[1]
            else:
                raw_date, description = item, None

            clean_date = str(raw_date).strip()
            if not clean_date:
                continue

            event_id = f"evt_{idx}_{clean_date}"
            instance = cls._compute_single_event(
                event_id=event_id,
                event_date=clean_date,
                description=description,
                bars=sorted_bars,
                bar_dates=bar_dates,
                horizons=target_horizons,
                max_horizon=max_horizon,
            )
            if instance is not None:
                event_results.append(instance)

        # Compute horizon statistics
        horizon_stats: Dict[int, HorizonSummaryStats] = {}
        for h in target_horizons:
            stats = cls._compute_horizon_stats(h, event_results)
            horizon_stats[h] = stats

        # Compute average daily trajectory
        avg_trajectory = cls._compute_average_trajectory(event_results, max_horizon)

        # Check guardrails
        total_events = len(event_results)
        if 0 < total_events < cls.MIN_SAMPLE_THRESHOLD:
            warnings.append(
                EventStudyGuardrailWarning(
                    code="UNRELIABLE_SAMPLE_SIZE",
                    message=(
                        f"Sample size is small (N={total_events} < {cls.MIN_SAMPLE_THRESHOLD}). "
                        "Historical patterns may have high variance and lack statistical significance."
                    ),
                )
            )

        # Check for partial horizons
        unfulfilled = any(
            any(inst.forward_returns.get(h) is None for h in target_horizons)
            for inst in event_results
        )
        if unfulfilled:
            warnings.append(
                EventStudyGuardrailWarning(
                    code="PARTIAL_HORIZONS",
                    message="Some recent events have not yet matured to the longest forward horizon.",
                )
            )

        return EventStudyReport(
            ticker=ticker,
            event_name=event_name,
            horizons=target_horizons,
            events=event_results,
            horizon_stats=horizon_stats,
            average_trajectory=avg_trajectory,
            warnings=warnings,
        )

    @classmethod
    def _compute_single_event(
        cls,
        event_id: str,
        event_date: str,
        description: Optional[str],
        bars: List[DailyPriceBar],
        bar_dates: List[str],
        horizons: List[int],
        max_horizon: int,
    ) -> Optional[EventInstanceResult]:
        """Resolves entry price and forward return trajectory for one event."""
        if not bar_dates or event_date < bar_dates[0]:
            # Event occurred before our earliest available bar
            return None

        # Find position of event_date in bar_dates
        idx = bisect.bisect_left(bar_dates, event_date)
        if idx >= len(bars):
            # Event occurred after our latest available bar
            return None

        # Determine entry bar (t+1)
        if bar_dates[idx] == event_date:
            # Event occurred on a trading day -> entry is the NEXT trading day
            entry_idx = idx + 1
        else:
            # Event occurred on a weekend/holiday -> entry is the first trading day on/after the event
            entry_idx = idx

        if entry_idx >= len(bars):
            # Next trading day has not occurred yet
            return None

        entry_bar = bars[entry_idx]
        entry_price = entry_bar.open if entry_bar.open > 0 else entry_bar.close
        if entry_price <= 0:
            return None

        entry_date = entry_bar.date

        # Compute day-by-day trajectory up to max_horizon
        daily_trajectory: List[EventPathPoint] = []
        forward_returns: Dict[int, Optional[float]] = {}
        is_win_by_horizon: Dict[int, Optional[bool]] = {}

        for offset in range(1, max_horizon + 1):
            bar_pos = entry_idx + (offset - 1)
            if bar_pos < len(bars):
                curr_bar = bars[bar_pos]
                ret = (curr_bar.close / entry_price) - 1.0
                ret_pct = round(ret * 100.0, 2)
                daily_trajectory.append(
                    EventPathPoint(
                        day_offset=offset,
                        date=curr_bar.date,
                        close=curr_bar.close,
                        cumulative_return_pct=ret_pct,
                    )
                )

        # Extract horizon returns
        for h in horizons:
            # Target bar is entry_idx + (h - 1)
            target_pos = entry_idx + (h - 1)
            if target_pos < len(bars):
                target_bar = bars[target_pos]
                ret = (target_bar.close / entry_price) - 1.0
                ret_pct = round(ret * 100.0, 2)
                forward_returns[h] = ret_pct
                is_win_by_horizon[h] = ret_pct > 0.0
            else:
                forward_returns[h] = None
                is_win_by_horizon[h] = None

        return EventInstanceResult(
            event_id=event_id,
            event_date=event_date,
            entry_date=entry_date,
            entry_price=round(entry_price, 2),
            description=description,
            forward_returns=forward_returns,
            daily_trajectory=daily_trajectory,
            is_win_by_horizon=is_win_by_horizon,
        )

    @classmethod
    def _compute_horizon_stats(
        cls,
        horizon: int,
        events: List[EventInstanceResult],
    ) -> HorizonSummaryStats:
        """Aggregates distribution metrics for a single forward horizon."""
        valid_returns = [
            e.forward_returns[horizon]
            for e in events
            if e.forward_returns.get(horizon) is not None
        ]

        n = len(valid_returns)
        if n == 0:
            return HorizonSummaryStats(
                horizon_days=horizon,
                sample_size=0,
                hit_rate_pct=0.0,
                avg_return_pct=0.0,
                median_return_pct=0.0,
            )

        wins = [r for r in valid_returns if r > 0.0]
        losses = [r for r in valid_returns if r < 0.0]
        win_count = len(wins)
        loss_count = len(losses)

        hit_rate = round((win_count / n) * 100.0, 2)
        avg_ret = round(sum(valid_returns) / n, 2)
        median_ret = round(statistics.median(valid_returns), 2)
        best_ret = round(max(valid_returns), 2)
        worst_ret = round(min(valid_returns), 2)

        avg_win = round(sum(wins) / win_count, 2) if win_count > 0 else None
        avg_loss = round(sum(losses) / loss_count, 2) if loss_count > 0 else None

        risk_reward: Optional[float] = None
        if avg_win is not None and avg_loss is not None and abs(avg_loss) > 0.001:
            risk_reward = round(avg_win / abs(avg_loss), 2)

        return HorizonSummaryStats(
            horizon_days=horizon,
            sample_size=n,
            hit_rate_pct=hit_rate,
            avg_return_pct=avg_ret,
            median_return_pct=median_ret,
            best_return_pct=best_ret,
            worst_return_pct=worst_ret,
            avg_win_pct=avg_win,
            avg_loss_pct=avg_loss,
            risk_reward_ratio=risk_reward,
            win_count=win_count,
            loss_count=loss_count,
        )

    @classmethod
    def _compute_average_trajectory(
        cls,
        events: List[EventInstanceResult],
        max_horizon: int,
    ) -> List[AverageTrajectoryPoint]:
        """Calculates average cumulative return path day-by-day across all events."""
        trajectory_by_day: Dict[int, List[float]] = {d: [] for d in range(1, max_horizon + 1)}

        for ev in events:
            for pt in ev.daily_trajectory:
                if pt.day_offset in trajectory_by_day:
                    trajectory_by_day[pt.day_offset].append(pt.cumulative_return_pct)

        result: List[AverageTrajectoryPoint] = []
        for d in range(1, max_horizon + 1):
            values = trajectory_by_day[d]
            if values:
                mean_val = round(sum(values) / len(values), 2)
                result.append(
                    AverageTrajectoryPoint(
                        day_offset=d,
                        avg_return_pct=mean_val,
                        sample_count=len(values),
                    )
                )
        return result
