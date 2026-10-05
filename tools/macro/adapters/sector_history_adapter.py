"""Adjusted ETF history adapter for the sector-rotation domain."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError, as_completed
from datetime import date
import logging
import time
from typing import Any

from application.macro.sector_rotation_ports import SectorHistoryBatch

from tools.market.market_calendar import get_last_completed_regular_session, is_us_trading_day
from tools.market.ohlcv.adapters.yfinance_adapter import YFinanceOhlcvAdapter
from tools.macro.sector_rotation.domain.universe import BENCHMARK, SECTOR_TICKERS


class SectorHistoryAdapter:
    """Fetch a complete, date-aligned input set using the existing OHLCV port."""

    def __init__(
        self,
        provider: Any | None = None,
        *,
        max_workers: int = 4,
        request_timeout_seconds: float = 12.0,
        batch_timeout_seconds: float = 45.0,
    ) -> None:
        self._provider = provider or YFinanceOhlcvAdapter(
            request_timeout=request_timeout_seconds, retry_attempts=1, raise_errors=True,
        )
        self._max_workers = max(1, min(int(max_workers), 4))
        self._batch_timeout_seconds = max(0.01, float(batch_timeout_seconds))

    def fetch(self) -> SectorHistoryBatch:
        cutoff = get_last_completed_regular_session()
        symbols = (*SECTOR_TICKERS, BENCHMARK)
        histories: dict[str, dict[str, float]] = {}
        reasons: dict[str, str] = {}
        deadline = time.monotonic() + self._batch_timeout_seconds
        executor = ThreadPoolExecutor(max_workers=self._max_workers, thread_name_prefix="sector-history")
        futures = {}
        try:
            futures = {
                executor.submit(self._provider.fetch_history, symbol, "5y", "1d", True): symbol
                for symbol in symbols
            }
            try:
                for future in as_completed(futures, timeout=self._batch_timeout_seconds):
                    symbol = futures[future]
                    try:
                        frame = future.result(timeout=max(0.0, deadline - time.monotonic()))
                        histories[symbol] = self._normalize(frame, cutoff)
                        if not histories[symbol]:
                            reasons[symbol] = "price_history_unavailable"
                    except Exception as exc:  # provider-specific errors become visible row state
                        histories[symbol] = {}
                        is_timeout = isinstance(exc, TimeoutError) or "timeout" in type(exc).__name__.lower()
                        reasons[symbol] = (
                            "provider_timeout" if is_timeout or time.monotonic() >= deadline
                            else f"provider_error:{type(exc).__name__}"
                        )
            except FuturesTimeoutError:
                logging.getLogger(__name__).warning("Sector history provider batch exceeded its %.1fs deadline", self._batch_timeout_seconds)
                for future, symbol in futures.items():
                    if symbol not in histories:
                        future.cancel()
                        histories[symbol] = {}
                        reasons[symbol] = "provider_timeout"
        finally:
            # Do not let executor shutdown turn the batch deadline into a wait for
            # futures that already missed it. Provider HTTP calls have their own timeout.
            executor.shutdown(wait=False, cancel_futures=True)
        observed = [session for history in histories.values() for session in history]
        start = min((date.fromisoformat(session) for session in observed), default=cutoff)
        expected_sessions = []
        current = start
        while current <= cutoff:
            if is_us_trading_day(current):
                expected_sessions.append(current.isoformat())
            current = date.fromordinal(current.toordinal() + 1)
        return SectorHistoryBatch(histories, reasons, cutoff, tuple(expected_sessions))

    @staticmethod
    def _normalize(frame: Any, cutoff: date) -> dict[str, float]:
        if frame is None or getattr(frame, "empty", True):
            return {}
        close = frame.get("Close") if hasattr(frame, "get") else None
        if close is None:
            return {}
        if hasattr(close, "items"):
            entries = close.items()
        else:
            return {}
        normalized: dict[str, float] = {}
        for index, value in entries:
            try:
                stamp = index
                if getattr(stamp, "tzinfo", None) is not None:
                    stamp = stamp.tz_convert("America/New_York")
                session = stamp.date() if hasattr(stamp, "date") else date.fromisoformat(str(stamp)[:10])
                price = float(value)
            except (TypeError, ValueError, AttributeError):
                continue
            if session > cutoff or not is_us_trading_day(session) or not price > 0:
                continue
            normalized[session.isoformat()] = price
        return dict(sorted(normalized.items()))
