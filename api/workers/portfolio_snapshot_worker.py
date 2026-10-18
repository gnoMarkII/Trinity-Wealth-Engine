"""Background Daily Snapshot Worker for Portfolios."""
from __future__ import annotations

import asyncio
from datetime import datetime
import logging
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from tools.portfolio.service import PortfolioService

log = logging.getLogger(__name__)


class PortfolioDailySnapshotWorker:
    """Asynchronous background worker that ensures all portfolios have a daily performance snapshot."""

    def __init__(
        self,
        service: PortfolioService,
        check_interval_seconds: float = 1800.0,
    ) -> None:
        self._service = service
        self._interval = check_interval_seconds
        self._running = False
        self._task: Optional[asyncio.Task] = None

    def snapshot_all_portfolios(self, refresh_prices: bool = False, force: bool = False) -> dict[str, str]:
        """Record performance snapshot for all portfolios that don't have one today."""
        today_str = datetime.now().strftime("%Y-%m-%d")
        results: dict[str, str] = {}
        try:
            portfolios = self._service.list_portfolios()
        except Exception as exc:
            log.warning("PortfolioDailySnapshotWorker: Failed to list portfolios: %s", exc)
            return {"error": f"Failed to list portfolios: {exc}"}

        for p in portfolios:
            pid = p.id
            try:
                if not force:
                    recent = self._service.get_structured_performance_history(days=2, portfolio_id=pid)
                    if any(r.get("Date") == today_str for r in recent):
                        results[pid] = f"Already snapshotted for {today_str}"
                        continue

                msg = self._service.record_performance_snapshot(refresh_prices=refresh_prices, portfolio_id=pid)
                results[pid] = msg
                log.info("PortfolioDailySnapshotWorker: Snapshot recorded for %s: %s", pid, msg)
            except Exception as exc:
                results[pid] = f"Error: {exc}"
                log.warning("PortfolioDailySnapshotWorker: Error snapshotting %s: %s", pid, exc)

        return results

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop(), name="portfolio_daily_snapshot_worker")
        log.info("PortfolioDailySnapshotWorker started (check_interval=%.1fs)", self._interval)

    async def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        log.info("PortfolioDailySnapshotWorker stopped")

    async def _loop(self) -> None:
        # Initial small delay to let server boot and settle
        await asyncio.sleep(3.0)
        while self._running:
            try:
                loop = asyncio.get_running_loop()
                # Run synchronous filesystem/portfolio work in threadpool executor
                await loop.run_in_executor(None, self.snapshot_all_portfolios, False, False)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.warning("Error in PortfolioDailySnapshotWorker loop: %s", exc)

            try:
                await asyncio.sleep(self._interval)
            except asyncio.CancelledError:
                break
