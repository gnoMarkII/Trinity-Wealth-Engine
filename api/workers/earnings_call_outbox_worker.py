"""Background Outbox Polling Worker for Earnings Call Saga."""
import asyncio
import logging
from typing import Optional

from application.earnings_call.service import EarningsCallApplicationService

log = logging.getLogger(__name__)


class EarningsCallOutboxWorker:
    """Asynchronous background worker that periodically polls and processes pending outbox events."""

    def __init__(
        self,
        service: Optional[EarningsCallApplicationService] = None,
        poll_interval_seconds: float = 3.0,
    ) -> None:
        self._service = service
        self._poll_interval = poll_interval_seconds
        self._running = False
        self._task: Optional[asyncio.Task] = None

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop(), name="earnings_call_outbox_worker")
        log.info("EarningsCallOutboxWorker started (interval=%.1fs)", self._poll_interval)

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
        log.info("EarningsCallOutboxWorker stopped")

    async def _loop(self) -> None:
        while self._running:
            try:
                await asyncio.sleep(self._poll_interval)
                if not self._running:
                    break

                service = self._service
                if service is None:
                    from api.dependencies import get_earnings_call_service

                    service = get_earnings_call_service()

                # Run blocking DB work in default executor
                loop = asyncio.get_running_loop()
                count = await loop.run_in_executor(None, service.process_outbox_batch, 10)
                if count > 0:
                    log.info("EarningsCallOutboxWorker processed %d outbox events", count)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.warning("Error in EarningsCallOutboxWorker loop: %s", exc, exc_info=False)
