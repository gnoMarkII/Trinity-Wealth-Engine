"""Background Outbox Polling Worker for Earnings Call Saga."""
import asyncio
import logging

from application.earnings_call.service import EarningsCallApplicationService

log = logging.getLogger(__name__)


class EarningsCallOutboxWorker:
    """Asynchronous background worker that periodically polls and processes pending outbox events."""

    def __init__(
        self,
        service: EarningsCallApplicationService,
        poll_interval_seconds: float = 3.0,
    ) -> None:
        """Create a worker with an already-composed application service.

        The worker is an infrastructure entrypoint, not a composition root.
        Requiring the service here prevents a background task from resolving
        dependencies through a module-level service locator after startup.
        """
        self._service = service
        self._poll_interval = poll_interval_seconds
        self._running = False
        self._task: asyncio.Task | None = None

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

                # Run blocking DB work in default executor
                loop = asyncio.get_running_loop()
                count = await loop.run_in_executor(None, self._service.process_outbox_batch, 10)
                if count > 0:
                    log.info("EarningsCallOutboxWorker processed %d outbox events", count)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.warning("Error in EarningsCallOutboxWorker loop: %s", exc, exc_info=False)
