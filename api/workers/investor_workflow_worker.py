"""Dedicated durable workflow worker for Investor Essence asynchronous background operations."""
from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any, Dict, Optional

from application.investor_essence import (
    BucketPlanningService,
    ClaimReviewService,
    InterviewService,
    InvestmentAxisService,
    InvestorEssenceError,
    InvestorRuntimeUowFactory,
)

log = logging.getLogger(__name__)


class InvestorWorkflowWorker:
    """Asynchronous background worker that periodically polls and processes pending investor operations."""

    def __init__(
        self,
        uow_factory: InvestorRuntimeUowFactory,
        interview_service: InterviewService,
        claim_review_service: ClaimReviewService,
        axis_service: InvestmentAxisService,
        bucket_service: BucketPlanningService,
        worker_id: Optional[str] = None,
        poll_interval_seconds: float = 2.0,
    ) -> None:
        self._uow_factory = uow_factory
        self._interview_service = interview_service
        self._claim_review_service = claim_review_service
        self._axis_service = axis_service
        self._bucket_service = bucket_service
        self._worker_id = worker_id or f"investor-worker-{uuid.uuid4().hex[:8]}"
        self._poll_interval = poll_interval_seconds
        self._running = False
        self._task: Optional[asyncio.Task] = None

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop(), name="investor_workflow_worker")
        log.info("InvestorWorkflowWorker started (id=%s, interval=%.1fs)", self._worker_id, self._poll_interval)

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
        log.info("InvestorWorkflowWorker stopped")

    def process_one_batch(self, limit: int = 10) -> int:
        """Synchronously processes up to `limit` pending leased operations."""
        processed_count = 0

        for _ in range(limit):
            with self._uow_factory.open() as uow:
                claim = uow.operations.claim_lease(worker_id=self._worker_id, lease_seconds=30)
                if not claim:
                    break

            op_id = claim["operation_id"]
            task_type = claim["task_type"]
            resource_id = claim["resource_id"]
            payload = claim.get("payload", {})
            fence = claim["fencing_token"]

            try:
                result = self._dispatch(task_type, resource_id, payload)
                with self._uow_factory.open() as uow:
                    uow.operations.complete_with_fence(op_id, fence, result)
                processed_count += 1
                log.info("Investor operation %s (%s) completed successfully", op_id, task_type)
            except Exception as exc:
                is_retryable = not isinstance(exc, InvestorEssenceError)
                err_msg = str(exc)
                log.warning("Investor operation %s (%s) failed: %s", op_id, task_type, err_msg)
                with self._uow_factory.open() as uow:
                    uow.operations.fail_with_fence(
                        operation_id=op_id,
                        fence=fence,
                        error_code=exc.__class__.__name__,
                        error_message=err_msg,
                        retryable=is_retryable,
                    )

        return processed_count

    def _dispatch(self, task_type: str, resource_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        prompt_version = payload.get("prompt_version", "1.0")
        expected_revision = payload.get("resource_revision")

        if task_type in ("interview_next", "advance_question"):
            view = self._interview_service.advance_next_question(
                session_id=resource_id,
                expected_revision=expected_revision,
                prompt_version=prompt_version,
            )
            return {"session_id": view.session_id, "questions_count": view.questions_count, "answers_count": view.answers_count}

        if task_type in ("essence_summary", "summarize"):
            view = self._claim_review_service.generate_summary(
                session_id=resource_id,
                expected_revision=expected_revision,
                prompt_version=prompt_version,
            )
            return {"summary_id": view.summary_id, "claims_count": len(view.claims), "statement": view.statement}

        if task_type in ("investment_axis", "axis_draft"):
            view = self._axis_service.create_axis_draft(
                portfolio_id=resource_id,
                prompt_version=prompt_version,
            )
            return {"draft_id": view.draft_id, "is_complete": view.is_complete}

        if task_type in ("bucket_proposal", "bucket_plan"):
            view = self._bucket_service.create_bucket_plan_draft(
                portfolio_id=resource_id,
                prompt_version=prompt_version,
            )
            return {"draft_id": view.draft_id, "buckets_count": len(view.purpose_buckets)}

        raise ValueError(f"Unknown investor operation task_type: {task_type}")

    async def _loop(self) -> None:
        while self._running:
            try:
                await asyncio.sleep(self._poll_interval)
                if not self._running:
                    break

                loop = asyncio.get_running_loop()
                count = await loop.run_in_executor(None, self.process_one_batch, 10)
                if count > 0:
                    log.info("InvestorWorkflowWorker processed %d operations", count)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.exception("Unexpected error in InvestorWorkflowWorker loop: %s", exc)
