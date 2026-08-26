"""Outbound Ports for NotebookLM Context."""
from pathlib import Path
from typing import Protocol, Optional, Dict, Any, Callable


class NotebookLMJobRepositoryPort(Protocol):
    """Abstract port for querying NotebookLM jobs."""

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        ...


class NotebookLMCardRepositoryPort(Protocol):
    def get_card(self, card_id: str) -> Optional[Dict[str, Any]]:
        ...

    def set_source(self, card_id: str, prompt: str, is_verified: bool) -> None:
        ...

    def move_card(self, card_id: str, column_name: str, job_id: Optional[str] = None) -> None:
        ...


class NotebookLMDispatchPort(Protocol):
    def dispatch(self, instruction: str, card_id: str, flow: str = "notebooklm", scope: str = "both") -> str:
        ...


class NotebookLMBinaryPort(Protocol):
    def check_available(self) -> None:
        ...


class NotebookLMSourceCatalogPort(Protocol):
    """Source discovery and path-boundary policy for NotebookLM inputs."""

    def list_sources(self) -> list[Dict[str, Any]]:
        ...

    def resolve_source(self, reference: str) -> Dict[str, Any]:
        ...


class NotebookLMManifestPort(Protocol):
    """Read-only manifest lookup for a dispatched NotebookLM source."""

    def get_for_source(self, source_path: str) -> Any:
        ...


class NotebookLMWorkerStatePort(Protocol):
    """State operations required by the post-production worker use case."""

    def append_log(self, job_id: str, node: str, message: str) -> None:
        ...

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        ...

    def get_card(self, card_id: str) -> Optional[Dict[str, Any]]:
        ...

    def mark_discord_events_sent(self, card_id: str, event_ids: list[str]) -> None:
        ...


class NotebookLMBriefingContentPort(Protocol):
    def read(self, path: str) -> str:
        ...


class NotebookLMNotificationPort(Protocol):
    def send(self, *, audio_path: Any, title: str, summary: str, source_ref: str) -> Any:
        ...


class NotebookLMNotificationOutboxPort(Protocol):
    """Durable idempotency boundary for external notifications."""

    def enqueue(
        self,
        *,
        event_id: str,
        idempotency_key: str,
        aggregate_type: str,
        aggregate_id: str,
        event_type: str,
        payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        ...

    def mark_sent(self, idempotency_key: str) -> None:
        ...

    def mark_failed(self, idempotency_key: str, error: str) -> None:
        ...

    def get(self, idempotency_key: str) -> Optional[Dict[str, Any]]:
        ...

    def list_pending(self, limit: int = 100) -> list[Dict[str, Any]]:
        ...


class NotebookLMPipelinePort(Protocol):
    """NotebookLM post-production pipeline boundary."""

    async def run(
        self,
        briefing_path: Path,
        *,
        confirm_generation: bool,
        notebooklm_prompts: Any,
        on_step: Callable[[str, str], None],
    ) -> Any:
        ...


class NotebookLMPromptExtractorPort(Protocol):
    def extract(self, raw_text: str) -> Any:
        ...
