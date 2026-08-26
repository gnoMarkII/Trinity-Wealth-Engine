"""Thin Adapter wrapping KanbanApplicationService for Earnings Call Context."""
from application.earnings_call.dto import KanbanCardResultDTO
from application.kanban.service import KanbanApplicationService
from core.logger import get_logger

log = get_logger(__name__)


class KanbanEarningsCallAdapter:
    """Implements EarningsCallKanbanPort by delegating to KanbanApplicationService."""

    def __init__(self, kanban_service: KanbanApplicationService) -> None:
        self._kanban_service = kanban_service

    def ensure_card(
        self, ticker: str, period: str, vault_relative_path: str, source_key: str
    ) -> KanbanCardResultDTO:
        title = f"[{ticker.upper()}] Earnings Call {period}"
        prompt = vault_relative_path

        card_dto, created = self._kanban_service.create_card(
            title=title,
            flow="manager",
            prompt=prompt,
            source_key=source_key,
            scope="both",
        )

        log.info(
            "Kanban card ensured | title=%s | card_id=%s | created=%s | source_key=%s",
            title,
            card_dto.card_id,
            created,
            source_key,
        )

        return KanbanCardResultDTO(
            card_id=card_dto.card_id,
            created=created,
            title=card_dto.title,
            status="ok",
        )
