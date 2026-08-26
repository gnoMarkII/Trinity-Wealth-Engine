"""Application Layer Service for Kanban Cards Lifecycle."""
import uuid
from typing import Optional, List, Tuple, Dict, Any

from application.kanban.ports import KanbanRepositoryPort
from application.kanban.dto import KanbanCardDTO


def _dict_to_dto(d: Dict[str, Any]) -> KanbanCardDTO:
    return KanbanCardDTO(
        card_id=d["card_id"],
        title=d["title"],
        prompt=d.get("prompt"),
        column_name=d["column_name"],
        job_id=d.get("job_id"),
        source_key=d.get("source_key"),
        flow=d["flow"],
        scope=d.get("scope", "both"),
        display_seq=d.get("display_seq"),
        discord_notify=bool(d["discord_notify"]) if d.get("discord_notify") is not None else True,
        is_verified=bool(d["is_verified"]) if d.get("is_verified") is not None else True,
        created_at=d["created_at"],
        updated_at=d["updated_at"],
    )


class KanbanApplicationService:
    """Application service for managing Kanban boards, columns, cards, and discord notifications."""

    def __init__(self, repo: KanbanRepositoryPort):
        self._repo = repo

    def list_cards(self) -> List[KanbanCardDTO]:
        rows = self._repo.list_kanban_cards()
        return [_dict_to_dto(r) for r in rows]

    def get_card(self, card_id: str) -> Optional[KanbanCardDTO]:
        row = self._repo.get_kanban_card(card_id)
        return _dict_to_dto(row) if row else None

    def create_card(
        self,
        title: str,
        flow: str = "manager",
        prompt: Optional[str] = None,
        source_key: Optional[str] = None,
        scope: str = "both",
    ) -> Tuple[KanbanCardDTO, bool]:
        title_clean = title.strip()
        prompt_clean = (prompt or "").strip() or None
        source_key_clean = (source_key or "").strip() or None

        if source_key_clean:
            existing = self._repo.find_kanban_card_by_source_key(source_key_clean)
            if existing is not None:
                return _dict_to_dto(existing), False

        existing = self._repo.find_kanban_card_by_title_in_column(
            title_clean, "backlog", prompt=prompt_clean
        )
        if existing is not None:
            return _dict_to_dto(existing), False

        card_id = str(uuid.uuid4())
        self._repo.create_kanban_card(
            card_id=card_id,
            title=title_clean,
            column_name="backlog",
            flow=flow,
            prompt=prompt_clean,
            source_key=source_key_clean,
            scope=scope,
        )
        row = self._repo.get_kanban_card(card_id)
        return _dict_to_dto(row or {}), True

    def update_card(
        self,
        card_id: str,
        title: str,
        flow: str,
        prompt: Optional[str] = None,
        scope: str = "both",
    ) -> Optional[KanbanCardDTO]:
        title_clean = title.strip()
        prompt_clean = (prompt or "").strip() or None

        existing = self._repo.get_kanban_card(card_id)
        if existing is None:
            return None
        self._repo.update_kanban_card(
            card_id=card_id,
            title=title_clean,
            prompt=prompt_clean,
            flow=flow,
            scope=scope,
        )
        updated = self._repo.get_kanban_card(card_id)
        return _dict_to_dto(updated or {})

    def delete_card(self, card_id: str) -> bool:
        return self._repo.delete_kanban_card(card_id)

    def move_card(
        self,
        card_id: str,
        target_column: str,
        new_display_seq: Optional[int] = None,
        job_id: Optional[str] = None,
    ) -> Optional[KanbanCardDTO]:
        if new_display_seq is not None:
            self._repo.update_kanban_display_seq(card_id, new_display_seq)
        row = self._repo.move_kanban_card(card_id, target_column, job_id=job_id)
        return _dict_to_dto(row) if row else None

    def update_display_seq(self, card_id: str, new_seq: int) -> None:
        self._repo.update_kanban_display_seq(card_id, new_seq)

    def toggle_discord(self, card_id: str, enabled: Optional[bool] = None) -> Optional[KanbanCardDTO]:
        row = self._repo.toggle_discord(card_id, enabled=enabled)
        return _dict_to_dto(row) if row else None
