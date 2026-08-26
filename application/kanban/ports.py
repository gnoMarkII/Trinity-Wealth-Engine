"""Outbound Ports for Kanban Context."""
from typing import Protocol, Optional, List, Dict, Any


class KanbanRepositoryPort(Protocol):
    """Abstract port for accessing and modifying Kanban persistence."""

    def list_kanban_cards(self) -> List[Dict[str, Any]]:
        ...

    def get_kanban_card(self, card_id: str) -> Optional[Dict[str, Any]]:
        ...

    def find_kanban_card_by_title_in_column(
        self, title: str, column_name: str, prompt: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        ...

    def find_kanban_card_by_source_key(
        self, source_key: str
    ) -> Optional[Dict[str, Any]]:
        ...

    def create_kanban_card(
        self,
        card_id: str,
        title: str,
        column_name: str,
        flow: str = "manager",
        prompt: Optional[str] = None,
        source_key: Optional[str] = None,
        scope: str = "both",
        discord_notify: bool = True,
        is_verified: bool = True,
    ) -> None:
        ...

    def update_kanban_card(
        self,
        card_id: str,
        title: str,
        flow: str,
        prompt: Optional[str] = None,
        scope: str = "both",
        discord_notify: Optional[bool] = None,
    ) -> None:
        ...

    def delete_kanban_card(self, card_id: str) -> bool:
        ...

    def move_kanban_card(
        self,
        card_id: str,
        target_column: str,
        job_id: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        ...

    def update_kanban_display_seq(self, card_id: str, new_seq: int) -> None:
        ...

    def toggle_discord(self, card_id: str, enabled: Optional[bool] = None) -> Optional[Dict[str, Any]]:
        ...
