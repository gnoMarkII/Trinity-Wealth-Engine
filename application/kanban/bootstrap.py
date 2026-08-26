"""Composition root for Kanban Application Context."""
from application.kanban.ports import KanbanRepositoryPort
from application.kanban.service import KanbanApplicationService


def build_kanban_service(
    repo: KanbanRepositoryPort,
) -> KanbanApplicationService:
    """Build and wire KanbanApplicationService with dependencies."""
    return KanbanApplicationService(repo=repo)
