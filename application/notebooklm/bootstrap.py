"""Composition root for NotebookLM Application Context."""
from pathlib import Path
from typing import Optional

from application.notebooklm.ports import (
    NotebookLMCardRepositoryPort,
    NotebookLMDispatchPort,
    NotebookLMBinaryPort,
    NotebookLMJobRepositoryPort,
    NotebookLMManifestPort,
    NotebookLMSourceCatalogPort,
)
from application.notebooklm.service import NotebookLMApplicationService


def build_notebooklm_service(
    repo: NotebookLMJobRepositoryPort,
    sources_dir: Optional[Path] = None,
    *,
    card_repo: Optional[NotebookLMCardRepositoryPort] = None,
    dispatcher: Optional[NotebookLMDispatchPort] = None,
    binary: Optional[NotebookLMBinaryPort] = None,
    source_catalog: Optional[NotebookLMSourceCatalogPort] = None,
    manifest_port: Optional[NotebookLMManifestPort] = None,
) -> NotebookLMApplicationService:
    """Build the use case from injected ports.

    This bootstrap deliberately does not construct filesystem, SQLite, queue,
    or CLI adapters.  Concrete wiring belongs to ``api.dependencies`` (the
    process composition root); tests and other entry points can inject fakes.
    """
    return NotebookLMApplicationService(
        repo=repo,
        sources_dir=sources_dir,
        card_repo=card_repo,
        dispatcher=dispatcher,
        binary=binary,
        source_catalog=source_catalog,
        manifest_port=manifest_port,
    )
