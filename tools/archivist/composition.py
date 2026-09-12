"""Approved infrastructure composition roots for Vault write adapters."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from application.knowledge.note_write_ports import KnowledgeNoteWritePort
from application.knowledge.write_ports import KnowledgeWritePort
from tools.archivist.knowledge_note_writer import BrokerKnowledgeNoteWriter
from tools.archivist.vault_paths import VaultPaths
from tools.archivist.write_broker import KnowledgeWriteBroker


def build_knowledge_write_port(
    *,
    vault_paths: Optional[VaultPaths] = None,
    runtime_base: Optional[str | Path] = None,
) -> KnowledgeWritePort:
    """Build the approved in-process durable write port."""
    paths = vault_paths or VaultPaths()
    return KnowledgeWriteBroker(vault_paths=paths, runtime_base=runtime_base)


def build_knowledge_note_writer(
    *,
    vault_paths: Optional[VaultPaths] = None,
    write_port: Optional[KnowledgeWritePort] = None,
    runtime_base: Optional[str | Path] = None,
) -> KnowledgeNoteWritePort:
    paths = vault_paths or VaultPaths()
    port = write_port or build_knowledge_write_port(vault_paths=paths, runtime_base=runtime_base)
    return BrokerKnowledgeNoteWriter(vault_paths=paths, write_port=port)
