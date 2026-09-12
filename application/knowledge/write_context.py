"""Optional process context for legacy-shaped use cases.

Production composition roots should pass ``KnowledgeNoteWritePort`` explicitly.
The context exists only for boundary adapters and tests that call old
function-shaped APIs; it never constructs infrastructure itself.
"""
from __future__ import annotations

from contextvars import ContextVar, Token
from pathlib import Path
from typing import Callable, Optional

from application.knowledge.note_write_ports import KnowledgeNoteWritePort


NoteWriterProvider = Callable[[Path], KnowledgeNoteWritePort]
_provider: ContextVar[Optional[NoteWriterProvider]] = ContextVar("knowledge_note_writer_provider", default=None)


def bind_note_writer_provider(provider: NoteWriterProvider) -> Token[Optional[NoteWriterProvider]]:
    return _provider.set(provider)


def reset_note_writer_provider(token: Token[Optional[NoteWriterProvider]]) -> None:
    _provider.reset(token)


def current_note_writer(vault_root: str | Path) -> KnowledgeNoteWritePort:
    provider = _provider.get()
    if provider is None:
        raise RuntimeError("KnowledgeNoteWritePort must be injected by a composition root")
    return provider(Path(vault_root).resolve())
