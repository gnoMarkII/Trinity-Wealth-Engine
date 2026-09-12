"""Application-facing port for note-shaped write use cases."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Protocol, Union


class KnowledgeNoteWritePort(Protocol):
    """Write note intent through an injected application write boundary."""

    def write_note(
        self,
        metadata: dict[str, Any],
        body: str,
        filename: Optional[str] = None,
        target_path: Optional[Union[str, Path]] = None,
        expected_current_hash: Optional[str] = None,
        companion_artifacts: Optional[Mapping[str, Union[str, bytes]]] = None,
    ) -> Any:
        ...

    def write_capture(self, metadata: dict[str, Any], body: str, *, filename: Optional[str] = None) -> Path:
        ...
