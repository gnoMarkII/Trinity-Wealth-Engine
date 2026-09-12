"""Application DTOs for NotebookLM Context."""
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class NotebookLMAvailableSourceDTO:
    file_path: str
    title: str
    date_part: Optional[str] = None
    is_verified: bool = True


@dataclass(frozen=True)
class NotebookLMStatusDTO:
    job_id: str
    status: str
    audio_path: Optional[str] = None
    notebook_id: Optional[str] = None
    error: Optional[str] = None
    recovery_status: Optional[str] = None
