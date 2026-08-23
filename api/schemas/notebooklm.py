"""NotebookLM Schemas."""
from typing import Optional
from pydantic import BaseModel

class NotebookLMAvailableSourceDTO(BaseModel):
    file_path: str
    title: str
    date_part: Optional[str] = None
    is_verified: bool


class NotebookLMGenerateRequest(BaseModel):
    card_id: str
    briefing_file_path: Optional[str] = None


class NotebookLMGenerateResponse(BaseModel):
    job_id: str
    status: str


class NotebookLMStatusDTO(BaseModel):
    job_id: str
    status: str
    audio_path: Optional[str] = None
    notebook_id: Optional[str] = None
    error: Optional[str] = None


