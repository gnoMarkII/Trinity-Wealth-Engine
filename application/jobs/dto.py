"""Application DTOs for Background Jobs & Execution Context."""
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass(frozen=True)
class SpecialistOutputDTO:
    node_name: str
    label: str
    content: str
    seq: int
    created_at: float


@dataclass(frozen=True)
class JobOutputsDTO:
    job_id: str
    status: str
    executive_summary: Optional[str]
    executive_summary_created_at: Optional[float]
    specialists: List[SpecialistOutputDTO]
    last_seq: int
    error_message: Optional[str] = None


@dataclass(frozen=True)
class JobStatusDTO:
    job_id: str
    status: str
    created_at: float
    updated_at: float
    error_message: Optional[str] = None
