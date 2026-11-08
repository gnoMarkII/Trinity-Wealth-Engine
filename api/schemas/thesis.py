"""Pydantic schemas and DTOs for Thesis and Catalyst Exception Queue (Chapter 7)."""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class CatalystExceptionDTO(BaseModel):
    """Factual catalyst exception item awaiting human PM review."""
    exception_id: str
    catalyst_id: str
    entity_id: str
    classification: str
    source_id: str
    observed_fact: str
    prior_date: Optional[str] = None
    new_date: Optional[str] = None
    analyst_field: str = "materiality_assessment"
    human_assessment: Optional[str] = None
    status: str = "REVIEW_REQUIRED"
    detected_utc: str = ""


class ReviewCatalystRequestDTO(BaseModel):
    """Payload submitted by PM to acknowledge and assess a catalyst exception."""
    human_assessment: str = Field(..., min_length=1, description="Materiality assessment entered by PM")

