"""Core domain package for Structured Thesis, Falsifiers & As-Of Research State."""
from core.thesis.evidence_taxonomy import EvidenceProvenance, EvidenceTier
from core.thesis.models import (
    AsOfResearchSnapshot,
    CatalystEvent,
    FalsifierRule,
    ThesisClaimRecord,
    ThesisExceptionRow,
    compute_thesis_checksum,
)
from core.thesis.evaluator import ThesisEvaluator

__all__ = [
    "EvidenceTier",
    "EvidenceProvenance",
    "FalsifierRule",
    "CatalystEvent",
    "ThesisClaimRecord",
    "ThesisExceptionRow",
    "AsOfResearchSnapshot",
    "compute_thesis_checksum",
    "ThesisEvaluator",
]

