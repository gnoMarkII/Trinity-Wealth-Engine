"""Six-Condition Task Routing Engine (Chapter 1 & Laboratory 1).

Architecture Rule (Layer 1 - Core):
Pure Python standard library only — strictly zero external dependencies per Hexagonal
Architecture AST Domain Layer Isolation rules.

Reference:
Claude Code for Hedge Funds, Chapter 1 (pp. 11-23), Laboratory 1 (pp. 81-83):
"Route the Work Before You Open Code. A routing table that makes alternatives credible.
Route eight real work shapes across four execution surfaces.
Never route financial math or capital allocation to unconstrained LLMs.
Your completed row answers four questions in one glance:
What is authoritative? What must be reconstructable?
What test or review can reject the output? What condition would reverse the route?"
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple


class WorkShape(str, Enum):
    """The eight canonical financial work shapes from Chapter 1 (pp. 21-23)."""
    ROUTINE_READING_OR_INGEST = "ROUTINE_READING_OR_INGEST"
    POLISHED_DOCUMENT_SYNTHESIS = "POLISHED_DOCUMENT_SYNTHESIS"
    LIVE_FINANCIAL_CALCULATION = "LIVE_FINANCIAL_CALCULATION"
    STATEFUL_RECURRING_AUDIT = "STATEFUL_RECURRING_AUDIT"
    CALENDAR_EVENT_TRACKING = "CALENDAR_EVENT_TRACKING"
    PIT_MACRO_SERIES = "PIT_MACRO_SERIES"
    PORTFOLIO_EXPOSURE_AND_HEDGING = "PORTFOLIO_EXPOSURE_AND_HEDGING"
    MNPI_OR_INCIDENT_REVIEW = "MNPI_OR_INCIDENT_REVIEW"


class ExecutionSurface(str, Enum):
    """The four target execution surfaces per institutional governance."""
    DETERMINISTIC_TOOL_OR_SCRIPT = "DETERMINISTIC_TOOL_OR_SCRIPT"
    LLM_RESEARCH_SYNTHESIS = "LLM_RESEARCH_SYNTHESIS"
    HUMAN_PM_DECISION = "HUMAN_PM_DECISION"
    QUARANTINE_OR_POLICY_REJECT = "QUARANTINE_OR_POLICY_REJECT"


@dataclass(frozen=True)
class TaskRoutingContext:
    """Pre-flight context evaluated during the Six-Condition Test."""
    has_authoritative_calculator: bool = False
    requires_reconstructability: bool = False
    has_rejection_test_gate: bool = False
    touches_capital_or_policy: bool = False
    is_data_approved_and_sanitized: bool = True
    contains_mnpi_risk: bool = False
    user_role: str = "analyst"


@dataclass(frozen=True)
class TaskRouteRecord:
    """Immutable audit record justifying the selected execution surface."""
    task_id: str
    task_description: str
    work_shape: WorkShape
    selected_surface: ExecutionSurface
    authoritative_system: str
    reconstructable_artifact: str
    rejection_gate: str
    reversal_condition: str
    rationale: str
    status: str = "REVIEW_REQUIRED"
    evaluated_at_utc: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_description": self.task_description,
            "work_shape": self.work_shape.value,
            "selected_surface": self.selected_surface.value,
            "authoritative_system": self.authoritative_system,
            "reconstructable_artifact": self.reconstructable_artifact,
            "rejection_gate": self.rejection_gate,
            "reversal_condition": self.reversal_condition,
            "rationale": self.rationale,
            "status": self.status,
            "evaluated_at_utc": self.evaluated_at_utc,
        }


def evaluate_task_route(
    task_id: str,
    task_description: str,
    work_shape: WorkShape,
    context: TaskRoutingContext,
    as_of_utc: Optional[str] = None,
) -> TaskRouteRecord:
    """Applies the Six-Condition Test to deterministically select the execution surface.
    
    Hard Invariants:
    1. MNPI Risk or Unsanitized Data -> QUARANTINE_OR_POLICY_REJECT immediately.
    2. Capital, position changes, or hedging approvals -> HUMAN_PM_DECISION.
    3. Deterministic math, exposure, or vintage reconstruction -> DETERMINISTIC_TOOL_OR_SCRIPT.
    4. Unstructured document reading & initial synthesis -> LLM_RESEARCH_SYNTHESIS.
    """
    now_iso = as_of_utc or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # 1. Immediate Policy Quarantine Gate
    is_mnpi = context.contains_mnpi_risk or (work_shape == WorkShape.MNPI_OR_INCIDENT_REVIEW)
    if is_mnpi or not context.is_data_approved_and_sanitized:
        return TaskRouteRecord(
            task_id=task_id,
            task_description=task_description,
            work_shape=work_shape,
            selected_surface=ExecutionSurface.QUARANTINE_OR_POLICY_REJECT,
            authoritative_system="Compliance & Security Review",
            reconstructable_artifact="review/security/quarantine_log.json",
            rejection_gate="Hostile input sanitizer or Reg FD MNPI compliance filter",
            reversal_condition="Document is formally redacted, sanitized, or approved by legal counsel",
            rationale="Untrusted instructions, injection risk, or potential MNPI cannot be submitted to AI surfaces.",
            status="STOP",
            evaluated_at_utc=now_iso,
        )

    # 2. Capital & Hedging Gate (Always Human)
    if work_shape == WorkShape.PORTFOLIO_EXPOSURE_AND_HEDGING or context.touches_capital_or_policy:
        return TaskRouteRecord(
            task_id=task_id,
            task_description=task_description,
            work_shape=work_shape,
            selected_surface=ExecutionSurface.HUMAN_PM_DECISION,
            authoritative_system="Portfolio Management / Human PM Review Gate",
            reconstructable_artifact="review/signoffs/trade_rebalance_decision.md",
            rejection_gate="PM explicit sign-off and risk parameter boundary check",
            reversal_condition="Task is purely analytical and completely disconnected from execution authority",
            rationale="Hedging approvals and capital rebalancing require human accountability and regulatory sign-off.",
            status="REVIEW_REQUIRED",
            evaluated_at_utc=now_iso,
        )

    # 3. Deterministic Financial Math & Exposure Gate
    if work_shape in (
        WorkShape.LIVE_FINANCIAL_CALCULATION,
        WorkShape.PIT_MACRO_SERIES,
        WorkShape.STATEFUL_RECURRING_AUDIT,
    ) or context.has_authoritative_calculator:
        return TaskRouteRecord(
            task_id=task_id,
            task_description=task_description,
            work_shape=work_shape,
            selected_surface=ExecutionSurface.DETERMINISTIC_TOOL_OR_SCRIPT,
            authoritative_system="Python Core Engine / Decimal Math Tools",
            reconstructable_artifact="data/state_manifest.json and calculation records",
            rejection_gate="Automated mathematical balance test and AST dependency gate",
            reversal_condition="Calculation requires qualitative narrative interpretation of unstructured disclosures",
            rationale="Yields, exposures, and vintage delta calculations must be computed deterministically with Decimal.",
            status="REVIEW_REQUIRED",
            evaluated_at_utc=now_iso,
        )

    # 4. LLM Synthesis & Reading Gate
    return TaskRouteRecord(
        task_id=task_id,
        task_description=task_description,
        work_shape=work_shape,
        selected_surface=ExecutionSurface.LLM_RESEARCH_SYNTHESIS,
        authoritative_system="Hermes LLM Progressive Tool Harness",
        reconstructable_artifact="memories/30_Knowledge_Base/ research drafts",
        rejection_gate="4-Tier Claim Evidence Validator and citation tie-out",
        reversal_condition="The task requires authoritative financial calculations or state ledger updates",
        rationale="Unstructured reading and narrative drafting benefit from LLM extraction wrapped in tool harnesses.",
        status="REVIEW_REQUIRED",
        evaluated_at_utc=now_iso,
    )

