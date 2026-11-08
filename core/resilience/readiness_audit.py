"""The 10-Question Institutional Readiness Audit & Decision Record (Appendix J).

Architecture Rule (Layer 1 - Core):
Pure Python standard library only — strictly zero external dependencies per Hexagonal
Architecture AST Domain Layer Isolation rules.

Reference:
Claude Code for Hedge Funds, Appendix J (pp. 136-139):
"A team is ready for the next stage only if it can answer yes to all 10 questions:
1. Can it name the authoritative system for every artifact?
2. Can it prove data rights and environment approval?
3. Can it reconstruct publication, retrieval, vintage, and approval time?
4. Can it reproduce calculations from frozen inputs?
5. Can it detect changed protected files?
6. Can it demonstrate a seeded failure for each critical validator?
7. Can it distinguish a documented feature from a directly tested workflow?
8. Can a fresh analyst rebuild the packet without the transcript?
9. Does every automated branch end REVIEW_REQUIRED or FAILED?
10. Can the team stop and preserve evidence when a control fails?"
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple


class AuditVerdict(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    CONDITIONAL = "CONDITIONAL"  # Note: Per book p. 137, "a conditional answer is not a pass"


class ReadinessDecision(str, Enum):
    READY_FOR_NAMED_STAGE = "READY_FOR_NAMED_STAGE"
    REMEDIATE = "REMEDIATE"
    STOP = "STOP"


@dataclass(frozen=True)
class AuditQuestionSpec:
    """Specification of an institutional readiness question."""
    question_number: int
    question_text: str
    controlling_layer: str
    target_proof: str


# The canonical 10 questions from Appendix J (pp. 136-138)
CANONICAL_QUESTIONS: Tuple[AuditQuestionSpec, ...] = (
    AuditQuestionSpec(
        question_number=1,
        question_text="Can it name the authoritative system for every artifact?",
        controlling_layer="Infrastructure / Storage",
        target_proof="SQLite State Database (data/webui_state.sqlite) and append-only ledgers",
    ),
    AuditQuestionSpec(
        question_number=2,
        question_text="Can it prove data rights and environment approval?",
        controlling_layer="Configuration / Security",
        target_proof="Environment boundary enforcement; zero secrets committed in prompts/repos",
    ),
    AuditQuestionSpec(
        question_number=3,
        question_text="Can it reconstruct publication, retrieval, vintage, and approval time?",
        controlling_layer="Core Provenance",
        target_proof="5-Clock system (event_time, available_time, observed_time, revised_time, compute_time)",
    ),
    AuditQuestionSpec(
        question_number=4,
        question_text="Can it reproduce calculations from frozen inputs?",
        controlling_layer="Core Financial Math",
        target_proof="Pure Python Decimal math with exact driver sum assertions and zero LLM mental math",
    ),
    AuditQuestionSpec(
        question_number=5,
        question_text="Can it detect changed protected files?",
        controlling_layer="Core State Integrity",
        target_proof="SHA-256 frozen state manifests and tamper-detection invariants",
    ),
    AuditQuestionSpec(
        question_number=6,
        question_text="Can it demonstrate a seeded failure for each critical validator?",
        controlling_layer="Core Resilience",
        target_proof="Institutional Failure Atlas with bounded repair mappings",
    ),
    AuditQuestionSpec(
        question_number=7,
        question_text="Can it distinguish a documented feature from a directly tested workflow?",
        controlling_layer="Core Thesis Taxonomy",
        target_proof="4-Tier Claim Evidence taxonomy ([DOCUMENTED], [TESTED], [INFERRED], [UNVERIFIED])",
    ),
    AuditQuestionSpec(
        question_number=8,
        question_text="Can a fresh analyst rebuild the packet without the transcript?",
        controlling_layer="Application Services",
        target_proof="Deterministic packet synthesis from stored SQLite state without LLM chat history",
    ),
    AuditQuestionSpec(
        question_number=9,
        question_text="Does every automated branch end REVIEW_REQUIRED or FAILED?",
        controlling_layer="Application / Portfolio Advisory",
        target_proof="Human-in-the-loop signoff gate; zero autonomous trade execution",
    ),
    AuditQuestionSpec(
        question_number=10,
        question_text="Can the team stop and preserve evidence when a control fails?",
        controlling_layer="Core Logging & Telemetry",
        target_proof="Immutable exception ledger and structured failure detail preservation",
    ),
)


@dataclass(frozen=True)
class QuestionEvaluation:
    """The result of evaluating a single institutional readiness question."""
    question_number: int
    question_text: str
    verdict: AuditVerdict
    evidence_artifact: str
    notes: str = ""
    smallest_repair: Optional[str] = None
    controlling_layer: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question_number": self.question_number,
            "question_text": self.question_text,
            "verdict": self.verdict.value,
            "evidence_artifact": self.evidence_artifact,
            "notes": self.notes,
            "smallest_repair": self.smallest_repair,
            "controlling_layer": self.controlling_layer,
        }


@dataclass(frozen=True)
class InstitutionalReadinessReport:
    """Structured report and decision record for institutional adoption readiness."""
    as_of_date: str
    decision: ReadinessDecision
    evaluations: Tuple[QuestionEvaluation, ...]
    scope: str = "RESEARCH_PREPARATION_AND_ADVISORY"
    signatory: str = "PORTFOLIO_MANAGER_HERMES"
    sha256_checksum: str = ""

    @property
    def pass_count(self) -> int:
        return sum(1 for e in self.evaluations if e.verdict == AuditVerdict.PASS)

    @property
    def fail_count(self) -> int:
        return sum(1 for e in self.evaluations if e.verdict != AuditVerdict.PASS)

    @property
    def is_fully_ready(self) -> bool:
        return self.pass_count == len(self.evaluations) and self.decision == ReadinessDecision.READY_FOR_NAMED_STAGE

    def to_dict(self) -> Dict[str, Any]:
        return {
            "as_of_date": self.as_of_date,
            "decision": self.decision.value,
            "scope": self.scope,
            "signatory": self.signatory,
            "pass_count": self.pass_count,
            "fail_count": self.fail_count,
            "is_fully_ready": self.is_fully_ready,
            "sha256_checksum": self.sha256_checksum,
            "evaluations": [e.to_dict() for e in self.evaluations],
        }

    def render_markdown(self) -> str:
        """Renders the official readiness-decision.md artifact per Appendix J format."""
        lines = [
            "# Institutional Readiness Decision Record",
            f"> **As-Of Date:** {self.as_of_date} | **Decision:** `{self.decision.value}` | **Scope:** `{self.scope}`",
            "",
            "## 1. 📋 Institutional Examination Questions",
            "",
            "| # | Question | Verdict | Controlling Layer | Evidence Artifact | Notes / Smallest Repair |",
            "|---|---|:---:|---|---|---|",
        ]

        for ev in self.evaluations:
            badge = "✅ PASS" if ev.verdict == AuditVerdict.PASS else "❌ FAIL"
            repair = ev.smallest_repair or ev.notes or "—"
            lines.append(
                f"| Q{ev.question_number} | {ev.question_text} | {badge} | {ev.controlling_layer} | `{ev.evidence_artifact}` | {repair} |"
            )

        lines.extend([
            "",
            "## 2. ⚖️ Decision & Governance Summary",
            "",
            f"- **Examination Result:** {self.pass_count}/{len(self.evaluations)} checks PASSED.",
            f"- **Official Decision:** `{self.decision.value}`",
            f"- **Authorizing Signatory:** `{self.signatory}`",
            "- **Scope Limitation:** Approved strictly for research-preparation, portfolio monitoring, and advisory packet generation.",
            "- **Non-Authority:** *Does not authorize autonomous recommendations, live portfolio changes, order routing, or external publication.*",
            "",
            "## 3. 🔒 Audit & Integrity Signature",
            f"```",
            f"AUDIT_INTEGRITY_CHECKSUM: {self.sha256_checksum or 'PENDING_FINALIZATION'}",
            f"SIGN_OFF_STATUS: {self.decision.value}",
            f"```",
            "",
            "> *\"A research project becomes trustworthy when its state can outlive the conversation that changed it. Keep the sources dated. Keep beliefs append-only. Keep exceptions visible. Keep decisions human.\"* — Dave Wang & Felipe Sinisterra",
        ])

        return "\n".join(lines)

