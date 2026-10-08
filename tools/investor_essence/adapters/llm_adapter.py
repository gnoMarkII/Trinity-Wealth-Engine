"""LLM driven adapter implementing generator ports for Investor Essence."""
from __future__ import annotations

import json
import logging
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from core.llm_factory import detect_provider, get_llm
from application.investor_essence.dto import (
    AllocationPlanRowProposal,
    AxisProposal,
    BucketPlanProposal,
    ClaimProposal,
    ContentOptionProposal,
    EssenceSummaryProposal,
    EvidenceSnapshot,
    NumericPolicyFieldProposal,
    PurposeBucketProposal,
    QuestionProposal,
    WeightedAllocationMappingProposal,
)
from application.investor_essence.errors import (
    ProviderUnavailableError,
    ValidationFailedError,
)
from application.investor_essence.ports import (
    AxisGeneratorPort,
    BucketGeneratorPort,
    EssenceGeneratorPort,
    InterviewGeneratorPort,
)

logger = logging.getLogger(__name__)

PROMPT_DIR = Path(__file__).resolve().parent.parent.parent.parent / "prompts" / "skills" / "investor_essence"


def _read_prompt(filename: str) -> str:
    path = PROMPT_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(f"Missing prompt template: {path}")
    return path.read_text(encoding="utf-8")


def _clean_json_text(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return cleaned.strip()


class LlmInvestorEssenceAdapter(
    InterviewGeneratorPort,
    EssenceGeneratorPort,
    AxisGeneratorPort,
    BucketGeneratorPort,
):
    """LangChain-backed adapter invoking models for adaptive interview, summary, axis, and buckets."""

    def __init__(self, llm: Optional[BaseChatModel] = None) -> None:
        self._llm = llm

    def _get_active_llm(self) -> BaseChatModel:
        if self._llm is not None:
            return self._llm
        provider = detect_provider("gemini-2.5-flash")
        return get_llm(provider, "gemini-2.5-flash", temperature=0.2)

    def _invoke_and_parse(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        try:
            llm = self._get_active_llm()
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
            response = llm.invoke(messages)
            content = response.content if hasattr(response, "content") else str(response)
            cleaned = _clean_json_text(content)
            return json.loads(cleaned)
        except json.JSONDecodeError as exc:
            logger.error("LLM failed to output valid JSON: %s", exc)
            raise ValidationFailedError(f"Model output could not be parsed as JSON: {exc}") from exc
        except Exception as exc:
            logger.error("LLM provider invocation failed: %s", exc)
            raise ProviderUnavailableError(f"LLM provider error: {exc}") from exc

    def generate_next_question(
        self, evidence: EvidenceSnapshot, prompt_version: str
    ) -> QuestionProposal:
        skill_prompt = _read_prompt("SKILL.md")
        stage_prompt = _read_prompt("INTERVIEW_NEXT.md")
        system = f"{skill_prompt}\n\n{stage_prompt}"

        context_data = {
            "session_id": evidence.session_id,
            "branch_id": evidence.branch_id,
            "revision": evidence.revision,
            "context_hash": evidence.context_hash,
            "qa_pairs": evidence.qa_pairs,
            "coverage_topics": evidence.coverage_topics,
            "prompt_version": prompt_version,
        }
        user = f"บริบทการสัมภาษณ์ปัจจุบัน:\n```json\n{json.dumps(context_data, ensure_ascii=False, indent=2)}\n```\n\nกรุณาสร้างคำถามข้อถัดไปพร้อม 4 ตัวเลือกตามข้อกำหนด"

        parsed = self._invoke_and_parse(system, user)
        raw_options = parsed.get("options", [])
        if len(raw_options) != 4:
            raise ValidationFailedError(f"Expected 4 options, got {len(raw_options)}")

        options = [
            ContentOptionProposal(key=opt["key"], text=opt["text"])
            for opt in raw_options
        ]
        return QuestionProposal(
            text=parsed["text"],
            options=options,
            coverage_topics=parsed.get("coverage_topics", []),
            evidence_type=parsed.get("evidence_type", "self_report"),
            is_clarification=False,
        )

    def generate_clarification(
        self, evidence: EvidenceSnapshot, unresolved_topic: str, prompt_version: str
    ) -> QuestionProposal:
        skill_prompt = _read_prompt("SKILL.md")
        stage_prompt = _read_prompt("ESSENCE_CLARIFICATION.md")
        system = f"{skill_prompt}\n\n{stage_prompt}"

        context_data = {
            "session_id": evidence.session_id,
            "unresolved_topic": unresolved_topic,
            "qa_pairs": evidence.qa_pairs,
            "prompt_version": prompt_version,
        }
        user = f"ประเด็นที่ยังไม่ชัดเจน: {unresolved_topic}\n\nบริบทก่อนหน้า:\n```json\n{json.dumps(context_data, ensure_ascii=False, indent=2)}\n```"

        parsed = self._invoke_and_parse(system, user)
        raw_options = parsed.get("options", [])
        options = [
            ContentOptionProposal(key=opt["key"], text=opt["text"])
            for opt in raw_options
        ]
        return QuestionProposal(
            text=parsed["text"],
            options=options,
            coverage_topics=[unresolved_topic],
            evidence_type="self_report",
            is_clarification=True,
        )

    def generate_summary(
        self, evidence: EvidenceSnapshot, prompt_version: str
    ) -> EssenceSummaryProposal:
        skill_prompt = _read_prompt("SKILL.md")
        stage_prompt = _read_prompt("ESSENCE_SUMMARY.md")
        system = f"{skill_prompt}\n\n{stage_prompt}"

        context_data = {
            "session_id": evidence.session_id,
            "qa_pairs": evidence.qa_pairs,
            "context_hash": evidence.context_hash,
            "prompt_version": prompt_version,
        }
        user = f"ประวัติการสัมภาษณ์ครบ 10 ข้อ:\n```json\n{json.dumps(context_data, ensure_ascii=False, indent=2)}\n```\n\nกรุณาสรุปแก่นแท้ในการลงทุนตามข้อกำหนด"

        parsed = self._invoke_and_parse(system, user)
        raw_claims = parsed.get("claims", [])
        claims = [
            ClaimProposal(
                text=c["text"],
                source_kind=c["source_kind"],
                supporting_question_ids=c.get("supporting_question_ids", []),
                quote=c.get("quote", ""),
                evidence_type=c.get("evidence_type", "self_report"),
            )
            for c in raw_claims
        ]

        return EssenceSummaryProposal(
            statement=parsed.get("statement", ""),
            claims=claims,
            unresolved_topics=parsed.get("unresolved_topics", []),
            coverage_report=parsed.get("coverage_report", []),
        )

    def generate_axis(
        self,
        accepted_claims: List[Dict[str, Any]],
        financial_context: Dict[str, Any],
        prompt_version: str,
    ) -> AxisProposal:
        skill_prompt = _read_prompt("SKILL.md")
        stage_prompt = _read_prompt("INVESTMENT_AXIS.md")
        system = f"{skill_prompt}\n\n{stage_prompt}"

        context_data = {
            "accepted_claims": accepted_claims,
            "financial_context": financial_context,
            "prompt_version": prompt_version,
        }
        user = f"ข้อมูลแก่นแท้ที่ยืนยันและบริบทการเงิน:\n```json\n{json.dumps(context_data, ensure_ascii=False, indent=2)}\n```\n\nกรุณาร่างแกนหลักการลงทุนครบ 8 หัวข้อ"

        parsed = self._invoke_and_parse(system, user)

        raw_limits = parsed.get("risk_limits", {})
        risk_limits = {}
        for k, v in raw_limits.items():
            val = str(v["value"]) if v.get("value") is not None else None
            risk_limits[k] = NumericPolicyFieldProposal(
                field_id=v.get("field_id", k),
                value=val,
                unit=v.get("unit", ""),
                calculation_basis=v.get("calculation_basis", ""),
                origin=v.get("origin", "ai_proposal"),
            )

        raw_rows = parsed.get("allocation_rows", [])
        allocation_rows = [
            AllocationPlanRowProposal(
                allocation_id=r.get("allocation_id", f"alloc_{idx}"),
                category_name=r.get("category_name", ""),
                target_percent=str(r.get("target_percent", "0")),
                role_description=r.get("role_description", ""),
            )
            for idx, r in enumerate(raw_rows)
        ]

        return AxisProposal(
            basic_policy=parsed.get("basic_policy", ""),
            risk_limits=risk_limits,
            invest_targets=parsed.get("invest_targets", []),
            exclude_targets=parsed.get("exclude_targets", []),
            primary_methods=parsed.get("primary_methods", []),
            secondary_methods=parsed.get("secondary_methods", []),
            investment_horizon=parsed.get("investment_horizon", ""),
            rebalance_frequency=parsed.get("rebalance_frequency", ""),
            allocation_basis=parsed.get("allocation_basis", "asset_class"),
            allocation_rows=allocation_rows,
            role_models=parsed.get("role_models", []),
            non_actions=parsed.get("non_actions", []),
        )

    def generate_buckets(
        self,
        confirmed_axis: Dict[str, Any],
        prompt_version: str,
        accepted_claims: Optional[List[Dict[str, Any]]] = None,
    ) -> BucketPlanProposal:
        skill_prompt = _read_prompt("SKILL.md")
        stage_prompt = _read_prompt("BUCKET_PROPOSAL.md")
        system = f"{skill_prompt}\n\n{stage_prompt}"

        context_data = {
            "confirmed_axis": confirmed_axis,
            "accepted_claims": accepted_claims or [],
            "prompt_version": prompt_version,
        }
        user = f"ข้อมูลแกนหลักและแก่นแท้:\n```json\n{json.dumps(context_data, ensure_ascii=False, indent=2)}\n```\n\nกรุณาร่าง Purpose Buckets พร้อม matrix การจัดสรร"

        parsed = self._invoke_and_parse(system, user)

        raw_buckets = parsed.get("purpose_buckets", [])
        purpose_buckets = [
            PurposeBucketProposal(
                bucket_id=b.get("bucket_id"),
                name=b.get("name", ""),
                role=b.get("role", ""),
                color=b.get("color") or b.get("color_hex", "#3B82F6"),
                target_percent=str(b.get("target_percent", "0")),
                source_value_ids=b.get("source_value_ids", []),
                source_axis_allocation_ids=b.get("source_axis_allocation_ids", []),
            )
            for b in raw_buckets
        ]

        raw_mapping = parsed.get("allocation_mapping") or parsed.get("mapping_weights", [])
        mapping_weights = [
            {
                "axis_allocation_id": m.get("axis_allocation_id", ""),
                "bucket_id": m.get("bucket_id", ""),
                "portfolio_weight_percent": str(m.get("portfolio_weight_percent", "0")),
            }
            for m in raw_mapping
        ]

        return BucketPlanProposal(
            purpose_buckets=purpose_buckets,
            allocation_basis=parsed.get("allocation_basis", "asset_class"),
            mapping_weights=mapping_weights,
            constraints=parsed.get("constraints", []),
        )

