"""Schemas สำหรับระบบ Evidence Contract, Briefing Book Generation, และ Research Quality Gate"""
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
from pathlib import Path
from pydantic import BaseModel, Field, model_validator, computed_field, ConfigDict, field_validator

from schemas.market_data_schemas import MacroObservation


class SourceRecord(BaseModel):
    model_config = ConfigDict(validate_assignment=True, extra="forbid")
    source_id: str = Field(description="Unique source ID (e.g. S01, S02)")
    original_title: str = Field(description="ชื่อข่าวหรือคลิปต้นทางจริง ห้ามแต่งเติม")
    publisher: str = Field(description="ชื่อสำนักข่าวหรือช่อง")
    host: str = Field(description="Domain host ของลิงก์ (e.g. finance.yahoo.com)")
    published_at: Optional[str] = Field(default=None, description="วันเผยแพร่จริง (ISO format) หรือ None หากไม่ทราบ")
    ingested_at: str = Field(description="วันที่ดึงข้อมูลเข้าสู่ระบบ")
    url: str = Field(description="URL อ้างอิงต้นทาง")
    source_type: Literal[
        "official", "data_provider", "wire_service",
        "analysis", "creator_commentary", "unknown"
    ] = Field(default="unknown")
    independence_key: str = Field(description="Key สำหรับกลุ่มผู้ผลิตสื่อเดียวกัน (e.g. finnomena_group)")
    verification_status: Literal["verified", "partial", "unverified"] = Field(default="unverified")


class EvidenceItem(BaseModel):
    model_config = ConfigDict(validate_assignment=True, extra="forbid")
    evidence_id: str = Field(description="Unique evidence ID (e.g. E01, E02)")
    claim: str = Field(description="ข้อความอ้างอิงหรือข้อมูลดิบ")
    classification: Literal[
        "verified_fact",
        "source_reported_fact",
        "consensus",
        "hypothesis",
        "speculation"
    ] = Field(description="ระดับชั้นความน่าเชื่อถือของหลักฐาน")
    value: Optional[float] = Field(default=None, description="ตัวเลขเชิงปริมาณ (ถ้ามี)")
    unit: Optional[str] = Field(default=None, description="หน่วยของตัวเลข (e.g. USD, %, Index)")
    observed_at: Optional[str] = Field(default=None, description="วันที่หรือช่วงเวลาของตัวเลข")
    reported_at: Optional[str] = Field(
        default=None,
        description="วันที่ที่แหล่งข่าวรายงานตัวเลข (แยกจากเวลาที่ตัวเลขเกิดขึ้น)",
    )
    time_semantics: Literal["observed", "reported", "fiscal_period", "forecast", "unknown"] = Field(
        default="unknown",
        description="ความหมายของเวลาใน observed_at/reported_at เพื่อไม่เปรียบเทียบคนละช่วงเวลา",
    )
    source_excerpt: Optional[str] = Field(
        default=None,
        description="ข้อความต้นทางสั้น ๆ ที่บอกบริบทของตัวเลข โดยไม่สร้างข้อเท็จจริงเพิ่ม",
    )
    source_ids: List[str] = Field(description="รายการ Source IDs ที่อ้างอิงหลักฐานนี้")
    confidence: Literal["high", "medium", "low"] = Field(default="medium")
    calculation_formula: Optional[str] = Field(default=None, description="สูตรคำนวณสำหรับDerived metric")
    input_evidence_ids: List[str] = Field(default_factory=list, description="Input evidence IDs ที่นำมาคำนวณ")
    caveat: str = Field(default="", description="คำเตือนหรือข้อจำกัดของหลักฐาน")


    metric_name: Optional[str] = Field(default=None, description="Canonical metric or entity name")
    period_type: Optional[Literal["instant", "daily_close", "fiscal_period", "forecast"]] = None
    claim_scope: Literal["exact", "derived", "hypothesis", "scenario_assumption"] = "exact"


class MacroAutopsySnapshot(BaseModel):
    observations: List[MacroObservation] = Field(default_factory=list)
    as_of: str = Field(default_factory=lambda: datetime.now().isoformat())
    is_complete: bool = Field(default=False)
    unavailable_reasons: List[str] = Field(default_factory=list)


class FinancialAutopsyPeriodRecord(BaseModel):
    """A serialisable financial-statement period used by the briefing renderer."""
    model_config = ConfigDict(extra="ignore")

    fiscal_period_end: str
    free_cash_flow: Optional[float] = None
    operating_cash_flow: Optional[float] = None
    capital_expenditure: Optional[float] = None
    total_debt: Optional[float] = None
    total_revenue: Optional[float] = None
    net_income: Optional[float] = None
    dividends_paid: Optional[float] = None
    payout_ratio_pct: Optional[float] = None


class FinancialAutopsySnapshotRef(BaseModel):
    symbol: str
    provider_symbol: Optional[str] = None
    status: Literal["success", "unavailable", "error"]
    currency: Optional[str] = None
    source: Optional[str] = None
    periods: List[FinancialAutopsyPeriodRecord] = Field(default_factory=list)
    market_cap: Optional[str] = None
    revenue: Optional[str] = None
    net_income: Optional[str] = None
    fcf: Optional[str] = None
    total_debt: Optional[str] = None
    health_notes: Optional[str] = None

    @field_validator("periods", mode="before")
    @classmethod
    def _coerce_periods(cls, v: Any) -> List[Any]:
        if not isinstance(v, list):
            return []
        coerced = []
        for item in v:
            if isinstance(item, FinancialAutopsyPeriodRecord):
                coerced.append(item)
            elif hasattr(item, "model_dump"):
                coerced.append(FinancialAutopsyPeriodRecord.model_validate(item.model_dump()))
            elif isinstance(item, dict):
                coerced.append(FinancialAutopsyPeriodRecord.model_validate(item))
            elif hasattr(item, "fiscal_period_end"):
                coerced.append(FinancialAutopsyPeriodRecord(
                    fiscal_period_end=str(getattr(item, "fiscal_period_end", "")),
                    free_cash_flow=getattr(item, "free_cash_flow", None),
                    operating_cash_flow=getattr(item, "operating_cash_flow", None),
                    capital_expenditure=getattr(item, "capital_expenditure", None),
                    total_debt=getattr(item, "total_debt", None),
                    total_revenue=getattr(item, "total_revenue", None),
                    net_income=getattr(item, "net_income", None),
                    dividends_paid=getattr(item, "dividends_paid", None),
                    payout_ratio_pct=getattr(item, "payout_ratio_pct", None),
                ))
            else:
                coerced.append(item)
        return coerced



class ScenarioRecord(BaseModel):
    scenario_id: str = Field(description="e.g. SC01, SC02")
    name: str = Field(description="ชื่อฉากทัศน์ e.g. De-escalation, Persistent Disruption, Tail-risk Escalation")
    description: str = Field(description="คำอธิบายรายละเอียดฉากทัศน์")
    probability_pct: float = Field(ge=0, le=100, description="ความน่าจะเป็น (%)")
    trigger_conditions: List[str] = Field(description="เงื่อนไขที่จะทำให้เกิดฉากทัศน์นี้")
    falsification_triggers: List[str] = Field(description="เงื่อนไขที่หักล้างฉากทัศน์นี้")
    evidence_ids: List[str] = Field(description="Evidence IDs ที่ใช้อ้างอิง")


    time_horizon: str = Field(default="", description="Time horizon for this scenario")
    threshold_basis: str = Field(default="", description="Evidence or methodology behind numeric thresholds")
    assumption_ids: List[str] = Field(default_factory=list, description="Scenario-assumption evidence IDs")


class AssetImpactRecord(BaseModel):
    symbol_or_name: str = Field(description="ชื่อสินทรัพย์หรือหุ้น e.g. [[NVDA]], Brent Crude")
    impact_type: Literal["direct_upside", "direct_downside", "second_order", "loser"] = Field(description="ประเภทผลกระทบ")
    reasoning: str = Field(description="เหตุผลเชิงวิเคราะห์")
    risk_factors: List[str] = Field(description="ความเสี่ยงสำคัญ")
    invalidation_conditions: List[str] = Field(description="เงื่อนไขที่จะทำให้สมมติฐานการลงทุนล้มเหลว (Risk & Invalidation)")
    evidence_ids: List[str] = Field(description="Evidence IDs ที่ใช้อ้างอิง")


class VisualEvidenceDirective(BaseModel):
    visual_id: str = Field(description="e.g. V01, V02")
    act: Literal["Act I", "Act II", "Act III"] = Field(description="Act ที่ปรากฏ")
    title: str = Field(description="ชื่อกราฟหรือภาพประกอบ")
    chart_type: str = Field(description="ประเภทกราฟ e.g. Rebased Comparison Chart, Time-Series Spread")
    series_keys: List[str] = Field(
        description="สัญลักษณ์ชุดข้อมูล e.g. ['BZ=F', '^GSPC']; when data_mode is evidence_table, this must be exactly ['EVIDENCE_TABLE']"
    )
    date_range: str = Field(description="ช่วงเวลาของข้อมูล e.g. 2026-06-20 ถึง 2026-07-21")
    sources: List[str] = Field(description="แหล่งข้อมูลตัวเลข e.g. Yahoo Finance BZ=F")
    annotation: str = Field(description="จุดเน้นบนกราฟ e.g. วันที่ Brent ทะลุ $90")
    evidence_ids: List[str] = Field(description="Evidence IDs ที่ใช้อ้างอิง")
    historical_comparison: bool = Field(default=False)
    data_mode: Literal["provider_series", "evidence_table", "event_timeline"] = Field(
        default="provider_series",
        description="ชนิดข้อมูลภาพ: provider_series, evidence_table, หรือ event_timeline; evidence_table requires series_keys=['EVIDENCE_TABLE']",
    )


class NotebookLMPromptRecord(BaseModel):
    prompt_id: str = Field(description="e.g. P01, P02")
    prompt_type: Literal["BLIND_SPOT", "SOCRATIC", "FEYNMAN", "RESEARCH"] = Field(description="ประเภท Prompt")
    question_or_prompt: str = Field(description="ข้อความคำถามหรือ Prompt")
    expected_output_format: str = Field(description="รูปแบบคำตอบที่คาดหวัง")


class QualityIssueRecord(BaseModel):
    code: str = Field(description="Issue code (e.g. MACRO_COVERAGE_FAILURE)")
    category: Literal["provenance", "macro", "financial", "numeric", "structure", "other"]
    severity: Literal["blocker", "cap", "warning"]
    description: str = Field(description="รายละเอียดของข้อผิดพลาด")
    evidence_ids: List[str] = Field(default_factory=list, description="Evidence IDs ที่เกี่ยวข้องกับปัญหานี้ (ถ้ามี)")
    bypassable: bool = Field(default=False, description="สามารถ bypass เป็น Unverified Draft ได้หรือไม่")


class ResearchQualityReport(BaseModel):
    model_config = ConfigDict(validate_assignment=True, extra="forbid")
    advisories: List[str] = Field(
        default_factory=list,
        description="Non-blocking cautions that do not reduce publishability",
    )
    score: int = Field(description="คะแนนคุณภาพ 0-100")
    issues: List[QualityIssueRecord] = Field(default_factory=list, description="รายการข้อผิดพลาดที่มีโครงสร้างชัดเจน")
    rubric_breakdown: Dict[str, int] = Field(default_factory=dict, description="คะแนนแยกตามหมวด")
    evaluated_at: str = Field(default_factory=lambda: datetime.now().isoformat())

    status: Literal["pass", "degraded", "fail"] = "fail"
    publishable: bool = False

    @computed_field
    @property
    def hard_blockers(self) -> List[str]:
        return [issue.description for issue in self.issues if issue.severity == "blocker"]

    @computed_field
    @property
    def warnings(self) -> List[str]:
        return [issue.description for issue in self.issues if issue.severity == "warning"]

    @computed_field
    @property
    def score_caps_applied(self) -> List[str]:
        return [issue.description for issue in self.issues if issue.severity == "cap"]


class InvestigativeBriefingBookDraft(BaseModel):
    model_config = ConfigDict(validate_assignment=True, extra="forbid")
    title: str = Field(description="ชื่อเอกสาร Briefing Book")
    executive_summary: str = Field(description="สรุปผู้บริหารเชิงลึก พร้อมอ้างอิง [E01], [E02]")
    causality_scenarios: List[ScenarioRecord] = Field(description="อย่างน้อย 3 ฉากทัศน์")
    asset_impacts: List[AssetImpactRecord] = Field(description="รายการผลกระทบต่อสินทรัพย์")
    bull_case: str = Field(description="มุมมองฝั่งกระทิงพร้อม Evidence IDs")
    bear_case: str = Field(description="มุมมองฝั่งหมีพร้อม Evidence IDs")
    falsification_triggers: List[str] = Field(description="เงื่อนไขจุดชี้ขาดในการหักล้างสมมติฐาน")
    act1_script: str = Field(description="โครงเรื่อง Act I พร้อม [VISUAL_EVIDENCE id=V01 evidence=E01,E02]")
    act2_script: str = Field(description="โครงเรื่อง Act II พร้อม [VISUAL_EVIDENCE id=V02 evidence=E03,E04]")
    act3_script: str = Field(description="โครงเรื่อง Act III พร้อม [VISUAL_EVIDENCE id=V03 evidence=E05,E06]")
    visual_directives: List[VisualEvidenceDirective] = Field(description="คำสั่งสร้างภาพประกอบที่ผลิตซ้ำได้ครบถ้วน")
    notebooklm_prompts: List[NotebookLMPromptRecord] = Field(description="ชุดคำถามวิจัยสำหรับ NotebookLM 5-8 ข้อ")
    audio_overview_directive: Optional[str] = Field(
        default=None,
        description=(
            "System-level Directive ภาษาไทย (มีคำศัพท์เทคนิคภาษาอังกฤษได้) สำหรับ NotebookLM Audio Overview "
            "ทำหน้าที่ควบคุมสไตล์การจัดรายการและความกระชับโดยรวม ไม่ให้พูดวน ต่างจาก notebooklm_prompts ที่เป็นรายการคำถามเจาะลึก 5-8 ข้อ"
        ),
    )



class BriefingEvidenceBundle(BaseModel):
    model_config = ConfigDict(validate_assignment=True, extra="forbid")
    pitch_id: str
    investigation_mode: Literal["stock", "macro", "mixed"] = "mixed"
    sources: List[SourceRecord] = Field(default_factory=list)
    evidence_items: List[EvidenceItem] = Field(default_factory=list)
    macro_snapshot: Optional[MacroAutopsySnapshot] = None
    financial_snapshots: List[FinancialAutopsySnapshotRef] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())


class PublishableBriefingResult(BaseModel):
    """Successful synthesis with its real quality gate report."""
    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    artifact_status: Literal["publishable"] = "publishable"
    content: str
    draft: InvestigativeBriefingBookDraft
    quality_report: ResearchQualityReport
    evidence_bundle: BriefingEvidenceBundle
    generated_at: str = Field(default_factory=lambda: datetime.now().isoformat())

    def __str__(self) -> str:
        return self.content

    def __contains__(self, value: object) -> bool:
        return value in self.content if isinstance(value, str) else False


class UnverifiedDraftOverrideAudit(BaseModel):
    job_id: str
    thread_id: str
    pitch_id: str
    policy_version: str
    reason: str
    server_timestamp: str
    token_hash: str
    source_readiness_snapshot: List[str]


class UnverifiedBriefingDraftResult(BaseModel):
    """Fallback draft generated bypassing strict provenance requirements."""
    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    content: str
    draft: InvestigativeBriefingBookDraft
    quality_report: ResearchQualityReport
    evidence_bundle: BriefingEvidenceBundle
    artifact_status: Literal["unverified_draft"] = "unverified_draft"
    trust_tier: Literal["unverified"] = "unverified"
    override_audit: UnverifiedDraftOverrideAudit
    generated_at: str = Field(default_factory=lambda: datetime.now().isoformat())

    @model_validator(mode="after")
    def _validate_draft_invariants(self):
        if not self.override_audit.source_readiness_snapshot:
            raise ValueError("Draft overrides must freeze source readiness state.")
        if self.quality_report.publishable:
            raise ValueError("Draft result must not be publishable")
        for issue in self.quality_report.issues:
            if issue.severity in ("blocker", "cap") and not issue.bypassable:
                raise ValueError(f"Draft result has non-bypassable issue: {issue.code}")
        return self

    def __str__(self) -> str:
        return self.content

    def __contains__(self, value: object) -> bool:
        return value in self.content if isinstance(value, str) else False


class RenderedVisualMarker(BaseModel):
    id: str
    evidence_ids: List[str]
    act: str = ""


class RenderedBriefing(BaseModel):
    content: str
    section_names: List[str]
    visual_markers: List[RenderedVisualMarker]
    cited_evidence_ids: set[str]
    cited_source_ids: set[str]


class SavedBriefingArtifact(BaseModel):
    path: Path
    quality_path: Path
    index_status: Literal["indexed", "pending", "failed"]
    index_error: str | None = None
