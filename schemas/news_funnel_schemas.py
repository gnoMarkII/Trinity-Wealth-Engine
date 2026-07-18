"""Pydantic schemas สำหรับระบบ News Funnel Architecture & Obsidian Linking Keys

รองรับ Ingestion (Title + Summary), Deduplication/Clustering, และ Fast LLM Triage (Impact Score)
"""
import re
from typing import List
from pydantic import BaseModel, Field, computed_field, field_validator


def strip_wikilink(text: str) -> str:
    """ลบเครื่องหมาย [[ ... ]] และ Alias [[File|Alias]] ออกจากข้อความ เหลือชื่อหลัก พร้อม trim whitespace"""
    clean = text.strip()
    match = re.search(r"\[\[([^|\]]+)(?:\|[^\]]+)?\]\]", clean)
    if match:
        return match.group(1).strip()
    return clean.split("|")[0].strip() if "|" in clean else clean


HIGH_IMPACT_THRESHOLD = 7


class MacroImpactTriageResult(BaseModel):
    """ผลลัพธ์การประเมินความสำคัญและคัดกรองข่าวโดย Fast LLM (Gemini Flash)"""
    macro_impact_score: int = Field(
        ge=1, le=10, description="คะแนนผลกระทบต่อเศรษฐกิจมหภาค 1-10"
    )
    asset_impact_score: int = Field(
        ge=1, le=10, description="คะแนนผลกระทบต่อราคาสินทรัพย์/หุ้นในตลาด 1-10"
    )
    primary_tags: List[str] = Field(
        default_factory=list, description="แท็กหลัก เช่น inflation, policy, earnings"
    )
    extracted_tickers: List[str] = Field(
        default_factory=list, description="สัญลักษณ์หุ้นหรือสินทรัพย์ที่เกี่ยวข้อง เช่น NVDA, PTT, Gold"
    )
    extracted_themes: List[str] = Field(
        default_factory=list, description="ธีมเศรษฐกิจที่เกี่ยวข้อง ตามมาตรฐาน ThemeCategory"
    )
    triage_reasoning: str = Field(
        default="", description="เหตุผลในการให้คะแนนและคัดกรอง"
    )
    thai_title: str = Field(
        default="", description="ชื่อหัวข้อข่าวแปลเป็นภาษาไทย"
    )
    thai_summary: str = Field(
        default="", description="สรุปข่าวกระชับ 2-3 ประโยคเป็นภาษาไทย สำหรับนักลงทุนไทย (ห้ามเขียนย่อหน้ายาว)"
    )

    @field_validator("extracted_tickers", "extracted_themes", mode="after")
    @classmethod
    def _clean_extracted_links(cls, values: List[str]) -> List[str]:
        return [strip_wikilink(v) for v in values if v and strip_wikilink(v)]

    @computed_field
    def is_high_impact(self) -> bool:
        """เกณฑ์ High-Impact: คะแนนสูงสุดของ Macro หรือ Asset >= HIGH_IMPACT_THRESHOLD"""
        return max(self.macro_impact_score, self.asset_impact_score) >= HIGH_IMPACT_THRESHOLD


class TriageBatchResult(BaseModel):
    """ผลลัพธ์ Batch LLM Triage สำหรับรายการข่าวหลายหัวข้อพร้อมกัน"""
    results: List[MacroImpactTriageResult] = Field(default_factory=list, description="ผลลัพธ์คัดกรองเรียงตามลำดับข่าวที่ส่งเข้า batch")
