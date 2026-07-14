"""Pydantic schemas สำหรับระบบ News Funnel Architecture & Obsidian Linking Keys

รองรับ Ingestion (Title + Summary), Deduplication/Clustering, Fast LLM Triage (Impact Score),
และ Twice-Daily 12h Synthesis สรุป 3 Key Macro Themes
"""
from datetime import datetime
import re
from typing import List, Optional
from pydantic import BaseModel, Field, computed_field, field_validator


def strip_wikilink(text: str) -> str:
    """ลบเครื่องหมาย [[ ... ]] และ Alias [[File|Alias]] ออกจากข้อความ เหลือชื่อหลัก พร้อม trim whitespace"""
    clean = text.strip()
    match = re.search(r"\[\[([^|\]]+)(?:\|[^\]]+)?\]\]", clean)
    if match:
        return match.group(1).strip()
    return clean.split("|")[0].strip() if "|" in clean else clean


class NewsFunnelRawItem(BaseModel):
    """ข้อมูลดิบแต่ละรายการที่ดึงมาจาก RSS feed"""
    title: str = Field(description="หัวข้อข่าว")
    summary: str = Field(default="", description="สรุปหรือรายละเอียดข่าวเบื้องต้นจาก RSS")
    link: str = Field(description="URL ของข่าว")
    source: str = Field(default="Unknown", description="ชื่อสำนักข่าว เช่น Investing.com")
    published_at: Optional[str] = Field(default=None, description="เวลาเผยแพร่")


class ClusteredNewsEvent(BaseModel):
    """เหตุการณ์ข่าวที่ผ่านการตัดซ้ำ/ยุบรวมจากหลายสำนักข่าวในหัวข้อเดียวกัน"""
    event_id: str = Field(description="รหัสระบุเหตุการณ์เฉพาะ (UUID หรือ Hash)")
    canonical_title: str = Field(description="หัวข้อข่าวตัวแทนที่เป็นกลางและชัดเจนที่สุด")
    comprehensive_summary: str = Field(description="สรุปเนื้อหาที่รวมข้อมูลจากทุกแหล่งข่าว")
    source_count: int = Field(default=1, description="จำนวนแหล่งข่าวที่รายงานเหตุการณ์นี้")
    sources: List[str] = Field(default_factory=list, description="รายชื่อสำนักข่าว")
    links: List[str] = Field(default_factory=list, description="รายการ URL ทั้งหมดในคลัสเตอร์นี้")
    oldest_pub_time: Optional[str] = Field(default=None)
    latest_pub_time: Optional[str] = Field(default=None)


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


class MacroThemeDigest(BaseModel):
    """สรุปรายธีมเศรษฐกิจสำหรับรายงานรอบ 12 ชั่วโมง"""
    theme_title: str = Field(description="ชื่อธีมหลัก เช่น การส่งสัญญาณคงดอกเบี้ยของ Fed และแรงกดดัน Inflation")
    key_takeaways: List[str] = Field(default_factory=list, description="ประเด็นสำคัญที่สังเคราะห์ได้")
    linked_assets: List[str] = Field(default_factory=list, description="รายการ Wikilink สินทรัพย์ เช่น [[NVDA]], [[PTT]]")
    linked_themes: List[str] = Field(default_factory=list, description="รายการ Wikilink ธีม เช่น [[Monetary Policy]], [[Inflation]]")
    policy_implications: str = Field(default="", description="ผลกระทบและนัยต่อนโยบายพอร์ตโฟลิโอ")

    @field_validator("linked_assets", "linked_themes", mode="after")
    @classmethod
    def _normalize_wikilinks(cls, values: List[str]) -> List[str]:
        result = []
        for v in values:
            if not v:
                continue
            clean = strip_wikilink(v)
            if clean:
                result.append(f"[[{clean}]]")
        return result


class ThemeSynthesisBatchResult(BaseModel):
    """ผลลัพธ์ Batch LLM Synthesis สูงสุด 3 ธีม"""
    themes: List[MacroThemeDigest] = Field(default_factory=list, description="รายการธีมหลักสูงสุด 3 ธีม")


class DailyFunnelReport(BaseModel):
    """รายงานสรุป Key Macro Themes ประจำรอบ 12 ชั่วโมง (เช้า/ค่ำ)"""
    report_title: str = Field(description="ชื่อรายงาน เช่น Macro Themes Digest - 2026-07-13 Morning")
    report_date: str = Field(description="วันที่ เช่น 2026-07-13")
    batch_period: str = Field(default="morning_12h", description="รอบเวลา เช่น morning_12h หรือ evening_12h")
    approved_by: str = Field(default="scheduled_auto", description="ผู้หรือระบบที่อนุมัติ เช่น scheduled_auto หรือ user_kanban_hitl")
    themes: List[MacroThemeDigest] = Field(default_factory=list, description="รายการธีมหลัก 3 ธีม")
    total_events_analyzed: int = Field(default=0, description="จำนวนข่าวที่นำมาวิเคราะห์ในรอบนี้")
    high_impact_event_ids: List[str] = Field(default_factory=list, description="รายการ event_id ที่นำมาสังเคราะห์")
