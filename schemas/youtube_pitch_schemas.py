"""Pydantic schemas สำหรับ YouTube Content Pitching & Research-Grade Briefing Book"""
import re
import unicodedata
from typing import Any, List, Literal, Optional
from pydantic import AliasChoices, BaseModel, Field, field_validator


class YouTubeContentPitchItem(BaseModel):
    """โครงสร้างไอเดียคลิป 1 หัวข้อ (Single Deep Dive Anchor Pitch)"""
    pitch_id: str = Field(..., description="UUID รหัสไอเดีย")
    working_titles: List[str] = Field(
        ...,
        min_length=3,
        max_length=3,
        description="รายชื่อหัวข้อคลิป 3 สไตล์: 1.คำถามเจาะลึก 2.วิเคราะห์สมมติฐาน 3.เตือนภัย/โอกาส",
    )
    target_audience: str = Field(..., description="กลุ่มคนดูเป้าหมาย")
    core_thesis: str = Field(
        ...,
        validation_alias=AliasChoices("core_thesis", "core_hook"),
        description="ใจความสำคัญหรือข้อสรุปหลัก 1 ประโยคภาษาไทย (Single Core Thesis)",
    )
    primary_anchor_event_id: str = Field(
        default="",
        description="รหัส event_id ของข่าวหลักที่เป็นแกน (Authoritative Primary Anchor Event ID)",
    )
    primary_anchor_title: Optional[str] = Field(
        default=None,
        description="ชื่อข่าวต้นทางของ Anchor event (Display Metadata ถูก hydrate จาก server)",
    )
    parking_lot_ideas: List[str] = Field(
        default_factory=list,
        max_length=5,
        description="ไอเดียประเด็นย่อยน่าสนใจที่แยกเก็บไว้สำหรับคลิปถัดไป (สูงสุด 5 รายการ)",
    )
    key_questions_to_answer: List[str] = Field(
        ...,
        min_length=3,
        description="คำถามสำคัญ 3-5 ข้อที่คลิปต้องตอบให้เคลียร์",
    )
    research_hypotheses: List[str] = Field(
        ...,
        min_length=2,
        description="สมมติฐานเชิงวิเคราะห์ 2-3 ข้อสำหรับให้ NotebookLM โหมด Research ค้นคว้าต่อ",
    )
    source_event_ids: List[str] = Field(..., description="รหัสอ้างอิงข่าว (event_id) จาก News Funnel Store")
    source_links: List[str] = Field(..., description="ลิงก์บทความต้นฉบับ")
    source_titles: List[str] = Field(..., description="ชื่อข่าวต้นทางที่นำมารวมใน Pitch นี้")
    recommended_format: str = Field(..., description="รูปแบบคลิป เช่น 'Deep Dive 15m' หรือ 'Quick Update 5m'")
    estimated_impact: str = Field(..., description="สรุปผลกระทบสั้นๆ 1 บรรทัด")
    investigation_mode: Literal["stock", "macro", "mixed"] = Field(
        default="mixed", description="ประเภทการสืบสวน: stock/macro/mixed"
    )
    target_symbols: List[str] = Field(
        default_factory=list, description="สัญลักษณ์หุ้นหรือสินทรัพย์ทางการเงินที่เกี่ยวข้อง"
    )
    counter_intuitive_lead: str = Field(
        default="", description="ข้อความเปิดเรื่อง/สมมติฐานค้านสายตาเชิงสืบสวน (Investigative Lead Narrative)"
    )
    analogy_generator: str = Field(
        default="", description="คำเปรียบเปรยเปรียบเทียบกับชีวิตประจำวัน"
    )
    thumbnail_concept: str = Field(
        default="", description="ไอเดียภาพหน้าปกคลิป (Visual/Text Concept) แบบดึงดูดสายตา"
    )
    audience_takeaway: str = Field(
        default="", description="WIIFM: สิ่งที่คนดูจะได้ประโยชน์หรือเอาไปใช้ต่อได้ทันที (Actionable Advice)"
    )
    presentation_style: Literal["narrative", "interview_qa"] = Field(
        default="narrative", description="สไตล์การนำเสนอเนื้อหาที่ส่งผลต่อการสร้าง Script (narrative/interview_qa)"
    )
    # This is deliberately advisory and backward-compatible.  It is populated
    # from the actual selected source records before the approval interrupt, so
    # a user never has to approve a pitch whose evidence cannot pass the hard
    # provenance gate.  Older checkpoints omit it and retain ``unknown``.
    source_readiness: Literal["ready", "needs_refresh", "blocked", "unknown"] = Field(
        default="unknown", description="ความพร้อมของแหล่งข้อมูลก่อนสร้าง Briefing Book"
    )
    source_readiness_issues: List[str] = Field(
        default_factory=list, description="เหตุผลที่ต้อง refresh หรือห้ามอนุมัติ Pitch นี้"
    )
    unverified_draft_issue_codes: List[str] = Field(
        default_factory=list, description="รหัสปัญหา Source Readiness สำหรับ Unverified Draft"
    )
    unverified_draft_eligible: bool = Field(
        default=False,
        description="Server-authorized eligibility for a provenance-only Unverified Draft",
    )
    unverified_draft_eligibility_token: str = Field(
        default="", description="Token รับรองความถูกต้องเพื่อ bypass ไปเป็น Unverified Draft"
    )

    @field_validator("parking_lot_ideas", mode="before")
    @classmethod
    def normalize_and_dedup_parking_lot(cls, v: Any) -> List[str]:
        if not isinstance(v, list):
            return []
        seen = set()
        normalized = []
        for item in v:
            if not isinstance(item, str):
                continue
            norm = unicodedata.normalize("NFC", item).strip()
            norm = re.sub(r"\s+", " ", norm)
            key = norm.casefold()
            if key and key not in seen:
                seen.add(key)
                normalized.append(norm)
        return normalized[:5]


class YouTubeContentPitchBatch(BaseModel):
    """รายการไอเดียทั้งหมดที่สกัดได้ตามช่วงวันที่เลือก"""
    pitches: List[YouTubeContentPitchItem] = Field(..., description="รายการไอเดียคลิป 3-5 หัวข้อ")
    date_range_summary: str = Field(..., description="สรุปช่วงวันที่ที่ใช้คัดกรองข่าว")
    total_source_events: int = Field(..., description="จำนวนเหตุการณ์ข่าวที่นำมาร่วมวิเคราะห์")


def validate_generated_pitch(batch: YouTubeContentPitchBatch) -> None:
    """ตรวจสอบความครบถ้วนของฟิลด์สืบสวน, Single Anchor, และสมมติฐานใน Pitch ที่เพิ่งสร้างใหม่จาก LLM"""
    for i, pitch in enumerate(batch.pitches):
        thesis = pitch.core_thesis.strip()
        lead = pitch.counter_intuitive_lead.strip()
        analogy = pitch.analogy_generator.strip()
        takeaway = pitch.audience_takeaway.strip()
        thumbnail = pitch.thumbnail_concept.strip()
        anchor_id = pitch.primary_anchor_event_id.strip()

        if len(thesis) < 15:
            raise ValueError(
                f"Pitch {i} ('{pitch.pitch_id}') missing or insufficient core_thesis (min 15 non-whitespace chars, got '{thesis}')"
            )
        if not anchor_id:
            raise ValueError(
                f"Pitch {i} ('{pitch.pitch_id}') missing primary_anchor_event_id"
            )
        if anchor_id not in pitch.source_event_ids:
            raise ValueError(
                f"Pitch {i} ('{pitch.pitch_id}') primary_anchor_event_id '{anchor_id}' is not in source_event_ids: {pitch.source_event_ids}"
            )
        if len(lead) < 10:
            raise ValueError(
                f"Pitch {i} ('{pitch.pitch_id}') missing or insufficient counter_intuitive_lead (min 10 non-whitespace chars)"
            )
        if len(analogy) < 10:
            raise ValueError(
                f"Pitch {i} ('{pitch.pitch_id}') missing or insufficient analogy_generator (min 10 non-whitespace chars)"
            )
        if len(takeaway) < 10:
            raise ValueError(
                f"Pitch {i} ('{pitch.pitch_id}') missing or insufficient audience_takeaway (min 10 non-whitespace chars)"
            )
        if len(thumbnail) < 5:
            raise ValueError(
                f"Pitch {i} ('{pitch.pitch_id}') missing or insufficient thumbnail_concept (min 5 non-whitespace chars)"
            )

