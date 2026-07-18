"""Pydantic schemas สำหรับ YouTube Content Pitching & Research-Grade Briefing Book"""
from typing import List
from pydantic import BaseModel, Field


class YouTubeContentPitchItem(BaseModel):
    """โครงสร้างไอเดียคลิป 1 หัวข้อ (Multi-source Pitch)"""
    pitch_id: str = Field(..., description="UUID รหัสไอเดีย")
    working_titles: List[str] = Field(
        ...,
        min_length=3,
        max_length=3,
        description="รายชื่อหัวข้อคลิป 3 สไตล์: 1.คำถามเจาะลึก 2.วิเคราะห์สมมติฐาน 3.เตือนภัย/โอกาส",
    )
    target_audience: str = Field(..., description="กลุ่มคนดูเป้าหมาย")
    core_hook: str = Field(..., description="ประโยค Hook 30 วินาทีแรกที่ดึงดูดความสนใจ")
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


class YouTubeContentPitchBatch(BaseModel):
    """รายการไอเดียทั้งหมดที่สกัดได้ตามช่วงวันที่เลือก"""
    pitches: List[YouTubeContentPitchItem] = Field(..., description="รายการไอเดียคลิป 3-5 หัวข้อ")
    date_range_summary: str = Field(..., description="สรุปช่วงวันที่ที่ใช้คัดกรองข่าว")
    total_source_events: int = Field(..., description="จำนวนเหตุการณ์ข่าวที่นำมาร่วมวิเคราะห์")
