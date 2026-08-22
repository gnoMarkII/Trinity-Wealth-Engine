"""Unit tests สำหรับ schemas/youtube_pitch_schemas.py"""
import pytest
from pydantic import ValidationError
from schemas.youtube_pitch_schemas import YouTubeContentPitchItem, YouTubeContentPitchBatch, validate_generated_pitch


def test_youtube_pitch_item_valid():
    item = YouTubeContentPitchItem(
        pitch_id="uuid-1",
        working_titles=["หัวข้อคำถามเจาะลึก", "หัวข้อวิเคราะห์สมมติฐาน", "หัวข้อเตือนภัยและโอกาส"],
        target_audience="นักลงทุนไทย",
        core_thesis="ใจความสำคัญหลัก 1 ประโยคที่ชัดเจนและยาวเกิน 15 ตัวอักษร",
        primary_anchor_event_id="ev-1",
        primary_anchor_title="ข่าวหลักเหตุการณ์ที่ 1",
        parking_lot_ideas=["ไอเดียเสริมที่ 1", "ไอเดียเสริมที่ 2"],
        key_questions_to_answer=["คำถาม 1", "คำถาม 2", "คำถาม 3"],
        research_hypotheses=["สมมติฐาน 1", "สมมติฐาน 2"],
        source_event_ids=["ev-1", "ev-2"],
        source_links=["http://example.com/1", "http://example.com/2"],
        source_titles=["ข่าวที่ 1", "ข่าวที่ 2"],
        recommended_format="Deep Dive 15m",
        estimated_impact="ผลกระทบสำคัญต่อตลาดหุ้นไทย",
    )
    assert item.pitch_id == "uuid-1"
    assert len(item.working_titles) == 3
    assert item.core_thesis == "ใจความสำคัญหลัก 1 ประโยคที่ชัดเจนและยาวเกิน 15 ตัวอักษร"
    assert item.primary_anchor_event_id == "ev-1"
    assert item.primary_anchor_title == "ข่าวหลักเหตุการณ์ที่ 1"
    assert len(item.parking_lot_ideas) == 2


def test_backward_input_alias_core_hook_to_core_thesis():
    # Payload เก่าที่มีเฉพาะ core_hook ต้อง deserialize ได้เป็น core_thesis โดยอัตโนมัติ
    old_payload = {
        "pitch_id": "uuid-legacy",
        "working_titles": ["หัวข้อ 1", "หัวข้อ 2", "หัวข้อ 3"],
        "target_audience": "นักลงทุน",
        "core_hook": "ประโยค Hook จาก checkpoint เก่า",
        "key_questions_to_answer": ["Q1", "Q2", "Q3"],
        "research_hypotheses": ["H1", "H2"],
        "source_event_ids": ["ev-1"],
        "source_links": ["http://example.com/1"],
        "source_titles": ["ข่าว 1"],
        "recommended_format": "Deep Dive 15m",
        "estimated_impact": "Impact",
    }
    item = YouTubeContentPitchItem.model_validate(old_payload)
    assert item.core_thesis == "ประโยค Hook จาก checkpoint เก่า"
    assert item.primary_anchor_event_id == ""
    assert item.parking_lot_ideas == []


def test_canonical_model_dump_emits_core_thesis():
    item = YouTubeContentPitchItem(
        pitch_id="uuid-dump",
        working_titles=["1", "2", "3"],
        target_audience="aud",
        core_thesis="ข้อความ Core Thesis ใหม่",
        primary_anchor_event_id="ev-1",
        key_questions_to_answer=["q1", "q2", "q3"],
        research_hypotheses=["h1", "h2"],
        source_event_ids=["ev-1"],
        source_links=["http://example.com/1"],
        source_titles=["news 1"],
        recommended_format="format",
        estimated_impact="impact",
    )
    dumped = item.model_dump()
    assert "core_thesis" in dumped
    assert dumped["core_thesis"] == "ข้อความ Core Thesis ใหม่"
    assert "core_hook" not in dumped


def test_dual_payload_prefers_core_thesis():
    payload = {
        "pitch_id": "uuid-dual",
        "working_titles": ["1", "2", "3"],
        "target_audience": "aud",
        "core_thesis": "ค่าที่ถูกต้องจาก core_thesis",
        "core_hook": "ค่าเก่าจาก core_hook",
        "key_questions_to_answer": ["q1", "q2", "q3"],
        "research_hypotheses": ["h1", "h2"],
        "source_event_ids": ["ev-1"],
        "source_links": ["http://example.com/1"],
        "source_titles": ["news 1"],
        "recommended_format": "format",
        "estimated_impact": "impact",
    }
    item = YouTubeContentPitchItem.model_validate(payload)
    assert item.core_thesis == "ค่าที่ถูกต้องจาก core_thesis"


def test_parking_lot_ideas_normalization_and_dedup():
    item = YouTubeContentPitchItem(
        pitch_id="uuid-parking",
        working_titles=["1", "2", "3"],
        target_audience="aud",
        core_thesis="ข้อความ Core Thesis ยาวพอสมควร",
        primary_anchor_event_id="ev-1",
        parking_lot_ideas=[
            "  ไอเดียข้อ 1  ",
            "ไอเดียข้อ 1",  # duplicate
            "   ",  # whitespace only
            "ไอเดียข้อ 2",
            "ไอเดียข้อ 3",
            "ไอเดียข้อ 4",
            "ไอเดียข้อ 5",
            "ไอเดียข้อ 6 เกินโควตา",
        ],
        key_questions_to_answer=["q1", "q2", "q3"],
        research_hypotheses=["h1", "h2"],
        source_event_ids=["ev-1"],
        source_links=["http://example.com/1"],
        source_titles=["news 1"],
        recommended_format="format",
        estimated_impact="impact",
    )
    # Should be normalized, deduplicated, stripped of empty items, and capped at 5
    assert item.parking_lot_ideas == [
        "ไอเดียข้อ 1",
        "ไอเดียข้อ 2",
        "ไอเดียข้อ 3",
        "ไอเดียข้อ 4",
        "ไอเดียข้อ 5",
    ]


def test_validate_generated_pitch_success():
    item = YouTubeContentPitchItem(
        pitch_id="uuid-val",
        working_titles=["111", "222", "333"],
        target_audience="aud",
        core_thesis="สรุปใจความสำคัญหลักหนึ่งประโยคที่เกิน 15 ตัวอักษรแน่นอน",
        primary_anchor_event_id="ev-1",
        primary_anchor_title="ข่าวเหตุการณ์หลัก",
        key_questions_to_answer=["q1", "q2", "q3"],
        research_hypotheses=["h1", "h2"],
        source_event_ids=["ev-1", "ev-2"],
        source_links=["http://example.com/1"],
        source_titles=["news 1"],
        recommended_format="format",
        estimated_impact="impact",
        investigation_mode="stock",
        counter_intuitive_lead="เบาะแสสำคัญค้านสายตา: ตลาดหุ้นเติบโตแต่กระแสเงินสดติดลบ",
        analogy_generator="คำเปรียบเปรย: เหมือนรถที่วิ่งด้วยความเร็วสูงแต่เชื้อเพลิงกำลังจะหมด",
        audience_takeaway="เก็บเงินสดสำรอง 6 เดือนไว้ก่อนตัดสินใจลงทุนเพิ่ม",
        thumbnail_concept="ภาพกราฟตลาดหุ้นพุ่งขึ้นแต่กระเป๋าเงินโล่ง",
    )
    batch = YouTubeContentPitchBatch(pitches=[item], date_range_summary="summary", total_source_events=2)
    validate_generated_pitch(batch)


def test_validate_generated_pitch_failure_when_core_thesis_short():
    item = YouTubeContentPitchItem(
        pitch_id="uuid-fail-thesis",
        working_titles=["111", "222", "333"],
        target_audience="aud",
        core_thesis="สั้นเกินไป",  # < 15 chars
        primary_anchor_event_id="ev-1",
        key_questions_to_answer=["q1", "q2", "q3"],
        research_hypotheses=["h1", "h2"],
        source_event_ids=["ev-1"],
        source_links=["http://example.com/1"],
        source_titles=["news 1"],
        recommended_format="format",
        estimated_impact="impact",
        counter_intuitive_lead="เบาะแสสำคัญค้านสายตา: ตลาดหุ้นเติบโตแต่กระแสเงินสดติดลบ",
        analogy_generator="คำเปรียบเปรย: เหมือนรถที่วิ่งด้วยความเร็วสูงแต่เชื้อเพลิงกำลังจะหมด",
        audience_takeaway="เก็บเงินสดสำรอง 6 เดือนไว้ก่อนตัดสินใจลงทุนเพิ่ม",
        thumbnail_concept="ภาพกราฟตลาดหุ้นพุ่งขึ้นแต่กระเป๋าเงินโล่ง",
    )
    batch = YouTubeContentPitchBatch(pitches=[item], date_range_summary="summary", total_source_events=1)
    with pytest.raises(ValueError, match="missing or insufficient core_thesis"):
        validate_generated_pitch(batch)


def test_validate_generated_pitch_failure_when_primary_anchor_missing_or_not_in_sources():
    # 1. Missing anchor
    item_missing = YouTubeContentPitchItem(
        pitch_id="uuid-missing-anchor",
        working_titles=["111", "222", "333"],
        target_audience="aud",
        core_thesis="สรุปใจความสำคัญหลักหนึ่งประโยคที่เกิน 15 ตัวอักษรแน่นอน",
        primary_anchor_event_id="",
        key_questions_to_answer=["q1", "q2", "q3"],
        research_hypotheses=["h1", "h2"],
        source_event_ids=["ev-1"],
        source_links=["http://example.com/1"],
        source_titles=["news 1"],
        recommended_format="format",
        estimated_impact="impact",
        counter_intuitive_lead="เบาะแสสำคัญค้านสายตา: ตลาดหุ้นเติบโตแต่กระแสเงินสดติดลบ",
        analogy_generator="คำเปรียบเปรย: เหมือนรถที่วิ่งด้วยความเร็วสูงแต่เชื้อเพลิงกำลังจะหมด",
        audience_takeaway="เก็บเงินสดสำรอง 6 เดือนไว้ก่อนตัดสินใจลงทุนเพิ่ม",
        thumbnail_concept="ภาพกราฟตลาดหุ้นพุ่งขึ้นแต่กระเป๋าเงินโล่ง",
    )
    batch1 = YouTubeContentPitchBatch(pitches=[item_missing], date_range_summary="summary", total_source_events=1)
    with pytest.raises(ValueError, match="missing primary_anchor_event_id"):
        validate_generated_pitch(batch1)

    # 2. Anchor not in source_event_ids
    item_mismatch = YouTubeContentPitchItem(
        pitch_id="uuid-mismatch-anchor",
        working_titles=["111", "222", "333"],
        target_audience="aud",
        core_thesis="สรุปใจความสำคัญหลักหนึ่งประโยคที่เกิน 15 ตัวอักษรแน่นอน",
        primary_anchor_event_id="ev-999",  # Not in source_event_ids
        key_questions_to_answer=["q1", "q2", "q3"],
        research_hypotheses=["h1", "h2"],
        source_event_ids=["ev-1", "ev-2"],
        source_links=["http://example.com/1"],
        source_titles=["news 1"],
        recommended_format="format",
        estimated_impact="impact",
        counter_intuitive_lead="เบาะแสสำคัญค้านสายตา: ตลาดหุ้นเติบโตแต่กระแสเงินสดติดลบ",
        analogy_generator="คำเปรียบเปรย: เหมือนรถที่วิ่งด้วยความเร็วสูงแต่เชื้อเพลิงกำลังจะหมด",
        audience_takeaway="เก็บเงินสดสำรอง 6 เดือนไว้ก่อนตัดสินใจลงทุนเพิ่ม",
        thumbnail_concept="ภาพกราฟตลาดหุ้นพุ่งขึ้นแต่กระเป๋าเงินโล่ง",
    )
    batch2 = YouTubeContentPitchBatch(pitches=[item_mismatch], date_range_summary="summary", total_source_events=2)
    with pytest.raises(ValueError, match="not in source_event_ids"):
        validate_generated_pitch(batch2)

