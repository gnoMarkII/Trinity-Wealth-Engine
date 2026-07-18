"""Unit tests สำหรับ schemas/youtube_pitch_schemas.py"""
import pytest
from pydantic import ValidationError
from schemas.youtube_pitch_schemas import YouTubeContentPitchItem, YouTubeContentPitchBatch


def test_youtube_pitch_item_valid():
    item = YouTubeContentPitchItem(
        pitch_id="uuid-1",
        working_titles=["หัวข้อคำถามเจาะลึก", "หัวข้อวิเคราะห์สมมติฐาน", "หัวข้อเตือนภัยและโอกาส"],
        target_audience="นักลงทุนไทย",
        core_hook="Hook ประโยคเปิดที่น่าสนใจมาก",
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


def test_youtube_pitch_item_invalid_working_titles_count():
    # working_titles น้อยกว่าหรือมากกว่า 3 ต้องเกิด ValidationError
    with pytest.raises(ValidationError):
        YouTubeContentPitchItem(
            pitch_id="uuid-2",
            working_titles=["หัวข้อที่ 1", "หัวข้อที่ 2"],  # แค่ 2
            target_audience="นักลงทุนไทย",
            core_hook="Hook",
            key_questions_to_answer=["คำถาม 1", "คำถาม 2", "คำถาม 3"],
            research_hypotheses=["สมมติฐาน 1", "สมมติฐาน 2"],
            source_event_ids=["ev-1"],
            source_links=["http://example.com/1"],
            source_titles=["ข่าวที่ 1"],
            recommended_format="Quick 5m",
            estimated_impact="Impact",
        )

    with pytest.raises(ValidationError):
        YouTubeContentPitchItem(
            pitch_id="uuid-3",
            working_titles=["1", "2", "3", "4"],  # 4 ข้อ
            target_audience="นักลงทุนไทย",
            core_hook="Hook",
            key_questions_to_answer=["คำถาม 1", "คำถาม 2", "คำถาม 3"],
            research_hypotheses=["สมมติฐาน 1", "สมมติฐาน 2"],
            source_event_ids=["ev-1"],
            source_links=["http://example.com/1"],
            source_titles=["ข่าวที่ 1"],
            recommended_format="Quick 5m",
            estimated_impact="Impact",
        )


def test_youtube_pitch_batch_valid():
    item = YouTubeContentPitchItem(
        pitch_id="uuid-1",
        working_titles=["หัวข้อคำถามเจาะลึก", "หัวข้อวิเคราะห์สมมติฐาน", "หัวข้อเตือนภัยและโอกาส"],
        target_audience="นักลงทุนไทย",
        core_hook="Hook",
        key_questions_to_answer=["คำถาม 1", "คำถาม 2", "คำถาม 3"],
        research_hypotheses=["สมมติฐาน 1", "สมมติฐาน 2"],
        source_event_ids=["ev-1"],
        source_links=["http://example.com/1"],
        source_titles=["ข่าวที่ 1"],
        recommended_format="Deep Dive 15m",
        estimated_impact="Impact",
    )
    batch = YouTubeContentPitchBatch(
        pitches=[item],
        date_range_summary="7 วันล่าสุด",
        total_source_events=1,
    )
    assert len(batch.pitches) == 1
    assert batch.total_source_events == 1
