"""Unit tests สำหรับ tools/content/notebooklm/prompts.py"""
from tools.content.notebooklm.prompts import (
    build_notebook_query,
    build_research_query,
    extract_notebooklm_prompts,
)

_REAL_SECTION = """## NotebookLM Prompts

- [SOCRATIC] ราคาน้ำมันที่ระดับ 100 ดอลลาร์ส่งผลกระทบต่อการตัดสินใจของเฟดอย่างไรในเชิงทฤษฎีเศรษฐศาสตร์?
  *Output Format: บทวิเคราะห์เชิงเปรียบเทียบ*

- [BLIND_SPOT] มีปัจจัยใดบ้างที่อาจทำให้ราคาน้ำมันพุ่งสูงขึ้นโดยที่ตลาดไม่ได้คาดการณ์ไว้?
  *Output Format: รายการปัจจัยเสี่ยง*

- [RESEARCH] วิเคราะห์ความสัมพันธ์ระหว่างดัชนี DXY ที่ 101.19 และราคาน้ำมันดิบในปัจจุบัน
  *Output Format: บทสรุปเชิงวิเคราะห์*

- [FEYNMAN] อธิบายให้เด็กอายุ 12 ปีฟังว่าทำไมความตึงเครียดในช่องแคบฮอร์มุซถึงทำให้หุ้นเทคโนโลยีตกได้
  *Output Format: คำอธิบายแบบง่าย*

- [RESEARCH] เปรียบเทียบผลกระทบของราคาน้ำมันต่อกลุ่มพลังงาน (XLE) และกลุ่มเทคโนโลยี (XLK) ในช่วง 90 วันที่ผ่านมา
  *Output Format: ตารางเปรียบเทียบผลตอบแทน*

## Evidence Ledger

- E01: ตัวอย่าง evidence ที่ไม่ควรถูกดึงเข้ามาด้วย
"""


def test_extract_parses_all_four_types_from_real_section():
    prompts = extract_notebooklm_prompts(_REAL_SECTION)
    assert len(prompts) == 5
    assert [p.prompt_type for p in prompts] == ["SOCRATIC", "BLIND_SPOT", "RESEARCH", "FEYNMAN", "RESEARCH"]
    assert prompts[0].question_or_prompt.startswith("ราคาน้ำมันที่ระดับ 100 ดอลลาร์")
    assert prompts[0].expected_output_format == "บทวิเคราะห์เชิงเปรียบเทียบ"
    assert [p.prompt_id for p in prompts] == ["P01", "P02", "P03", "P04", "P05"]


def test_extract_stops_at_next_heading_does_not_leak_into_other_sections():
    prompts = extract_notebooklm_prompts(_REAL_SECTION)
    assert not any("E01" in p.question_or_prompt for p in prompts)


def test_extract_returns_empty_list_when_section_missing():
    text = "# Briefing\n\nไฟล์รุ่นเก่าที่ไม่มี section นี้เลย\n\n## Evidence Ledger\n- E01: x\n"
    assert extract_notebooklm_prompts(text) == []


def test_extract_skips_unknown_tag_without_crashing():
    text = (
        "## NotebookLM Prompts\n\n"
        "- [UNKNOWN_TAG] คำถามที่มี tag แปลกๆ\n"
        "  *Output Format: อะไรก็ได้*\n\n"
        "- [SOCRATIC] คำถามปกติ\n"
        "  *Output Format: สั้นๆ*\n"
    )
    prompts = extract_notebooklm_prompts(text)
    assert len(prompts) == 1
    assert prompts[0].prompt_type == "SOCRATIC"


def test_build_research_query_combines_research_tagged_only():
    prompts = extract_notebooklm_prompts(_REAL_SECTION)
    query = build_research_query(prompts)
    assert query == (
        "วิเคราะห์ความสัมพันธ์ระหว่างดัชนี DXY ที่ 101.19 และราคาน้ำมันดิบในปัจจุบัน; "
        "เปรียบเทียบผลกระทบของราคาน้ำมันต่อกลุ่มพลังงาน (XLE) และกลุ่มเทคโนโลยี (XLK) ในช่วง 90 วันที่ผ่านมา"
    )


def test_build_research_query_returns_none_when_no_research_tagged_prompt():
    text = "## NotebookLM Prompts\n\n- [SOCRATIC] คำถาม\n  *Output Format: สั้นๆ*\n"
    prompts = extract_notebooklm_prompts(text)
    assert len(prompts) == 1
    assert prompts[0].prompt_type == "SOCRATIC"
    assert build_research_query(prompts) is None


def test_build_notebook_query_includes_output_format_hint():
    prompts = extract_notebooklm_prompts(_REAL_SECTION)
    query = build_notebook_query(prompts[0])
    assert prompts[0].question_or_prompt in query
    assert "บทวิเคราะห์เชิงเปรียบเทียบ" in query


def test_build_notebook_query_without_output_format_returns_question_only():
    from schemas.briefing_book_schemas import NotebookLMPromptRecord
    prompt = NotebookLMPromptRecord(
        prompt_id="P01", prompt_type="SOCRATIC", question_or_prompt="คำถามเปล่า", expected_output_format="",
    )
    assert build_notebook_query(prompt) == "คำถามเปล่า"
