import pytest
from core.text_utils import strip_html_tags, repair_mojibake


def test_strip_html_tags_basic():
    html_input = "<p>บริษัท <b>เดลต้า</b> รายงานกำไรสุทธิโตต่อเนื่อง</p>"
    result = strip_html_tags(html_input)
    assert result == "บริษัท เดลต้า รายงานกำไรสุทธิโตต่อเนื่อง"


def test_strip_html_tags_preserves_financial_comparison_operators():
    # Crucial test: mathematical/financial '<' and '>' must NOT be treated as HTML tags
    financial_text = "คาดการณ์ P/E < 15.5 เท่า ขณะที่ Dividend Yield > 4.2% และ Gross Margin > 25%"
    result = strip_html_tags(financial_text)
    assert "P/E < 15.5 เท่า" in result
    assert "Dividend Yield > 4.2%" in result
    assert "Gross Margin > 25%" in result
    assert result == financial_text


def test_strip_html_tags_with_mixed_tags_and_operators():
    mixed = "<div><b>Valuation Check:</b> P/E < 15 เท่า และ <i>ROE > 18%</i></div>"
    result = strip_html_tags(mixed)
    assert "Valuation Check:" in result
    assert "P/E < 15 เท่า" in result
    assert "ROE > 18%" in result


def test_strip_html_tags_strips_scripts_and_styles():
    html_with_script = (
        "<div>ข่าวเด่นประจำวัน<script>alert('malicious')</script>"
        "<style>.hide { display: none; }</style>"
        "<p>ตลาดหุ้นปิดบวก</p></div>"
    )
    result = strip_html_tags(html_with_script)
    assert "alert" not in result
    assert "display" not in result
    assert "ข่าวเด่นประจำวัน ตลาดหุ้นปิดบวก" in result


def test_strip_html_tags_handles_entities():
    html_entities = "PTT &amp; CPALL &quot;Outperform&quot; rating &lt;target 50 THB&gt;"
    result = strip_html_tags(html_entities)
    assert result == 'PTT & CPALL "Outperform" rating <target 50 THB>'
