from core.utils import normalize_content


def test_normalize_str():
    assert normalize_content("hello") == "hello"
    assert normalize_content("  hello  ") == "hello"


def test_normalize_empty():
    assert normalize_content("") == ""
    assert normalize_content(None) == ""


def test_normalize_gemini_list_format():
    content = [{"text": "part1"}, {"text": "part2"}]
    assert normalize_content(content) == "part1 part2"


def test_normalize_mixed_list():
    content = [{"text": "a"}, "b", {"other": "x"}]
    out = normalize_content(content)
    assert "a" in out
    assert "b" in out
