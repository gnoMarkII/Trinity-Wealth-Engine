"""Tests for archivist_tools helpers — §3.1 Atomic Storage invariant"""
import os
import pytest
from pathlib import Path

from tools.archivist_tools import _atomic_write_text, _sanitize_filename


class TestAtomicWriteText:
    def test_creates_file_with_correct_content(self, tmp_path):
        target = tmp_path / "test.md"
        _atomic_write_text(target, "hello world")
        assert target.read_text(encoding="utf-8") == "hello world"

    def test_creates_parent_directories(self, tmp_path):
        target = tmp_path / "a" / "b" / "c.md"
        _atomic_write_text(target, "nested")
        assert target.read_text(encoding="utf-8") == "nested"

    def test_overwrites_existing_file(self, tmp_path):
        target = tmp_path / "file.md"
        target.write_text("old content", encoding="utf-8")
        _atomic_write_text(target, "new content")
        assert target.read_text(encoding="utf-8") == "new content"

    def test_no_temp_file_left_after_success(self, tmp_path):
        target = tmp_path / "clean.md"
        _atomic_write_text(target, "content")
        assert list(tmp_path.glob("*.tmp")) == []

    def test_unicode_content(self, tmp_path):
        target = tmp_path / "thai.md"
        content = "ภาษาไทย: สวัสดีครับ 🎉"
        _atomic_write_text(target, content)
        assert target.read_text(encoding="utf-8") == content

    def test_original_preserved_on_replace_failure(self, tmp_path, monkeypatch):
        """§3.1: if os.replace raises, original must be intact and temp file cleaned up"""
        target = tmp_path / "original.md"
        target.write_text("original content", encoding="utf-8")

        def mock_replace(src, dst):
            raise OSError("disk full")

        monkeypatch.setattr(os, "replace", mock_replace)

        with pytest.raises(OSError, match="disk full"):
            _atomic_write_text(target, "replacement")

        assert target.read_text(encoding="utf-8") == "original content"
        assert list(tmp_path.glob("*.tmp")) == []

    def test_temp_file_in_same_directory(self, tmp_path, monkeypatch):
        """Temp file must be on same filesystem to guarantee atomic rename"""
        seen_dirs = []
        original_mkstemp = __import__("tempfile").mkstemp

        def capture_mkstemp(*args, **kwargs):
            fd, name = original_mkstemp(*args, **kwargs)
            seen_dirs.append(Path(name).parent)
            return fd, name

        monkeypatch.setattr("tempfile.mkstemp", capture_mkstemp)
        target = tmp_path / "sub" / "file.md"
        _atomic_write_text(target, "data")
        assert seen_dirs and seen_dirs[0] == target.parent


class TestSanitizeFilename:
    def test_clean_name_unchanged(self):
        assert _sanitize_filename("hello-world") == "hello-world"

    def test_invalid_chars_replaced(self):
        result = _sanitize_filename('file<>:"/|?*name')
        for ch in '<>:"/|?*':
            assert ch not in result

    def test_backslash_replaced(self):
        result = _sanitize_filename("path\\file")
        assert "\\" not in result

    def test_leading_trailing_spaces_stripped(self):
        assert _sanitize_filename("  hello  ") == "hello"

    def test_leading_trailing_dots_stripped(self):
        assert _sanitize_filename("...hello...") == "hello"

    def test_consecutive_dashes_collapsed(self):
        assert _sanitize_filename("a---b") == "a-b"

    def test_leading_trailing_dashes_stripped(self):
        assert _sanitize_filename("-hello-") == "hello"

    def test_empty_string_returns_untitled(self):
        assert _sanitize_filename("") == "untitled"

    def test_all_invalid_chars_returns_untitled(self):
        assert _sanitize_filename("<>:/") == "untitled"

    def test_thai_chars_preserved(self):
        result = _sanitize_filename("บันทึก-2025-01-15")
        assert "บันทึก" in result
        assert "2025-01-15" in result

    def test_ticker_with_dot_preserved(self):
        # Dot is NOT an invalid char — mid-name dots must survive
        result = _sanitize_filename("BRK.B")
        assert result == "BRK.B"

    def test_multiple_invalid_chars_all_replaced(self):
        result = _sanitize_filename("a:b/c")
        assert ":" not in result
        assert "/" not in result
        assert "a" in result
        assert "b" in result
        assert "c" in result
