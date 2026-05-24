"""Tests for core/agent_log.py — §4 Prefix Logging contract"""
import pytest

import core.agent_log as agent_log


class TestTruncate:
    def test_short_text_unchanged(self):
        assert agent_log._truncate("hello") == "hello"

    def test_long_text_truncated_with_ellipsis(self):
        text = "a" * 300
        result = agent_log._truncate(text)
        assert result.endswith("...")
        assert len(result) <= 203  # 200 chars + "..."

    def test_text_at_exact_limit_not_truncated(self):
        text = "a" * 200
        result = agent_log._truncate(text)
        assert result == text
        assert not result.endswith("...")

    def test_empty_string_returns_empty(self):
        assert agent_log._truncate("") == ""

    def test_whitespace_collapsed(self):
        result = agent_log._truncate("hello   world\n\t foo")
        assert result == "hello world foo"

    def test_custom_limit(self):
        result = agent_log._truncate("hello world", limit=5)
        assert result == "hello..."

    def test_long_whitespace_text_truncated(self):
        text = "word " * 100  # collapses to ~499 chars
        result = agent_log._truncate(text, limit=20)
        assert result.endswith("...")
        assert len(result) <= 23


class TestLabel:
    def test_user_lowercase_preserved(self):
        assert agent_log._label("user") == "user"

    def test_manager_capitalized(self):
        assert agent_log._label("manager") == "Manager"

    def test_archivist_capitalized(self):
        assert agent_log._label("archivist") == "Archivist"

    def test_bookkeeper_capitalized(self):
        assert agent_log._label("bookkeeper") == "Bookkeeper"

    def test_already_capitalized_stays(self):
        assert agent_log._label("Manager") == "Manager"

    def test_researcher_capitalized(self):
        assert agent_log._label("researcher") == "Researcher"


class TestLogRouting:
    def test_creates_log_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(agent_log, "_LOG_DIR", tmp_path)
        agent_log.log_routing("manager", "bookkeeper")
        assert len(list(tmp_path.glob("Agent_Log_*.md"))) == 1

    def test_header_contains_arrow_and_labels(self, tmp_path, monkeypatch):
        monkeypatch.setattr(agent_log, "_LOG_DIR", tmp_path)
        agent_log.log_routing("manager", "bookkeeper")
        content = list(tmp_path.glob("Agent_Log_*.md"))[0].read_text(encoding="utf-8")
        assert "Manager → Bookkeeper" in content

    def test_reason_included(self, tmp_path, monkeypatch):
        monkeypatch.setattr(agent_log, "_LOG_DIR", tmp_path)
        agent_log.log_routing("manager", "archivist", reason="save_to_vault")
        content = list(tmp_path.glob("Agent_Log_*.md"))[0].read_text(encoding="utf-8")
        assert "reason: save_to_vault" in content

    def test_no_reason_no_reason_line(self, tmp_path, monkeypatch):
        monkeypatch.setattr(agent_log, "_LOG_DIR", tmp_path)
        agent_log.log_routing("user", "manager")
        content = list(tmp_path.glob("Agent_Log_*.md"))[0].read_text(encoding="utf-8")
        assert "reason:" not in content

    def test_long_content_truncated(self, tmp_path, monkeypatch):
        monkeypatch.setattr(agent_log, "_LOG_DIR", tmp_path)
        agent_log.log_routing("user", "manager", content="word " * 100)
        content = list(tmp_path.glob("Agent_Log_*.md"))[0].read_text(encoding="utf-8")
        assert "..." in content

    def test_user_label_lowercase(self, tmp_path, monkeypatch):
        monkeypatch.setattr(agent_log, "_LOG_DIR", tmp_path)
        agent_log.log_routing("user", "manager")
        content = list(tmp_path.glob("Agent_Log_*.md"))[0].read_text(encoding="utf-8")
        assert "user → Manager" in content

    def test_frontmatter_written_to_new_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr(agent_log, "_LOG_DIR", tmp_path)
        agent_log.log_routing("manager", "researcher")
        content = list(tmp_path.glob("Agent_Log_*.md"))[0].read_text(encoding="utf-8")
        assert "entity_type: agent_log" in content
        assert "tags: [log, agents, system]" in content

    def test_appends_multiple_entries(self, tmp_path, monkeypatch):
        monkeypatch.setattr(agent_log, "_LOG_DIR", tmp_path)
        agent_log.log_routing("manager", "bookkeeper", reason="first_call")
        agent_log.log_routing("bookkeeper", "manager", reason="second_call")
        content = list(tmp_path.glob("Agent_Log_*.md"))[0].read_text(encoding="utf-8")
        assert "first_call" in content
        assert "second_call" in content

    def test_time_format_in_header(self, tmp_path, monkeypatch):
        import re
        monkeypatch.setattr(agent_log, "_LOG_DIR", tmp_path)
        agent_log.log_routing("manager", "archivist")
        content = list(tmp_path.glob("Agent_Log_*.md"))[0].read_text(encoding="utf-8")
        # ### [HH:MM:SS] Source → Target
        assert re.search(r"### \[\d{2}:\d{2}:\d{2}\]", content)
