"""Tests for core/agent_log.py — Lifecycle and Foldable Callouts"""
import pytest
import json

import core.agent_log as agent_log

class TestSmartFormatAndTruncate:
    def test_json_formatting(self):
        result = agent_log._smart_format_and_truncate('{"a": 1}')
        assert "```json" in result
        assert '"a": 1' in result

    def test_text_formatting(self):
        result = agent_log._smart_format_and_truncate("hello world")
        assert "```text" in result
        assert "hello world" in result

    def test_markdown_passed_through(self):
        result = agent_log._smart_format_and_truncate("### Heading\n\nSome text")
        assert "```text" not in result
        assert "### Heading" in result

    def test_truncation(self):
        long_text = "a" * 4000
        result = agent_log._smart_format_and_truncate(long_text, limit=3000)
        assert "... (truncated, original length: 4000 chars)" in result
        assert len(result) < 3200

    def test_inner_backticks(self):
        result = agent_log._smart_format_and_truncate('{"a": "```"}')
        assert "````json" in result
        assert "```" in result

class TestPrefixMultiline:
    def test_multiline_callout_prefix(self):
        text = "line1\nline2\n\nline3"
        result = agent_log._prefix_multiline(text)
        assert result == "> line1\n> line2\n> \n> line3"

class TestLogRouting:
    def test_legacy_log_routing_no_turn_id(self, tmp_path, monkeypatch):
        monkeypatch.setattr(agent_log, "_LOG_DIR", tmp_path)
        agent_log.log_routing("manager", "bookkeeper")
        content = list(tmp_path.glob("Agent_Log_*.md"))[0].read_text(encoding="utf-8")
        assert "Legacy routing" in content
        assert "Turn: `legacy`" in content
        assert "Source: manager" in content

    def test_elapsed_sec_none(self, tmp_path, monkeypatch):
        monkeypatch.setattr(agent_log, "_LOG_DIR", tmp_path)
        agent_log.log_worker_result("test-turn", "researcher", "result", elapsed_sec=None)
        content = list(tmp_path.glob("Agent_Log_*.md"))[0].read_text(encoding="utf-8")
        import re
        assert re.search(r"Worker: Researcher \[\d{2}:\d{2}:\d{2}\]", content)

    def test_elapsed_sec_provided(self, tmp_path, monkeypatch):
        monkeypatch.setattr(agent_log, "_LOG_DIR", tmp_path)
        agent_log.log_worker_result("test-turn", "researcher", "result", elapsed_sec=2.5)
        content = list(tmp_path.glob("Agent_Log_*.md"))[0].read_text(encoding="utf-8")
        assert " | 2.5s]" in content

    def test_replan_warning(self, tmp_path, monkeypatch):
        monkeypatch.setattr(agent_log, "_LOG_DIR", tmp_path)
        agent_log.log_system_action("test-turn", "Re-plan Triggered", "Error info", status="warning")
        content = list(tmp_path.glob("Agent_Log_*.md"))[0].read_text(encoding="utf-8")
        assert "> [!warning] 🔄 System: Re-plan Triggered" in content
        assert "Error info" in content
