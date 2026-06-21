"""Tests for manager_agent pure helpers — no LLM calls"""
import pytest

from agents.manager_agent import _has_researcher_frontmatter, _msg_role
from langchain_core.messages import HumanMessage, AIMessage


class TestHasResearcherFrontmatter:
    def test_valid_frontmatter(self):
        text = "---\nentity_type: article_note\ntitle: Test\n---\n\n# Body"
        assert _has_researcher_frontmatter(text)

    def test_missing_entity_type(self):
        text = "---\ntitle: Test\n---\n\n# Body"
        assert not _has_researcher_frontmatter(text)

    def test_no_frontmatter_at_all(self):
        assert not _has_researcher_frontmatter("Just plain text")

    def test_markdown_horizontal_rule_not_mistaken(self):
        # "---" as HR in middle of text must not match
        text = "Some content\n\n---\n\nMore content"
        assert not _has_researcher_frontmatter(text)

    def test_horizontal_rule_after_whitespace_not_mistaken(self):
        text = "\n\n---\n\nMore content without entity_type"
        assert not _has_researcher_frontmatter(text)

    def test_entity_type_in_first_30_lines(self):
        lines = ["---"] + [f"field_{i}: val" for i in range(28)] + ["entity_type: book_note", "---"]
        assert _has_researcher_frontmatter("\n".join(lines))

    def test_entity_type_beyond_30_lines_not_matched(self):
        lines = ["---"] + [f"field_{i}: val" for i in range(35)] + ["entity_type: article_note", "---"]
        assert not _has_researcher_frontmatter("\n".join(lines))

    def test_empty_string(self):
        assert not _has_researcher_frontmatter("")

    def test_only_dashes(self):
        assert not _has_researcher_frontmatter("---")

    def test_leading_whitespace_ignored(self):
        text = "   \n---\nentity_type: holding\n---"
        assert _has_researcher_frontmatter(text)


class TestMsgRole:
    def test_human_message_returns_human(self):
        assert _msg_role(HumanMessage(content="hi")) == "human"

    def test_ai_message_returns_assistant(self):
        assert _msg_role(AIMessage(content="response")) == "assistant"

    def test_other_object_returns_assistant(self):
        class FakeMsg:
            content = "data"
        assert _msg_role(FakeMsg()) == "assistant"
