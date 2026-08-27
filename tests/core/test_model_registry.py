"""Tests for core/model_registry.py — centralized LLM model config."""
from core.model_registry import REGISTRY, get_model_name, get_registry_summary


def test_registry_has_exactly_15_slots():
    assert len(REGISTRY) == 15


def test_registry_has_10_agent_and_5_tool_slots():
    agent_slots = [s for s in REGISTRY.values() if s.layer == "agent"]
    tool_slots = [s for s in REGISTRY.values() if s.layer == "tool"]
    assert len(agent_slots) == 10
    assert len(tool_slots) == 5


def test_all_expected_slot_keys_present():
    expected = {
        "manager", "router", "archivist", "bookkeeper",
        "macro_quant", "economist", "allocator",
        "equity_quant", "equity_narrative", "equity_synthesizer",
        "extractor", "youtube_pitch", "news_triage", "thai_title_translation",
        "earnings_call_summarizer",
    }
    assert set(REGISTRY.keys()) == expected


def test_get_model_name_returns_default_when_env_unset(monkeypatch):
    monkeypatch.delenv("EXTRACTOR_MODEL", raising=False)
    assert get_model_name("extractor") == "gemini-3.1-flash-lite-preview"


def test_get_model_name_returns_override_when_env_set(monkeypatch):
    monkeypatch.setenv("EXTRACTOR_MODEL", "claude-opus-5")
    assert get_model_name("extractor") == "claude-opus-5"


def test_get_model_name_unknown_slot_raises_keyerror():
    import pytest
    with pytest.raises(KeyError):
        get_model_name("does_not_exist")


def test_get_registry_summary_returns_15_entries():
    summary = get_registry_summary()
    assert len(summary) == 15


def test_get_registry_summary_marks_overridden_slot(monkeypatch):
    monkeypatch.setenv("NEWS_FUNNEL_TRIAGE_MODEL", "custom-model")
    monkeypatch.delenv("NEWS_FUNNEL_SYNTHESIS_MODEL", raising=False)
    summary = {row["slot"]: row for row in get_registry_summary()}

    assert summary["news_triage"]["is_overridden"] is True
    assert summary["news_triage"]["resolved_model"] == "custom-model"
    assert summary["thai_title_translation"]["is_overridden"] is False
    assert summary["thai_title_translation"]["resolved_model"] == "gemini-3.1-flash-lite-preview"


def test_registry_summary_entries_have_all_expected_fields():
    row = get_registry_summary()[0]
    assert set(row.keys()) == {"slot", "env_var", "resolved_model", "default", "is_overridden", "purpose", "layer"}
