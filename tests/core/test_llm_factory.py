import pytest
import os
from unittest.mock import patch, MagicMock

from core.llm_factory import (
    detect_provider,
    _build_primary,
    get_llm,
    _fetch_google_models,
    _fetch_anthropic_models,
    _fetch_openrouter_models,
    list_available_models,
)

def test_detect_provider(monkeypatch):
    assert detect_provider("claude-sonnet") == "anthropic"
    assert detect_provider("gemini-pro") == "google"
    assert detect_provider("models/gemini-pro") == "google"
    assert detect_provider("openai/gpt-4") == "openrouter"
    assert detect_provider("unknown-model") == "google"
    
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    assert detect_provider("gemini-pro") == "anthropic"

def test_build_primary(monkeypatch):
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_anthropic import ChatAnthropic
    from langchain_openai import ChatOpenAI
    
    monkeypatch.setenv("GOOGLE_API_KEY", "fake")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake")
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake")
    
    with patch("core.llm_factory.os.getenv", return_value="fake_key"):
        assert isinstance(_build_primary("google", "gemini", 0.0), ChatGoogleGenerativeAI)
        assert isinstance(_build_primary("anthropic", "claude", 0.0), ChatAnthropic)
        assert isinstance(_build_primary("openrouter", "openai/gpt", 0.0), ChatOpenAI)
    
    with pytest.raises(ValueError, match="Unknown provider"):
        _build_primary("invalid", "model", 0.0)

def test_get_llm():
    from langchain_core.runnables import RunnableWithFallbacks
    
    with patch("core.llm_factory._build_primary") as mock_build:
        mock_primary = MagicMock()
        mock_fallback_chain = MagicMock()
        mock_primary.with_fallbacks.return_value = mock_fallback_chain
        
        mock_build.side_effect = [mock_primary, MagicMock()] # primary, then fallback
        
        # Test no fallback
        res1 = get_llm("google", "gemini")
        assert res1 == mock_primary
        assert mock_build.call_count == 1
        
        mock_build.reset_mock()
        mock_build.side_effect = [mock_primary, MagicMock()]
        
        # Test with fallback
        res2 = get_llm("google", "gemini", use_fallback=True)
        assert res2 == mock_fallback_chain
        assert mock_build.call_count == 2
        
        mock_build.reset_mock()
        mock_build.side_effect = [mock_primary, MagicMock()]
        
        # Test with fallback but model is already fallback model
        from core.llm_factory import FALLBACK_MODEL
        res3 = get_llm("google", FALLBACK_MODEL, use_fallback=True)
        assert res3 == mock_primary
        assert mock_build.call_count == 1

@patch("core.llm_factory.genai.Client")
def test_fetch_google_models(mock_client_cls):
    mock_client = MagicMock()
    mock_model = MagicMock()
    mock_model.name = "models/gemini-pro"
    mock_model_bad = MagicMock()
    mock_model_bad.name = "other-model"
    
    mock_client.models.list.return_value = [mock_model, mock_model_bad]
    mock_client_cls.return_value = mock_client
    
    assert _fetch_google_models() == ["models/gemini-pro"]
    
    # Test failure
    mock_client_cls.side_effect = Exception("API error")
    assert _fetch_google_models() == []

@patch("core.llm_factory.anthropic.Anthropic")
def test_fetch_anthropic_models(mock_anthropic_cls):
    mock_client = MagicMock()
    mock_model = MagicMock()
    mock_model.id = "claude-sonnet"
    mock_client.models.list().data = [mock_model]
    mock_anthropic_cls.return_value = mock_client
    
    assert _fetch_anthropic_models() == ["claude-sonnet"]
    
    # Test failure
    mock_anthropic_cls.side_effect = Exception("API error")
    assert _fetch_anthropic_models() == []

@patch("httpx.get")
def test_fetch_openrouter_models(mock_get):
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"data": [{"id": "openai/gpt"}]}
    mock_get.return_value = mock_resp
    
    assert _fetch_openrouter_models() == ["openai/gpt"]
    
    # Test failure
    mock_get.side_effect = Exception("Network error")
    assert _fetch_openrouter_models() == []

@patch("core.llm_factory._fetch_google_models", return_value=["g1"])
@patch("core.llm_factory._fetch_anthropic_models", return_value=["a1"])
@patch("core.llm_factory._fetch_openrouter_models", return_value=["o1"])
def test_list_available_models(mock_or, mock_ant, mock_goo):
    assert list_available_models("google") == ["g1"]
    assert list_available_models("anthropic") == ["a1"]
    assert list_available_models("openrouter") == ["o1"]
    
    all_models = list_available_models(None)
    assert all_models["google"] == ["g1"]
    assert all_models["anthropic"] == ["a1"]
    assert all_models["openrouter"] == ["o1"]
    
    with pytest.raises(ValueError, match="Unknown provider"):
        list_available_models("invalid")
