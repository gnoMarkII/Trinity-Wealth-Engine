"""Pytest fixtures shared across the suite."""
import os
import sys
from pathlib import Path

# ทำให้ทุก test resolve absolute imports ได้ (agents/, tools/, core/, schemas/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


import pytest  # noqa: E402


@pytest.fixture
def tmp_vault(tmp_path, monkeypatch):
    """แยก Vault per test — set OBSIDIAN_VAULT_PATH ให้ชี้ tmp_path"""
    monkeypatch.setenv("OBSIDIAN_VAULT_PATH", str(tmp_path))
    yield tmp_path


@pytest.fixture
def isolated_portfolio(tmp_vault, monkeypatch):
    """โหลด portfolio_tools fresh ทุกครั้ง ด้วย Vault path ใหม่ — module ใช้ env ตอน import"""
    import importlib

    # remove cached modules ที่ pin path ไว้ตอน import แรก
    for mod_name in list(sys.modules):
        if mod_name.startswith("tools.portfolio_tools"):
            del sys.modules[mod_name]

    portfolio_tools = importlib.import_module("tools.portfolio_tools")
    return portfolio_tools
