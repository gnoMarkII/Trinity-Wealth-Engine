"""Unit tests for Trinity MCP Tools & DI Providers (17 Tools across 8 Domains)."""
from __future__ import annotations

import json
from pathlib import Path
import pytest

from tools.hermes_bridge.trinity_mcp_server import (
    get_wrapped_trinity_tools,
    get_latest_macro_summary,
    get_latest_daily_digest,
    get_upcoming_economic_events,
    get_equity_report_summary,
    get_equity_promises_and_reviews,
    get_portfolio_overview,
    get_portfolio_look_through,
    get_portfolio_advisory_and_alerts,
    get_portfolio_opportunities,
    get_investor_essence_and_buckets,
    query_terminal_v2_screener,
    search_knowledge_vault,
    save_insight_to_vault,
    trigger_deep_research_job,
    run_sandbox_calculation,
    save_sandbox_skill,
    update_advisor_memory,
    set_macro_provider,
    set_portfolio_provider,
    set_advisory_provider,
    set_opportunity_provider,
    set_calendar_provider,
    set_digest_provider,
    set_equity_report_provider,
    set_equity_promises_provider,
    set_look_through_provider,
    set_essence_buckets_provider,
    set_screener_provider,
    set_research_job_dispatcher,
)
from api.dependencies import get_manager_chat_service, get_hermes_sync_service


def test_wrapped_trinity_tools_count_and_names():
    """Verify that exactly 17 tools are registered and wrapped."""
    tools = get_wrapped_trinity_tools()
    assert len(tools) == 17
    tool_names = {t.name for t in tools}
    expected_names = {
        "get_latest_macro_summary",
        "get_latest_daily_digest",
        "get_upcoming_economic_events",
        "get_equity_report_summary",
        "get_equity_promises_and_reviews",
        "get_portfolio_overview",
        "get_portfolio_look_through",
        "get_portfolio_advisory_and_alerts",
        "get_portfolio_opportunities",
        "get_investor_essence_and_buckets",
        "query_terminal_v2_screener",
        "search_knowledge_vault",
        "save_insight_to_vault",
        "trigger_deep_research_job",
        "run_sandbox_calculation",
        "save_sandbox_skill",
        "update_advisor_memory",
    }
    assert tool_names == expected_names


def test_sandbox_tools_invocation(tmp_path, monkeypatch):
    """Test the three sandbox tools in isolation using tmp_path."""
    import tools.hermes_bridge.trinity_mcp_server as server_mod
    from tools.hermes_bridge.sandbox_adapter import SandboxAdapter

    temp_sandbox = SandboxAdapter(sandbox_root=tmp_path)
    monkeypatch.setattr(server_mod, "_sandbox", temp_sandbox)

    # 1. Calculation
    calc_res = run_sandbox_calculation.invoke({
        "filename": "test_calc.py",
        "python_code": "import math\nprint(math.sqrt(16))",
    })
    assert "4.0" in calc_res

    # 2. Memory update
    mem_res = update_advisor_memory.invoke({
        "target": "USER",
        "markdown_content": "# Isolated Test Investor Profile\nRisk tolerance: High",
    })
    assert "เรียบร้อย" in mem_res
    assert (tmp_path / "memory" / "USER.md").is_file()

    # 3. Save skill
    skill_res = save_sandbox_skill.invoke({
        "skill_name": "isolated_test_skill",
        "skill_markdown": "# Isolated Test Skill\nDescription of skill",
    })
    assert "เรียบร้อย" in skill_res
    assert (tmp_path / "skills" / "_quarantine" / "isolated_test_skill" / "SKILL.md").is_file()


def test_vault_tools_invocation():
    """Test search and save tools with LangChain StructuredTool contract."""
    # Search should return string (even if empty or no results) without TypeError
    search_res = search_knowledge_vault.invoke({"query": "investment"})
    assert isinstance(search_res, str)

    # Save note should return string without TypeError
    save_res = save_insight_to_vault.invoke({
        "title": "Unit Test Insight Note",
        "content": "Key takeaways from portfolio review",
    })
    assert isinstance(save_res, str)


def test_job_dispatch_tool():
    """Test research dispatcher tool."""
    set_research_job_dispatcher(lambda ticker, rtype: f"mock_job_{ticker}")
    res = trigger_deep_research_job.invoke({"ticker": "TSLA", "research_type": "equity_intel"})
    assert "mock_job_TSLA" in res


def test_mock_domain_providers_wiring():
    """Test that all domain providers return expected JSON output."""
    set_macro_provider(lambda: json.dumps({"regime": "Expansion"}))
    set_digest_provider(lambda: json.dumps({"title": "Morning Digest"}))
    set_calendar_provider(lambda: json.dumps({"events": ["FOMC"]}))
    set_equity_report_provider(lambda t: json.dumps({"ticker": t, "score": 85}))
    set_equity_promises_provider(lambda t: json.dumps({"ticker": t, "promises": 5}))
    set_portfolio_provider(lambda: json.dumps({"nav": 1000000}))
    set_look_through_provider(lambda: json.dumps({"look_through": True}))
    set_advisory_provider(lambda: json.dumps({"status": "healthy"}))
    set_opportunity_provider(lambda: json.dumps({"candidates": []}))
    set_essence_buckets_provider(lambda: json.dumps({"pillars": 5}))
    set_screener_provider(lambda tickers: json.dumps([{"ticker": t} for t in tickers]))

    assert "Expansion" in get_latest_macro_summary.invoke({})
    assert "Morning Digest" in get_latest_daily_digest.invoke({})
    assert "FOMC" in get_upcoming_economic_events.invoke({})
    assert "NVDA" in get_equity_report_summary.invoke({"ticker": "NVDA"})
    assert "NVDA" in get_equity_promises_and_reviews.invoke({"ticker": "NVDA"})
    assert "1000000" in get_portfolio_overview.invoke({})
    assert "look_through" in get_portfolio_look_through.invoke({})
    assert "healthy" in get_portfolio_advisory_and_alerts.invoke({})
    assert "candidates" in get_portfolio_opportunities.invoke({})
    assert "pillars" in get_investor_essence_and_buckets.invoke({})
    assert "AAPL" in query_terminal_v2_screener.invoke({"tickers_comma_separated": "AAPL, MSFT"})


def test_get_manager_chat_service_wiring():
    """Verify that get_manager_chat_service composes successfully and wires real providers."""
    svc = get_manager_chat_service()
    assert svc is not None

    # Test invoking the wired providers through the tools
    macro_out = get_latest_macro_summary.invoke({})
    assert isinstance(macro_out, str)

    digest_out = get_latest_daily_digest.invoke({})
    assert isinstance(digest_out, str)

    cal_out = get_upcoming_economic_events.invoke({})
    assert isinstance(cal_out, str)

    port_out = get_portfolio_overview.invoke({})
    assert isinstance(port_out, str)

    adv_out = get_portfolio_advisory_and_alerts.invoke({})
    assert isinstance(adv_out, str)

    opp_out = get_portfolio_opportunities.invoke({})
    assert isinstance(opp_out, str)

    eq_out = get_equity_report_summary.invoke({"ticker": "NVDA"})
    assert isinstance(eq_out, str)

    prom_out = get_equity_promises_and_reviews.invoke({"ticker": "NVDA"})
    assert isinstance(prom_out, str)

    lt_out = get_portfolio_look_through.invoke({})
    assert isinstance(lt_out, str)

    ess_out = get_investor_essence_and_buckets.invoke({})
    assert isinstance(ess_out, str)

    scr_out = query_terminal_v2_screener.invoke({"tickers_comma_separated": "NVDA, MSFT"})
    assert isinstance(scr_out, str)

    dispatch_out = trigger_deep_research_job.invoke({"ticker": "INTC", "research_type": "equity_intel"})
    assert isinstance(dispatch_out, str)
    assert "INTC" in dispatch_out
    assert "Job ID:" in dispatch_out

    vault_search_out = search_knowledge_vault.invoke({"query": "crypto"})
    assert isinstance(vault_search_out, str)

    vault_save_out = save_insight_to_vault.invoke({"title": "Test Wired Note", "content": "Wired note test"})
    assert isinstance(vault_save_out, str)


def test_create_mcp_server_fastmcp():
    """Verify FastMCP server instance creates and registers all 17 tools."""
    import asyncio
    from tools.hermes_bridge.trinity_mcp_server import create_mcp_server

    server = create_mcp_server()
    assert server is not None
    tools = asyncio.run(server.list_tools())
    assert len(tools) == 17


def test_hermes_sync_service_singleton():
    """Verify HermesSyncService provider returns a singleton."""
    s1 = get_hermes_sync_service()
    s2 = get_hermes_sync_service()
    assert s1 is s2
    status = s1.get_sync_status()
    assert hasattr(status, "is_installed")



def test_parse_skill_frontmatter_and_snapshot(tmp_path):
    """Verify that YAML frontmatter is parsed cleanly and snapshot does not expose '---'."""
    from tools.hermes_bridge.sandbox_adapter import SandboxAdapter, parse_skill_frontmatter

    skill_content = """---
name: my_test_skill
description: Comprehensive financial analysis skill
---
# Instructions
Do something smart.
"""
    meta, body = parse_skill_frontmatter(skill_content)
    assert meta["name"] == "my_test_skill"
    assert meta["description"] == "Comprehensive financial analysis skill"
    assert "Do something smart" in body

    # Test snapshot description
    adapter = SandboxAdapter(sandbox_root=tmp_path)
    adapter.save_skill("my_test_skill", skill_content)
    snap = adapter.get_snapshot()
    assert len(snap.quarantined_skills) == 1
    assert snap.quarantined_skills[0].description == "Comprehensive financial analysis skill"
    assert snap.quarantined_skills[0].description != "---"


def test_promote_quarantined_skill_blocks_sql_mutation(tmp_path):
    """Verify that skills with SQL mutation commands or transactions are blocked from promotion."""
    from tools.hermes_bridge.sandbox_adapter import SandboxAdapter

    adapter = SandboxAdapter(sandbox_root=tmp_path)

    # 1. Malicious skill with DROP TABLE
    bad_skill = """---
name: evil_skill
description: Tries to drop tables
---
cursor.execute("DROP TABLE advisor_messages")
"""
    adapter.save_skill("evil_skill", bad_skill)
    res_bad = adapter.promote_quarantined_skill("evil_skill")
    assert res_bad is False

    # 2. Malicious skill with direct commit
    bad_skill_commit = """---
name: commit_skill
description: Tries to commit
---
conn.commit()
"""
    adapter.save_skill("commit_skill", bad_skill_commit)
    res_commit = adapter.promote_quarantined_skill("commit_skill")
    assert res_commit is False

    # 3. Safe skill
    safe_skill = """---
name: good_skill
description: Safe pure calculation skill
---
# Good Skill
Compute DCF with Decimal.
"""
    adapter.save_skill("good_skill", safe_skill)
    res_good = adapter.promote_quarantined_skill("good_skill")
    assert res_good is True
    snap = adapter.get_snapshot()
    assert any(s.name == "good_skill" and s.description == "Safe pure calculation skill" for s in snap.active_skills)
