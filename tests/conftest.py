"""Pytest fixtures shared across the suite."""
import os
import sys
from pathlib import Path

# Disable LangSmith tracing globally during tests to avoid Rate Limit Errors
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["APP_SECRET_KEY"] = "test-secret-key-123456789"
os.environ["UNVERIFIED_DRAFT_SIGNING_KEY"] = "test-secret-key-123456789-with-enough-entropy-for-tests"
os.environ["WEBUI_PASSWORD"] = "test-password"
os.environ["SESSION_SECRET_KEY"] = "test-secret-key-123456789-with-enough-entropy"

import hashlib
import tempfile

_GLOBAL_TEST_TEMP = Path(tempfile.gettempdir()) / "invest_agents_global_test_env"
_GLOBAL_TEST_TEMP.mkdir(parents=True, exist_ok=True)

os.environ["OBSIDIAN_VAULT_PATH"] = str(_GLOBAL_TEST_TEMP / "vault")
os.environ["WEBUI_STATE_DB_PATH"] = str(_GLOBAL_TEST_TEMP / "webui_state.sqlite")
os.environ["CHECKPOINT_DB_PATH"] = str(_GLOBAL_TEST_TEMP / "checkpoints.sqlite")
os.environ["NEWS_FUNNEL_STORE_PATH"] = str(_GLOBAL_TEST_TEMP / "news_funnel_state.json")

# ทำให้ทุก test resolve absolute imports ได้ (agents/, tools/, core/, schemas/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


import pytest  # noqa: E402


def _snapshot_protected_dirs():
    """บันทึก snapshot ของ memories/ และ data/ เพื่อป้องกัน test leakage"""
    snapshot = {}
    for d_name in ("memories", "data"):
        d = Path(d_name)
        if d.exists():
            for f in d.glob("**/*"):
                if f.is_file() and ".obsidian" not in f.parts:
                    try:
                        content_hash = hashlib.sha256(f.read_bytes()).hexdigest()
                        snapshot[str(f.resolve())] = content_hash
                    except Exception:
                        pass
    return snapshot


@pytest.fixture(scope="session", autouse=True)
def enforce_vault_isolation():
    """Safety check: ensure tests do not modify production vault or data files"""
    initial_snapshot = _snapshot_protected_dirs()

    yield

    final_snapshot = _snapshot_protected_dirs()

    diffs = []
    # Check modified or deleted
    for path, h in initial_snapshot.items():
        if path not in final_snapshot:
            diffs.append(f"DELETED: {path}")
        elif final_snapshot[path] != h:
            diffs.append(f"MODIFIED: {path}")

    # Check newly created
    for path in final_snapshot:
        if path not in initial_snapshot:
            diffs.append(f"CREATED: {path}")

    assert not diffs, "Production vault or data directory was modified during test run! Changes:\n" + "\n".join(diffs)

@pytest.fixture
def tmp_vault(tmp_path, monkeypatch):
    """แยก Vault per test — set OBSIDIAN_VAULT_PATH ให้ชี้ tmp_path"""
    monkeypatch.setenv("OBSIDIAN_VAULT_PATH", str(tmp_path))
    yield tmp_path

@pytest.fixture
def equity_tmp_vault(tmp_path, monkeypatch):
    """แยก Vault per test สำหรับ Equity (Monkeypatch ตัวแปร VAULT_PATH ให้ครบทุก archivist submodule)"""
    import tools.archivist.core as core
    import tools.archivist.writer as writer
    import tools.archivist.indexer as indexer
    import tools.archivist.search as search
    import tools.archivist.linter as linter
    import tools.archivist.parser as parser
    import api.routes_equity

    vpath = tmp_path.resolve()
    monkeypatch.setenv("OBSIDIAN_VAULT_PATH", str(vpath))
    for mod in [core, writer, indexer, search, linter, parser]:
        monkeypatch.setattr(mod, "VAULT_PATH", vpath, raising=False)
    monkeypatch.setattr(api.routes_equity, "VAULT_PATH", vpath, raising=False)

    yield tmp_path


@pytest.fixture(autouse=True)
def _pin_portfolio_vault_baseline(monkeypatch):
    """Safety net (defense-in-depth): ก่อน**ทุก**test ไม่ว่าจะ opt-in portfolio isolation หรือไม่
    บังคับให้ tools.portfolio.* module ที่ live อยู่ใน sys.modules ตอนนี้ (เผื่อค้างจาก test อื่น
    ในเซสชันเดียวกัน) ชี้ VAULT_PATH ไปที่ safe global test path เสมอ และให้
    api.routes_portfolio.portfolio_* (ถ้าโหลดอยู่) ชี้ตาม module เดียวกันนี้ — กัน test ที่ไม่ได้ตั้งใจ
    แตะ portfolio (เช่น test_auth.py, test_equity.py ที่ใช้ authed_client เฉยๆ) หลุดไปอ้างอิง module
    reference ค้างจาก test isolated ก่อนหน้า ซึ่งเป็น root cause ของ vault leak ที่เจอตอนรัน full suite
    (pytest instantiates autouse fixtures ก่อน fixture ที่ระบุชื่อตรงๆ ในสโคปเดียวกันเสมอ — ดังนั้น
    isolated_portfolio/isolated_mutation_portfolio/isolated_calendar_portfolio ที่ test เรียกใช้เองจะ
    รันทับ baseline นี้ทีหลังเสมอ ไม่ชนกัน)"""
    safe_vault = _GLOBAL_TEST_TEMP / "vault"

    mod_key_by_name = {
        "tools.portfolio.constants": "constants",
        "tools.portfolio.core": "core",
        "tools.portfolio.trading": "trading",
        "tools.portfolio.watchlist": "watchlist",
        "tools.portfolio.goals": "goals",
        "tools.portfolio.journal": "journal",
        "tools.portfolio.prices": "prices",
        "tools.portfolio.performance": "perf",
        "tools.portfolio.dividends": "dividends",
        "tools.portfolio.ledger_replay": "ledger_replay",
    }
    live_mods = {}
    for mod_name, key in mod_key_by_name.items():
        mod = sys.modules.get(mod_name)
        if mod is not None:
            monkeypatch.setattr(mod, "VAULT_PATH", safe_vault, raising=False)
            monkeypatch.setattr(mod, "PORTFOLIOS_DIR", safe_vault / "20_Portfolio_Management/Current_Holdings/Portfolios", raising=False)
            monkeypatch.setattr(mod, "GOALS_PATH", safe_vault / "20_Portfolio_Management/Goals/Goals.md", raising=False)
            live_mods[key] = mod

    rp = sys.modules.get("api.routes_portfolio")
    if rp is not None:
        rp_attr_by_key = {
            "core": "portfolio_core",
            "trading": "portfolio_trading",
            "watchlist": "portfolio_watchlist",
            "goals": "portfolio_goals",
            "perf": "portfolio_perf",
            "journal": "portfolio_journal",
            "prices": "portfolio_prices",
            "dividends": "portfolio_dividends",
            "ledger_replay": "portfolio_ledger_replay",
        }
        for key, rp_attr in rp_attr_by_key.items():
            mod = live_mods.get(key)
            if mod is not None and hasattr(rp, rp_attr):
                monkeypatch.setattr(rp, rp_attr, mod, raising=False)

    yield


def _reset_portfolio_modules(tmp_vault, monkeypatch):
    """Reset and reimport all tools.portfolio.* submodules, patch paths, and reattach to api.routes_portfolio."""
    from types import SimpleNamespace
    import sys

    for mod_name in list(sys.modules):
        if mod_name.startswith("tools.portfolio.") or mod_name.startswith("tools.portfolio_tools"):
            del sys.modules[mod_name]

    import tools.portfolio.constants as constants
    import tools.portfolio.core as core
    import tools.portfolio.trading as trading
    import tools.portfolio.watchlist as watchlist
    import tools.portfolio.goals as goals
    import tools.portfolio.journal as journal
    import tools.portfolio.prices as prices
    import tools.portfolio.performance as perf
    import tools.portfolio.dividends as dividends
    import tools.portfolio.ledger_replay as ledger_replay

    vpath = Path(tmp_vault).resolve()
    for mod in [constants, core, trading, watchlist, goals, journal, prices, perf, dividends, ledger_replay]:
        monkeypatch.setattr(mod, "VAULT_PATH", vpath, raising=False)
        monkeypatch.setattr(mod, "PORTFOLIOS_DIR", vpath / "20_Portfolio_Management/Current_Holdings/Portfolios", raising=False)
        monkeypatch.setattr(mod, "GOALS_PATH", vpath / "20_Portfolio_Management/Goals/Goals.md", raising=False)

    if "api.routes_portfolio" in sys.modules:
        import api.routes_portfolio as rp
        # monkeypatch.setattr (ไม่ใช่ plain assignment) เพื่อให้ revert อัตโนมัติตอน test จบ —
        # ป้องกัน rp.portfolio_* ค้างชี้ไปที่ module ของ test ก่อนหน้า (root cause ของ vault leak
        # ที่เจอตอนรัน full suite: plain assignment ไม่ revert ทำให้ test อื่นในเซสชันเดียวกัน
        # ที่ไม่ได้ opt-in isolation ไปเจอ module reference ค้างจาก test isolated ก่อนหน้า)
        monkeypatch.setattr(rp, "portfolio_core", core, raising=False)
        monkeypatch.setattr(rp, "portfolio_trading", trading, raising=False)
        monkeypatch.setattr(rp, "portfolio_watchlist", watchlist, raising=False)
        monkeypatch.setattr(rp, "portfolio_goals", goals, raising=False)
        monkeypatch.setattr(rp, "portfolio_perf", perf, raising=False)
        monkeypatch.setattr(rp, "portfolio_journal", journal, raising=False)
        monkeypatch.setattr(rp, "portfolio_prices", prices, raising=False)
        monkeypatch.setattr(rp, "portfolio_dividends", dividends, raising=False)
        monkeypatch.setattr(rp, "portfolio_ledger_replay", ledger_replay, raising=False)

    return SimpleNamespace(
        constants=constants,
        core=core,
        trading=trading,
        watchlist=watchlist,
        goals=goals,
        journal=journal,
        prices=prices,
        perf=perf,
        dividends=dividends,
        ledger_replay=ledger_replay,
    )


@pytest.fixture
def isolated_portfolio(tmp_vault, monkeypatch):
    """โหลด portfolio submodules fresh ทุกครั้ง"""
    from types import SimpleNamespace
    mods = _reset_portfolio_modules(tmp_vault, monkeypatch)
    core = mods.core
    trading = mods.trading
    watchlist = mods.watchlist
    goals = mods.goals
    journal = mods.journal
    prices = mods.prices
    perf = mods.perf
    dividends = mods.dividends
    constants = mods.constants
    ledger_replay = mods.ledger_replay

    pt = SimpleNamespace()
    pt.Holding = core.Holding
    pt.PortfolioState = core.PortfolioState
    pt._recalc_holding = core._recalc_holding
    pt._recalc_all = core._recalc_all
    pt._load_or_init = core._load_or_init
    pt._find_holding = core._find_holding
    pt._save = core._save
    pt.compute_allocation_breakdown = core.compute_allocation_breakdown
    pt.get_portfolio_state = core.get_portfolio_state

    pt._compute_total_cost = trading._compute_total_cost
    pt._record_income_locked = trading._record_income_locked
    pt._execute_trade_locked = trading._execute_trade_locked
    pt._manage_cash_flow_locked = trading._manage_cash_flow_locked
    pt._update_fx_rate_locked = trading._update_fx_rate_locked
    pt._edit_holding_locked = trading._edit_holding_locked
    pt.record_income = trading.record_income
    pt.structured_execute_trade = trading.structured_execute_trade
    pt.sync_dividends_from_history = dividends.sync_dividends_from_history
    pt.fetch_fx_rate = prices.fetch_fx_rate
    pt.trading = trading
    pt.prices = prices
    pt.dividends = dividends
    pt.core = core
    pt.ledger_replay = ledger_replay
    pt.edit_transaction = ledger_replay.edit_transaction
    pt.delete_transaction = ledger_replay.delete_transaction
    pt.execute_trade = trading.execute_trade
    pt.manage_cash_flow = trading.manage_cash_flow
    pt.update_fx_rate = trading.update_fx_rate
    pt.edit_holding = trading.edit_holding
    pt.batch_import_holdings = trading.batch_import_holdings

    pt.sync_market_prices = prices.sync_market_prices
    pt._refresh_prices = prices._refresh_prices

    pt.add_to_watchlist = watchlist.add_to_watchlist
    pt.remove_from_watchlist = watchlist.remove_from_watchlist
    pt.read_watchlist = watchlist.read_watchlist

    pt.record_performance_snapshot = perf.record_performance_snapshot
    pt.read_performance_history = perf.read_performance_history

    pt.append_trading_journal = journal.append_trading_journal
    pt.read_trading_journal = journal.read_trading_journal

    pt.set_goal = goals.set_goal
    pt.remove_goal = goals.remove_goal
    pt.get_goals_progress = goals.get_goals_progress
    pt.GOALS_PATH = goals.GOALS_PATH
    pt.GOALS_ITEMS_DIR = goals.GOALS_ITEMS_DIR

    from tools.portfolio.models import AllocationTarget
    pt.AllocationTarget = AllocationTarget
    pt.get_structured_portfolio_state = core.get_structured_portfolio_state
    pt.get_structured_bucket_allocation = core.get_structured_bucket_allocation
    pt._recalc_fundamentals_derived = core._recalc_fundamentals_derived
    pt._fetch_fundamentals = prices._fetch_fundamentals
    pt.get_structured_watchlist = watchlist.get_structured_watchlist
    pt.get_structured_performance_history = perf.get_structured_performance_history
    pt.get_structured_journal = journal.get_structured_journal
    pt.get_structured_goals = goals.get_structured_goals

    return pt


@pytest.fixture
def isolated_archivist(tmp_vault, monkeypatch):
    import importlib
    import unittest.mock
    from types import SimpleNamespace
    
    # Use existing modules


    import tools.archivist.core as core
    import tools.archivist.writer as writer
    import tools.archivist.indexer as indexer
    import tools.archivist.search as search
    import tools.archivist.linter as linter
    import tools.archivist.parser as parser
    
    for mod in [core, writer, indexer, search, linter, parser]:
        monkeypatch.setattr(mod, "VAULT_PATH", tmp_vault.resolve())

    monkeypatch.setattr(search, "CHROMA_PATH", tmp_vault.resolve() / ".chroma_index")
    monkeypatch.setattr(search, "_CHROMA_MTIME_FILE", tmp_vault.resolve() / ".chroma_mtime")
    
    at = SimpleNamespace()
    at.init_vault_structure = core.init_vault_structure
    at.read_file = core.read_file
    at.save_memory = writer.save_memory
    at.write_raw_markdown = writer.write_raw_markdown
    at.update_master_index = indexer.update_master_index
    at.search_all_memories = search.search_all_memories
    at.search_graph_context = search.search_graph_context
    at.lint_structural_health = linter.lint_structural_health
    at.lint_semantic_conflict = linter.lint_semantic_conflict
    
    monkeypatch.setattr(search, "get_embeddings", unittest.mock.MagicMock())
    
    return at


@pytest.fixture(autouse=True)
def _no_real_llm_keys(monkeypatch):
    """ป้องกันการเรียก network / API จริงระหว่างรัน unit tests ทั้ง suite

    ครอบคลุม DISCORD_WEBHOOK_URL ด้วย — ไม่ใช่แค่ LLM keys: api/main.py เรียก load_dotenv()
    ตอน import โดยไม่มีเงื่อนไข และหลาย test fixture (เช่น tests/api/conftest.py::client) import
    api.main เข้ามา ทำให้ค่าจริงจาก .env รั่วเข้า os.environ ได้ตลอด process หลังจากนั้น ถ้าไม่ clear
    ตรงนี้ test ของ News Funnel synthesis / article ingest ที่ไม่ได้ mock Discord เอง (เพราะไม่ได้
    ตั้งใจทดสอบ Discord) จะยิง webhook จริงโดยไม่ตั้งใจ — เจอเป็น incident จริงจาก live test run
    """
    for k in (
        "GOOGLE_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY",
        "DISCORD_WEBHOOK_URL", "DISCORD_TAG_ID_ULTRA", "DISCORD_TAG_ID_HIGH", "DISCORD_TAG_ID_WARNING",
        "NOTEBOOKLM_AUTH_DIR",
    ):
        monkeypatch.delenv(k, raising=False)
