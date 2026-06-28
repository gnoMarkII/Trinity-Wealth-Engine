"""Test portfolio_tools: recalc, execute_trade, record_income — ใช้ Vault แยกผ่าน tmp_path"""
import json

import pytest


class TestReadWatchlist:
    def test_empty_watchlist(self, isolated_portfolio):
        pt = isolated_portfolio
        result = pt.read_watchlist.invoke({})
        data = json.loads(result)
        assert data["n_items"] == 0
        assert data["items"] == []

    def test_returns_added_items(self, isolated_portfolio):
        pt = isolated_portfolio
        pt.add_to_watchlist.invoke({
            "symbol": "NVDA", "asset_type": "Stock",
            "target_price": 100.0, "notes": "wait for dip"
        })
        pt.add_to_watchlist.invoke({
            "symbol": "PTT", "asset_type": "Stock", "target_price": 28.0
        })
        result = pt.read_watchlist.invoke({})
        data = json.loads(result)
        assert data["n_items"] == 2
        symbols = {it["symbol"] for it in data["items"]}
        assert symbols == {"NVDA", "PTT"}
        nvda = next(it for it in data["items"] if it["symbol"] == "NVDA")
        assert nvda["target_price"] == 100.0
        assert nvda["notes"] == "wait for dip"



import pytest
import os
import tools.portfolio.watchlist as watchlist

class TestAddToWatchlist:
    def test_add_empty_symbol_raises(self, isolated_portfolio):
        pt = isolated_portfolio
        result = pt.add_to_watchlist.func(symbol="", asset_type="Stock")

        assert isinstance(result, str) and result.startswith("Error:")
        assert "symbol ต้องไม่ว่าง" in result
    def test_add_negative_target_raises(self, isolated_portfolio):
        pt = isolated_portfolio
        result = pt.add_to_watchlist.func(symbol="PTT", asset_type="Stock", target_price=-10.0)

        assert isinstance(result, str) and result.startswith("Error:")
        assert "target_price ต้องมากกว่า 0" in result
    def test_add_existing_updates(self, isolated_portfolio):
        pt = isolated_portfolio
        pt.add_to_watchlist.func(symbol="PTT", asset_type="Stock", target_price=30.0)
        result = pt.add_to_watchlist.func(symbol="PTT", asset_type="Stock", target_price=40.0)
        assert "[WATCH UPD]" in result
        
        data = pt.read_watchlist.func()
        import json
        parsed = json.loads(data)
        assert len(parsed["items"]) == 1
        assert parsed["items"][0]["target_price"] == 40.0

    def test_add_lock_timeout(self, isolated_portfolio, monkeypatch):
        pt = isolated_portfolio
        def mock_lock(*args, **kwargs):
            from filelock import Timeout
            raise Timeout("mock")
        monkeypatch.setattr("tools.portfolio.watchlist._watchlist_lock.acquire", mock_lock)
        
        result = pt.add_to_watchlist.func(symbol="PTT", asset_type="Stock")

        assert isinstance(result, str) and result.startswith("Error:")
        assert "watchlist lock" in result
class TestRemoveFromWatchlist:
    def test_remove_success(self, isolated_portfolio):
        pt = isolated_portfolio
        pt.add_to_watchlist.func(symbol="PTT", asset_type="Stock")
        result = pt.remove_from_watchlist.func(symbol="PTT")
        assert "[WATCH DEL]" in result
        assert "remaining: 0" in result

    def test_remove_empty_symbol_raises(self, isolated_portfolio):
        pt = isolated_portfolio
        result = pt.remove_from_watchlist.func(symbol="")

        assert isinstance(result, str) and result.startswith("Error:")
        assert "symbol ต้องไม่ว่าง" in result
    def test_remove_not_found_raises(self, isolated_portfolio):
        pt = isolated_portfolio
        result = pt.remove_from_watchlist.func(symbol="PTT")

        assert isinstance(result, str) and result.startswith("Error:")
        assert "ไม่พบ PTT ใน Watchlist" in result
    def test_remove_lock_timeout(self, isolated_portfolio, monkeypatch):
        pt = isolated_portfolio
        def mock_lock(*args, **kwargs):
            from filelock import Timeout
            raise Timeout("mock")
        monkeypatch.setattr("tools.portfolio.watchlist._watchlist_lock.acquire", mock_lock)
        
        result = pt.remove_from_watchlist.func(symbol="PTT")

        assert isinstance(result, str) and result.startswith("Error:")
        assert "watchlist lock" in result
class TestReadWatchlistExceptions:
    def test_read_lock_timeout(self, isolated_portfolio, monkeypatch):
        pt = isolated_portfolio
        def mock_lock(*args, **kwargs):
            from filelock import Timeout
            raise Timeout("mock")
        monkeypatch.setattr("tools.portfolio.watchlist._watchlist_lock.acquire", mock_lock)
        
        result = pt.read_watchlist.func()
        import json
        data = json.loads(result)
        assert "error" in data
        assert "watchlist lock" in data["error"]

class TestLoadOrInitWatchlist:
    def test_load_no_metadata(self, isolated_portfolio):
        pt = isolated_portfolio
        # Use the patched path from the loaded module
        import tools.portfolio.watchlist as wl
        # Create a blank markdown file
        wl.WATCHLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(wl.WATCHLIST_PATH, "w", encoding="utf-8") as f:
            f.write("Hello World")
        
        # It should log warning and init new state
        post, state = wl._load_or_init_watchlist()
        assert len(state.items) == 0

class TestAtomicWriteException:
    def test_atomic_write_exception(self, isolated_portfolio, monkeypatch):
        pt = isolated_portfolio
        import os
        def mock_replace(*args, **kwargs):
            raise OSError("Mock disk error")
        monkeypatch.setattr(os, "replace", mock_replace)
        
        with pytest.raises(OSError, match="Mock disk error"):
            pt.add_to_watchlist.func(symbol="PTT", asset_type="Stock")

class TestSyncWatchlistSidecars:
    def test_sidecar_deletion(self, isolated_portfolio):
        pt = isolated_portfolio
        pt.add_to_watchlist.func(symbol="PTT", asset_type="Stock")
        pt.add_to_watchlist.func(symbol="AAPL", asset_type="Stock")
        
        import tools.portfolio.watchlist as wl
        sidecars = list(wl.WATCHLIST_ITEMS_DIR.glob("*.md"))
        assert len(sidecars) == 2
        
        pt.remove_from_watchlist.func(symbol="PTT")
        sidecars_after = list(wl.WATCHLIST_ITEMS_DIR.glob("*.md"))
        assert len(sidecars_after) == 1
        assert sidecars_after[0].stem == "AAPL"
