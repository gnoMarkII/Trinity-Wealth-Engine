"""Test portfolio_tools: recalc, execute_trade, record_income — ใช้ Vault แยกผ่าน tmp_path"""
import json

import pytest


class TestGoals:
    def test_set_goal_new(self, isolated_portfolio):
        pt = isolated_portfolio
        result = pt.set_goal.invoke({
            "name": "พอร์ต 10 ล้าน",
            "goal_type": "nav_target",
            "target_amount_thb": 10_000_000.0,
            "deadline": "2031-12-31",
        })
        assert "[GOAL SET]" in result
        assert "พอร์ต 10 ล้าน" in result
        assert "10,000,000.00 THB" in result
        assert "2031-12-31" in result
        assert "total: 1" in result

    def test_set_goal_update_existing(self, isolated_portfolio):
        pt = isolated_portfolio
        pt.set_goal.invoke({
            "name": "เป้า", "goal_type": "nav_target", "target_amount_thb": 1_000_000.0,
        })
        result = pt.set_goal.invoke({
            "name": "เป้า", "goal_type": "nav_target", "target_amount_thb": 2_000_000.0,
        })
        assert "[GOAL UPD]" in result
        assert "2,000,000.00 THB" in result
        assert "total: 1" in result  # ไม่ซ้ำ ยังเป็น 1

    def test_set_goal_multiple(self, isolated_portfolio):
        pt = isolated_portfolio
        pt.set_goal.invoke({"name": "A", "goal_type": "nav_target", "target_amount_thb": 1e6})
        pt.set_goal.invoke({"name": "B", "goal_type": "cash_target", "target_amount_thb": 5e5})
        result = pt.set_goal.invoke({
            "name": "C", "goal_type": "passive_income_ytd", "target_amount_thb": 3e5,
        })
        assert "total: 3" in result

    def test_set_goal_invalid_name_raises(self, isolated_portfolio):
        pt = isolated_portfolio
        result = pt.set_goal.invoke({
            "name": "   ", "goal_type": "nav_target", "target_amount_thb": 1e6,
        })

        assert isinstance(result, str) and result.startswith("Error:")
        assert "name" in result
    def test_set_goal_invalid_target_raises(self, isolated_portfolio):
        pt = isolated_portfolio
        result = pt.set_goal.invoke({
            "name": "X", "goal_type": "nav_target", "target_amount_thb": 0.0,
        })

        assert isinstance(result, str) and result.startswith("Error:")
    def test_set_goal_invalid_deadline_raises(self, isolated_portfolio):
        pt = isolated_portfolio
        result = pt.set_goal.invoke({
            "name": "X", "goal_type": "nav_target",
            "target_amount_thb": 1e6, "deadline": "31-12-2031",
        })

        assert isinstance(result, str) and result.startswith("Error:")
        assert "deadline" in result
    def test_set_goal_years_from_now(self, isolated_portfolio):
        pt = isolated_portfolio
        from datetime import datetime
        result = pt.set_goal.invoke({
            "name": "Future", "goal_type": "nav_target",
            "target_amount_thb": 2e6, "years_from_now": 5,
        })
        expected_year = datetime.now().year + 5
        assert f"{expected_year}-12-31" in result

    def test_set_goal_years_from_now_invalid_raises(self, isolated_portfolio):
        pt = isolated_portfolio
        result = pt.set_goal.invoke({
            "name": "Bad", "goal_type": "nav_target",
            "target_amount_thb": 1e6, "years_from_now": 0,
        })

        assert isinstance(result, str) and result.startswith("Error:")
        assert "years_from_now" in result
    def test_set_goal_years_from_now_and_deadline_raises(self, isolated_portfolio):
        pt = isolated_portfolio
        result = pt.set_goal.invoke({
            "name": "Conflict", "goal_type": "nav_target",
            "target_amount_thb": 1e6, "deadline": "2030-12-31", "years_from_now": 5,
        })

        assert isinstance(result, str) and result.startswith("Error:")
    def test_set_goal_preserves_created_date_on_update(self, isolated_portfolio):
        pt = isolated_portfolio
        pt.set_goal.invoke({"name": "G", "goal_type": "nav_target", "target_amount_thb": 1e6})

        # อ่าน created_date จาก Goals.md
        import frontmatter
        goals_path = pt.GOALS_PATH
        post = frontmatter.load(goals_path)
        original_date = post.metadata["goals"][0]["created_date"]

        pt.set_goal.invoke({"name": "G", "goal_type": "nav_target", "target_amount_thb": 2e6})
        post2 = frontmatter.load(goals_path)
        assert post2.metadata["goals"][0]["created_date"] == original_date

    def test_remove_goal(self, isolated_portfolio):
        pt = isolated_portfolio
        pt.set_goal.invoke({"name": "Del", "goal_type": "nav_target", "target_amount_thb": 1e6})
        result = pt.remove_goal.invoke({"name": "Del"})
        assert "[GOAL DEL]" in result
        assert "remaining: 0" in result

    def test_remove_goal_not_found_raises(self, isolated_portfolio):
        pt = isolated_portfolio
        result = pt.remove_goal.invoke({"name": "ไม่มีอยู่จริง"})

        assert isinstance(result, str) and result.startswith("Error:")
        assert "ไม่พบ" in result
    def test_get_goals_progress_nav_target(self, isolated_portfolio):
        pt = isolated_portfolio
        # เติมเงินสด → NAV = 200,000
        pt._manage_cash_flow_locked(200_000.0, "deposit", "THB")
        pt.set_goal.invoke({
            "name": "NAV เป้า", "goal_type": "nav_target", "target_amount_thb": 1_000_000.0,
        })
        result = pt.get_goals_progress.invoke({})
        data = json.loads(result)
        assert data["n_goals"] == 1
        g = data["goals"][0]
        assert g["goal_type"] == "nav_target"
        assert g["current_amount_thb"] == pytest.approx(200_000.0, rel=1e-3)
        assert g["progress_pct"] == pytest.approx(20.0, rel=1e-2)

    def test_get_goals_progress_cash_target(self, isolated_portfolio):
        pt = isolated_portfolio
        pt._manage_cash_flow_locked(300_000.0, "deposit", "THB")
        pt.set_goal.invoke({
            "name": "เงินฉุกเฉิน", "goal_type": "cash_target", "target_amount_thb": 400_000.0,
        })
        result = pt.get_goals_progress.invoke({})
        data = json.loads(result)
        g = data["goals"][0]
        assert g["goal_type"] == "cash_target"
        assert g["current_amount_thb"] == pytest.approx(300_000.0, rel=1e-3)
        assert g["progress_pct"] == pytest.approx(75.0, rel=1e-2)

    def test_get_goals_progress_passive_income(self, isolated_portfolio):
        pt = isolated_portfolio
        pt._manage_cash_flow_locked(500_000.0, "deposit", "THB")
        pt._execute_trade_locked(
            symbol="PTT", asset_type="Stock", action="buy",
            units=1000, price=30.0, currency="THB",
        )
        pt._record_income_locked("Dividend", 50_000.0, "PTT")
        pt.set_goal.invoke({
            "name": "passive 500k", "goal_type": "passive_income_ytd",
            "target_amount_thb": 500_000.0,
        })
        result = pt.get_goals_progress.invoke({})
        data = json.loads(result)
        g = data["goals"][0]
        assert g["goal_type"] == "passive_income_ytd"
        assert g["current_amount_thb"] == pytest.approx(50_000.0, rel=1e-3)
        assert g["progress_pct"] == pytest.approx(10.0, rel=1e-2)

    def test_get_goals_progress_deadline_days_left(self, isolated_portfolio):
        pt = isolated_portfolio
        pt.set_goal.invoke({
            "name": "D", "goal_type": "nav_target",
            "target_amount_thb": 1e6, "deadline": "2031-12-31",
        })
        result = pt.get_goals_progress.invoke({})
        data = json.loads(result)
        g = data["goals"][0]
        assert "deadline" in g
        assert "deadline_days_left" in g
        assert g["deadline_days_left"] > 0

    def test_get_goals_progress_empty(self, isolated_portfolio):
        pt = isolated_portfolio
        result = pt.get_goals_progress.invoke({})
        data = json.loads(result)
        assert data["n_goals"] == 0
        assert data["goals"] == []

    def test_sidecar_files_created(self, isolated_portfolio):
        """set_goal ต้องสร้าง sidecar .md ไฟล์ใน Goals/Items/"""
        pt = isolated_portfolio
        pt.set_goal.invoke({"name": "My Goal", "goal_type": "nav_target", "target_amount_thb": 1e6})
        items_dir = pt.GOALS_ITEMS_DIR
        sidecars = list(items_dir.glob("*.md"))
        assert len(sidecars) == 1
        assert sidecars[0].stem == "My_Goal"
        import frontmatter
        with sidecars[0].open("r", encoding="utf-8") as f:
            post = frontmatter.load(f)
        assert post.metadata.get("status") is None
        assert post.metadata.get("schema_version") == 1
        assert post.metadata.get("derived") is True

    def test_sidecar_deleted_on_remove(self, isolated_portfolio):
        """remove_goal ต้องลบ sidecar ออกตามแผน (ไม่ต้อง archive)"""
        pt = isolated_portfolio
        pt.set_goal.invoke({"name": "Del Goal", "goal_type": "nav_target", "target_amount_thb": 1e6})
        pt.remove_goal.invoke({"name": "Del Goal"})
        items_dir = pt.GOALS_ITEMS_DIR
        sidecars = list(items_dir.glob("*.md"))
        assert len(sidecars) == 0



import pytest
import tools.portfolio.goals as goals

class TestAtomicWriteGoalsException:
    def test_atomic_write_exception(self, isolated_portfolio, monkeypatch):
        pt = isolated_portfolio
        import os
        def mock_replace(*args, **kwargs):
            raise OSError("Mock disk error")
        monkeypatch.setattr(os, "replace", mock_replace)
        
        with pytest.raises(OSError, match="Mock disk error"):
            pt.set_goal.func(name="A", goal_type="nav_target", target_amount_thb=1e6)

class TestGoalItemToMd:
    def test_notes_included(self, isolated_portfolio):
        pt = isolated_portfolio
        import tools.portfolio.models as models
        import tools.portfolio.goals as gl
        goal = models.GoalItem(name="A", goal_type="nav_target", target_amount_thb=100.0, created_date="2026-01-01", notes="Some notes")
        md = gl._goal_item_to_md(goal)
        assert 'notes: "Some notes"' in md

class TestLoadOrInitGoals:
    def test_load_no_metadata(self, isolated_portfolio):
        pt = isolated_portfolio
        import tools.portfolio.goals as gl
        
        gl_path = pt.GOALS_PATH
        gl_path.parent.mkdir(parents=True, exist_ok=True)
        with open(gl_path, "w", encoding="utf-8") as f:
            f.write("Hello World")
            
        post, state = gl._load_or_init_goals()
        assert len(state.goals) == 0

class TestSetGoalExceptions:
    def test_set_goal_lock_timeout(self, isolated_portfolio, monkeypatch):
        pt = isolated_portfolio
        def mock_lock(*args, **kwargs):
            from filelock import Timeout
            raise Timeout("mock")
        monkeypatch.setattr("tools.portfolio.goals._goals_lock.acquire", mock_lock)
        
        result = pt.set_goal.func(name="A", goal_type="nav_target", target_amount_thb=1e6)

        assert isinstance(result, str) and result.startswith("Error:")
        assert "goals lock" in result
class TestRemoveGoalExceptions:
    def test_remove_empty_name(self, isolated_portfolio):
        pt = isolated_portfolio
        result = pt.remove_goal.func(name="   ")

        assert isinstance(result, str) and result.startswith("Error:")
        assert "name ต้องไม่ว่าง" in result
    def test_remove_lock_timeout(self, isolated_portfolio, monkeypatch):
        pt = isolated_portfolio
        def mock_lock(*args, **kwargs):
            from filelock import Timeout
            raise Timeout("mock")
        monkeypatch.setattr("tools.portfolio.goals._goals_lock.acquire", mock_lock)
        
        result = pt.remove_goal.func(name="A")

        assert isinstance(result, str) and result.startswith("Error:")
        assert "goals lock" in result
class TestGetGoalsProgressExceptions:
    def test_get_progress_lock_timeout(self, isolated_portfolio, monkeypatch):
        pt = isolated_portfolio
        def mock_lock(*args, **kwargs):
            from filelock import Timeout
            raise Timeout("mock")
        monkeypatch.setattr("tools.portfolio.goals._portfolio_lock.acquire", mock_lock)
        
        result = pt.get_goals_progress.func()
        import json
        data = json.loads(result)
        assert "error" in data
        assert "lock timeout" in data["error"]
        
    def test_bad_deadline_format_fallback(self, isolated_portfolio):
        pt = isolated_portfolio
        pt.set_goal.func(name="A", goal_type="nav_target", target_amount_thb=1e6)
        
        # Manually alter the YAML to inject bad deadline
        import frontmatter
        post = frontmatter.load(pt.GOALS_PATH)
        post.metadata["goals"][0]["deadline"] = "BAD-DATE"
        post.metadata["goals"][0]["notes"] = "Has notes"
        with open(pt.GOALS_PATH, "w", encoding="utf-8") as f:
            f.write(frontmatter.dumps(post))
            
        result = pt.get_goals_progress.func()
        import json
        data = json.loads(result)
        g = data["goals"][0]
        assert g["deadline"] == "BAD-DATE"
        assert "deadline_days_left" not in g
        assert g["notes"] == "Has notes"
