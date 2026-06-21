"""Test portfolio_tools: recalc, execute_trade, record_income — ใช้ Vault แยกผ่าน tmp_path"""
import json

import pytest


class TestReadTradingJournal:
    def test_no_file_returns_error_msg(self, isolated_portfolio):
        pt = isolated_portfolio
        result = pt.read_trading_journal.invoke({"days": 30})
        data = json.loads(result)
        assert "error" in data and "ยังไม่มี" in data["error"]

    def test_invalid_days_raises(self, isolated_portfolio):
        pt = isolated_portfolio
        with pytest.raises(ValueError, match="days"):
            pt.read_trading_journal.invoke({"days": 0})

    def test_invalid_limit_raises(self, isolated_portfolio):
        pt = isolated_portfolio
        with pytest.raises(ValueError, match="limit"):
            pt.read_trading_journal.invoke({"limit": 0})

    def test_reads_entries_newest_first(self, isolated_portfolio):
        pt = isolated_portfolio
        pt.append_trading_journal.invoke({"entry": "first entry — buy AAPL"})
        pt.append_trading_journal.invoke({"entry": "second entry — sell PTT"})
        pt.append_trading_journal.invoke({"entry": "third entry — market crash"})

        result = pt.read_trading_journal.invoke({})
        data = json.loads(result)
        assert data["n_total_in_window"] == 3
        assert data["n_returned"] == 3
        # newest first → third comes before first
        contents = [e["content"] for e in data["entries"]]
        assert "third entry" in contents[0]
        assert "first entry" in contents[2]

    def test_keyword_filter_case_insensitive(self, isolated_portfolio):
        pt = isolated_portfolio
        pt.append_trading_journal.invoke({"entry": "Bought AAPL because earnings beat"})
        pt.append_trading_journal.invoke({"entry": "Sold PTT due to dividend cut"})
        pt.append_trading_journal.invoke({"entry": "AAPL split announced"})

        result = pt.read_trading_journal.invoke({"keyword": "aapl"})
        data = json.loads(result)
        assert data["n_total_in_window"] == 2
        for e in data["entries"]:
            assert "aapl" in e["content"].lower()

    def test_limit_caps_entries(self, isolated_portfolio):
        pt = isolated_portfolio
        for i in range(5):
            pt.append_trading_journal.invoke({"entry": f"entry {i}"})
        result = pt.read_trading_journal.invoke({"limit": 2})
        data = json.loads(result)
        assert data["n_total_in_window"] == 5
        assert data["n_returned"] == 2

    def test_days_filter_excludes_old(self, isolated_portfolio, tmp_vault):
        pt = isolated_portfolio
        # เขียน journal entries ผสม — ใหม่และเก่า (เก่าเขียน timestamp ด้วยมือเลย)
        journal_path = tmp_vault / "20_Portfolio_Management" / "Journals_and_Reports" / "Trading_Journal.md"
        journal_path.parent.mkdir(parents=True, exist_ok=True)
        journal_path.write_text(
            "\n## [2020-01-01 10:00:00]\n\nold entry far in past\n"
            "\n## [2026-05-22 14:00:00]\n\nentry today\n",
            encoding="utf-8",
        )
        # days=30 → 2020 entry ตกขอบ, 2026 อยู่ใน window (assuming current date >= 2026-04-22)
        # Note: ทดสอบนี้พึ่งวันที่ system — ใช้ days=10000 เพื่อให้ปลอดภัย แล้วเช็คว่า old ก็ติด
        result_all = pt.read_trading_journal.invoke({"days": 10_000})
        data_all = json.loads(result_all)
        assert data_all["n_total_in_window"] == 2

        result_recent = pt.read_trading_journal.invoke({"days": 40})
        data_recent = json.loads(result_recent)
        # 2020 ต้องถูก filter ออก
        assert data_recent["n_total_in_window"] == 1
        assert "today" in data_recent["entries"][0]["content"]

    def test_edit_holding_entries_appear_in_journal(self, isolated_portfolio):
        """integration: edit_holding auto-logs → read_trading_journal เห็น"""
        pt = isolated_portfolio
        pt._manage_cash_flow_locked(500_000.0, "deposit", "THB")
        pt._execute_trade_locked(
            symbol="PTT", asset_type="Stock", action="buy",
            units=1000, price=30.0, currency="THB",
        )
        pt._edit_holding_locked(
            "PTT", units=950.0, avg_cost=None,
            accumulated_dividend_thb=None, asset_type=None,
            reason="bonus share correction",
        )
        result = pt.read_trading_journal.invoke({"keyword": "EDIT PTT"})
        data = json.loads(result)
        assert data["n_total_in_window"] >= 1



import pytest
import tools.portfolio.journal as journal

class TestInjectJournalWikilinks:
    def test_inject_journal_wikilinks_ignores_cash(self):
        # Normal stock should get wikilink
        res1 = journal._inject_journal_wikilinks("**[BUY]** AAPL **[10 units]**")
        assert "[[AAPL]]" in res1
        
        # Cash should be ignored
        res2 = journal._inject_journal_wikilinks(f"**[DEPOSIT]** {journal.CASH_THB_SYMBOL} **[100 THB]**")
        assert f"[[{journal.CASH_THB_SYMBOL}]]" not in res2
        assert "CASH_THB" in res2

class TestAppendTradingJournalExceptions:
    def test_append_empty_raises(self, isolated_portfolio):
        pt = isolated_portfolio
        with pytest.raises(ValueError, match="entry ต้องไม่ว่าง"):
            pt.append_trading_journal.func(entry="   ")

class TestReadTradingJournalExceptions:
    def test_read_journal_bad_datetime(self, isolated_portfolio, tmp_vault):
        pt = isolated_portfolio
        journal_path = tmp_vault / "20_Portfolio_Management" / "Journals_and_Reports" / "Trading_Journal.md"
        journal_path.parent.mkdir(parents=True, exist_ok=True)
        # Write one good and one bad datetime block
        journal_path.write_text(
            "\n## [2026-05-22 14:00:00]\n\ngood entry\n"
            "\n## [2026-99-99 99:99:99]\n\nbad datetime entry\n",
            encoding="utf-8",
        )
        
        # It should ignore the bad datetime and only return the good one
        result = pt.read_trading_journal.func(days=10000)
        import json
        data = json.loads(result)
        assert data["n_total_in_window"] == 1
        assert "good entry" in data["entries"][0]["content"]
