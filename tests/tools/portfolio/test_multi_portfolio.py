import json
import pytest
from pathlib import Path


def test_multi_portfolio_crud_and_isolation(isolated_portfolio):
    import tools.portfolio.core as core
    import tools.portfolio.trading as trading
    import tools.portfolio.goals as goals

    # 1. List initial portfolios (should have 'default')
    portfolios = core.list_portfolios()
    assert any(p.id == "default" for p in portfolios)

    # 2. Create emergency fund portfolio
    p_emergency = core.create_portfolio(name="Emergency Fund", portfolio_id="emergency_fund")
    assert p_emergency.id == "emergency_fund"
    assert p_emergency.name == "Emergency Fund"

    # List should now include emergency_fund
    portfolios = core.list_portfolios()
    assert len(portfolios) >= 2
    assert any(p.id == "emergency_fund" for p in portfolios)

    # 3. Add cash to emergency_fund portfolio
    trading.structured_manage_cash_flow(
        amount=100000.0, action="deposit", currency="THB", portfolio_id="emergency_fund"
    )

    # Verify default portfolio has 0 cash THB while emergency_fund has 100000
    state_default = core.get_structured_portfolio_state(portfolio_id="default")
    state_emergency = core.get_structured_portfolio_state(portfolio_id="emergency_fund")

    cash_default = next(h.units for h in state_default.holdings if h.symbol == "CASH_THB")
    cash_emergency = next(h.units for h in state_emergency.holdings if h.symbol == "CASH_THB")

    assert cash_default == 0.0
    assert cash_emergency == 100000.0

    # 4. Create Goals linked to different portfolios
    goals.structured_upsert_goal(
        name="Main Port 10M",
        goal_type="nav_target",
        target_amount_thb=10000000.0,
        portfolio_id="default",
    )
    goals.structured_upsert_goal(
        name="Emergency Cash 100k",
        goal_type="cash_target",
        target_amount_thb=100000.0,
        portfolio_id="emergency_fund",
    )

    all_goals = goals.get_structured_goals()
    g_main = next(g for g in all_goals if g["name"] == "Main Port 10M")
    g_emer = next(g for g in all_goals if g["name"] == "Emergency Cash 100k")

    assert g_main["current_amount_thb"] == 0.0
    assert g_emer["current_amount_thb"] == 100000.0
    assert g_emer["progress_pct"] == 100.0

    # 5. Clean up / delete portfolio
    core.delete_portfolio("emergency_fund")
    portfolios_after = core.list_portfolios()
    assert not any(p.id == "emergency_fund" for p in portfolios_after)


def test_create_portfolio_thai_name_does_not_produce_garbage_id(isolated_portfolio):
    """ชื่อพอร์ตภาษาไทยล้วน (ไม่มี ASCII เลย) ต้อง fallback เป็น id แบบ timestamp
    ไม่ใช่ id ที่เป็นขีดล่างล้วน (regex sub ตัวอักษรไทยทุกตัวเป็น '_') เพราะจะอ่านไม่ออก
    และชนกันได้ง่ายกับชื่อไทยอื่นที่ความยาวเท่ากัน
    """
    import re
    import tools.portfolio.core as core

    p1 = core.create_portfolio(name="พอร์ตเงินสำรองฉุกเฉิน")
    assert p1.name == "พอร์ตเงินสำรองฉุกเฉิน"
    assert re.search(r"[a-zA-Z0-9]", p1.id), f"id ต้องมีตัวอักษร/ตัวเลขจริงอย่างน้อย 1 ตัว ได้ {p1.id!r}"
    assert not re.fullmatch(r"_+", p1.id)

    # ชื่อไทยอีกชื่อที่ความยาวเท่ากันเป๊ะ (คนละความหมาย) ต้องไม่ชน id กับ p1
    p2 = core.create_portfolio(name="พอร์ตกองทุนสำรองบำนาญ")
    assert p2.id != p1.id

    portfolios = core.list_portfolios()
    assert any(p.id == p1.id for p in portfolios)
    assert any(p.id == p2.id for p in portfolios)


def test_read_tools_respect_portfolio_id(isolated_portfolio):
    """get_portfolio_state / compute_allocation_breakdown / read_performance_history
    ต้องอ่านค่าของพอร์ตที่ระบุ ไม่ใช่ default เสมอ (เดิมไม่มีพารามิเตอร์ portfolio_id เลย)
    """
    import tools.portfolio.core as core
    import tools.portfolio.trading as trading
    import tools.portfolio.performance as perf

    core.create_portfolio(name="Retirement", portfolio_id="retirement")
    trading.structured_manage_cash_flow(
        amount=50000.0, action="deposit", currency="THB", portfolio_id="retirement"
    )

    state_data = json.loads(
        core.get_portfolio_state.invoke({"refresh_prices": False, "portfolio_id": "retirement"})
    )
    cash_retirement = next(h["units"] for h in state_data["holdings"] if h["symbol"] == "CASH_THB")
    assert cash_retirement == 50000.0

    default_data = json.loads(
        core.get_portfolio_state.invoke({"refresh_prices": False, "portfolio_id": "default"})
    )
    cash_default = next(h["units"] for h in default_data["holdings"] if h["symbol"] == "CASH_THB")
    assert cash_default == 0.0

    breakdown = json.loads(
        core.compute_allocation_breakdown.invoke({"group_by": "asset_type", "portfolio_id": "retirement"})
    )
    assert breakdown["total_nav_thb"] == 50000.0

    perf.record_performance_snapshot.invoke({"refresh_prices": False, "portfolio_id": "retirement"})
    hist = json.loads(
        perf.read_performance_history.invoke({"days": 30, "portfolio_id": "retirement"})
    )
    assert hist["latest_nav"] == 50000.0


def test_trades_log_split_per_portfolio(isolated_portfolio):
    """Trades_Log.csv ของพอร์ตรองต้องแยกไฟล์จาก default ไม่ปนกัน"""
    import tools.portfolio.core as core
    import tools.portfolio.trading as trading

    core.create_portfolio(name="Retirement", portfolio_id="retirement")
    trading.structured_manage_cash_flow(
        amount=100000.0, action="deposit", currency="THB", portfolio_id="retirement"
    )
    trading.execute_trade.invoke({
        "symbol": "PG", "asset_type": "Stock", "action": "buy",
        "units": 10, "price": 100.0, "currency": "THB", "portfolio_id": "retirement",
    })

    default_log = trading._get_trades_log_filepath("default")
    retirement_log = trading._get_trades_log_filepath("retirement")

    assert retirement_log != default_log
    assert retirement_log.exists()
    assert "PG" in retirement_log.read_text(encoding="utf-8")
    # default log ต้องไม่มีแถวจากพอร์ต retirement ปนอยู่
    if default_log.exists():
        assert "PG" not in default_log.read_text(encoding="utf-8")


def test_sidecars_sync_for_non_default_portfolio_without_collision(isolated_portfolio):
    """Holdings/Watchlist sidecar ของพอร์ตรองต้อง sync จริง และไม่ชนกับ symbol เดียวกันในพอร์ตอื่น"""
    import tools.portfolio.core as core
    import tools.portfolio.trading as trading
    import tools.portfolio.watchlist as watchlist

    core.create_portfolio(name="Retirement", portfolio_id="retirement")
    for pid in ("default", "retirement"):
        trading.structured_manage_cash_flow(amount=100000.0, action="deposit", currency="THB", portfolio_id=pid)
        trade_result = trading.execute_trade.invoke({
            "symbol": "AAPL", "asset_type": "Stock", "action": "buy",
            "units": 1, "price": 100.0, "currency": "THB", "portfolio_id": pid,
        })
        assert not trade_result.startswith("Error:"), trade_result
        watchlist.add_to_watchlist.invoke({"symbol": "TSLA", "asset_type": "Stock", "portfolio_id": pid})

    default_holdings_dir = core._get_holdings_dir("default")
    retirement_holdings_dir = core._get_holdings_dir("retirement")
    assert default_holdings_dir != retirement_holdings_dir
    assert (default_holdings_dir / "AAPL.md").exists()
    assert (retirement_holdings_dir / "AAPL.md").exists()

    default_wl_dir = watchlist._get_watchlist_items_dir("default")
    retirement_wl_dir = watchlist._get_watchlist_items_dir("retirement")
    assert default_wl_dir != retirement_wl_dir
    assert (default_wl_dir / "TSLA.md").exists()
    assert (retirement_wl_dir / "TSLA.md").exists()


def test_watchlist_guards_against_nonexistent_portfolio(isolated_portfolio):
    """เรียก watchlist tool ด้วย portfolio_id ที่ไม่เคยสร้างจริง ต้องได้ error ไม่ใช่สร้างไฟล์กำพร้า"""
    import tools.portfolio.watchlist as watchlist

    result = watchlist.add_to_watchlist.invoke({
        "symbol": "AAPL", "asset_type": "Stock", "portfolio_id": "never_created",
    })
    assert result.startswith("Error:")

    ghost_watchlist_path = watchlist._get_watchlist_filepath("never_created")
    assert not ghost_watchlist_path.exists()


def test_portfolio_id_rejects_path_traversal(isolated_portfolio):
    """portfolio_id ต้องถูก sanitize ก่อนสร้าง filesystem path เสมอ — กัน path traversal
    (เช่น '../../evil') ที่อาจหลุดออกนอกโฟลเดอร์ Portfolios/ ได้ ไม่งั้น delete_portfolio's
    shutil.rmtree จะลบโฟลเดอร์นอกเป้าหมายได้
    """
    import tools.portfolio.core as core

    malicious_ids = ["../../evil", "..\\..\\evil", "a/b", "a\\b", "..", "."]
    for bad_id in malicious_ids:
        with pytest.raises(ValueError):
            core._get_portfolio_filepath(bad_id)
        with pytest.raises(ValueError):
            core._normalize_portfolio_id(bad_id)

    # ค่าที่ถูกต้องต้องยังผ่านปกติ
    assert core._normalize_portfolio_id("emergency_fund-2") == "emergency_fund-2"
    assert core._normalize_portfolio_id("default") == "default"
    assert core._normalize_portfolio_id(None) == "default"


def test_load_or_init_rejects_nonexistent_non_default_portfolio(isolated_portfolio):
    """เรียก _load_or_init กับ portfolio_id ที่ไม่เคยสร้างผ่าน create_portfolio() ต้อง error
    ไม่ใช่สร้างพอร์ตผีขึ้นมาเงียบๆ (default ยังคง auto-bootstrap ได้ตามปกติ)
    """
    import tools.portfolio.core as core

    with pytest.raises(ValueError, match="ไม่พบพอร์ตไอดี"):
        core._load_or_init(portfolio_id="never_created_portfolio")

    # default ต้องยัง auto-bootstrap ได้ตามเดิม (ไม่ error)
    post, state = core._load_or_init(portfolio_id="default")
    assert state is not None


def test_journal_entries_land_in_correct_portfolio_not_default(isolated_portfolio):
    """เทรด/ฝากเงิน/บันทึกรายได้/แก้ไข/ลบ holding ในพอร์ตที่ไม่ใช่ default ต้องเขียนบันทึก
    อัตโนมัติลง Trading Journal ของพอร์ตนั้นเอง ไม่ใช่ของ default (บั๊กเดิม: trading.py
    เรียก _write_journal_entry โดยไม่ส่ง portfolio_id เลยตกไปที่ default เสมอ)
    """
    import tools.portfolio.core as core
    import tools.portfolio.journal as journal
    import tools.portfolio.trading as trading

    core.create_portfolio(name="Emergency Fund", portfolio_id="emergency_fund")
    trading.structured_manage_cash_flow(
        amount=100000.0, action="deposit", currency="THB",
        date="2026-01-01", notes="เติมเงินก้อนแรก", portfolio_id="emergency_fund",
    )
    trading.structured_execute_trade(
        symbol="PTT", asset_type="Stock", action="buy", units=100, price=30.0,
        currency="THB", date="2026-01-02", notes="ซื้อ PTT", portfolio_id="emergency_fund",
    )

    emergency_journal_path = journal._get_journal_filepath("emergency_fund")
    default_journal_path = journal._get_journal_filepath("default")

    assert emergency_journal_path.exists()
    emergency_content = emergency_journal_path.read_text(encoding="utf-8")
    assert "CASH FLOW NOTE" in emergency_content
    assert "TRADE NOTE" in emergency_content
    assert "PTT" in emergency_content

    # default journal ต้องไม่มีข้อมูลของ emergency_fund ปนอยู่ (ไฟล์ไม่มีอยู่เลยหรือไม่มีคำว่า PTT)
    if default_journal_path.exists():
        assert "PTT" not in default_journal_path.read_text(encoding="utf-8")


def test_reset_clean_slate_preserves_portfolio_name_and_backs_up_trades(isolated_portfolio):
    """การ RESET พอร์ตต้องคงชื่อพอร์ตที่ผู้ใช้ตั้งไว้ (ไม่กลายเป็น portfolio_id)
    และต้องสำรองทั้ง Portfolio_Holdings.md และ Trades_Log.csv ไว้ใน .backups เสมอ
    """
    import tools.portfolio.core as core
    import tools.portfolio.trading as trading

    pid = "pvd_retirement"
    port_name = "กองทุนสำรองเลี้ยงชีพ"
    core.create_portfolio(name=port_name, portfolio_id=pid)

    # บันทึกเงินฝากและเทรด
    trading.structured_manage_cash_flow(
        amount=500_000.0, action="deposit", currency="THB", portfolio_id=pid
    )
    trading.structured_execute_trade(
        symbol="PTT", asset_type="Stock", action="buy", units=1000, price=35.0,
        currency="THB", portfolio_id=pid
    )

    state_before = core.get_structured_portfolio_state(portfolio_id=pid)
    assert any(h.symbol == "PTT" for h in state_before.holdings)

    # สั่ง Reset clean slate
    new_state = core.structured_reset_clean_slate(portfolio_id=pid)

    # 1. ยืนยันว่า new_state และ list_portfolios ยังคงรักษาชื่อพอร์ตภาษาไทยไว้
    assert new_state.name == port_name
    portfolios = core.list_portfolios()
    matched = next((p for p in portfolios if p.id == pid), None)
    assert matched is not None
    assert matched.name == port_name

    # 2. ยืนยันว่า holdings ถูก reset สะอาด
    assert len(new_state.holdings) == 2
    assert all(h.symbol in ("CASH_THB", "CASH_USD") for h in new_state.holdings)

    # 3. ยืนยันว่า .backups สำรองทั้ง Portfolio_Holdings.md และ Trades_Log.csv
    portfolio_file = core._get_portfolio_filepath(pid)
    backups_dir = portfolio_file.parent / ".backups"
    assert backups_dir.exists()
    backup_subdirs = sorted(backups_dir.glob("*"))
    assert len(backup_subdirs) >= 1
    latest_backup = backup_subdirs[-1]
    assert (latest_backup / "Portfolio_Holdings.md").exists()
    assert (latest_backup / "Trades_Log.csv").exists()
    assert (latest_backup / "Holdings" / "PTT.md").exists()

    # 4. ตรวจสอบว่า Trades_Log.csv ปัจจุบันถูกล้าง (ไม่มีรายการ PTT อยู่)
    current_trades_path = trading._get_trades_log_filepath(pid)
    assert current_trades_path.exists()
    assert "PTT" not in current_trades_path.read_text(encoding="utf-8")


def test_rename_goal_in_portfolio_and_cross_portfolio_isolation(isolated_portfolio):
    """ทดสอบการเปลี่ยนชื่อเป้าหมาย:
    1. เปลี่ยนชื่อสำเร็จและคง created_date เดิม
    2. ชนกับชื่อเป้าหมายในพอร์ตเดียวกันต้อง raise ValueError
    3. ตั้งชื่อซ้ำกับเป้าหมายในอีกพอร์ตหนึ่งได้โดยไม่ชนกัน
    """
    import tools.portfolio.core as core
    import tools.portfolio.goals as goals

    core.create_portfolio(name="พอร์ตเกษียณ", portfolio_id="retire")

    # สร้างเป้าหมายใน default
    goals.structured_upsert_goal(
        name="Target 1",
        goal_type="nav_target",
        target_amount_thb=1_000_000.0,
        portfolio_id="default",
    )
    goals.structured_upsert_goal(
        name="Target 2",
        goal_type="cash_target",
        target_amount_thb=500_000.0,
        portfolio_id="default",
    )

    # 1. เปลี่ยนชื่อ Target 1 -> Target 1 Renamed
    res = goals.structured_upsert_goal(
        name="Target 1",
        new_name="Target 1 Renamed",
        goal_type="nav_target",
        target_amount_thb=1_200_000.0,
        portfolio_id="default",
    )
    renamed = next((g for g in res if g["name"] == "Target 1 Renamed"), None)
    assert renamed is not None
    assert renamed["target_amount_thb"] == 1_200_000.0
    assert not any(g["name"] == "Target 1" for g in res)

    # 2. เปลี่ยนชื่อไปชน Target 2 ในพอร์ตเดียวกัน -> ต้อง raise ValueError
    import pytest
    with pytest.raises(ValueError, match="มีอยู่แล้วในพอร์ตนี้"):
        goals.structured_upsert_goal(
            name="Target 1 Renamed",
            new_name="Target 2",
            goal_type="nav_target",
            target_amount_thb=1_200_000.0,
            portfolio_id="default",
        )

    # 3. สร้างหรือเปลี่ยนชื่อเป้าหมายในอีกพอร์ตหนึ่งเป็น "Target 2" ได้ ไม่ชน
    res_retire = goals.structured_upsert_goal(
        name="Retire Goal",
        new_name="Target 2",
        goal_type="nav_target",
        target_amount_thb=5_000_000.0,
        portfolio_id="retire",
    )
    retire_target = next((g for g in res_retire if g["name"] == "Target 2" and g["portfolio_id"] == "retire"), None)
    assert retire_target is not None
    assert retire_target["target_amount_thb"] == 5_000_000.0

