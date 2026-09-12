"""Seed test data ลง Obsidian vault — deterministic, ไม่ต้อง network/LLM

ใช้:
    uv run python scripts/seed_test_data.py

ผลลัพธ์:
    - Portfolio_Holdings.md: dual-currency holdings (TH + US), realistic prices
    - Trading_Journal.md: 6+ entries (mix recent + เก่า)
    - Watchlist.md: 3 ตัว
    - Performance_Log.csv: 5 snapshots ข้ามวัน (สำหรับ read_performance_history)

ลบไฟล์เดิมก่อนเริ่ม (ตามที่ user สั่ง overwrite). ถ้าต้องการ restore ใช้ .bak
"""
import csv
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Script อยู่ใน scripts/ — เพิ่ม project root เข้า path ก่อน import tools.*
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from tools.portfolio_tools import (
    PERFORMANCE_LOG_PATH,
    PORTFOLIO_PATH,
    TRADING_JOURNAL_PATH,
    WATCHLIST_PATH,
    _PERFORMANCE_LOG_HEADER,
    _edit_holding_locked,
    _execute_trade_locked,
    _load_or_init,
    _manage_cash_flow_locked,
    _record_income_locked,
    _save,
    _update_fx_rate_locked,
    _write_journal_entry,
    add_to_watchlist,
)


def reset_files() -> None:
    for p in [PORTFOLIO_PATH, WATCHLIST_PATH, TRADING_JOURNAL_PATH, PERFORMANCE_LOG_PATH]:
        if p.exists():
            p.unlink()
            print(f"  removed {p}")


def set_current_price(symbol: str, price: float, currency: str) -> None:
    """บังคับ current_price เพื่อจำลอง market move หลัง buy → unrealized P/L มีค่า"""
    post, state = _load_or_init()
    h = next(h for h in state.holdings if h.symbol == symbol)
    if currency == "USD":
        h.current_price_usd = price
    else:
        h.current_price_thb = price
    _save(post, state)


def seed_holdings() -> None:
    """Initial capital, FX, buys, sells, income"""
    print("\n[1] Cash deposits")
    print(" ", _manage_cash_flow_locked(5_000_000.0, "deposit", "THB"))
    print(" ", _manage_cash_flow_locked(20_000.0, "deposit", "USD"))

    print("\n[2] FX rate manual update (skip yfinance for deterministic seed)")
    print(" ", _update_fx_rate_locked(35.80))

    print("\n[3] Buy US stocks (debit CASH_USD)")
    for sym, qty, price in [
        ("AAPL", 30, 180.0),
        ("MSFT", 15, 400.0),
        ("GOOGL", 10, 150.0),
    ]:
        print(" ", _execute_trade_locked(sym, "Stock", "buy", qty, price, "USD"))

    print("\n[4] Buy TH stocks (debit CASH_THB)")
    for sym, qty, price in [
        ("PTT", 3000, 35.0),
        ("KBANK", 1000, 140.0),
        ("AOT", 2000, 55.0),
        ("CPALL", 1500, 55.0),
        ("ADVANC", 300, 200.0),
    ]:
        print(" ", _execute_trade_locked(sym, "Stock", "buy", qty, price, "THB"))

    print("\n[5] Simulate market moves (manual current_price)")
    moves_usd = {"AAPL": 220.0, "MSFT": 460.0, "GOOGL": 195.0}
    moves_thb = {"PTT": 38.5, "KBANK": 155.0, "AOT": 48.0, "CPALL": 52.0, "ADVANC": 245.0}
    for sym, price in moves_usd.items():
        set_current_price(sym, price, "USD")
    for sym, price in moves_thb.items():
        set_current_price(sym, price, "THB")
    print(f"  updated prices: {len(moves_usd)} USD + {len(moves_thb)} THB")

    print("\n[6] Partial sell (realize profit on PTT)")
    print(" ", _execute_trade_locked("PTT", "Stock", "sell", 500, 38.5, "THB"))

    print("\n[7] Edit correction (simulate bonus share)")
    print(" ", _edit_holding_locked(
        "ADVANC", units=315.0, avg_cost=None,
        accumulated_dividend_thb=None, asset_type=None,
        reason="ADVANC bonus share 5% — adjusted units 300 → 315",
    ))

    print("\n[8] Passive income (non-dividend per current architecture)")
    print(" ", _record_income_locked("Interest", 8_500.0, None))
    print(" ", _record_income_locked("Rental", 22_000.0, None))


def seed_watchlist() -> None:
    print("\n[9] Watchlist (3 items)")
    for item in [
        {"symbol": "NVDA", "asset_type": "Stock", "target_price": 130.0,
         "notes": "wait for AI hype correction"},
        {"symbol": "TSLA", "asset_type": "Stock", "target_price": 180.0,
         "notes": "support level + delivery beat"},
        {"symbol": "DELTA", "asset_type": "Stock", "target_price": 95.0,
         "notes": "อิเล็กทรอนิกส์ไทย — รอ pullback"},
    ]:
        print(" ", add_to_watchlist.invoke(item))


def seed_journal() -> None:
    """หลายเทคนิค: ใช้ _write_journal_entry ตรงๆ + เขียน timestamp เก่าด้วยมือสำหรับ days filter"""
    print("\n[10] Trading journal entries (recent + historical)")

    # Recent entries via the helper (uses current timestamp)
    recent = [
        "**[BUY AAPL]** เข้าซื้อ AAPL เพราะ Q4 earnings เกินคาด + iPhone cycle ใหม่. ตั้ง stop-loss ที่ $165",
        "**[BUY GOOGL]** Cloud growth + AI search dominance. เข้าตอน P/E ต่ำกว่า peers",
        "**[SELL PTT partial]** ขายบางส่วนหลังราคาแตะ ฿38.5 — lock profit + รอ pullback ซื้อกลับ",
        "**[OBSERVATION]** ตลาด US ใกล้ ATH — เพิ่ม cash position สำรอง 20% เผื่อ correction",
    ]
    for entry in recent:
        ts = _write_journal_entry(entry)
        print(f"  recent [{ts}]")

    # Historical entries — write directly เพราะ _write_journal_entry ใช้ now()
    TRADING_JOURNAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    today = datetime.now()
    historical = [
        (today - timedelta(days=60),
         "**[ANALYSIS]** ดู KBANK งบ Q3 — NIM ขยายดี แต่ NPL เริ่มขึ้น. ระวังครึ่งหลังปีหน้า"),
        (today - timedelta(days=120),
         "**[MISTAKE]** เข้า NVDA แพงไป $750 ก่อนหุ้นปรับ — บทเรียน: ห้าม FOMO ตามกระแส AI"),
    ]
    with TRADING_JOURNAL_PATH.open("a", encoding="utf-8") as f:
        for date, entry in historical:
            ts_str = date.strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"\n## [{ts_str}]\n\n{entry}\n")
            print(f"  historical [{ts_str}]")


def seed_performance_log() -> None:
    """เขียน Performance_Log.csv 5 rows ย้อนหลัง — สำหรับ read_performance_history"""
    print("\n[11] Performance log snapshots (5 days)")
    PERFORMANCE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    today = datetime.now()
    # Trend: ขึ้น → peak → drawdown → recovery
    rows = []
    base_nav = 6_200_000.0
    for i, (offset_days, nav, cash) in enumerate([
        (4, base_nav, 1_200_000),
        (3, base_nav + 60_000, 1_180_000),
        (2, base_nav + 110_000, 1_180_000),  # peak
        (1, base_nav + 50_000, 1_180_000),   # drawdown -0.95% from peak
        (0, base_nav + 90_000, 1_180_000),
    ]):
        date = (today - timedelta(days=offset_days)).strftime("%Y-%m-%d")
        unrealized = nav - 6_000_000  # rough
        rows.append([date, f"{nav:.2f}", f"{nav - unrealized:.2f}", f"{unrealized:.2f}", f"{cash:.2f}"])

    with PERFORMANCE_LOG_PATH.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(_PERFORMANCE_LOG_HEADER)
        w.writerows(rows)
    print(f"  wrote {len(rows)} snapshots → {PERFORMANCE_LOG_PATH}")


def main():
    print("=" * 60)
    print("Seeding test data — overwrite existing portfolio/journal/watchlist/perf log")
    print("=" * 60)

    print("\n[0] Reset existing files (use .bak ที่ backup ไว้แล้ว ถ้าต้อง restore)")
    reset_files()

    seed_holdings()
    seed_watchlist()
    seed_journal()
    seed_performance_log()

    print("\n" + "=" * 60)
    print("Seed complete. ลองรัน main.py แล้วถาม Bookkeeper เช่น:")
    print("  - ดูสถานะพอร์ต")
    print("  - หุ้นนอก vs ไทย กี่ %")
    print("  - drawdown สูงสุดเดือนนี้")
    print("  - ดู watchlist")
    print("  - ทำไมขาย PTT")
    print("=" * 60)


if __name__ == "__main__":
    main()
