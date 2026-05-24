import concurrent.futures
import csv
import json
import os
import re
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

import frontmatter
import yfinance as yf
from filelock import FileLock, Timeout
from langchain_core.tools import tool
from pydantic import BaseModel, ConfigDict, Field, field_validator

from core.logger import get_logger

log = get_logger(__name__)

VAULT_PATH = Path(os.getenv("OBSIDIAN_VAULT_PATH", "./memories"))
PORTFOLIO_REL = os.getenv("PORTFOLIO_FILE", "20_Portfolio_Management/Current_Holdings/Portfolio_Holdings.md")
PORTFOLIO_PATH = VAULT_PATH / PORTFOLIO_REL
TRADING_JOURNAL_REL = os.getenv(
    "TRADING_JOURNAL_FILE",
    "20_Portfolio_Management/Journals_and_Reports/Trading_Journal.md",
)
TRADING_JOURNAL_PATH = VAULT_PATH / TRADING_JOURNAL_REL
WATCHLIST_REL = os.getenv(
    "WATCHLIST_FILE",
    "20_Portfolio_Management/Current_Holdings/Watchlist.md",
)
WATCHLIST_PATH = VAULT_PATH / WATCHLIST_REL
PERFORMANCE_LOG_REL = os.getenv(
    "PERFORMANCE_LOG_FILE",
    "20_Portfolio_Management/Journals_and_Reports/Performance_Log.csv",
)
PERFORMANCE_LOG_PATH = VAULT_PATH / PERFORMANCE_LOG_REL
_PERFORMANCE_LOG_HEADER = ["Date", "Total_NAV", "Total_Cost", "Unrealized_PnL", "Cash_Balance"]

# Derived sidecar folders (master = Portfolio_Holdings.md / Watchlist.md)
HOLDINGS_DIR = VAULT_PATH / "20_Portfolio_Management/Current_Holdings/Holdings"
WATCHLIST_ITEMS_DIR = VAULT_PATH / "20_Portfolio_Management/Current_Holdings/WatchlistItems"

CASH_THB_SYMBOL = "CASH_THB"
CASH_USD_SYMBOL = "CASH_USD"
_CASH_SYMBOLS = (CASH_THB_SYMBOL, CASH_USD_SYMBOL)
# Back-compat alias — call sites and tests still reference CASH_SYMBOL
CASH_SYMBOL = CASH_THB_SYMBOL

_FLOAT_EPS = 1e-6
_MONEY_DP = 2
_COST_DP = 4
_PCT_DP = 2

_LOCK_TIMEOUT = 15  # seconds — wait up to 15s for another process to release
_PRICE_FETCH_TIMEOUT = 6  # seconds per symbol when refreshing

_PORTFOLIO_LOCK_PATH = str(PORTFOLIO_PATH) + ".lock"
_portfolio_lock = FileLock(_PORTFOLIO_LOCK_PATH, timeout=_LOCK_TIMEOUT)

_WATCHLIST_LOCK_PATH = str(WATCHLIST_PATH) + ".lock"
_watchlist_lock = FileLock(_WATCHLIST_LOCK_PATH, timeout=_LOCK_TIMEOUT)


class Holding(BaseModel):
    model_config = ConfigDict(extra="allow")

    symbol: str
    asset_type: str
    units: float

    avg_cost_thb: float | None = None
    avg_cost_usd: float | None = None
    current_price_thb: float | None = None
    current_price_usd: float | None = None
    fx_rate: float | None = None

    market_value_thb: float = 0.0
    unrealized_pnl_percent: float | None = None
    accumulated_dividend_thb: float | None = None


class Summary(BaseModel):
    model_config = ConfigDict(extra="allow")

    total_value_thb: float = 0.0
    total_unrealized_profit: float = 0.0
    total_realized_profit_ytd: float = 0.0
    passive_income_ytd: float = 0.0
    total_accumulated_dividend: float = 0.0


def _coerce_iso_string(v):
    """PyYAML implicit-types ISO 8601 strings → datetime; coerce กลับเป็น str
    กัน ValidationError เมื่อไฟล์ถูกแก้ด้วยมือ (ไม่มี quotes รอบค่า)
    """
    if hasattr(v, "isoformat"):
        return v.isoformat(timespec="seconds") if hasattr(v, "hour") else v.isoformat()
    return v


class PortfolioState(BaseModel):
    model_config = ConfigDict(extra="allow")

    doc_type: Literal["portfolio_master"] = "portfolio_master"
    last_updated: str
    base_currency: str = "THB"
    summary: Summary = Field(default_factory=Summary)
    fx_rates: dict[str, float] = Field(default_factory=lambda: {"USDTHB": 36.5})
    holdings: list[Holding] = Field(default_factory=list)

    @field_validator("last_updated", mode="before")
    @classmethod
    def _validate_last_updated(cls, v):
        return _coerce_iso_string(v)


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _initial_state() -> PortfolioState:
    return PortfolioState(
        last_updated=_now_iso(),
        summary=Summary(),
        fx_rates={"USDTHB": 36.5},
        holdings=[
            Holding(
                symbol=CASH_THB_SYMBOL,
                asset_type="Cash",
                units=0.0,
                market_value_thb=0.0,
            ),
            Holding(
                symbol=CASH_USD_SYMBOL,
                asset_type="Cash",
                units=0.0,
                market_value_thb=0.0,
            ),
        ],
    )


def _load_or_init() -> tuple[frontmatter.Post, PortfolioState]:
    if not PORTFOLIO_PATH.exists():
        PORTFOLIO_PATH.parent.mkdir(parents=True, exist_ok=True)
        post = frontmatter.Post(content="")  # YAML state only — no body
        state = _initial_state()
        _save(post, state)
        return post, state

    with PORTFOLIO_PATH.open("r", encoding="utf-8") as f:
        post = frontmatter.load(f)

    if not post.metadata:
        log.warning("Portfolio_Holdings.md ไม่มี YAML frontmatter — บูตข้อมูลใหม่")
        state = _initial_state()
        _save(post, state)
        return post, state

    state = PortfolioState.model_validate(post.metadata)
    return post, state


def _atomic_write(serialized: str) -> None:
    """เขียนไฟล์แบบ atomic: temp file ใน folder เดียวกัน → os.replace()
    os.replace() เป็น atomic บนทั้ง Windows และ POSIX เมื่ออยู่ filesystem เดียวกัน
    """
    parent = PORTFOLIO_PATH.parent
    parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(
        prefix=".portfolio_", suffix=".md.tmp", dir=str(parent)
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(serialized)
        os.replace(tmp_path, PORTFOLIO_PATH)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _atomic_write_to(path: Path, content: str) -> None:
    """Generic atomic write: temp file → os.replace() — ใช้ได้กับไฟล์ใดก็ได้"""
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".tmp_", suffix=".md.tmp", dir=str(parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _holding_to_md(h: Holding) -> str:
    """สร้าง YAML frontmatter สำหรับ sidecar file ของ Holding รายตัว"""
    if h.avg_cost_usd is not None:
        currency = "USD"
        avg_cost = h.avg_cost_usd
        current_price = h.current_price_usd
    else:
        currency = "THB"
        avg_cost = h.avg_cost_thb
        current_price = h.current_price_thb

    lines = [
        "---",
        "entity_type: holding",
        f"symbol: {h.symbol}",
        f"asset_type: {h.asset_type}",
        f"currency: {currency}",
        f"units: {h.units}",
    ]
    if avg_cost is not None:
        lines.append(f"avg_cost: {avg_cost}")
    if current_price is not None:
        lines.append(f"current_price: {current_price}")
    lines.append(f"market_value_thb: {h.market_value_thb}")
    if h.unrealized_pnl_percent is not None:
        lines.append(f"unrealized_pnl_pct: {h.unrealized_pnl_percent}")
    if h.accumulated_dividend_thb is not None:
        lines.append(f"dividend_thb: {h.accumulated_dividend_thb}")
    lines.append("---")
    lines.append("")
    return "\n".join(lines)


def _sync_holding_sidecars(state: "PortfolioState") -> None:
    """Sync derived sidecar files ใน HOLDINGS_DIR จาก master PortfolioState
    เขียน 1 ไฟล์ต่อ holding (ข้าม Cash) — ลบ sidecar เก่าที่ holding ถูกลบแล้ว
    """
    HOLDINGS_DIR.mkdir(parents=True, exist_ok=True)
    live: set[str] = set()

    for h in state.holdings:
        if h.asset_type == "Cash":
            continue
        safe = h.symbol.replace("/", "_")
        _atomic_write_to(HOLDINGS_DIR / f"{safe}.md", _holding_to_md(h))
        live.add(safe)

    for old in HOLDINGS_DIR.glob("*.md"):
        if old.stem not in live:
            old.unlink(missing_ok=True)
            log.debug("[SIDECAR DEL] | holdings/%s", old.name)


def _recalc_holding(h: Holding, current_fx: float) -> None:
    """คำนวณ market_value_thb และ unrealized_pnl_percent ของ holding รายตัว

    fx_rate ใน Holding คือ FX ตอน trade (cost-basis); market_value ใช้ current_fx ของพอร์ต
    """
    if h.asset_type == "Cash":
        if h.symbol == CASH_USD_SYMBOL:
            h.market_value_thb = round(h.units * current_fx, _MONEY_DP)
        else:
            h.market_value_thb = round(h.units, _MONEY_DP)
        h.unrealized_pnl_percent = None
        h.accumulated_dividend_thb = None
        return

    if h.avg_cost_usd is not None and h.current_price_usd is not None:
        h.market_value_thb = round(h.units * h.current_price_usd * current_fx, _MONEY_DP)
        h.unrealized_pnl_percent = (
            round((h.current_price_usd - h.avg_cost_usd) / h.avg_cost_usd * 100, _PCT_DP)
            if h.avg_cost_usd
            else 0.0
        )
    elif h.avg_cost_thb is not None and h.current_price_thb is not None:
        h.market_value_thb = round(h.units * h.current_price_thb, _MONEY_DP)
        h.unrealized_pnl_percent = (
            round((h.current_price_thb - h.avg_cost_thb) / h.avg_cost_thb * 100, _PCT_DP)
            if h.avg_cost_thb
            else 0.0
        )
    else:
        log.warning(
            "Holding %s has incomplete cost/price pair — market value reset to 0", h.symbol
        )
        h.market_value_thb = 0.0
        h.unrealized_pnl_percent = None
        return


def _recalc_summary(state: PortfolioState, current_fx: float) -> None:
    """รวม total_value_thb และ total_unrealized_profit จาก holdings (ใช้ current_fx)"""
    total_value = 0.0
    total_unrealized = 0.0

    for h in state.holdings:
        total_value += h.market_value_thb
        if h.asset_type == "Cash":
            continue
        if h.avg_cost_usd is not None and h.current_price_usd is not None:
            total_unrealized += (h.current_price_usd - h.avg_cost_usd) * h.units * current_fx
        elif h.avg_cost_thb is not None and h.current_price_thb is not None:
            total_unrealized += (h.current_price_thb - h.avg_cost_thb) * h.units

    state.summary.total_value_thb = round(total_value, _MONEY_DP)
    state.summary.total_unrealized_profit = round(total_unrealized, _MONEY_DP)


def _recalc_all(state: PortfolioState) -> None:
    """Anti-Drift: คำนวณใหม่ทั้งหมดโดยใช้ fx_rates.USDTHB ปัจจุบันของพอร์ต"""
    current_fx = state.fx_rates.get("USDTHB", 0.0) or 0.0
    if current_fx <= 0:
        log.warning("fx_rates.USDTHB missing or invalid — USD holdings will compute as 0")
    for h in state.holdings:
        _recalc_holding(h, current_fx)
    _recalc_summary(state, current_fx)


def _yf_symbol(symbol: str, currency: str) -> str:
    """แปลง symbol → ticker ที่ yfinance รู้จัก (THB → เติม .BK)"""
    if currency == "THB" and not symbol.endswith(".BK"):
        return f"{symbol}.BK"
    return symbol


def _fetch_last_price(symbol: str) -> float | None:
    """ดึง last_price จาก yfinance — คืน None ถ้า fail"""
    try:
        tk = yf.Ticker(symbol)
        fi = tk.fast_info
        last = getattr(fi, "last_price", None)
        if last is not None:
            return float(last)
        hist = tk.history(period="1d")
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
    except Exception as e:
        log.warning("fetch price failed for %s: %s", symbol, e)
    return None


_USDTHB_TICKER = "USDTHB=X"


def _fetch_fx_rate() -> float | None:
    """ดึง USD/THB exchange rate ล่าสุดจาก yfinance (ticker USDTHB=X)
    Pattern เดียวกับ _fetch_last_price: fast_info ก่อน → history fallback
    คืน None ถ้าดึงไม่ได้ (log warning อัตโนมัติ)
    """
    return _fetch_last_price(_USDTHB_TICKER)


def fetch_latest_price(symbol: str, currency: Literal["THB", "USD"]) -> float | None:
    """Public helper: ดึงราคาล่าสุดจาก yfinance (แปลง symbol THB → .BK ให้)

    Args:
        symbol: ticker เช่น 'AAPL', 'PTT' (ห้ามมี .BK suffix สำหรับ THB — ระบบเติมให้)
        currency: 'THB' (เติม .BK) หรือ 'USD' (ใช้ symbol ตรงๆ) — *required*
                  ไม่มี default เพื่อกัน silent fail กับหุ้นไทยที่ต้องเติม .BK

    Returns:
        last_price (float) หรือ None ถ้าดึงไม่ได้ (log warning อัตโนมัติ)
    """
    if currency not in ("THB", "USD"):
        raise ValueError(f"currency ต้องเป็น 'THB' หรือ 'USD' (got '{currency}')")
    return _fetch_last_price(_yf_symbol(symbol, currency))


def _refresh_prices(state: PortfolioState) -> dict[str, str]:
    """Refresh current_price_* ของทุก holding ที่ไม่ใช่ Cash — best-effort

    Returns: dict ของ {symbol: status_msg} สำหรับ logging/observability
    """
    targets: list[tuple[Holding, str, str]] = []
    for h in state.holdings:
        if h.asset_type == "Cash":
            continue
        if h.avg_cost_usd is not None:
            targets.append((h, h.symbol, "USD"))
        elif h.avg_cost_thb is not None:
            targets.append((h, h.symbol, "THB"))

    if not targets:
        return {}

    results: dict[str, str] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(targets))) as ex:
        future_map = {
            ex.submit(_fetch_last_price, _yf_symbol(sym, cur)): (h, sym, cur)
            for h, sym, cur in targets
        }
        try:
            for future in concurrent.futures.as_completed(
                future_map, timeout=_PRICE_FETCH_TIMEOUT * 2
            ):
                h, sym, cur = future_map[future]
                try:
                    price = future.result()
                except Exception as e:
                    results[sym] = f"error: {e}"
                    continue
                if price is None:
                    results[sym] = "no_data"
                    continue
                if cur == "USD":
                    h.current_price_usd = price
                else:
                    h.current_price_thb = price
                results[sym] = "ok"
        except concurrent.futures.TimeoutError:
            for f, (h, sym, cur) in future_map.items():
                if not f.done():
                    results.setdefault(sym, "timeout")
    return results


_TOP_LEVEL_KEY_ORDER = (
    "doc_type",
    "last_updated",
    "base_currency",
    "summary",
    "fx_rates",
    "holdings",
)


def _save(post: frontmatter.Post, state: PortfolioState) -> None:
    """[CRITICAL] Atomic save พร้อม Anti-Drift recalculation
    เขียน temp file ก่อนเสมอ → os.replace() เป็น atomic operation
    """
    _recalc_all(state)
    state.last_updated = _now_iso()

    dump = state.model_dump(exclude_none=True)

    ordered: dict = {}
    for key in _TOP_LEVEL_KEY_ORDER:
        if key in dump:
            ordered[key] = dump.pop(key)
    # คงค่า extra fields ที่ผู้ใช้อาจเติมเองใน YAML ไว้ท้าย
    ordered.update(dump)

    post.metadata.clear()
    post.metadata.update(ordered)
    post.content = ""  # YAML-only contract — เนื้อหาอื่นห้ามปนใน Portfolio_Holdings.md

    serialized = frontmatter.dumps(post, sort_keys=False)
    _atomic_write(serialized)
    _sync_holding_sidecars(state)


def _find_holding(state: PortfolioState, symbol: str) -> Holding | None:
    return next((h for h in state.holdings if h.symbol == symbol), None)


def _require_cash(state: PortfolioState, currency: Literal["THB", "USD"] = "THB") -> Holding:
    """หา cash holding ตาม currency — lazy-create ถ้าไฟล์เก่ายังไม่มี CASH_USD"""
    sym = CASH_THB_SYMBOL if currency == "THB" else CASH_USD_SYMBOL
    cash = _find_holding(state, sym)
    if cash is None:
        cash = Holding(symbol=sym, asset_type="Cash", units=0.0, market_value_thb=0.0)
        state.holdings.append(cash)
    return cash


def _require_fx(state: PortfolioState) -> float:
    fx = state.fx_rates.get("USDTHB")
    if fx is None or fx <= 0:
        raise ValueError("ไม่พบ fx_rates.USDTHB ที่ valid ใน portfolio")
    return fx


@tool
def get_portfolio_state(refresh_prices: bool = True) -> str:
    """อ่านสถานะ portfolio ปัจจุบันคืนเป็น JSON string

    เมื่อ refresh_prices=True (ค่าเริ่มต้น) — ดึงราคาตลาดล่าสุดจาก yfinance ทุก holding
    (USD ใช้ ticker ตรง, THB เติม .BK) แล้ว recalc + persist ลงไฟล์ทันที
    เมื่อ refresh_prices=False — แค่ recalc จากราคาที่บันทึกไว้ (ใช้เมื่อไม่ต้องการ network call)

    Args:
        refresh_prices: True = ดึงราคาตลาดล่าสุดก่อน recalc (default), False = skip network

    Returns:
        JSON string ของ PortfolioState — มี field _price_refresh พิเศษบอกผลการดึงราคา
    """
    try:
        with _portfolio_lock:
            post, state = _load_or_init()
            refresh_info: dict[str, str] = {}
            if refresh_prices:
                refresh_info = _refresh_prices(state)
            if refresh_info:
                _save(post, state)
            else:
                _recalc_all(state)
    except Timeout:
        return json.dumps(
            {"error": f"portfolio lock timeout ({_LOCK_TIMEOUT}s) — another op in progress"},
            ensure_ascii=False,
        )

    dump = state.model_dump(exclude_none=True)
    if refresh_info:
        dump["_price_refresh"] = refresh_info
    return json.dumps(dump, ensure_ascii=False, indent=2)


@tool
def execute_trade(
    symbol: str,
    asset_type: str,
    action: Literal["buy", "sell"],
    units: float,
    price: float,
    currency: Literal["THB", "USD"] = "THB",
) -> str:
    """ดำเนินการเทรดซื้อหรือขายสินทรัพย์ พร้อมจัดการเงินสด weighted-avg cost และ realized P&L

    [Buy] เช็คเงินสด → หักจาก CASH_THB → ถ้า holding ใหม่ให้สร้างพร้อม fields ครบ
          ถ้ามีอยู่แล้วให้คำนวณ weighted-average cost ตามสกุลเงิน
    [Sell] เช็คว่าขายไม่เกินที่มี → คำนวณ realized profit (USD คูณ fx_rate ปัจจุบัน)
           บวกเข้า summary.total_realized_profit_ytd → คืนเงินเข้า CASH_THB
           ถ้า units เหลือ 0 จะลบ holding ออกจาก list

    Args:
        symbol: ticker หรือชื่อย่อสินทรัพย์ เช่น 'AAPL', 'PTT' (ห้ามใช้ CASH_THB)
        asset_type: ประเภทสินทรัพย์ เช่น 'Stock', 'ETF', 'REIT', 'Bond'
        action: 'buy' หรือ 'sell'
        units: จำนวนหน่วยที่เทรด (>0)
        price: ราคาต่อหน่วยในสกุลเงินของสินทรัพย์
        currency: 'THB' (ดีฟอลต์) หรือ 'USD' — ระบุสกุลเงินของ price/avg_cost ของสินทรัพย์นี้

    Raises:
        ValueError: ถ้าซื้อเกินเงินสด, ขายเกินที่มี, หรือ currency ไม่ตรงกับ holding เดิม
    """
    if symbol.strip().upper() in _CASH_SYMBOLS:
        raise ValueError(
            f"ห้ามเทรด cash sentinel ({'/'.join(_CASH_SYMBOLS)}) ผ่าน execute_trade "
            f"— ใช้ manage_cash_flow แทน"
        )
    if units <= 0:
        raise ValueError("units ต้องมากกว่า 0")
    if price <= 0:
        raise ValueError("price ต้องมากกว่า 0")
    if action not in ("buy", "sell"):
        raise ValueError("action ต้องเป็น 'buy' หรือ 'sell'")

    try:
        with _portfolio_lock:
            return _execute_trade_locked(symbol, asset_type, action, units, price, currency)
    except Timeout:
        raise ValueError(
            f"portfolio lock timeout ({_LOCK_TIMEOUT}s) — มี operation อื่นกำลังทำงาน"
        )


def _execute_trade_locked(
    symbol: str,
    asset_type: str,
    action: Literal["buy", "sell"],
    units: float,
    price: float,
    currency: Literal["THB", "USD"],
) -> str:
    post, state = _load_or_init()
    cash = _require_cash(state, currency)
    target = _find_holding(state, symbol)

    fx_rate = _require_fx(state) if currency == "USD" else None
    # ทุก trade เคลื่อนเงินสดในสกุลเงินของตัวเอง (USD trade → CASH_USD, THB trade → CASH_THB)
    amount_native = units * price

    if action == "buy":
        if cash.units + _FLOAT_EPS < amount_native:
            raise ValueError(
                f"Insufficient cash balance — มี {cash.units:,.2f} {currency} "
                f"ต้องใช้ {amount_native:,.2f} {currency}"
            )

        if target is None:
            new_h = Holding(
                symbol=symbol,
                asset_type=asset_type,
                units=units,
                accumulated_dividend_thb=0.0,
            )
            if currency == "USD":
                new_h.avg_cost_usd = round(price, _COST_DP)
                new_h.current_price_usd = price
                new_h.fx_rate = fx_rate
            else:
                new_h.avg_cost_thb = round(price, _COST_DP)
                new_h.current_price_thb = price
            state.holdings.append(new_h)
            target = new_h
            avg_cost_note = ""
        else:
            if currency == "USD":
                if target.avg_cost_usd is None:
                    raise ValueError(
                        f"{symbol} เป็นสินทรัพย์ THB อยู่แล้ว ไม่สามารถซื้อเพิ่มเป็น USD ได้"
                    )
                old_basis = target.units * target.avg_cost_usd
                new_basis = units * price
                target.units += units
                target.avg_cost_usd = round((old_basis + new_basis) / target.units, _COST_DP)
                target.current_price_usd = price
                # fx_rate ของ holding คือ historical cost-basis FX — ห้าม update ตอน weighted-avg buy
                # (ถ้าต้องการ true weighted-avg FX ต้อง refactor ที่ Holding model ทั้งหมด)
                avg_cost_note = f" (Avg cost updated to ${target.avg_cost_usd:.2f})"
            else:
                if target.avg_cost_thb is None:
                    raise ValueError(
                        f"{symbol} เป็นสินทรัพย์ USD อยู่แล้ว ไม่สามารถซื้อเพิ่มเป็น THB ได้"
                    )
                old_basis = target.units * target.avg_cost_thb
                new_basis = units * price
                target.units += units
                target.avg_cost_thb = round((old_basis + new_basis) / target.units, _COST_DP)
                target.current_price_thb = price
                avg_cost_note = f" (Avg cost updated to ฿{target.avg_cost_thb:.2f})"

        cash.units = round(cash.units - amount_native, _MONEY_DP)
        realized = 0.0
        action_tag = "[BUY]"
        cashflow_sign = "-"

    else:  # sell
        if target is None:
            raise ValueError(f"Insufficient units to sell — {symbol} ไม่มีในพอร์ต")
        if units > target.units + _FLOAT_EPS:
            raise ValueError(
                f"Insufficient units to sell — มี {symbol} {target.units} หน่วย แต่สั่งขาย {units} หน่วย"
            )

        if currency == "USD":
            if target.avg_cost_usd is None:
                raise ValueError(f"{symbol} cost เป็น THB ไม่สามารถขายด้วย USD ได้")
            realized = (price - target.avg_cost_usd) * units * fx_rate
            target.current_price_usd = price
            # ไม่ update target.fx_rate — เป็น historical cost-basis ของหน่วยที่เหลือ
        else:
            if target.avg_cost_thb is None:
                raise ValueError(f"{symbol} cost เป็น USD ไม่สามารถขายด้วย THB ได้")
            realized = (price - target.avg_cost_thb) * units
            target.current_price_thb = price

        target.units = round(target.units - units, _COST_DP)
        cash.units = round(cash.units + amount_native, _MONEY_DP)
        state.summary.total_realized_profit_ytd = round(
            state.summary.total_realized_profit_ytd + realized, _MONEY_DP
        )

        if target.units < _FLOAT_EPS:
            state.holdings.remove(target)

        avg_cost_note = ""
        action_tag = "[SELL]"
        cashflow_sign = "+"

    _save(post, state)

    cash_sym = CASH_USD_SYMBOL if currency == "USD" else CASH_THB_SYMBOL
    cash_after = _find_holding(state, cash_sym)
    cash_after_units = cash_after.units if cash_after else 0.0

    price_str = f"${price:.2f}" if currency == "USD" else f"฿{price:.2f}"
    parts = [
        f"{action_tag} {symbol} {units:g} units @ {price_str}{avg_cost_note}",
        f"กระแสเงินสด: {cashflow_sign}{amount_native:,.2f} {currency}",
    ]
    if action == "sell":
        parts.append(f"Realized P/L: {realized:+,.2f} THB")
    parts.append(f"{cash_sym} คงเหลือ: {cash_after_units:,.2f} {currency}")
    return " | ".join(parts)


@tool
def record_income(
    income_type: Literal["Dividend", "Interest", "Rental", "Other"],
    amount_thb: float,
    source_symbol: str | None = None,
) -> str:
    """บันทึกรายรับ passive (เงินปันผล/ดอกเบี้ย/ค่าเช่า/อื่นๆ) เข้า portfolio

    Effects:
        - บวกเงินเข้า CASH_THB.units
        - บวกเข้า summary.passive_income_ytd
        - บวกเข้า summary.total_accumulated_dividend
        - ถ้าระบุ source_symbol → บวกเข้า holdings[source].accumulated_dividend_thb ด้วย

    Args:
        income_type: ประเภทรายรับ — 'Dividend' / 'Interest' / 'Rental' / 'Other'
        amount_thb: จำนวนเงินใน THB (>0)
        source_symbol: ticker ของสินทรัพย์ที่จ่ายรายรับนี้ (ถ้ามี) เช่น 'PTT'
                       ห้ามใส่ CASH_THB

    Raises:
        ValueError: ถ้า amount ไม่ถูกต้อง, ไม่พบ source_symbol, หรือ source เป็น CASH_THB
    """
    if amount_thb <= 0:
        raise ValueError("amount_thb ต้องมากกว่า 0")

    try:
        with _portfolio_lock:
            return _record_income_locked(income_type, amount_thb, source_symbol)
    except Timeout:
        raise ValueError(
            f"portfolio lock timeout ({_LOCK_TIMEOUT}s) — มี operation อื่นกำลังทำงาน"
        )


def _record_income_locked(
    income_type: Literal["Dividend", "Interest", "Rental", "Other"],
    amount_thb: float,
    source_symbol: str | None,
) -> str:
    post, state = _load_or_init()
    cash = _require_cash(state)

    target: Holding | None = None
    if source_symbol:
        src_norm = source_symbol.strip().upper()
        if src_norm in _CASH_SYMBOLS:
            raise ValueError(f"source_symbol ห้ามเป็น cash sentinel ({'/'.join(_CASH_SYMBOLS)})")
        target = _find_holding(state, source_symbol)
        if target is None:
            raise ValueError(f"ไม่พบ {source_symbol} ใน portfolio")

    cash.units = round(cash.units + amount_thb, _MONEY_DP)
    state.summary.passive_income_ytd = round(
        state.summary.passive_income_ytd + amount_thb, _MONEY_DP
    )
    state.summary.total_accumulated_dividend = round(
        state.summary.total_accumulated_dividend + amount_thb, _MONEY_DP
    )

    if target is not None:
        current = target.accumulated_dividend_thb or 0.0
        target.accumulated_dividend_thb = round(current + amount_thb, _MONEY_DP)

    _save(post, state)

    tag = {"Dividend": "[DIV]", "Interest": "[INT]", "Rental": "[RENT]"}.get(income_type, "[INCOME]")
    src_note = f" จาก {source_symbol}" if source_symbol else ""
    return (
        f"{tag} +{amount_thb:,.2f} THB{src_note} | "
        f"passive_income_ytd: {state.summary.passive_income_ytd:,.2f} | "
        f"เงินสดคงเหลือ: {cash.units:,.2f} บาท"
    )


class WatchlistItem(BaseModel):
    model_config = ConfigDict(extra="allow")

    symbol: str
    asset_type: str
    target_price: float | None = None
    notes: str | None = None
    added_date: str


class WatchlistState(BaseModel):
    model_config = ConfigDict(extra="allow")

    doc_type: Literal["watchlist"] = "watchlist"
    last_updated: str
    items: list[WatchlistItem] = Field(default_factory=list)

    @field_validator("last_updated", mode="before")
    @classmethod
    def _validate_last_updated(cls, v):
        return _coerce_iso_string(v)


_WATCHLIST_KEY_ORDER = ("doc_type", "last_updated", "items")


def _atomic_write_watchlist(serialized: str) -> None:
    """Atomic write สำหรับ Watchlist.md — pattern เดียวกับ portfolio _atomic_write"""
    parent = WATCHLIST_PATH.parent
    parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".watchlist_", suffix=".md.tmp", dir=str(parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(serialized)
        os.replace(tmp_path, WATCHLIST_PATH)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _save_watchlist(post: frontmatter.Post, state: WatchlistState) -> None:
    state.last_updated = _now_iso()
    dump = state.model_dump(exclude_none=True)

    ordered: dict = {}
    for key in _WATCHLIST_KEY_ORDER:
        if key in dump:
            ordered[key] = dump.pop(key)
    ordered.update(dump)

    post.metadata.clear()
    post.metadata.update(ordered)
    post.content = ""

    _atomic_write_watchlist(frontmatter.dumps(post, sort_keys=False))
    _sync_watchlist_sidecars(state)


def _watchlist_item_to_md(item: WatchlistItem) -> str:
    """สร้าง YAML frontmatter สำหรับ sidecar file ของ WatchlistItem"""
    lines = [
        "---",
        "entity_type: watchlist_item",
        f"symbol: {item.symbol}",
        f"asset_type: {item.asset_type}",
        f"added_date: {item.added_date}",
    ]
    if item.target_price is not None:
        lines.append(f"target_price: {item.target_price}")
    if item.notes is not None:
        notes_escaped = item.notes.replace('"', '\\"')
        lines.append(f'notes: "{notes_escaped}"')
    lines.append("---")
    lines.append("")
    return "\n".join(lines)


def _sync_watchlist_sidecars(state: WatchlistState) -> None:
    """Sync derived sidecar files ใน WATCHLIST_ITEMS_DIR จาก master WatchlistState"""
    WATCHLIST_ITEMS_DIR.mkdir(parents=True, exist_ok=True)
    live: set[str] = set()

    for item in state.items:
        safe = item.symbol.replace("/", "_")
        _atomic_write_to(WATCHLIST_ITEMS_DIR / f"{safe}.md", _watchlist_item_to_md(item))
        live.add(safe)

    for old in WATCHLIST_ITEMS_DIR.glob("*.md"):
        if old.stem not in live:
            old.unlink(missing_ok=True)
            log.debug("[SIDECAR DEL] | watchlist/%s", old.name)


def _load_or_init_watchlist() -> tuple[frontmatter.Post, WatchlistState]:
    if not WATCHLIST_PATH.exists():
        WATCHLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
        post = frontmatter.Post(content="")
        state = WatchlistState(last_updated=_now_iso())
        _save_watchlist(post, state)
        return post, state

    with WATCHLIST_PATH.open("r", encoding="utf-8") as f:
        post = frontmatter.load(f)

    if not post.metadata:
        log.warning("Watchlist.md ไม่มี YAML frontmatter — บูตข้อมูลใหม่")
        state = WatchlistState(last_updated=_now_iso())
        _save_watchlist(post, state)
        return post, state

    state = WatchlistState.model_validate(post.metadata)
    return post, state


def _compute_total_cost(state: PortfolioState, current_fx: float) -> float:
    """รวมต้นทุนทุก holding ใน THB — ใช้ current_fx แปลง USD cost ให้สอดคล้องกับ market_value
    (Anti-Drift: NAV - Cost = Unrealized P/L แบบเป๊ะตามที่ _recalc_summary คำนวณ)

    *Pair-required guard*: นับ holding เฉพาะตอนมี cost+price ครบคู่
    (สอดคล้องกับ _recalc_summary ที่ skip holding incomplete เช่นกัน)
    """
    total = 0.0
    for h in state.holdings:
        if h.asset_type == "Cash":
            if h.symbol == CASH_USD_SYMBOL:
                total += h.units * current_fx
            else:
                total += h.units
            continue
        if h.avg_cost_usd is not None and h.current_price_usd is not None:
            total += h.units * h.avg_cost_usd * current_fx
        elif h.avg_cost_thb is not None and h.current_price_thb is not None:
            total += h.units * h.avg_cost_thb
    return round(total, _MONEY_DP)


@tool
def add_to_watchlist(
    symbol: str,
    asset_type: str,
    target_price: float | None = None,
    notes: str | None = None,
) -> str:
    """เพิ่มสินทรัพย์เข้า Watchlist — ถ้า symbol มีอยู่แล้วจะอัปเดต fields ทับ (idempotent upsert)

    Args:
        symbol: ticker เช่น 'NVDA', 'PTT' (auto upper + strip)
        asset_type: ประเภท เช่น 'Stock', 'ETF', 'REIT', 'Crypto'
        target_price: ราคาเป้าหมายที่ตั้งใจซื้อ (optional)
        notes: หมายเหตุ/เหตุผลที่จับตา (optional)

    Raises:
        ValueError: ถ้า symbol ว่าง หรือ target_price <= 0
    """
    sym = symbol.strip().upper()
    if not sym:
        raise ValueError("symbol ต้องไม่ว่าง")
    if target_price is not None and target_price <= 0:
        raise ValueError("target_price ต้องมากกว่า 0")

    try:
        with _watchlist_lock:
            post, state = _load_or_init_watchlist()
            today = datetime.now().strftime("%Y-%m-%d")

            existing_idx = next(
                (i for i, it in enumerate(state.items) if it.symbol == sym),
                None,
            )
            preserved_date = (
                state.items[existing_idx].added_date if existing_idx is not None else today
            )
            new_item = WatchlistItem(
                symbol=sym,
                asset_type=asset_type,
                target_price=target_price,
                notes=notes,
                added_date=preserved_date,
            )
            if existing_idx is not None:
                state.items[existing_idx] = new_item
                action = "[WATCH UPD]"
            else:
                state.items.append(new_item)
                action = "[WATCH ADD]"
            _save_watchlist(post, state)
            total_items = len(state.items)
    except Timeout:
        raise ValueError(f"watchlist lock timeout ({_LOCK_TIMEOUT}s) — มี operation อื่นทำงาน")

    tp_note = f" target ฿{target_price:.2f}" if target_price else ""
    return f"{action} {sym} ({asset_type}){tp_note} | total: {total_items}"


@tool
def remove_from_watchlist(symbol: str) -> str:
    """ลบสินทรัพย์ออกจาก Watchlist

    Args:
        symbol: ticker ที่ต้องการลบ

    Raises:
        ValueError: ถ้า symbol ว่าง หรือไม่พบใน watchlist
    """
    sym = symbol.strip().upper()
    if not sym:
        raise ValueError("symbol ต้องไม่ว่าง")

    try:
        with _watchlist_lock:
            post, state = _load_or_init_watchlist()
            existing = next((it for it in state.items if it.symbol == sym), None)
            if existing is None:
                raise ValueError(f"ไม่พบ {sym} ใน Watchlist")
            state.items.remove(existing)
            _save_watchlist(post, state)
            remaining = len(state.items)
    except Timeout:
        raise ValueError(f"watchlist lock timeout ({_LOCK_TIMEOUT}s) — มี operation อื่นทำงาน")

    return f"[WATCH DEL] {sym} | remaining: {remaining}"


@tool
def read_watchlist() -> str:
    """อ่าน Watchlist ทั้งหมดคืนเป็น JSON string

    เรียกเมื่อ user ถาม "ดู watchlist", "สินทรัพย์จับตา", "หุ้นที่ตั้ง target ไว้"

    Returns:
        JSON string: {n_items, last_updated, items:[{symbol, asset_type, target_price, notes, added_date}]}
    """
    try:
        with _watchlist_lock:
            _, state = _load_or_init_watchlist()
            dump = state.model_dump(exclude_none=True)
    except Timeout:
        return json.dumps(
            {"error": f"watchlist lock timeout ({_LOCK_TIMEOUT}s)"},
            ensure_ascii=False,
        )

    items = dump.get("items", [])
    return json.dumps(
        {
            "n_items": len(items),
            "last_updated": dump.get("last_updated"),
            "items": items,
        },
        ensure_ascii=False,
        indent=2,
    )


@tool
def batch_import_holdings(
    assets_list: list[dict],
    mode: Literal["overwrite", "merge"] = "merge",
    reset_cash_usd: bool = False,
) -> str:
    """Smart Paste Import: บันทึก holdings หลายรายการพร้อมกัน + auto-fetch ราคาตลาดให้ที่ขาด

    รับ list ของ asset dict แต่ละตัวต้องมี keys:
      symbol (str), asset_type (str), units (float), avg_cost (float), currency ('THB'|'USD')
      current_price (float, optional) — ถ้าไม่ส่งมา ระบบจะ parallel-fetch จาก yfinance ให้
                                         ถ้าดึงไม่ได้ใช้ avg_cost เป็น fallback (unrealized P/L = 0 ชั่วคราว)

    Mode:
      'merge' (default) — อัปเดต holding ที่มี symbol ตรงกัน, ที่ไม่มีให้เพิ่ม (เก็บ accumulated_dividend_thb ของเดิม)
      'overwrite' — ล้าง holdings ที่ไม่ใช่ Cash ทั้งหมดก่อน

    reset_cash_usd: True = ศูนย์ CASH_USD.units ด้วย (ใช้คู่กับ mode='overwrite' ตอน full portfolio reset)
                    False (default) = คง CASH_USD ไว้ (backward-compatible)

    Anti-Drift: หลัง mutate จะรัน _recalc_all bottom-up อัตโนมัติผ่าน _save (Market Value → Summary)
    Atomic Storage: บันทึก Portfolio_Holdings.md ผ่าน tempfile + os.replace

    Args:
        assets_list: รายการสินทรัพย์ที่จะ import (ห้ามมี CASH_THB/CASH_USD, ห้ามมี symbol ซ้ำใน list)
        mode: 'merge' หรือ 'overwrite'
        reset_cash_usd: True = ล้าง CASH_USD เป็น 0 ด้วย (สำหรับ full reset เท่านั้น)

    Raises:
        ValueError: ถ้า list ว่าง, validation ล้มเหลว, symbol ซ้ำ, หรือมี cash sentinel
    """
    if mode not in ("overwrite", "merge"):
        raise ValueError("mode ต้องเป็น 'overwrite' หรือ 'merge'")
    if not isinstance(assets_list, list):
        raise ValueError("assets_list ต้องเป็น list")
    if not assets_list and mode != "overwrite":
        raise ValueError("assets_list ต้องเป็น list ที่ไม่ว่าง")

    # ─── 1. Pre-validate + normalize + duplicate check (ไม่ต้องล็อก) ───
    normalized: list[dict] = []
    seen: set[str] = set()
    for i, a in enumerate(assets_list):
        if not isinstance(a, dict):
            raise ValueError(f"item ลำดับ {i} ไม่ใช่ dict")
        try:
            sym = str(a["symbol"]).strip().upper()
            asset_type = str(a["asset_type"]).strip()
            units = float(a["units"])
            avg_cost = float(a["avg_cost"])
            currency = str(a["currency"]).strip().upper()
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"item ลำดับ {i}: field ขาดหรือ format ผิด — {e}")

        if not sym:
            raise ValueError(f"item ลำดับ {i}: symbol ว่าง")
        if sym in _CASH_SYMBOLS:
            raise ValueError(
                f"ห้าม import cash sentinel ({'/'.join(_CASH_SYMBOLS)}) ผ่าน batch_import "
                f"— ใช้ manage_cash_flow แทน"
            )
        if units <= 0:
            raise ValueError(f"{sym}: units ต้องมากกว่า 0")
        if avg_cost <= 0:
            raise ValueError(f"{sym}: avg_cost ต้องมากกว่า 0")
        if currency not in ("THB", "USD"):
            raise ValueError(f"{sym}: currency ต้องเป็น 'THB' หรือ 'USD' (got '{currency}')")
        if sym in seen:
            raise ValueError(f"symbol '{sym}' ซ้ำใน assets_list")
        seen.add(sym)

        cp_raw = a.get("current_price")
        current_price: float | None = None
        if cp_raw is not None:
            try:
                current_price = float(cp_raw)
            except (TypeError, ValueError) as e:
                raise ValueError(f"{sym}: current_price format ผิด — {e}")
            if current_price <= 0:
                raise ValueError(f"{sym}: current_price ต้องมากกว่า 0 (หรือส่ง None เพื่อให้ระบบดึงให้)")

        normalized.append({
            "symbol": sym,
            "asset_type": asset_type,
            "units": units,
            "avg_cost": avg_cost,
            "currency": currency,
            "current_price": current_price,
            "_source": "provided" if current_price is not None else "pending",
        })

    # ─── 2. Parallel fetch ราคาที่ขาด (นอก lock เพื่อไม่บล็อก ops อื่น) ───
    to_fetch = [a for a in normalized if a["_source"] == "pending"]
    if to_fetch:
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(to_fetch))) as ex:
            future_map = {
                ex.submit(fetch_latest_price, a["symbol"], a["currency"]): a
                for a in to_fetch
            }
            try:
                for future in concurrent.futures.as_completed(
                    future_map, timeout=_PRICE_FETCH_TIMEOUT * 2
                ):
                    asset = future_map[future]
                    try:
                        price = future.result()
                    except Exception:
                        price = None
                    if price is not None:
                        asset["current_price"] = price
                        asset["_source"] = "fetched"
                    else:
                        asset["current_price"] = asset["avg_cost"]
                        asset["_source"] = "fallback_avg_cost"
            except concurrent.futures.TimeoutError:
                for f, asset in future_map.items():
                    if not f.done():
                        asset["current_price"] = asset["avg_cost"]
                        asset["_source"] = "fallback_timeout"

    # ─── 3. Acquire lock → mutate → save (atomic + recalc) ───
    try:
        with _portfolio_lock:
            post, state = _load_or_init()
            current_fx = _require_fx(state)

            if mode == "overwrite":
                state.holdings = [h for h in state.holdings if h.asset_type == "Cash"]
                if reset_cash_usd:
                    cash_usd = _find_holding(state, CASH_USD_SYMBOL)
                    if cash_usd is not None:
                        cash_usd.units = 0.0

            for asset in normalized:
                sym = asset["symbol"]
                existing = next((h for h in state.holdings if h.symbol == sym), None)
                preserved_div = (existing.accumulated_dividend_thb if existing else None) or 0.0

                new_h = Holding(
                    symbol=sym,
                    asset_type=asset["asset_type"],
                    units=round(asset["units"], _COST_DP),
                    accumulated_dividend_thb=preserved_div,
                )
                if asset["currency"] == "USD":
                    new_h.avg_cost_usd = round(asset["avg_cost"], _COST_DP)
                    new_h.current_price_usd = float(asset["current_price"])
                    new_h.fx_rate = current_fx
                else:
                    new_h.avg_cost_thb = round(asset["avg_cost"], _COST_DP)
                    new_h.current_price_thb = float(asset["current_price"])

                if existing is not None:
                    state.holdings[state.holdings.index(existing)] = new_h
                else:
                    state.holdings.append(new_h)

            _save(post, state)  # ← _recalc_all bottom-up + atomic write
            total_nav = state.summary.total_value_thb
            non_cash = sum(1 for h in state.holdings if h.asset_type != "Cash")
    except Timeout:
        raise ValueError(f"portfolio lock timeout ({_LOCK_TIMEOUT}s)")

    # ─── 4. รายงานผล (Channel B format §4.2) ───
    counts = {"provided": 0, "fetched": 0, "fallback_avg_cost": 0, "fallback_timeout": 0}
    for a in normalized:
        counts[a["_source"]] = counts.get(a["_source"], 0) + 1
    fallback_total = counts["fallback_avg_cost"] + counts["fallback_timeout"]

    return (
        f"[IMPORT {mode.upper()}] {len(normalized)} assets | "
        f"prices(provided={counts['provided']}, fetched={counts['fetched']}, "
        f"fallback={fallback_total}) | "
        f"holdings: {non_cash} non-cash | NAV: {total_nav:,.2f} THB"
    )


@tool
def record_performance_snapshot(refresh_prices: bool = True) -> str:
    """บันทึก snapshot สถานะพอร์ตวันนี้ลง Performance_Log.csv (append-only time-series)

    คอลัมน์: Date, Total_NAV, Total_Cost, Unrealized_PnL, Cash_Balance
    ใช้สำหรับติดตามการเติบโตของพอร์ตข้ามวัน — ควรเรียกวันละครั้ง (เช่นปลายวัน)

    Args:
        refresh_prices: True (default) = ดึงราคาตลาดล่าสุดก่อน snapshot
                        False = ใช้ราคาที่บันทึกไว้ (สำหรับ test/replay)
    """
    try:
        with _portfolio_lock:
            post, state = _load_or_init()
            if refresh_prices:
                _refresh_prices(state)
                _save(post, state)
            else:
                _recalc_all(state)

            current_fx = _require_fx(state)
            total_nav = state.summary.total_value_thb
            unrealized = state.summary.total_unrealized_profit
            total_cost = _compute_total_cost(state, current_fx)

            # Cash_Balance ต้องรวมทั้ง CASH_THB และ CASH_USD ใน THB equivalent
            # ไม่งั้น Cash_Balance + non_cash_assets ≠ Total_NAV (drift)
            cash_balance = round(
                sum(h.market_value_thb for h in state.holdings if h.asset_type == "Cash"),
                _MONEY_DP,
            )
    except Timeout:
        raise ValueError(f"portfolio lock timeout ({_LOCK_TIMEOUT}s) — มี operation อื่นทำงาน")

    today = datetime.now().strftime("%Y-%m-%d")
    row = [
        today,
        f"{total_nav:.2f}",
        f"{total_cost:.2f}",
        f"{unrealized:.2f}",
        f"{cash_balance:.2f}",
    ]

    PERFORMANCE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    needs_header = (
        not PERFORMANCE_LOG_PATH.exists() or PERFORMANCE_LOG_PATH.stat().st_size == 0
    )
    with PERFORMANCE_LOG_PATH.open("a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if needs_header:
            writer.writerow(_PERFORMANCE_LOG_HEADER)
        writer.writerow(row)

    return (
        f"[PERF] {today} | NAV: {total_nav:,.2f} | "
        f"Cost: {total_cost:,.2f} | PnL: {unrealized:+,.2f} | "
        f"Cash: {cash_balance:,.2f}"
    )


@tool
def sync_market_prices() -> str:
    """ดึงราคาตลาดล่าสุดของทุก holding (ยกเว้น Cash) จาก yfinance แล้วบันทึกลงไฟล์

    Anti-Drift: หลัง fetch ราคาเสร็จ จะรัน _recalc_all bottom-up อัตโนมัติผ่าน _save
    (Per-Asset market_value → Summary total_value_thb → Summary total_unrealized_profit)
    Atomic Storage: บันทึกผ่าน tempfile + os.replace (กัน partial-write)

    เรียกใช้เมื่อผู้ใช้ขอ "อัปเดตพอร์ต / refresh ราคา / sync ตลาด / ดึงราคาล่าสุด"

    Returns:
        prefix-token format — [SYNC] | refreshed N/M [issues...] | NAV before → after | unrealized: ±X
    """
    try:
        with _portfolio_lock:
            post, state = _load_or_init()
            nav_before = state.summary.total_value_thb
            refresh_info = _refresh_prices(state)
            _save(post, state)
            nav_after = state.summary.total_value_thb
            unrealized_after = state.summary.total_unrealized_profit
    except Timeout:
        raise ValueError(f"portfolio lock timeout ({_LOCK_TIMEOUT}s) — มี operation อื่นทำงาน")

    total = len(refresh_info)
    if total == 0:
        return f"[SYNC] | no non-cash holdings | NAV: {nav_after:,.2f} THB"

    ok_count = sum(1 for s in refresh_info.values() if s == "ok")
    issues = {sym: s for sym, s in refresh_info.items() if s != "ok"}
    issue_note = ""
    if issues:
        sample = ", ".join(f"{s}={st}" for s, st in list(issues.items())[:3])
        more = f" +{len(issues) - 3} more" if len(issues) > 3 else ""
        issue_note = f" [issues: {sample}{more}]"

    return (
        f"[SYNC] | refreshed {ok_count}/{total}{issue_note} | "
        f"NAV: {nav_before:,.2f} → {nav_after:,.2f} THB | "
        f"unrealized: {unrealized_after:+,.2f} THB"
    )


@tool
def manage_cash_flow(
    amount: float,
    action: Literal["deposit", "withdraw"],
    currency: Literal["THB", "USD"] = "THB",
) -> str:
    """ฝาก/ถอนเงินสดเข้า/ออก cash pot ของพอร์ต (CASH_THB หรือ CASH_USD)

    Effects:
        deposit  → +amount เข้า CASH_{currency} (เม็ดเงินใหม่เข้าพอร์ต)
        withdraw → -amount จาก CASH_{currency} (ต้องมีเงินพอ)

    Anti-Drift: ฐานเงินทุน (cost basis) คำนวณ bottom-up อัตโนมัติผ่าน _compute_total_cost
    — cash sentinel นับเป็น cost = units (THB) หรือ units × fx (USD)
    ดังนั้นการ mutate cash.units อย่างเดียวก็ทำให้ทั้ง NAV และ cost basis sync กัน
    Unrealized P/L ไม่เพี้ยน (เพราะ NAV เพิ่ม = cost เพิ่ม เท่ากัน)

    Args:
        amount: จำนวนเงิน (>0) ในสกุล currency
        action: 'deposit' หรือ 'withdraw'
        currency: 'THB' (default) หรือ 'USD'

    Raises:
        ValueError: amount < 0, action/currency invalid, withdraw เกิน cash ที่มี
    """
    if amount <= 0:
        raise ValueError("amount ต้องมากกว่า 0")
    if action not in ("deposit", "withdraw"):
        raise ValueError("action ต้องเป็น 'deposit' หรือ 'withdraw'")
    if currency not in ("THB", "USD"):
        raise ValueError(f"currency ต้องเป็น 'THB' หรือ 'USD' (got '{currency}')")

    try:
        with _portfolio_lock:
            return _manage_cash_flow_locked(amount, action, currency)
    except Timeout:
        raise ValueError(
            f"portfolio lock timeout ({_LOCK_TIMEOUT}s) — มี operation อื่นทำงาน"
        )


def _manage_cash_flow_locked(
    amount: float,
    action: Literal["deposit", "withdraw"],
    currency: Literal["THB", "USD"],
) -> str:
    post, state = _load_or_init()
    cash = _require_cash(state, currency)

    if action == "deposit":
        cash.units = round(cash.units + amount, _MONEY_DP)
        tag = "[DEPOSIT]"
        sign = "+"
    else:  # withdraw
        if cash.units + _FLOAT_EPS < amount:
            raise ValueError(
                f"Insufficient cash balance — มี {cash.units:,.2f} {currency} "
                f"ต้องถอน {amount:,.2f} {currency}"
            )
        cash.units = round(cash.units - amount, _MONEY_DP)
        tag = "[WITHDRAW]"
        sign = "-"

    _save(post, state)

    cash_sym = CASH_USD_SYMBOL if currency == "USD" else CASH_THB_SYMBOL
    return (
        f"{tag} {currency} | {sign}{amount:,.2f} {currency} | "
        f"{cash_sym} คงเหลือ: {cash.units:,.2f} {currency}"
    )


@tool
def update_fx_rate(rate: float | None = None) -> str:
    """อัปเดต USD/THB exchange rate ของพอร์ต — manual rate หรือ auto-fetch จาก yfinance

    เรียกเมื่อผู้ใช้บอกค่าเงินบาทเปลี่ยน หรือ "อัปเดต FX / refresh อัตราแลกเปลี่ยน"

    Effects:
        - mutate state.fx_rates['USDTHB']
        - _save → _recalc_all bottom-up: ทุก USD market_value, CASH_USD value,
          total NAV, unrealized P/L ถูกคำนวณใหม่ด้วย fx ใหม่ทันที (Anti-Drift §3.2)

    Args:
        rate: USDTHB rate ใหม่ (>0). ถ้า None → auto-fetch จาก yfinance (USDTHB=X)

    Raises:
        ValueError: rate <= 0, auto-fetch ไม่สำเร็จ, หรือ lock timeout

    Returns:
        prefix-token — [FX] | USDTHB: old → new (±X%) | NAV: before → after | unrealized: before → after
    """
    if rate is not None and rate <= 0:
        raise ValueError("rate ต้องมากกว่า 0")

    try:
        with _portfolio_lock:
            return _update_fx_rate_locked(rate)
    except Timeout:
        raise ValueError(
            f"portfolio lock timeout ({_LOCK_TIMEOUT}s) — มี operation อื่นทำงาน"
        )


def _update_fx_rate_locked(rate: float | None) -> str:
    post, state = _load_or_init()
    old_rate = state.fx_rates.get("USDTHB", 0.0) or 0.0
    nav_before = state.summary.total_value_thb
    unrealized_before = state.summary.total_unrealized_profit

    if rate is None:
        fetched = _fetch_fx_rate()
        if fetched is None or fetched <= 0:
            raise ValueError(
                f"auto-fetch FX ล้มเหลว ({_USDTHB_TICKER}) — ลองระบุ rate manual"
            )
        new_rate = fetched
        source = "yfinance"
    else:
        new_rate = float(rate)
        source = "manual"

    state.fx_rates["USDTHB"] = new_rate
    _save(post, state)

    nav_after = state.summary.total_value_thb
    unrealized_after = state.summary.total_unrealized_profit
    pct_change = ((new_rate - old_rate) / old_rate * 100) if old_rate > 0 else 0.0

    return (
        f"[FX {source}] | USDTHB: {old_rate:.4f} → {new_rate:.4f} ({pct_change:+.2f}%) | "
        f"NAV: {nav_before:,.2f} → {nav_after:,.2f} THB | "
        f"unrealized: {unrealized_before:+,.2f} → {unrealized_after:+,.2f} THB"
    )


@tool
def read_performance_history(days: int = 30) -> str:
    """อ่าน Performance_Log.csv ย้อนหลัง N วัน + คำนวณ metrics สำหรับรายงาน trend

    เรียกเมื่อ user ถาม "NAV ขึ้นกี่ % เดือนนี้", "ผลตอบแทนช่วงนี้", "drawdown สูงสุดเท่าไหร่",
    "พอร์ตเป็นยังไงช่วงที่ผ่านมา"

    Metrics ทุกตัวคำนวณ deterministically ใน Python — ห้าม agent คิดเลขเอง (No Mental Math §2.2)
        latest_nav: NAV ล่าสุดในช่วง
        first_nav: NAV แรกในช่วง (baseline)
        change_abs: latest - first (THB)
        change_pct: (latest - first) / first × 100
        max_nav / min_nav: peak / trough ในช่วง
        max_drawdown_pct: drop สูงสุดจาก peak ที่ผ่านมา (running peak vs current)
        n_observations: จำนวน snapshot ที่อ่านได้

    Args:
        days: จำนวนวันย้อนหลังที่จะอ่าน (>0, default 30) — tail rows จาก CSV

    Returns:
        JSON string ของ metrics + raw rows (Date, Total_NAV, Total_Cost, Unrealized_PnL, Cash_Balance)
    """
    if days <= 0:
        raise ValueError("days ต้องมากกว่า 0")

    if not PERFORMANCE_LOG_PATH.exists():
        return json.dumps(
            {"error": "ยังไม่มี Performance_Log.csv — ใช้ record_performance_snapshot ก่อน"},
            ensure_ascii=False,
        )

    with PERFORMANCE_LOG_PATH.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        all_rows = list(reader)

    if not all_rows:
        return json.dumps(
            {"error": "Performance_Log.csv ว่างเปล่า"},
            ensure_ascii=False,
        )

    rows = all_rows[-days:]

    navs = [float(r["Total_NAV"]) for r in rows]
    first_nav = navs[0]
    latest_nav = navs[-1]
    change_abs = latest_nav - first_nav
    change_pct = (change_abs / first_nav * 100) if first_nav > 0 else 0.0

    # Max drawdown: running peak จาก left → right, เทียบ current value
    peak = navs[0]
    max_dd = 0.0
    for nav in navs:
        if nav > peak:
            peak = nav
        dd = ((nav - peak) / peak * 100) if peak > 0 else 0.0
        if dd < max_dd:
            max_dd = dd

    return json.dumps(
        {
            "window_days": days,
            "n_observations": len(rows),
            "first_date": rows[0]["Date"],
            "latest_date": rows[-1]["Date"],
            "first_nav": round(first_nav, _MONEY_DP),
            "latest_nav": round(latest_nav, _MONEY_DP),
            "change_abs": round(change_abs, _MONEY_DP),
            "change_pct": round(change_pct, _PCT_DP),
            "max_nav": round(max(navs), _MONEY_DP),
            "min_nav": round(min(navs), _MONEY_DP),
            "max_drawdown_pct": round(max_dd, _PCT_DP),
            "rows": rows,
        },
        ensure_ascii=False,
        indent=2,
    )


def _holding_currency(h: Holding) -> str:
    """Derive currency tag ของ holding — Cash ใช้ symbol, asset อื่นใช้ avg_cost_* field ที่มี"""
    if h.symbol == CASH_USD_SYMBOL:
        return "USD"
    if h.symbol == CASH_THB_SYMBOL:
        return "THB"
    if h.avg_cost_usd is not None:
        return "USD"
    if h.avg_cost_thb is not None:
        return "THB"
    return "UNKNOWN"


@tool
def compute_allocation_breakdown(
    group_by: Literal["asset_type", "currency"] = "asset_type",
) -> str:
    """คำนวณสัดส่วน Asset Allocation ของพอร์ตเป็น % โดยจัดกลุ่มตามมิติที่เลือก

    เรียกเมื่อ user ถาม "หุ้น vs เงินสด % เท่าไหร่", "หุ้นนอก vs หุ้นไทย",
    "TECH/REIT/Bond คิดเป็นกี่ %", "allocation พอร์ต"

    Group dimensions:
        'asset_type' (default) → Stock, ETF, REIT, Bond, Cash, ฯลฯ (จาก field asset_type)
        'currency' → THB, USD (จาก symbol ของ cash + avg_cost field ของ holding)

    Anti-Drift: read-only — ไม่ mutate state. Recalc ก่อนคำนวณเพื่อให้ market_value_thb fresh

    Returns:
        JSON string: {group_by, total_nav_thb, breakdown: [{group, value_thb, pct, count}], generated_at}
        breakdown เรียงจาก value_thb มากไปน้อย
    """
    if group_by not in ("asset_type", "currency"):
        raise ValueError("group_by ต้องเป็น 'asset_type' หรือ 'currency'")

    try:
        with _portfolio_lock:
            post, state = _load_or_init()
            _recalc_all(state)
            total_nav = state.summary.total_value_thb

            buckets: dict[str, dict] = {}
            for h in state.holdings:
                key = h.asset_type if group_by == "asset_type" else _holding_currency(h)
                b = buckets.setdefault(key, {"value_thb": 0.0, "count": 0})
                b["value_thb"] += h.market_value_thb
                b["count"] += 1
    except Timeout:
        raise ValueError(f"portfolio lock timeout ({_LOCK_TIMEOUT}s) — มี operation อื่นทำงาน")

    breakdown = [
        {
            "group": k,
            "value_thb": round(v["value_thb"], _MONEY_DP),
            "pct": round((v["value_thb"] / total_nav * 100) if total_nav > 0 else 0.0, _PCT_DP),
            "count": v["count"],
        }
        for k, v in buckets.items()
    ]
    breakdown.sort(key=lambda x: x["value_thb"], reverse=True)

    return json.dumps(
        {
            "group_by": group_by,
            "total_nav_thb": round(total_nav, _MONEY_DP),
            "breakdown": breakdown,
            "generated_at": _now_iso(),
        },
        ensure_ascii=False,
        indent=2,
    )


_EDITABLE_HOLDING_FIELDS = ("units", "avg_cost", "accumulated_dividend_thb", "asset_type")


@tool
def edit_holding(
    symbol: str,
    units: float | None = None,
    avg_cost: float | None = None,
    accumulated_dividend_thb: float | None = None,
    asset_type: str | None = None,
    reason: str = "",
) -> str:
    """แก้ไขข้อมูล holding ที่บันทึกผิด (correction tool) — เช่น พิมพ์ผิด, ลืมหารโบนัสหุ้น

    เปิดให้แก้เฉพาะ fields ปลอดภัย — ไม่แตะ:
        - market_value_thb / unrealized_pnl_percent → computed (auto-recalc)
        - current_price_* → ใช้ sync_market_prices
        - fx_rate ของ holding → historical cost-basis (ห้ามเขียนทับ)
        - symbol → ใช้ ลบ + import ใหม่แทน
        - summary.total_realized_profit_ytd → historical, ห้าม rewrite (per-trade booking)

    avg_cost auto-route ไป avg_cost_usd หรือ avg_cost_thb ตามสกุลที่ holding เดิมใช้

    Auto-side-effect: append เหตุผลลง Trading_Journal เพื่อ audit trail

    Args:
        symbol: ticker ของ holding ที่จะแก้ (ห้ามเป็น CASH_THB / CASH_USD — ใช้ manage_cash_flow)
        units: จำนวนหน่วยใหม่ (>0)
        avg_cost: avg cost ใหม่ (>0) — ระบบเลือก THB/USD field ตาม holding เดิม
        accumulated_dividend_thb: ยอดสะสมปันผลใหม่ (>=0)
        asset_type: เปลี่ยนหมวด (Stock/ETF/REIT/Bond ฯลฯ)
        reason: เหตุผลที่ต้องแก้ (audit log) — ควรระบุชัดเสมอ

    Raises:
        ValueError: symbol ไม่พบ, symbol เป็น cash, ไม่ส่ง field ไหนมาแก้, validation ค่าไม่ถูกต้อง
    """
    sym = symbol.strip().upper()
    if sym in _CASH_SYMBOLS:
        raise ValueError(
            f"ห้ามแก้ {sym} ผ่าน edit_holding — ใช้ manage_cash_flow สำหรับ cash"
        )
    if all(v is None for v in (units, avg_cost, accumulated_dividend_thb, asset_type)):
        raise ValueError(f"ต้องระบุอย่างน้อย 1 field ที่จะแก้ ({', '.join(_EDITABLE_HOLDING_FIELDS)})")
    if units is not None and units <= 0:
        raise ValueError("units ต้องมากกว่า 0")
    if avg_cost is not None and avg_cost <= 0:
        raise ValueError("avg_cost ต้องมากกว่า 0")
    if accumulated_dividend_thb is not None and accumulated_dividend_thb < 0:
        raise ValueError("accumulated_dividend_thb ต้อง >= 0")

    try:
        with _portfolio_lock:
            return _edit_holding_locked(
                sym, units, avg_cost, accumulated_dividend_thb, asset_type, reason
            )
    except Timeout:
        raise ValueError(
            f"portfolio lock timeout ({_LOCK_TIMEOUT}s) — มี operation อื่นทำงาน"
        )


def _edit_holding_locked(
    symbol: str,
    units: float | None,
    avg_cost: float | None,
    accumulated_dividend_thb: float | None,
    asset_type: str | None,
    reason: str,
) -> str:
    post, state = _load_or_init()
    target = _find_holding(state, symbol)
    if target is None:
        raise ValueError(f"ไม่พบ {symbol} ใน portfolio")
    if target.asset_type == "Cash":
        raise ValueError(f"{symbol} เป็น Cash holding — ใช้ manage_cash_flow")

    nav_before = state.summary.total_value_thb
    changes: list[str] = []

    if units is not None and units != target.units:
        changes.append(f"units: {target.units:g} → {units:g}")
        target.units = round(units, _COST_DP)

    if avg_cost is not None:
        if target.avg_cost_usd is not None:
            if avg_cost != target.avg_cost_usd:
                changes.append(f"avg_cost_usd: ${target.avg_cost_usd:.4f} → ${avg_cost:.4f}")
                target.avg_cost_usd = round(avg_cost, _COST_DP)
        elif target.avg_cost_thb is not None:
            if avg_cost != target.avg_cost_thb:
                changes.append(f"avg_cost_thb: ฿{target.avg_cost_thb:.4f} → ฿{avg_cost:.4f}")
                target.avg_cost_thb = round(avg_cost, _COST_DP)
        else:
            raise ValueError(
                f"{symbol} ไม่มี avg_cost_thb/usd เดิม — สร้าง holding ใหม่ผ่าน execute_trade แทน"
            )

    if accumulated_dividend_thb is not None:
        current = target.accumulated_dividend_thb or 0.0
        if accumulated_dividend_thb != current:
            changes.append(
                f"accumulated_dividend_thb: {current:,.2f} → {accumulated_dividend_thb:,.2f}"
            )
            target.accumulated_dividend_thb = round(accumulated_dividend_thb, _MONEY_DP)

    if asset_type is not None:
        new_type = asset_type.strip()
        if new_type and new_type != target.asset_type:
            changes.append(f"asset_type: {target.asset_type} → {new_type}")
            target.asset_type = new_type

    if not changes:
        raise ValueError("ค่าใหม่เหมือนเดิมทั้งหมด — ไม่มีอะไรต้องแก้")

    _save(post, state)
    nav_after = state.summary.total_value_thb

    reason_text = reason.strip() or "(no reason given)"
    _write_journal_entry(
        f"**[EDIT {symbol}]** {' | '.join(changes)}\n\nReason: {reason_text}"
    )

    return (
        f"[EDIT {symbol}] | {' | '.join(changes)} | reason: {reason_text} | "
        f"NAV: {nav_before:,.2f} → {nav_after:,.2f} THB"
    )


# จับ title แบบ "**[ACTION SYMBOL ...]**" → กลุ่ม 2 คือ SYMBOL
# Cash actions (CASH_THB/CASH_USD) จะถูก skip ผ่าน post-filter เพราะไม่ใช่ entity
_TRADE_TITLE_RE = re.compile(r'(\*\*\[\w+\s+)([A-Z][\w.\-]*)([^\]]*\]\*\*)(?!\s*—\s*\[\[)')


def _inject_journal_wikilinks(content: str) -> str:
    """เติม ' — [[SYMBOL]]' หลัง trade title เพื่อเชื่อม Layer 3 → Layer 1 ใน Graph
    Skip cash holdings (CASH_THB/CASH_USD) — ไม่ใช่ entity ที่ต้องการ hub node
    """
    def _replace(m: re.Match) -> str:
        symbol = m.group(2)
        if symbol in _CASH_SYMBOLS:
            return m.group(0)
        return f"{m.group(1)}{symbol}{m.group(3)} — [[{symbol}]]"
    return _TRADE_TITLE_RE.sub(_replace, content)


def _write_journal_entry(content: str) -> str:
    """Append timestamped block ลง Trading_Journal.md — คืน timestamp ที่ใช้
    ใช้ได้ทั้งจาก @tool append_trading_journal และจากภายใน locked ops อื่น (เช่น edit_holding)
    """
    TRADING_JOURNAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    linked = _inject_journal_wikilinks(content)
    block = f"\n## [{timestamp}]\n\n{linked}\n"
    with TRADING_JOURNAL_PATH.open("a", encoding="utf-8") as f:
        f.write(block)
    return timestamp


# ─── Goals ──────────────────────────────────────────────────────────────────

GOALS_REL = os.getenv("GOALS_FILE", "20_Portfolio_Management/Goals/Goals.md")
GOALS_PATH = VAULT_PATH / GOALS_REL
GOALS_ITEMS_DIR = VAULT_PATH / "20_Portfolio_Management/Goals/Items"

_GOALS_LOCK_PATH = str(GOALS_PATH) + ".lock"
# Lock ordering: ต้อง acquire _portfolio_lock ก่อน _goals_lock เสมอ เพื่อป้องกัน deadlock
_goals_lock = FileLock(_GOALS_LOCK_PATH, timeout=_LOCK_TIMEOUT)


class GoalItem(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    goal_type: Literal["nav_target", "cash_target", "passive_income_ytd"]
    target_amount_thb: float
    deadline: str | None = None
    notes: str | None = None
    created_date: str


class GoalsState(BaseModel):
    model_config = ConfigDict(extra="allow")

    doc_type: Literal["goals"] = "goals"
    last_updated: str
    goals: list[GoalItem] = Field(default_factory=list)

    @field_validator("last_updated", mode="before")
    @classmethod
    def _validate_last_updated(cls, v):
        return _coerce_iso_string(v)


_GOALS_KEY_ORDER = ("doc_type", "last_updated", "goals")


def _atomic_write_goals(serialized: str) -> None:
    parent = GOALS_PATH.parent
    parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".goals_", suffix=".md.tmp", dir=str(parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(serialized)
        os.replace(tmp_path, GOALS_PATH)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _goal_item_to_md(goal: GoalItem) -> str:
    lines = [
        "---",
        "entity_type: goal",
        f"name: {goal.name}",
        f"goal_type: {goal.goal_type}",
        f"target_amount_thb: {goal.target_amount_thb}",
        f"created_date: {goal.created_date}",
    ]
    if goal.deadline is not None:
        lines.append(f"deadline: {goal.deadline}")
    if goal.notes is not None:
        notes_escaped = goal.notes.replace('"', '\\"')
        lines.append(f'notes: "{notes_escaped}"')
    lines.append("---")
    lines.append("")
    return "\n".join(lines)


def _sync_goals_sidecars(state: GoalsState) -> None:
    GOALS_ITEMS_DIR.mkdir(parents=True, exist_ok=True)
    live: set[str] = set()

    for goal in state.goals:
        safe = goal.name.replace("/", "_").replace(" ", "_")
        _atomic_write_to(GOALS_ITEMS_DIR / f"{safe}.md", _goal_item_to_md(goal))
        live.add(safe)

    for old in GOALS_ITEMS_DIR.glob("*.md"):
        if old.stem not in live:
            old.unlink(missing_ok=True)
            log.debug("[SIDECAR DEL] | goals/%s", old.name)


def _save_goals(post: frontmatter.Post, state: GoalsState) -> None:
    state.last_updated = _now_iso()
    dump = state.model_dump(exclude_none=True)

    ordered: dict = {}
    for key in _GOALS_KEY_ORDER:
        if key in dump:
            ordered[key] = dump.pop(key)
    ordered.update(dump)

    post.metadata.clear()
    post.metadata.update(ordered)
    post.content = ""

    _atomic_write_goals(frontmatter.dumps(post, sort_keys=False))
    _sync_goals_sidecars(state)


def _load_or_init_goals() -> tuple[frontmatter.Post, GoalsState]:
    if not GOALS_PATH.exists():
        GOALS_PATH.parent.mkdir(parents=True, exist_ok=True)
        post = frontmatter.Post(content="")
        state = GoalsState(last_updated=_now_iso())
        _save_goals(post, state)
        return post, state

    with GOALS_PATH.open("r", encoding="utf-8") as f:
        post = frontmatter.load(f)

    if not post.metadata:
        log.warning("Goals.md ไม่มี YAML frontmatter — บูตข้อมูลใหม่")
        state = GoalsState(last_updated=_now_iso())
        _save_goals(post, state)
        return post, state

    state = GoalsState.model_validate(post.metadata)
    return post, state


@tool
def set_goal(
    name: str,
    goal_type: Literal["nav_target", "cash_target", "passive_income_ytd"],
    target_amount_thb: float,
    deadline: str | None = None,
    notes: str | None = None,
) -> str:
    """บันทึก/อัปเดตเป้าหมายทางการเงิน (idempotent upsert ตาม name)

    goal_type:
        'nav_target'          — เป้าหมาย NAV รวมพอร์ต (summary.total_value_thb)
        'cash_target'         — เป้าหมายเงินสดสะสม (CASH_THB + CASH_USD×fx ในหน่วย THB)
        'passive_income_ytd'  — เป้าหมาย passive income รายปี
                                 (YTD counter รีเซ็ตทุก 1 มกราคม — intentional behavior)

    progress ไม่ถูก store ลงไฟล์ — คำนวณเฉพาะตอนเรียก get_goals_progress

    Args:
        name: ชื่อเป้าหมาย (ใช้เป็น key — ถ้าซ้ำจะ overwrite)
        goal_type: ประเภทเป้าหมาย (ดูด้านบน)
        target_amount_thb: ยอดเป้าหมายใน THB (>0)
        deadline: วันกำหนด format 'YYYY-MM-DD' (optional)
        notes: หมายเหตุ (optional)

    Raises:
        ValueError: name ว่าง, target_amount_thb <= 0, deadline format ผิด
    """
    nm = name.strip()
    if not nm:
        raise ValueError("name ต้องไม่ว่าง")
    if target_amount_thb <= 0:
        raise ValueError("target_amount_thb ต้องมากกว่า 0")
    if deadline is not None:
        try:
            datetime.strptime(deadline, "%Y-%m-%d")
        except ValueError:
            raise ValueError(f"deadline ต้องอยู่ในรูปแบบ 'YYYY-MM-DD' (got '{deadline}')")

    try:
        with _goals_lock:
            post, state = _load_or_init_goals()
            today = datetime.now().strftime("%Y-%m-%d")

            existing_idx = next(
                (i for i, g in enumerate(state.goals) if g.name == nm), None
            )
            preserved_date = (
                state.goals[existing_idx].created_date if existing_idx is not None else today
            )
            new_goal = GoalItem(
                name=nm,
                goal_type=goal_type,
                target_amount_thb=target_amount_thb,
                deadline=deadline,
                notes=notes,
                created_date=preserved_date,
            )
            if existing_idx is not None:
                state.goals[existing_idx] = new_goal
                action = "[GOAL UPD]"
            else:
                state.goals.append(new_goal)
                action = "[GOAL SET]"
            _save_goals(post, state)
            total = len(state.goals)
    except Timeout:
        raise ValueError(f"goals lock timeout ({_LOCK_TIMEOUT}s) — มี operation อื่นทำงาน")

    dl_note = f" | deadline: {deadline}" if deadline else ""
    return f"{action} {nm} ({goal_type}) target: {target_amount_thb:,.2f} THB{dl_note} | total: {total}"


@tool
def remove_goal(name: str) -> str:
    """ลบเป้าหมายทางการเงินออกจากระบบ

    Args:
        name: ชื่อเป้าหมายที่ต้องการลบ

    Raises:
        ValueError: name ว่าง หรือไม่พบเป้าหมาย
    """
    nm = name.strip()
    if not nm:
        raise ValueError("name ต้องไม่ว่าง")

    try:
        with _goals_lock:
            post, state = _load_or_init_goals()
            existing = next((g for g in state.goals if g.name == nm), None)
            if existing is None:
                raise ValueError(f"ไม่พบเป้าหมาย '{nm}'")
            state.goals.remove(existing)
            _save_goals(post, state)
            remaining = len(state.goals)
    except Timeout:
        raise ValueError(f"goals lock timeout ({_LOCK_TIMEOUT}s) — มี operation อื่นทำงาน")

    return f"[GOAL DEL] {nm} | remaining: {remaining}"


@tool
def get_goals_progress() -> str:
    """คำนวณ progress ของเป้าหมายทางการเงินทั้งหมด เทียบกับสถานะพอร์ตปัจจุบัน

    Progress คำนวณจาก portfolio state ณ ขณะเรียก — ไม่ถูก cache หรือ store

    goal_type progress:
        nav_target          → summary.total_value_thb / target × 100%
        cash_target         → (CASH_THB + CASH_USD×fx) / target × 100%
                              (combined liquidity in THB — same as Performance_Log.Cash_Balance)
        passive_income_ytd  → summary.passive_income_ytd / target × 100%
                              (YTD resets every Jan 1 — intentional for annual goal tracking)

    Lock ordering: _portfolio_lock acquired first, then _goals_lock (prevents deadlock)

    Returns:
        JSON string: {n_goals, goals: [{name, goal_type, target_amount_thb,
                      current_amount_thb, progress_pct, deadline, deadline_days_left, notes}],
                      generated_at}
    """
    try:
        # portfolio lock ก่อนเสมอ — consistent ordering ป้องกัน deadlock
        with _portfolio_lock:
            _, port_state = _load_or_init()
            _recalc_all(port_state)
            current_fx = port_state.fx_rates.get("USDTHB", 0.0) or 0.0
            cash_thb = next(
                (h.units for h in port_state.holdings if h.symbol == CASH_THB_SYMBOL), 0.0
            )
            cash_usd = next(
                (h.units for h in port_state.holdings if h.symbol == CASH_USD_SYMBOL), 0.0
            )
            total_cash_thb = round(cash_thb + cash_usd * current_fx, _MONEY_DP)
            nav = port_state.summary.total_value_thb
            passive_ytd = port_state.summary.passive_income_ytd

            with _goals_lock:
                _, goals_state = _load_or_init_goals()

    except Timeout:
        return json.dumps(
            {"error": f"lock timeout ({_LOCK_TIMEOUT}s) — มี operation อื่นทำงาน"},
            ensure_ascii=False,
        )

    now = datetime.now()
    results = []
    for g in goals_state.goals:
        if g.goal_type == "nav_target":
            current = nav
        elif g.goal_type == "cash_target":
            current = total_cash_thb
        else:  # passive_income_ytd
            current = passive_ytd

        pct = round(
            (current / g.target_amount_thb * 100) if g.target_amount_thb > 0 else 0.0,
            _PCT_DP,
        )

        entry: dict = {
            "name": g.name,
            "goal_type": g.goal_type,
            "target_amount_thb": g.target_amount_thb,
            "current_amount_thb": round(current, _MONEY_DP),
            "progress_pct": pct,
        }
        if g.deadline:
            try:
                dl = datetime.strptime(g.deadline, "%Y-%m-%d")
                entry["deadline"] = g.deadline
                entry["deadline_days_left"] = (dl - now).days
            except ValueError:
                entry["deadline"] = g.deadline
        if g.notes:
            entry["notes"] = g.notes
        results.append(entry)

    return json.dumps(
        {
            "n_goals": len(results),
            "goals": results,
            "generated_at": _now_iso(),
        },
        ensure_ascii=False,
        indent=2,
    )


@tool
def append_trading_journal(entry: str) -> str:
    """ต่อท้ายบันทึกการเทรดลง Trading_Journal.md พร้อม timestamp อัตโนมัติ

    ใช้บันทึก เหตุผลซื้อ/ขาย, การวิเคราะห์เชิงคุณภาพ, สภาพตลาด, learning, mistakes
    ที่ไม่สามารถ encode เป็น YAML structured state ได้ — เก็บแยกจาก Portfolio_Holdings.md
    เพื่อรักษา YAML-only contract ของ source-of-truth file

    Args:
        entry: เนื้อหาบันทึกในรูปแบบ Markdown ทั่วไป (ไม่ต้องใส่ timestamp เอง ระบบใส่ให้)

    Returns:
        บรรทัด confirmation พร้อม timestamp ที่บันทึก
    """
    content = (entry or "").strip()
    if not content:
        raise ValueError("entry ต้องไม่ว่าง")

    timestamp = _write_journal_entry(content)
    return f"[JOURNAL] บันทึกสำเร็จ | [{timestamp}] | {len(content)} chars"


# จับ block ที่ขึ้นด้วย '## [YYYY-MM-DD HH:MM:SS]' จนถึงก่อนหน้า heading ถัดไป หรือ EOF
_JOURNAL_BLOCK_RE = re.compile(
    r"^##\s+\[(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]\s*\n(?P<body>.*?)(?=\n##\s+\[\d{4}-\d{2}-\d{2}|\Z)",
    re.DOTALL | re.MULTILINE,
)


@tool
def read_trading_journal(
    days: int = 30,
    keyword: str | None = None,
    limit: int = 20,
) -> str:
    """อ่าน entries ใน Trading_Journal.md ย้อนหลัง + filter ตาม keyword

    เรียกเมื่อ user ถาม "ดูบันทึก", "ทบทวน mistake", "ทำไมซื้อ X", "บันทึกอะไรไว้บ้าง"

    Args:
        days: ดู entries ย้อนหลังกี่วัน (>0, default 30)
        keyword: ค้น substring case-insensitive ใน content (optional)
        limit: คืนสูงสุดกี่ entries (>0, default 20) — เรียงใหม่สุดก่อน

    Returns:
        JSON string: {n_total_in_window, n_returned, filters_used, entries:[{timestamp, content}]}
        entries เรียงจากใหม่สุดไปเก่าสุด
    """
    if days <= 0:
        raise ValueError("days ต้องมากกว่า 0")
    if limit <= 0:
        raise ValueError("limit ต้องมากกว่า 0")

    if not TRADING_JOURNAL_PATH.exists():
        return json.dumps(
            {"error": "ยังไม่มี Trading_Journal.md — ใช้ append_trading_journal บันทึกก่อน"},
            ensure_ascii=False,
        )

    text = TRADING_JOURNAL_PATH.read_text(encoding="utf-8")
    cutoff = datetime.now() - timedelta(days=days)
    kw = keyword.strip().lower() if keyword else None

    entries: list[dict] = []
    for m in _JOURNAL_BLOCK_RE.finditer(text):
        ts_str = m.group("ts")
        try:
            ts = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue
        if ts < cutoff:
            continue
        body = m.group("body").strip()
        if kw and kw not in body.lower():
            continue
        entries.append({"timestamp": ts_str, "content": body})

    # ใหม่สุดก่อน — ใช้ file order (append-only) ไม่ใช่ timestamp
    # เพราะ entries ที่เขียนติดกันใน 1 วินาทีจะ tie-break ไม่ได้ด้วย sort(timestamp)
    entries.reverse()
    n_total = len(entries)
    returned = entries[:limit]

    return json.dumps(
        {
            "n_total_in_window": n_total,
            "n_returned": len(returned),
            "filters_used": {"days": days, "keyword": keyword, "limit": limit},
            "entries": returned,
        },
        ensure_ascii=False,
        indent=2,
    )
