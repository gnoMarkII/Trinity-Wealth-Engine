from typing import Literal, Optional
from langchain_core.tools import tool
import json

from tools.tool_errors import CASH_VIA_MANAGE, LOCK_TIMEOUT, validation_error
from tools.portfolio.domain.constants import _CASH_SYMBOLS


def _get_service():
    from tools.portfolio import get_default_service
    return get_default_service()

@tool
def get_portfolio_state(refresh_prices: bool = True, portfolio_id: str = 'default') -> str:
    """อ่านสถานะ Portfolio ปัจจุบันคืนเป็น JSON string (Read-only)

    [Usage/When to use]
    ใช้เมื่อต้องการสรุปภาพรวมพอร์ตโฟลิโอ สินทรัพย์ที่ถือครอง หรือคำนวณ NAV
    - ดึงราคาตลาดล่าสุดจาก yfinance อัตโนมัติ (ยกเว้นสั่ง refresh_prices=False)

    [Caution]
    - ไม่ทำการเปลี่ยนแปลงสถานะพอร์ต

    Args:
        refresh_prices (bool): True (ดึงราคาตลาดล่าสุด, default), False (ใช้ราคาเดิม)
        portfolio_id (str): พอร์ตการลงทุนที่ต้องการอ่าน (ค่าเริ่มต้น 'default')

    Returns:
        str: JSON string ของ PortfolioState พร้อมสถานะการอัปเดตราคา (_price_refresh)
    """
    return _get_service().get_portfolio_state(refresh_prices=refresh_prices, portfolio_id=portfolio_id)

@tool
def compute_allocation_breakdown(group_by: Literal['asset_type', 'currency'] = 'asset_type', portfolio_id: str = 'default') -> str:
    """Calculate portfolio Asset Allocation breakdown.

    Args:
        group_by: Dimension to group by ('asset_type' or 'currency'). Defaults to 'asset_type'.
        portfolio_id (str): พอร์ตการลงทุนที่ต้องการคำนวณ (ค่าเริ่มต้น 'default')

    Returns:
        JSON string: {group_by, total_nav_thb, breakdown: [{group, value_thb, pct, count}], generated_at}
    """
    if group_by not in ("asset_type", "currency"):
        return validation_error("group_by ต้องเป็น 'asset_type' หรือ 'currency'")
    return _get_service().compute_allocation_breakdown(group_by=group_by, portfolio_id=portfolio_id)

@tool
def tool_list_portfolios() -> str:
    """เรียกดูรายการพอร์ตการลงทุนทั้งหมดในระบบ

    [Usage/When to use]
    ใช้เพื่อดูว่ามีพอร์ตอะไรบ้าง และหา `portfolio_id` ที่ถูกต้องก่อนเรียก tool อื่นที่ต้องระบุพอร์ต
    - พอร์ตหลักเสมอมี id เป็น 'default'

    Returns:
        str: JSON list ของพอร์ตทั้งหมด (id, name, is_default, created_at)
    """
    try:
        items = _get_service().list_portfolios()
        return json.dumps([i.model_dump() for i in items], ensure_ascii=False, indent=2)
    except Exception as e:
        return f"Error: {e}"

@tool
def tool_create_portfolio(name: str, portfolio_id: str | None = None) -> str:
    """สร้างพอร์ตการลงทุนใหม่ เช่น พอร์ตเงินสำรองฉุกเฉิน หรือพอร์ตเกษียณ

    [Usage/When to use]
    ใช้เมื่อผู้ใช้ต้องการแยกเงิน/สินทรัพย์ออกเป็นอีกพอร์ตหนึ่งที่เป็นอิสระจากพอร์ตหลักโดยสิ้นเชิง

    [Caution]
    - ห้ามใช้ id 'default' (สงวนไว้สำหรับพอร์ตหลัก)
    - ถ้าไม่ระบุ portfolio_id ระบบจะสร้างให้อัตโนมัติจากชื่อ

    Args:
        name (str): ชื่อพอร์ตที่จะแสดงผล (ภาษาไทยได้)
        portfolio_id (str | None): รหัสพอร์ตที่ต้องการกำหนดเอง (ตัวอักษร/ตัวเลข/ขีดล่างเท่านั้น หากเป็น None จะสร้างอัตโนมัติ)
    """
    try:
        meta = _get_service().create_portfolio(name=name, portfolio_id=portfolio_id)
        return f"[PORT CREATE] {meta.name} (id: {meta.id})"
    except ValueError as e:
        return validation_error(str(e))
    except Exception as e:
        return f"Error: {e}"

@tool
def tool_delete_portfolio(portfolio_id: str) -> str:
    """ลบพอร์ตการลงทุนออกจากระบบ

    [Usage/When to use]
    ใช้เมื่อผู้ใช้ต้องการปิด/ลบพอร์ตที่ไม่ใช้แล้ว

    [Caution]
    - ลบไฟล์ทันทีถาวร ไม่มี undo — ไม่สามารถลบพอร์ตหลัก (default) ได้

    Args:
        portfolio_id (str): รหัสพอร์ตที่ต้องการลบ (ดูได้จาก tool_list_portfolios)
    """
    try:
        _get_service().delete_portfolio(portfolio_id=portfolio_id)
        return f"[PORT DEL] {portfolio_id}"
    except ValueError as e:
        return validation_error(str(e))
    except Exception as e:
        return f"Error: {e}"

@tool
def tool_rename_portfolio(portfolio_id: str, new_name: str) -> str:
    """เปลี่ยนชื่อพอร์ตการลงทุน

    [Usage/When to use]
    ใช้เมื่อผู้ใช้ต้องการเปลี่ยนชื่อพอร์ตที่มีอยู่

    Args:
        portfolio_id (str): รหัสพอร์ตที่ต้องการเปลี่ยนชื่อ (เช่น 'default' หรือ 'port_xxx')
        new_name (str): ชื่อใหม่ของพอร์ต
    """
    try:
        meta = _get_service().update_portfolio_name(portfolio_id=portfolio_id, name=new_name)
        return f"[PORT RENAME] {portfolio_id} -> {meta.name}"
    except ValueError as e:
        return validation_error(str(e))
    except Exception as e:
        return f"Error: {e}"

@tool
def execute_trade(symbol: str, asset_type: str, action: Literal['buy', 'sell'], units: float, price: float, currency: Literal['THB', 'USD'] = 'THB', notes: str = '', portfolio_id: str = 'default') -> str:
    """ดำเนินการเทรดซื้อหรือขายสินทรัพย์ พร้อมจัดการเงินสดและคำนวณต้นทุน/กำไรอัตโนมัติ

    [Usage/When to use]
    ใช้เมื่อผู้ใช้สั่งซื้อ (buy) หรือขาย (sell) สินทรัพย์
    - [Buy] หักเงินสด (CASH) อัตโนมัติและสร้าง/อัปเดต Holding พร้อมคำนวณ Weighted-average cost
    - [Sell] เพิ่มเงินสด (CASH) อัตโนมัติและคำนวณ Realized P&L
    - ห้ามใช้เพื่อเพิ่มสินทรัพย์แบบดื้อๆ โดยไม่หักเงิน ให้ใช้ `batch_import_holdings` ถ้าเป็นการย้ายพอร์ตมา

    [Caution]
    - ต้องมีเงินสด (CASH_THB/USD) เพียงพอสำหรับการซื้อ หากไม่พอ Trade จะถูกปฏิเสธ
    - การแก้ไขข้อผิดพลาดในการเทรดต้องใช้ `edit_holding` หรือถอนเงินเข้า/ออกผ่าน `manage_cash_flow`

    Args:
        symbol (str): Ticker ของสินทรัพย์ (เช่น 'AAPL', 'PTT')
        asset_type (str): ประเภทสินทรัพย์ (เช่น 'Stock', 'ETF', 'Bond')
        action (Literal["buy", "sell"]): ประเภทคำสั่ง
        units (float): จำนวนหน่วยที่ทำรายการ (>0)
        price (float): ราคาต่อหน่วย
        currency (Literal["THB", "USD"]): สกุลเงินที่ใช้เทรด (มีผลกับบัญชี Cash ที่หัก/รับ)
        notes (str): บันทึกเพิ่มเติมสำหรับรายการเทรดนี้
        portfolio_id (str): พอร์ตการลงทุนที่ต้องการเทรด (ค่าเริ่มต้น 'default')
    """
    if symbol.strip().upper() in _CASH_SYMBOLS:
        return CASH_VIA_MANAGE.format(symbols="/".join(_CASH_SYMBOLS))
    if units <= 0:
        return validation_error("units ต้องมากกว่า 0")
    if price <= 0:
        return validation_error("price ต้องมากกว่า 0")
    if action not in ("buy", "sell"):
        return validation_error("action ต้องเป็น 'buy' หรือ 'sell'")
    return _get_service().execute_trade(symbol=symbol, asset_type=asset_type, action=action, units=units, price=price, currency=currency, notes=notes, portfolio_id=portfolio_id)

@tool
def record_income(income_type: Literal['Dividend', 'Interest', 'Rental', 'Other'], amount_thb: float, source_symbol: str | None = None, portfolio_id: str = 'default') -> str:
    """บันทึกรายรับ Passive Income (เช่น เงินปันผล, ดอกเบี้ย)

    [Usage/When to use]
    ใช้เมื่อผู้ใช้ได้รับเงินปันผล หรือรายรับอื่นๆ ที่ไม่ต้องขายสินทรัพย์
    - ระบบจะบวกเงินสดเข้า CASH_THB อัตโนมัติ และอัปเดตสถิติ Passive Income YTD
    - หากระบุ `source_symbol` จะบันทึกเป็นเงินปันผลสะสมของสินทรัพย์นั้นๆ ด้วย

    Args:
        income_type (Literal["Dividend", "Interest", "Rental", "Other"]): ประเภทรายได้
        amount_thb (float): จำนวนเงิน (บาท)
        source_symbol (str | None): Ticker ต้นทางที่จ่ายปันผล (ถ้ามี)
        portfolio_id (str): พอร์ตการลงทุนที่ต้องการบันทึกรายรับ (ค่าเริ่มต้น 'default')
    """
    if amount_thb <= 0:
        return validation_error("amount_thb ต้องมากกว่า 0")
    return _get_service().record_income(income_type=income_type, amount_thb=amount_thb, source_symbol=source_symbol, portfolio_id=portfolio_id)

@tool
def batch_import_holdings(assets_list: list[dict], mode: Literal['overwrite', 'merge'] = 'merge', reset_cash_usd: bool = False, portfolio_id: str = 'default') -> str:
    """นำเข้า (Import) รายการสินทรัพย์หลายรายการพร้อมกัน

    [Usage/When to use]
    ใช้เมื่อผู้ใช้ต้องการย้ายพอร์ต หรือบันทึกสินทรัพย์เริ่มต้นโดยไม่ต้องผ่านการเทรดแบบหัก Cash
    - โหมด 'merge' จะอัปเดตสินทรัพย์เดิมและเพิ่มสินทรัพย์ใหม่
    - โหมด 'overwrite' จะล้างพอร์ตเดิม (ยกเว้นเงินสด) และทับด้วยข้อมูลใหม่

    [Caution]
    - การใช้โหมด 'overwrite' จะลบประวัติพอร์ตเก่า ต้องระวัง!

    Args:
        assets_list (list[dict]): รายการสินทรัพย์ (symbol, asset_type, units, avg_cost, currency)
        mode (Literal["overwrite", "merge"]): โหมดนำเข้า ค่าเริ่มต้นคือ 'merge'
        reset_cash_usd (bool): หาก True จะตั้งค่าเงินสด USD เป็น 0 ด้วย
        portfolio_id (str): พอร์ตการลงทุนที่ต้องการนำเข้า (ค่าเริ่มต้น 'default')
    """
    return _get_service().batch_import_holdings(assets_list=assets_list, mode=mode, reset_cash_usd=reset_cash_usd, portfolio_id=portfolio_id)

@tool
def manage_cash_flow(amount: float, action: Literal['deposit', 'withdraw'], currency: Literal['THB', 'USD'] = 'THB', portfolio_id: str = 'default') -> str:
    """ฝาก (Deposit) หรือ ถอน (Withdraw) เงินสดเข้า/ออกจากพอร์ตโฟลิโอ

    [Usage/When to use]
    ใช้เมื่อผู้ใช้เติมเงินเข้าพอร์ต (Deposit) หรือถอนเงินออกไปใช้จ่าย (Withdraw)
    - เปลี่ยนแปลงยอดเงินใน CASH_THB หรือ CASH_USD ทันที

    [Caution]
    - ไม่ใช่การเทรดสินทรัพย์ ใช้จัดการเฉพาะเงินสดที่รอลงทุนเท่านั้น

    Args:
        amount (float): จำนวนเงิน
        action (Literal["deposit", "withdraw"]): ฝากหรือถอน
        currency (Literal["THB", "USD"]): สกุลเงิน
        portfolio_id (str): พอร์ตการลงทุนที่ต้องการฝาก/ถอน (ค่าเริ่มต้น 'default')
    """
    if amount <= 0:
        return validation_error("amount ต้องมากกว่า 0")
    if action not in ("deposit", "withdraw"):
        return validation_error("action ต้องเป็น 'deposit' หรือ 'withdraw'")
    if currency not in ("THB", "USD"):
        return validation_error("currency ต้องเป็น 'THB' หรือ 'USD'")
    return _get_service().manage_cash_flow(amount=amount, action=action, currency=currency, portfolio_id=portfolio_id)

@tool
def update_fx_rate(rate: float | None = None, portfolio_id: str = 'default') -> str:
    """อัปเดตอัตราแลกเปลี่ยน USD/THB ของพอร์ต

    [Usage/When to use]
    ใช้เมื่อต้องการอัปเดตอัตราแลกเปลี่ยน (FX Rate) เพื่อให้มูลค่าพอร์ตที่เป็น USD ถูกคำนวณกลับมาเป็น THB อย่างแม่นยำ
    - ดึงข้อมูลจาก yfinance อัตโนมัติหากไม่ระบุ `rate`
    - ทำให้ Unrealized P/L และ NAV ถูกคำนวณใหม่ทั้งพอร์ตทันที

    Args:
        rate (float | None): อัตราแลกเปลี่ยนใหม่ที่กำหนดเอง (หากเป็น None จะดึงอัตโนมัติ)
        portfolio_id (str): พอร์ตการลงทุนที่ต้องการอัปเดต FX (ค่าเริ่มต้น 'default')
    """
    if rate is not None and rate <= 0:
        return validation_error("rate ต้องมากกว่า 0")
    return _get_service().update_fx_rate(rate=rate, portfolio_id=portfolio_id)

@tool
def edit_holding(symbol: str, units: float | None = None, avg_cost: float | None = None, accumulated_dividend_thb: float | None = None, asset_type: str | None = None, reason: str = '', portfolio_id: str = 'default') -> str:
    """แก้ไขข้อมูล Holding ที่บันทึกผิด (Correction Tool)

    [Usage/When to use]
    ใช้เมื่อผู้ใช้พิมพ์ผิด แจ้งต้นทุนผิด หรือต้องการแก้จำนวนหุ้นหลังจาก Corporate Action (เช่น แตกพาร์)
    - แก้ไขได้เฉพาะข้อมูลดิบ (units, avg_cost, ฯลฯ) ข้อมูลที่คำนวณจะอัปเดตอัตโนมัติ

    [Caution]
    - หลีกเลี่ยงการใช้คำสั่งนี้แทนการซื้อ/ขาย (`execute_trade`)

    Args:
        symbol (str): Ticker ที่ต้องการแก้ไข
        units (float | None): จำนวนหน่วยใหม่
        avg_cost (float | None): ต้นทุนเฉลี่ยใหม่
        accumulated_dividend_thb (float | None): เงินปันผลสะสมใหม่
        asset_type (str | None): ประเภทสินทรัพย์ใหม่
        reason (str): เหตุผลที่แก้ไข (เพื่อบันทึกลง Log)
        portfolio_id (str): พอร์ตการลงทุนที่ต้องการแก้ไข (ค่าเริ่มต้น 'default')
    """
    clean_sym = symbol.strip().upper()
    if clean_sym in _CASH_SYMBOLS:
        return validation_error("ห้ามแก้ไข CASH holding โดยตรง — ใช้ manage_cash_flow")
    if units is None and avg_cost is None and accumulated_dividend_thb is None and asset_type is None:
        return validation_error("ต้องระบุข้อมูลที่ต้องการแก้ไขอย่างน้อย 1 field")
    if units is not None and units <= 0:
        return validation_error("units ต้องมากกว่า 0")
    if avg_cost is not None and avg_cost <= 0:
        return validation_error("avg_cost ต้องมากกว่า 0")
    if accumulated_dividend_thb is not None and accumulated_dividend_thb < 0:
        return validation_error("accumulated_dividend_thb ต้องไม่ติดลบ (>= 0)")
    return _get_service().edit_holding(symbol=symbol, units=units, avg_cost=avg_cost, accumulated_dividend_thb=accumulated_dividend_thb, asset_type=asset_type, reason=reason, portfolio_id=portfolio_id)


@tool
def sync_market_prices(portfolio_id: str = "default") -> str:
    """ดึงราคาตลาดล่าสุดของทุกสินทรัพย์ในพอร์ตโฟลิโอ"""
    return _get_service().sync_market_prices(portfolio_id=portfolio_id)


@tool
def append_trading_journal(entry: str, portfolio_id: str = 'default') -> str:
    """บันทึกการเทรดและข้อคิดเห็น (Trading Journal)

    [Usage/When to use]
    ใช้จดบันทึกเหตุผลที่ซื้อ/ขาย สภาพตลาด บทเรียนที่ได้ หรือข้อผิดพลาด (Mistakes)
    - เพื่อแยกข้อมูล Qualitative (เหตุผล) ออกจากข้อมูล Quantitative (ตัวเลข Portfolio)

    [Caution]
    - ไม่ใช้เพื่อแก้ข้อมูลพอร์ต ให้ใช้บันทึกเป็น Text เท่านั้น

    Args:
        entry (str): เนื้อหาที่จะบันทึก (ระบบจะลง Timestamp ให้อัตโนมัติ)
        portfolio_id (str): พอร์ตการลงทุนที่ต้องการบันทึก (ค่าเริ่มต้น 'default')
    """
    content = (entry or "").strip()
    if not content:
        return validation_error("entry ต้องไม่ว่าง")
    return _get_service().append_trading_journal(entry=entry, portfolio_id=portfolio_id)

@tool
def read_trading_journal(days: int = 30, keyword: str | None = None, limit: int = 20, portfolio_id: str = 'default') -> str:
    """อ่านบันทึกการเทรด (Trading Journal) ย้อนหลัง

    [Usage/When to use]
    ใช้ดึงประวัติการบันทึกการลงทุน (Journal) เพื่อทบทวนเหตุผล ข้อคิด หรือสรุปบทเรียน
    - สามารถระบุ keyword เพื่อกรองเฉพาะบันทึกที่เกี่ยวข้องได้

    Args:
        days (int): จำนวนวันย้อนหลังที่ต้องการดึง
        keyword (str | None): คำที่ต้องการค้นหา
        limit (int): จำนวนบันทึกสูงสุดที่จะแสดง
        portfolio_id (str): พอร์ตการลงทุนที่ต้องการอ่าน (ค่าเริ่มต้น 'default')
    Returns:
        str: JSON string: {n_total_in_window, n_returned, filters_used, entries:[{timestamp, content}]}
        entries เรียงจากใหม่สุดไปเก่าสุด
    """
    return _get_service().read_trading_journal(days=days, keyword=keyword, limit=limit, portfolio_id=portfolio_id)

@tool
def add_to_watchlist(symbol: str, asset_type: str, target_price: float | None = None, notes: str | None = None, portfolio_id: str = 'default') -> str:
    """เพิ่มหรืออัปเดตสินทรัพย์ใน Watchlist

    [Usage/When to use]
    ใช้เมื่อต้องการจับตาสินทรัพย์ที่สนใจลงทุนในอนาคต พร้อมระบุราคาเป้าหมาย (Target Price)
    - สามารถอัปเดต Target Price สำหรับสินทรัพย์ที่มีอยู่แล้วได้ (Upsert)

    Args:
        symbol (str): Ticker ของสินทรัพย์
        asset_type (str): ประเภทสินทรัพย์
        target_price (float | None): ราคาที่ต้องการแจ้งเตือนเมื่อถึงเป้า
        notes (str | None): บันทึกเตือนความจำเพิ่มเติม
        portfolio_id (str): พอร์ตการลงทุนที่ต้องการเพิ่ม (ค่าเริ่มต้น 'default')
    """
    clean_sym = symbol.strip().upper()
    if not clean_sym:
        return validation_error("symbol ต้องไม่ว่าง")
    if target_price is not None and target_price <= 0:
        return validation_error("target_price ต้องมากกว่า 0")
    return _get_service().add_to_watchlist(symbol=symbol, asset_type=asset_type, target_price=target_price, notes=notes, portfolio_id=portfolio_id)

@tool
def remove_from_watchlist(symbol: str, portfolio_id: str = 'default') -> str:
    """ลบสินทรัพย์ออกจาก Watchlist

    [Usage/When to use]
    ใช้ลบสินทรัพย์ที่เลิกสนใจติดตามแล้ว

    Args:
        symbol (str): Ticker ที่ต้องการลบ
        portfolio_id (str): พอร์ตการลงทุนที่ต้องการลบ (ค่าเริ่มต้น 'default')
    """
    clean_sym = symbol.strip().upper()
    if not clean_sym:
        return validation_error("symbol ต้องไม่ว่าง")
    return _get_service().remove_from_watchlist(symbol=symbol, portfolio_id=portfolio_id)

@tool
def read_watchlist(portfolio_id: str = 'default') -> str:
    """อ่านรายการสินทรัพย์ที่อยู่ใน Watchlist

    [Usage/When to use]
    ใช้เพื่อดูรายการสินทรัพย์ที่จับตาดูอยู่และราคาเป้าหมาย

    Args:
        portfolio_id (str): พอร์ตการลงทุนที่ต้องการอ่าน (ค่าเริ่มต้น 'default')

    Returns:
        str: ข้อมูล Watchlist ในรูปแบบ JSON String
    """
    return _get_service().read_watchlist(portfolio_id=portfolio_id)

@tool
def set_goal(name: str, goal_type: Literal['nav_target', 'cash_target', 'passive_income_ytd', 'bucket_target'], target_amount_thb: float, deadline: str | None = None, years_from_now: int | None = None, notes: str | None = None, portfolio_id: str = 'default', bucket_id: str | None = None) -> str:
    """บันทึกหรืออัปเดตเป้าหมายทางการเงิน (Financial Goals)

    [Usage/When to use]
    ใช้ตั้งเป้าหมายทางการเงิน เช่น เป้าหมายมูลค่าพอร์ต (NAV), จำนวนเงินสดสำรอง, หรือรายได้ Passive Income

    [Caution]
    - ข้อมูลเป้าหมายจะถูกใช้เมื่อสั่งคำนวณ Progress

    Args:
        name (str): ชื่อเป้าหมาย
        goal_type (Literal["nav_target", "cash_target", "passive_income_ytd", "bucket_target"]): ประเภทเป้าหมาย
        target_amount_thb (float): จำนวนเป้าหมาย (บาท)
        deadline (str | None): กำหนดเวลา (ถ้ามี)
        years_from_now (int | None): จำนวนปีจากปัจจุบัน (ถ้ามี)
        notes (str | None): บันทึกเพิ่มเติม
        portfolio_id (str): พอร์ตการลงทุนที่ต้องการให้เป้าหมายนี้ติดตาม NAV/เงินสด (ค่าเริ่มต้น 'default')
        bucket_id (str | None): ระบุ Strategy Bucket ที่ต้องการติดตาม (สำหรับ goal_type='bucket_target')
    Returns:
        str: ข้อความยืนยันการบันทึกเป้าหมาย

    Raises:
        ValueError: name ว่าง, target_amount_thb <= 0, deadline format ผิด
    """
    clean_name = (name or "").strip()
    if not clean_name:
        return validation_error("name ต้องไม่ว่าง")
    if target_amount_thb <= 0:
        return validation_error("target_amount_thb ต้องมากกว่า 0")
    if goal_type not in ("nav_target", "cash_target", "passive_income_ytd", "bucket_target"):
        return validation_error("goal_type ไม่ถูกต้อง")
    return _get_service().set_goal(name=name, goal_type=goal_type, target_amount_thb=target_amount_thb, deadline=deadline, years_from_now=years_from_now, notes=notes, portfolio_id=portfolio_id, bucket_id=bucket_id)

@tool
def remove_goal(name: str) -> str:
    """ลบเป้าหมายทางการเงินออกจากระบบ

    [Usage/When to use]
    ใช้เมื่อต้องการยกเลิกหรือลบเป้าหมายที่ไม่ต้องการติดตามแล้ว

    Args:
        name (str): ชื่อเป้าหมายที่ต้องการลบ
    """
    clean_name = (name or "").strip()
    if not clean_name:
        return validation_error("name ต้องไม่ว่าง")
    return _get_service().remove_goal(name=name)

@tool
def get_goals_progress(portfolio_id: str | None = None) -> str:
    """เรียกดูความคืบหน้าของเป้าหมายทั้งหมด

    [Usage/When to use]
    ใช้เมื่อต้องการคำนวณ Progress (%) เทียบยอดเงินใน Portfolio กับเป้าหมายที่บันทึกไว้

    Args:
        portfolio_id (str | None): กรองเฉพาะเป้าหมายที่ผูกกับพอร์ตนี้ (None = ดูทุกพอร์ต)

    Returns:
        str: JSON string ประกอบด้วยสถานะของแต่ละเป้าหมาย
    """
    return _get_service().get_goals_progress(portfolio_id=portfolio_id)

@tool
def record_performance_snapshot(refresh_prices: bool = True, portfolio_id: str = 'default') -> str:
    """บันทึก Snapshot สถานะพอร์ตโฟลิโอ ณ สิ้นวัน (Performance Logging)

    [Usage/When to use]
    ใช้บันทึกประวัติการเติบโตของพอร์ตประจำวัน (Time-series) ลงใน CSV
    - บันทึก Date, Total_NAV, Total_Cost, Unrealized_PnL, Cash_Balance
    - ทำงานภายใต้ portfolio lock และแทนที่บรรทัดเดิมทันทีหากเป็นวันเดียวกัน (Atomic Upsert)

    Args:
        refresh_prices (bool): True (อัปเดตราคาล่าสุดก่อนบันทึก, default)
        portfolio_id (str): พอร์ตการลงทุนที่ต้องการบันทึก snapshot (ค่าเริ่มต้น 'default')
    """
    return _get_service().record_performance_snapshot(refresh_prices=refresh_prices, portfolio_id=portfolio_id)

@tool
def read_performance_history(days: int = 30, portfolio_id: str = 'default') -> str:
    """อ่านประวัติและวิเคราะห์ผลตอบแทนของพอร์ตโฟลิโอ (Performance Analytics)

    [Usage/When to use]
    ใช้เมื่อต้องการวิเคราะห์การเติบโต NAV, Drawdown, หรือผลตอบแทนย้อนหลัง
    - ระบบจะคำนวณ metrics เช่น P&L, Drawdown ให้อัตโนมัติ

    Args:
        days (int): จำนวนวันย้อนหลังที่ต้องการวิเคราะห์ (default 30)
        portfolio_id (str): พอร์ตการลงทุนที่ต้องการอ่าน (ค่าเริ่มต้น 'default')

    Returns:
        str: สรุปผลตอบแทนและ Metrics ในรูปแบบ Markdown
    """
    return _get_service().read_performance_history(days=days, portfolio_id=portfolio_id)
