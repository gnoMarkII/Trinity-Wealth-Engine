import pytest
from decimal import Decimal
from tools.portfolio.adapters.arr_gold.arr_gold_parser_adapter import (
    ArrGoldParserAdapter,
    GOLD_SYMBOL,
    GOLD_ASSET_TYPE,
    GOLD_SOURCE,
    GOLD_CURRENCY,
)

SAMPLE_HTML_2026_BUY = """
<!DOCTYPE html>
<html>
<body>
<p>ยืนยันการทำรายการ [ซื้อทองคำ 99.99%] บริษัท ออโรร่า เทรดดิ้ง จำกัด ขอยืนยันการทำรายการของลูกค้าผ่านแอปพลิเคชั่นเป๋าตัง ดังนี้</p>
<p>รหัส/ชื่อสมาชิก 200202413 กวินภพ จิโน</p>
<p>วันที่ทำรายการ 2026-06-12 16:23:30 เลขที่อ้างอิง B26061208401</p>
<table>
  <tr><td>รายการ</td><td>ราคาซื้อหน่วยละ</td><td>น้ำหนักทองคำ</td><td>คิดเป็นจำนวนเงิน</td><td>น้ำหนักทองคงเหลือ</td></tr>
  <tr><td>จำนวน</td><td>4,221.53</td><td>+ 1</td><td>- 4,221.53</td><td>1.00.</td></tr>
  <tr><td>หน่วย</td><td>USD</td><td>Oz</td><td>USD</td><td>Oz</td></tr>
</table>
</body>
</html>
"""

SAMPLE_HTML_2026_SELL = """
<!DOCTYPE html>
<html>
<body>
<p>ยืนยันการทำรายการ [ขายทองคำ 99.99%] บริษัท ออโรร่า เทรดดิ้ง จำกัด ขอยืนยันการทำรายการของลูกค้าผ่านแอปพลิเคชั่นเป๋าตัง ดังนี้</p>
<p>รหัส/ชื่อสมาชิก 200202413 กวินภพ จิโน</p>
<p>วันที่ทำรายการ 2026-05-18 14:51:29 เลขที่อ้างอิง S26051805698</p>
<table>
  <tr><td>รายการ</td><td>ราคาขายหน่วยละ</td><td>น้ำหนักทองคำ</td><td>คิดเป็นจำนวนเงิน</td><td>ยอดคงเหลือ</td><td>น้ำหนักทองคงเหลือ</td></tr>
  <tr><td>จำนวน</td><td>4,540.27</td><td>- 1</td><td>+ 4,540.27</td><td>+ 4,540.27</td><td>0.00</td></tr>
  <tr><td>หน่วย</td><td>USD</td><td>Oz</td><td>USD</td><td>USD</td><td>Oz</td></tr>
</table>
</body>
</html>
"""

SAMPLE_HTML_2025_BUY = """
<!DOCTYPE html>
<html>
<body>
<div>ยืนยันการทำรายการ</div>
<div>[ซื้อทองคำ 99.99%]</div>
<div>บริษัท ออโรร่า เทรดดิ้ง จำกัด ขอยืนยันการทำรายการของลูกค้าผ่านแอปพลิเคชั่นเป๋าตัง ดังนี้</div>
<div>รหัส/ชื่อสมาชิก</div>
<div>วันที่ทำรายการ</div>
<div>200202413</div>
<div>2025-09-26 10:47:04</div>
<div>กวินภพ จิโน</div>
<div>เลขที่อ้างอิง:</div>
<div>B25092602371</div>
<div>ราคาซื้อหน่วยละ</div>
<div>3,745.75</div>
<div>USD</div>
<div>น้ำหนักทองคำ</div>
<div>+</div>
<div>1</div>
<div>(Oz)</div>
<div>คิดเป็นจำนวนเงิน</div>
<div>-</div>
<div>3,745.75</div>
<div>USD</div>
<div>น้ำหนักทองคงเหลือ</div>
<div>1.00</div>
<div>(Oz)</div>
</body>
</html>
"""

SAMPLE_HTML_2025_SELL_FRACTIONAL = """
<!DOCTYPE html>
<html>
<body>
<div>ยืนยันการทำรายการ</div>
<div>[ขายทองคำ 99.99%]</div>
<div>บริษัท ออโรร่า เทรดดิ้ง จำกัด ขอยืนยันการทำรายการของลูกค้าผ่านแอปพลิเคชั่นเป๋าตัง ดังนี้</div>
<div>รหัส/ชื่อสมาชิก</div>
<div>วันที่ทำรายการ</div>
<div>200202413</div>
<div>2025-09-26 10:10:10</div>
<div>กวินภพ จิโน</div>
<div>เลขที่อ้างอิง:</div>
<div>S25092601596</div>
<div>ราคาขายหน่วยละ</div>
<div>3,735.54</div>
<div>USD</div>
<div>น้ำหนักทองคำ</div>
<div>-</div>
<div>0.5</div>
<div>(Oz)</div>
<div>คิดเป็นจำนวนเงิน</div>
<div>+</div>
<div>1,867.77</div>
<div>USD</div>
<div>น้ำหนักทองคงเหลือ</div>
<div>0.00</div>
<div>(Oz)</div>
</body>
</html>
"""


def test_parse_2026_buy_gold():
    parser = ArrGoldParserAdapter()
    items = parser.parse_trade_html(SAMPLE_HTML_2026_BUY, "Buy Gold (B26061208401)")
    assert len(items) == 1
    it = items[0]
    assert it.action == "BUY"
    assert it.confirmation_no == "B26061208401"
    assert it.trade_date == "2026-06-12"
    assert it.units == Decimal("1.000000")
    assert it.price == Decimal("4221.5300")
    assert it.net_amount == Decimal("4221.53")
    assert it.currency == GOLD_CURRENCY
    assert it.symbol == GOLD_SYMBOL
    assert it.asset_type == GOLD_ASSET_TYPE
    assert it.source == GOLD_SOURCE
    assert it.cash_adjusted is False


def test_parse_2026_sell_gold():
    parser = ArrGoldParserAdapter()
    items = parser.parse_trade_html(SAMPLE_HTML_2026_SELL, "Sell Gold (S26051805698)")
    assert len(items) == 1
    it = items[0]
    assert it.action == "SELL"
    assert it.confirmation_no == "S26051805698"
    assert it.trade_date == "2026-05-18"
    assert it.units == Decimal("1.000000")
    assert it.price == Decimal("4540.2700")
    assert it.net_amount == Decimal("4540.27")
    assert it.currency == GOLD_CURRENCY


def test_parse_2025_buy_gold():
    parser = ArrGoldParserAdapter()
    items = parser.parse_trade_html(SAMPLE_HTML_2025_BUY, "Buy Gold")
    assert len(items) == 1
    it = items[0]
    assert it.action == "BUY"
    assert it.confirmation_no == "B25092602371"
    assert it.trade_date == "2025-09-26"
    assert it.units == Decimal("1.000000")
    assert it.price == Decimal("3745.7500")
    assert it.net_amount == Decimal("3745.75")


def test_parse_2025_sell_fractional_gold():
    parser = ArrGoldParserAdapter()
    items = parser.parse_trade_html(SAMPLE_HTML_2025_SELL_FRACTIONAL, "Sell Gold")
    assert len(items) == 1
    it = items[0]
    assert it.action == "SELL"
    assert it.confirmation_no == "S25092601596"
    assert it.trade_date == "2025-09-26"
    assert it.units == Decimal("0.500000")
    assert it.price == Decimal("3735.5400")
    assert it.net_amount == Decimal("1867.77")


def test_parse_empty_or_invalid():
    parser = ArrGoldParserAdapter()
    assert parser.parse_trade_html("", "") == []
    assert parser.parse_trade_html("<html><body>Hello World</body></html>", "Newsletter") == []
    assert parser.parse_daily_statement_pdf(b"not a pdf") == []


def test_unpack_units_and_amount():
    from tools.portfolio.adapters.arr_gold.arr_gold_parser_adapter import _unpack_units_and_amount

    # Fractional Sell 2025: 0.50 Oz and 1,867.77 USD
    u, a = _unpack_units_and_amount("0.501,867.77", Decimal("3735.54"))
    assert u == Decimal("0.50")
    assert a == Decimal("1867.77")

    # Buy 2025: 1.00 Oz and -3,745.75 USD
    u, a = _unpack_units_and_amount("1.00-3,745.75", Decimal("3745.75"))
    assert u == Decimal("1.00")
    assert a == Decimal("3745.75")

    # Sell 2026: 1.00 Oz and 4,540.27 USD
    u, a = _unpack_units_and_amount("1.004,540.27", Decimal("4540.27"))
    assert u == Decimal("1.00")
    assert a == Decimal("4540.27")

    # Buy 2026: 1.00 Oz and -4,221.53 USD
    u, a = _unpack_units_and_amount("1.00-4,221.53", Decimal("4221.53"))
    assert u == Decimal("1.00")
    assert a == Decimal("4221.53")

