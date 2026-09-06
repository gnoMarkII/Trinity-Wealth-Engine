"""Isolated Dime PDF Parser Driven Adapter (Hexagonal Architecture).

Executes pypdf in a dedicated child process with hard wall-clock timeout
to safeguard the primary server against malformed, quadratic-time, or malicious PDFs.
"""
import hashlib
import io
import multiprocessing
import os
import re
import uuid
from decimal import Decimal
from typing import List, Optional, Tuple, Dict, Any

import pypdf

from tools.portfolio.domain.calculations import (
    validate_reconciliation_invariant,
    allocate_document_fees_pro_rata,
)
from tools.portfolio.domain.errors import TradeReconciliationError
from tools.portfolio.domain.models import (
    TradeImportItem,
    TradeFeeBreakdown,
    quantize_decimal,
    MONEY_QUANTUM,
    PRICE_QUANTUM,
    UNITS_QUANTUM,
)
from tools.portfolio.ports.trade_ingestion_port import TradeDocumentParserPort

MAX_PDF_BYTES = 10 * 1024 * 1024  # 10 MB limit
PARSER_TIMEOUT_SECONDS = 5.0      # 5 seconds wall-clock limit


def _worker_parse_pdf(pdf_bytes: bytes, password: Optional[str], queue: Any) -> None:
    """Top-level worker function spawned in an isolated child process."""
    try:
        reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
        if reader.is_encrypted:
            if not password:
                queue.put({"success": False, "error": "ไฟล์ PDF นี้ถูกเข้ารหัส กรุณาระบุรหัสผ่าน (Password)"})
                return
            decrypt_res = reader.decrypt(password)
            if not decrypt_res:
                queue.put({"success": False, "error": "รหัสผ่านสำหรับเปิดไฟล์ PDF ไม่ถูกต้อง"})
                return

        extracted_text = ""
        extracted_text = ""
        for page in reader.pages:
            t = page.extract_text(extraction_mode="layout")
            if t:
                extracted_text += t + "\n"

        if not extracted_text.strip():
            queue.put({"success": False, "error": "ไม่สามารถอ่านข้อความจากไฟล์ PDF ได้ (ไฟล์อาจเป็นรูปภาพล้วนหรือเสียหาย)"})
            return

        # Check for corporate actions (fail-closed)
        corp_keywords = ["stock dividend", "spinoff", "spin-off", "merger", "reverse split", "rights issue"]
        lower_text = extracted_text.lower()
        for kw in corp_keywords:
            if kw in lower_text:
                queue.put({
                    "success": False,
                    "error": f"พบรายการ Corporate Action ({kw}) ซึ่งยังไม่รองรับการนำเข้าอัตโนมัติ",
                })
                return

        items_dict = _parse_dime_text_to_dicts(extracted_text)
        queue.put({"success": True, "items": items_dict})
    except Exception as e:
        queue.put({"success": False, "error": str(e)})


def _parse_dime_text_to_dicts(text: str) -> List[Dict[str, Any]]:
    """Parse extracted Dime text into dictionary rows.
    
    Supports:
    1. Dime US Stocks (Limited Margin Account confirmation note)
    2. Dime Mutual Funds (Mutual Fund confirmation note)
    3. Generic fallback for synthetic and legacy trade documents
    """
    # 1. Tax Invoice / Confirmation No.
    conf_no = ""
    tax_inv_match = re.search(r"(?:Tax\s*Invoice\s*No\.?|เลขที่ใบกำกับภาษี)\s*[:]?\s*([A-Za-z0-9\-_]+)", text, re.IGNORECASE)
    if tax_inv_match:
        conf_no = tax_inv_match.group(1).strip()
    else:
        doc_no_match = re.search(r"(?:Confirmation\s*No\.?|Doc\s*No\.?|No\.\s*|เลขที่ใบยืนยัน|เลขที่\s*)\s*[:]?\s*([A-Za-z0-9\-_]+)", text, re.IGNORECASE)
        if doc_no_match:
            conf_no = doc_no_match.group(1).strip()
        else:
            raise ValueError("ไม่พบเลขที่เอกสารยืนยัน (Confirmation No. / Tax Invoice No.) ในเอกสาร PDF")

    # 2. Dates
    eff_date = ""
    eff_match = re.search(r"(?:Effective\s*Date|วันที่คำสั่งมีผล)\s*[:]?\s*(\d{2}/\d{2}/\d{4})", text, re.IGNORECASE)
    if eff_match:
        d, m, y = eff_match.group(1).split("/")
        eff_date = f"{y}-{m}-{d}"

    issue_date = ""
    issue_match = re.search(r"(?:Issue\s*Date|วันที่ออกใบกำกับภาษี)\s*[:]?\s*(\d{2}/\d{2}/\d{4})", text, re.IGNORECASE)
    if issue_match:
        d, m, y = issue_match.group(1).split("/")
        issue_date = f"{y}-{m}-{d}"

    trade_date = eff_date or issue_date
    if not trade_date:
        raise ValueError("ไม่พบวันที่ทำรายการ (Trade Date / Effective Date / Issue Date) ในเอกสาร PDF")

    # Exchange Rate
    exchange_rate = None
    fx_match = re.search(r"(?:THB/USD\s*=|THB\s*=\s*)\s*([\d\.]+)", text, re.IGNORECASE)
    if fx_match:
        try:
            exchange_rate = str(Decimal(fx_match.group(1)))
        except Exception:
            pass
    if not exchange_rate:
        fx_alt_match = re.search(r"(?:BOT as of[^\n]*\n\s*)([\d\.]+)", text, re.IGNORECASE)
        if fx_alt_match:
            try:
                exchange_rate = str(Decimal(fx_alt_match.group(1)))
            except Exception:
                pass

    items: List[Dict[str, Any]] = []
    line_idx = 0

    # Pattern A: Dime US Stock Confirmation
    # Order ID | Settlement Date | BUY/SEL/SELL/REW | Symbol | Units | Price | Currency | Gross | Fee | Tax | Net
    us_stock_pattern = re.compile(
        r"^\s*(\d+)\s+(\d{2}/\d{2}/\d{4})\s+(BUY|SEL|SELL|REW)\s+([A-Za-z0-9\.\-_]+)\s+([\d\.,]+)\s+([\d\.,]+)\s+([A-Z]{3})\s+([\d\.,]+)\s+([\d\.,]+)\s+([\d\.,]+)\s+([\d\.,]+)",
        re.MULTILINE,
    )
    for m in us_stock_pattern.finditer(text):
        order_id = m.group(1).strip()
        raw_settle = m.group(2)
        d, mo, y = raw_settle.split("/")
        settlement_date = f"{y}-{mo}-{d}"

        raw_action = m.group(3).upper()
        action = "BUY" if raw_action in ("BUY", "REW") else "SELL"
        symbol = m.group(4).upper()
        units = Decimal(m.group(5).replace(",", ""))
        price = Decimal(m.group(6).replace(",", ""))
        curr = m.group(7).upper()
        gross = Decimal(m.group(8).replace(",", ""))
        fee = Decimal(m.group(9).replace(",", ""))
        tax = Decimal(m.group(10).replace(",", ""))
        net = Decimal(m.group(11).replace(",", ""))

        fp_raw = f"{conf_no}|{order_id}|{symbol}|{action}|{units}|{price}|{trade_date}|{net}"
        fingerprint = hashlib.sha256(fp_raw.encode("utf-8")).hexdigest()

        items.append({
            "item_id": f"item_{uuid.uuid4().hex[:8]}",
            "trade_date": trade_date,
            "settlement_date": settlement_date,
            "symbol": symbol,
            "action": action,
            "units": str(units),
            "price": str(price),
            "gross_amount": str(gross),
            "fees": {
                "commission": str(fee),
                "vat": "0.00",
                "other_fees": str(tax),
                "fee_currency": curr,
            },
            "net_amount": str(net),
            "currency": curr,
            "exchange_rate": exchange_rate,
            "confirmation_no": conf_no,
            "order_id": order_id,
            "source": "DIME",
            "fingerprint": fingerprint,
            "line_index": line_idx,
            "cash_adjusted": True,
            "asset_type": "Stock",
        })
        line_idx += 1

    # Pattern B: Dime Mutual Fund Confirmation
    # Order ID | SUB/RED/SWI/SWO | Fund Name (can have spaces/hyphens) | Units | NAV/Unit | Total Amount | Fee Include Vat
    mf_pattern = re.compile(
        r"^\s*(\d+)\s+(SUB|RED|SWI|SWO)\s+([A-Za-z0-9\-_\s/&().]+?)\s{2,}([\d\.,]+)\s+([\d\.,]+)\s+([\d\.,]+)\s+([\d\.,]+)",
        re.MULTILINE,
    )
    for m in mf_pattern.finditer(text):
        order_id = m.group(1).strip()
        raw_action = m.group(2).upper()
        action = "BUY" if raw_action in ("SUB", "SWI") else "SELL"
        symbol = m.group(3).strip().upper()
        units = Decimal(m.group(4).replace(",", ""))
        price = Decimal(m.group(5).replace(",", ""))
        total_amount = Decimal(m.group(6).replace(",", ""))
        curr = "THB"
        gross = total_amount
        net = total_amount

        fp_raw = f"{conf_no}|{order_id}|{symbol}|{action}|{units}|{price}|{trade_date}|{net}"
        fingerprint = hashlib.sha256(fp_raw.encode("utf-8")).hexdigest()

        items.append({
            "item_id": f"item_{uuid.uuid4().hex[:8]}",
            "trade_date": trade_date,
            "settlement_date": None,
            "symbol": symbol,
            "action": action,
            "units": str(units),
            "price": str(price),
            "gross_amount": str(gross),
            "fees": {
                "commission": "0.00",
                "vat": "0.00",
                "other_fees": "0.00",
                "fee_currency": curr,
            },
            "net_amount": str(net),
            "currency": curr,
            "exchange_rate": "1.0",
            "confirmation_no": conf_no,
            "order_id": order_id,
            "source": "DIME",
            "fingerprint": fingerprint,
            "line_index": line_idx,
            "cash_adjusted": True,
            "asset_type": "Fund",
        })
        line_idx += 1

    if not items:
        raise ValueError("ไม่พบตารางรายการซื้อขายที่รองรับในเอกสาร หรือรูปแบบตัวเลขไม่ถูกต้อง")

    return items


class IsolatedDimePdfParserAdapter(TradeDocumentParserPort):
    """Isolated child-process parser for Dime confirmation PDFs."""

    def __init__(self, timeout_seconds: float = PARSER_TIMEOUT_SECONDS):
        self.timeout_seconds = timeout_seconds

    def parse_confirmation_pdf(
        self, pdf_bytes: bytes, password: Optional[str] = None
    ) -> List[TradeImportItem]:
        # Fallback to DIME_PDF_PASSWORD if not provided explicitly
        effective_password = (password.strip() if password and password.strip() else None) or os.getenv("DIME_PDF_PASSWORD")

        # 1. Bounded size check
        if len(pdf_bytes) > MAX_PDF_BYTES:
            raise ValueError(f"ไฟล์ PDF มีขนาด {len(pdf_bytes) / 1024 / 1024:.1f}MB ซึ่งเกินขีดจำกัด 10MB")

        # 2. Magic byte check
        if not pdf_bytes.startswith(b"%PDF-"):
            raise ValueError("ไฟล์ไม่ใช่เอกสาร PDF ที่ถูกต้อง (Header '%PDF-' ไม่ถูกต้อง)")

        # 3. Spawn child process with bounded queue
        ctx = multiprocessing.get_context("spawn")
        queue = ctx.Queue()
        proc = ctx.Process(target=_worker_parse_pdf, args=(pdf_bytes, effective_password, queue))
        proc.start()

        # Wait with wall-clock timeout
        proc.join(timeout=self.timeout_seconds)

        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=0.5)
            if proc.is_alive():
                proc.kill()
            raise TimeoutError(f"การอ่านและประมวลผล PDF ใช้เวลานานเกินกำหนด ({self.timeout_seconds:.1f}s Timeout)")

        if queue.empty():
            raise RuntimeError("Child process สิ้นสุดการทำงานโดยไม่มีผลลัพธ์ส่งกลับ (Crash หรือถูก Kill)")

        res = queue.get()
        if not res.get("success"):
            err_msg = res.get("error", "ไม่สามารถประมวลผล PDF ได้")
            raise ValueError(err_msg)

        raw_items = res.get("items", [])
        parsed_items: List[TradeImportItem] = []

        for r in raw_items:
            order_id = r.get("order_id")
            if not order_id:
                raise ValueError(f"รายการ {r.get('symbol')} ขาด Order ID ไม่สามารถระบุตัวตนของธุรกรรมได้ (ห้ามใช้ fallback line_index สำหรับ Dime)")

            fees_dict = r.get("fees", {})
            fees = TradeFeeBreakdown(
                commission=Decimal(str(fees_dict.get("commission", "0.00"))),
                vat=Decimal(str(fees_dict.get("vat", "0.00"))),
                other_fees=Decimal(str(fees_dict.get("other_fees", "0.00"))),
                fee_currency=fees_dict.get("fee_currency", r.get("currency", "THB")),
            )
            item = TradeImportItem(
                item_id=r["item_id"],
                trade_date=r["trade_date"],
                settlement_date=r.get("settlement_date"),
                symbol=r["symbol"],
                action=r["action"],
                units=quantize_decimal(Decimal(str(r["units"])), UNITS_QUANTUM),
                price=quantize_decimal(Decimal(str(r["price"])), PRICE_QUANTUM),
                gross_amount=quantize_decimal(Decimal(str(r["gross_amount"])), MONEY_QUANTUM),
                fees=fees,
                net_amount=quantize_decimal(Decimal(str(r["net_amount"])), MONEY_QUANTUM),
                currency=r.get("currency", "THB"),
                exchange_rate=Decimal(str(r["exchange_rate"])) if r.get("exchange_rate") else None,
                confirmation_no=r["confirmation_no"],
                order_id=order_id,
                source="DIME",
                fingerprint=r["fingerprint"],
                line_index=int(r.get("line_index", 0)),
                cash_adjusted=bool(r.get("cash_adjusted", True)),
                asset_type=r.get("asset_type", "Stock"),
            )

            # Validate reconciliation invariant
            ok, msg = validate_reconciliation_invariant(
                units=item.units,
                price=item.price,
                gross_amount=item.gross_amount,
                fees=item.fees,
                net_amount=item.net_amount,
                action=item.action,
            )
            if not ok:
                raise TradeReconciliationError(f"รายการ {item.symbol} ไม่ผ่านเกณฑ์ Reconciliation: {msg}")

            parsed_items.append(item)

        return parsed_items
