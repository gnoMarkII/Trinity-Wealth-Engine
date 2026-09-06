"""Isolated WealthX PDF Parser Driven Adapter (Hexagonal Architecture).

Executes pypdf in a dedicated child process with hard wall-clock timeout
to safeguard the primary server against malformed, quadratic-time, or malicious PDFs.
Parses Trade Confirmation Notes (ใบยืนยันการซื้อขาย) from WealthX (noreply@wealthx.co).
"""
import hashlib
import io
import multiprocessing
import os
import re
import uuid
from decimal import Decimal
from typing import List, Optional, Dict, Any

import pypdf

from tools.portfolio.domain.calculations import validate_reconciliation_invariant
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


def _worker_parse_wealthx_pdf(pdf_bytes: bytes, password: Optional[str], queue: Any) -> None:
    """Top-level worker function spawned in an isolated child process."""
    try:
        reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
        if reader.is_encrypted:
            # Try provided password, or fallback to environment variable
            effective_password = password or os.getenv("WEALTHX_PDF_PASSWORD")
            if not effective_password:
                queue.put({"success": False, "error": "ไฟล์ PDF นี้ถูกเข้ารหัส กรุณาระบุรหัสผ่าน (Password) หรือตั้งค่า WEALTHX_PDF_PASSWORD ใน .env"})
                return
            decrypt_res = reader.decrypt(effective_password)
            if not decrypt_res:
                queue.put({"success": False, "error": "รหัสผ่านสำหรับเปิดไฟล์ PDF ของ WealthX ไม่ถูกต้อง"})
                return

        extracted_text = ""
        for page in reader.pages:
            t = page.extract_text(extraction_mode="layout") or page.extract_text()
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

        items_dict = _parse_wealthx_text_to_dicts(extracted_text)
        queue.put({"success": True, "items": items_dict})
    except Exception as e:
        queue.put({"success": False, "error": str(e)})


KNOWN_AMCS = [
    "TALISAM", "LHFUND", "SCBAM", "KASIKORN", "KTAM", "TMBAM", "ONEAM",
    "PRINCIPAL", "UOBAM", "DAOL", "XSPRING", "BBLAM", "ABERDEEN", "KSAM", "TISCO"
]


def _parse_wealthx_text_to_dicts(text: str) -> List[Dict[str, Any]]:
    """Parse extracted WealthX confirmation note text into dictionary rows.
    
    Supports:
    1. Table Format with multi-row confirmation (e.g. DN 2026xxxxxxx)
    2. Single Form Format (Confirmation Note with colon-separated key-values)
    """
    # 1. Settlement No. (Confirmation No.)
    conf_no = ""
    settle_match = re.search(
        r"\b(DN\s*\d{10,14})\b",
        text,
        re.IGNORECASE,
    )
    if settle_match:
        conf_no = re.sub(r"\s+", " ", settle_match.group(1).strip())
    else:
        doc_match = re.search(
            r"(?:เลขที่เอกสาร|เลขที่ใบยืนยัน|Settlement\s*No\.?|Confirmation\s*No\.?)[^:\n]*:\s*([A-Za-z0-9\-_]+(?:\s+[A-Za-z0-9\-_]+)*)",
            text,
            re.IGNORECASE,
        )
        if doc_match:
            conf_no = re.sub(r"\s+", " ", doc_match.group(1).strip())
        else:
            raise ValueError("ไม่พบเลขที่ใบยืนยัน (Settlement No.) ในเอกสาร PDF ของ WealthX")

    # 2. Dates
    trade_date = ""
    settlement_date = None
    dates = re.findall(r"\b(\d{2}/\d{2}/\d{4})\b", text)
    if len(dates) >= 2:
        d1, m1, y1 = dates[0].split("/")
        d2, m2, y2 = dates[1].split("/")
        trade_date = f"{y1}-{m1}-{d1}"
        settlement_date = f"{y2}-{m2}-{d2}"
    elif len(dates) == 1:
        d1, m1, y1 = dates[0].split("/")
        trade_date = f"{y1}-{m1}-{d1}"
    else:
        trade_match = re.search(r"(?:วันที่ส่งคำ[า]?สั่ง|Trade\s*Date)[^\d\n]*(\d{2}/\d{2}/\d{4})", text, re.IGNORECASE)
        if trade_match:
            d, m, y = trade_match.group(1).split("/")
            trade_date = f"{y}-{m}-{d}"
        else:
            raise ValueError("ไม่พบวันที่ส่งคำสั่ง (Trade Date) ในเอกสาร PDF ของ WealthX")

    # 3. Try Table Parsing (Multi-row line-based chunking)
    lines = text.split("\n")
    num_pattern = re.compile(r"([\d,]+\.\d{4})\s+([\d,]+\.\d{4})\s+([\d\.,]+)\s+([\d\.,]+)")
    
    row_chunks = []
    current_chunk = []
    in_table = False
    
    for line in lines:
        if num_pattern.search(line):
            if current_chunk:
                row_chunks.append(current_chunk)
            current_chunk = [line]
            in_table = True
        elif in_table:
            if any(k in line for k in ["รวมมูลค่า", "Total Buy", "Total Sell", "Total Fee"]):
                if current_chunk:
                    row_chunks.append(current_chunk)
                    current_chunk = []
                break
            else:
                current_chunk.append(line)
                
    if current_chunk:
        row_chunks.append(current_chunk)
        
    items = []
    curr = "THB"
    
    if row_chunks:
        for line_idx, chunk in enumerate(row_chunks):
            header_line = chunk[0]
            m = num_pattern.search(header_line)
            units = Decimal(m.group(1).replace(",", ""))
            price = Decimal(m.group(2).replace(",", ""))
            fee = Decimal(m.group(3).replace(",", ""))
            net_amount = Decimal(m.group(4).replace(",", ""))
            gross_amount = net_amount
            
            full_chunk_text = "\n".join(chunk)
            
            # Action
            action = "BUY"
            if any(w in full_chunk_text for w in ["SELL", "RED", "ขาย"]):
                action = "SELL"
                
            # Order ID (Reference No.) - 16 digits starting with 239 or fallback
            order_id = ""
            tokens = full_chunk_text.split()
            for idx, tok in enumerate(tokens):
                tok_clean = re.sub(r"\D", "", tok)
                if tok_clean.startswith("239"):
                    curr_id = tok_clean
                    j = idx + 1
                    while len(curr_id) < 16 and j < len(tokens):
                        next_clean = re.sub(r"\D", "", tokens[j])
                        if next_clean:
                            curr_id += next_clean
                        j += 1
                    if len(curr_id) >= 16:
                        order_id = curr_id[:16]
                        break
                    elif len(curr_id) >= 10:
                        order_id = curr_id
                        break
                        
            if not order_id:
                digits_m = re.search(r"\b(\d{14,18})\b", full_chunk_text)
                if digits_m:
                    order_id = digits_m.group(1)
                else:
                    order_id = f"{conf_no}_{line_idx}"

            # Symbol reconstruction
            amc_pattern = r"(" + "|".join(KNOWN_AMCS) + r")"
            amc_match = re.search(amc_pattern, header_line, re.IGNORECASE)
            sym_part1 = ""
            if amc_match:
                before_amc = header_line[:amc_match.start()].strip()
                sym_part1 = before_amc.split()[-1] if before_amc.split() else ""
            else:
                act_match = re.search(r"\b(BUY|SELL|SUB|RED)\b", header_line)
                if act_match:
                    before_act = header_line[:act_match.start()].strip()
                    sym_part1 = before_act.split()[-1] if before_act.split() else ""

            sym_part2 = ""
            for subsequent_line in chunk[1:]:
                line_tokens = subsequent_line.split()
                if line_tokens:
                    cand = line_tokens[0]
                    if any(c.isalpha() for c in cand) and not any(k in cand for k in KNOWN_AMCS + ["BUY", "SELL", "SUB"]):
                        sym_part2 += cand

            symbol = (sym_part1 + sym_part2).strip()
            if not symbol or len(symbol) < 3:
                broad_sym = re.search(r"\b(TLWORLD-X|TLNDQINCOME-UH-X|LHGRID|[A-Z0-9\-]{4,20})\b", full_chunk_text)
                symbol = broad_sym.group(1) if broad_sym else "UNKNOWN"

            fp_raw = f"{conf_no}|{order_id}|{symbol}|{action}|{units}|{price}|{trade_date}|{net_amount}"
            fingerprint = hashlib.sha256(fp_raw.encode("utf-8")).hexdigest()

            items.append({
                "item_id": f"item_{uuid.uuid4().hex[:8]}",
                "trade_date": trade_date,
                "settlement_date": settlement_date,
                "symbol": symbol,
                "action": action,
                "units": str(units),
                "price": str(price),
                "gross_amount": str(gross_amount),
                "fees": {
                    "commission": "0.00",
                    "vat": "0.00",
                    "other_fees": "0.00",
                    "fee_currency": curr,
                },
                "net_amount": str(net_amount),
                "currency": curr,
                "exchange_rate": "1.0",
                "confirmation_no": conf_no,
                "order_id": order_id,
                "source": "WEALTHX",
                "fingerprint": fingerprint,
                "line_index": line_idx,
                "cash_adjusted": True,
                "asset_type": "Fund",
            })
        return items

    # Fallback for Form Format (Single trade per PDF with explicit key-value fields)
    action = "BUY"
    if re.search(r"\b(SELL|RED|ขาย|ขายคืน)\b", text, re.IGNORECASE):
        action = "SELL"

    units = None
    price = None
    num_match = re.search(r"\b([\d,]+\.\d{4})\s+([\d,]+\.\d{4})\b", text)
    if num_match:
        units = Decimal(num_match.group(1).replace(",", ""))
        price = Decimal(num_match.group(2).replace(",", ""))
    else:
        units_match = re.search(r"(?:จำ[า]?นวนหน่วย|Allocated\s*Units|No\.\s*of\s*Units)[^:\d\n]*[:]?\s*([\d\.,]+)", text, re.IGNORECASE)
        price_match = re.search(r"(?:ราคา/หน่วย|ราคาต่อหน่วย|Unit\s*Price|Price\s*per\s*Unit|NAV)[^:\d\n]*[:]?\s*([\d\.,]+)", text, re.IGNORECASE)
        if units_match and price_match:
            units = Decimal(units_match.group(1).replace(",", ""))
            price = Decimal(price_match.group(1).replace(",", ""))

    if units is None or price is None:
        raise ValueError("ไม่พบจำนวนหน่วย (Units) หรือราคาต่อหน่วย (NAV) ในเอกสาร PDF ของ WealthX")

    net_amount = None
    tot_match = re.search(r"(?:Total\s*Buy|รวมมูลค่าซื้อ|จำนวนเงินสุทธิ|Net\s*Amount)[^\d\n]*([\d\.,]+)", text, re.IGNORECASE)
    if tot_match:
        net_amount = Decimal(tot_match.group(1).replace(",", ""))
    else:
        amt_match = re.search(r"(?:จำ[า]?นวนเงิน(?!\s*สุทธิ)|Amount)[^:\d\n]*[:]?\s*([\d\.,]+)", text, re.IGNORECASE)
        if amt_match:
            net_amount = Decimal(amt_match.group(1).replace(",", ""))

    if net_amount is None:
        net_amount = (units * price).quantize(Decimal("0.01"))

    order_id = ""
    m_ref = re.search(r"\b(239\d{13})\b", text)
    if m_ref:
        order_id = m_ref.group(1)
    else:
        m_ref_split = re.search(r"\b(239\d{6,14})\s*\n\s*(\d{1,8})\b", text)
        if m_ref_split:
            order_id = m_ref_split.group(1) + m_ref_split.group(2)
        else:
            ref_match = re.search(r"(?:เลขอ้างอิง|Reference\s*No\.?)[^:\n]*:\s*([A-Za-z0-9\-_]+)", text, re.IGNORECASE)
            if ref_match:
                order_id = ref_match.group(1).strip()
            else:
                any_digits = re.search(r"\b(\d{15,18})\b", text.replace("\n", " "))
                if any_digits:
                    order_id = any_digits.group(1)
                else:
                    raise ValueError("ไม่พบเลขอ้างอิงคำสั่งซื้อ (Reference No.) ในเอกสาร PDF ของ WealthX")

    symbol = ""
    sym_paren = re.search(r"\(([A-Z0-9\-]{4,25})\)", text)
    if sym_paren:
        symbol = sym_paren.group(1).upper()
    else:
        broad_sym = re.search(r"\b(TLWORLD-X|TLNDQINCOME-UH-X|LHGRID|[A-Z0-9\-]{4,20})\b", text)
        if broad_sym:
            symbol = broad_sym.group(1)
        else:
            raise ValueError("ไม่สามารถระบุรหัสกองทุน (Fund Code / Symbol) จากเอกสาร PDF ของ WealthX ได้")

    gross_amount = net_amount
    fp_raw = f"{conf_no}|{order_id}|{symbol}|{action}|{units}|{price}|{trade_date}|{net_amount}"
    fingerprint = hashlib.sha256(fp_raw.encode("utf-8")).hexdigest()

    item_dict = {
        "item_id": f"item_{uuid.uuid4().hex[:8]}",
        "trade_date": trade_date,
        "settlement_date": settlement_date,
        "symbol": symbol,
        "action": action,
        "units": str(units),
        "price": str(price),
        "gross_amount": str(gross_amount),
        "fees": {
            "commission": "0.00",
            "vat": "0.00",
            "other_fees": "0.00",
            "fee_currency": curr,
        },
        "net_amount": str(net_amount),
        "currency": curr,
        "exchange_rate": "1.0",
        "confirmation_no": conf_no,
        "order_id": order_id,
        "source": "WEALTHX",
        "fingerprint": fingerprint,
        "line_index": 0,
        "cash_adjusted": True,
        "asset_type": "Fund",
    }

    return [item_dict]


class WealthXPdfParserAdapter(TradeDocumentParserPort):
    """Isolated child-process parser for WealthX confirmation PDFs."""

    def __init__(self, timeout_seconds: float = PARSER_TIMEOUT_SECONDS):
        self.timeout_seconds = timeout_seconds

    def parse_confirmation_pdf(
        self, pdf_bytes: bytes, password: Optional[str] = None
    ) -> List[TradeImportItem]:
        # Fallback to WEALTHX_PDF_PASSWORD if not provided explicitly
        effective_password = (
            password.strip() if password and password.strip() else None
        ) or os.getenv("WEALTHX_PDF_PASSWORD")

        # 1. Bounded size check
        if len(pdf_bytes) > MAX_PDF_BYTES:
            raise ValueError(f"ไฟล์ PDF มีขนาด {len(pdf_bytes) / 1024 / 1024:.1f}MB ซึ่งเกินขีดจำกัด 10MB")

        # 2. Magic byte check
        if not pdf_bytes.startswith(b"%PDF-"):
            raise ValueError("ไฟล์ไม่ใช่เอกสาร PDF ที่ถูกต้อง (Header '%PDF-' ไม่ถูกต้อง)")

        # 3. Spawn child process with timeout to protect against crashes/quadratic CPU
        ctx = multiprocessing.get_context("spawn")
        queue = ctx.Queue()
        proc = ctx.Process(
            target=_worker_parse_wealthx_pdf,
            args=(pdf_bytes, effective_password, queue),
        )
        proc.start()
        proc.join(timeout=self.timeout_seconds)

        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=1.0)
            if proc.is_alive():
                proc.kill()
            raise ValueError(f"การประมวลผล PDF ของ WealthX หมดเวลาเกินกำหนด ({self.timeout_seconds} วินาที)")

        if queue.empty():
            raise ValueError("โพรเซสถอดรหัส PDF สิ้นสุดการทำงานโดยไม่ส่งข้อมูลกลับ (อาจเกิดการ crash ในระดับ process)")

        res = queue.get()
        if not res.get("success"):
            err_msg = res.get("error", "ไม่สามารถประมวลผล PDF ของ WealthX ได้")
            raise ValueError(err_msg)

        raw_items = res.get("items", [])
        parsed_items: List[TradeImportItem] = []

        for r in raw_items:
            order_id = r.get("order_id")
            if not order_id:
                raise ValueError(f"รายการ {r.get('symbol')} ขาด Order ID (Reference No.) ไม่สามารถระบุตัวตนของธุรกรรมได้")

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
                source="WEALTHX",
                fingerprint=r["fingerprint"],
                line_index=int(r.get("line_index", 0)),
                cash_adjusted=bool(r.get("cash_adjusted", True)),
                asset_type=r.get("asset_type", "Fund"),
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
