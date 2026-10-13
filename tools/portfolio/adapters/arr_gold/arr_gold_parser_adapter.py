"""ARR Gold Trading Parser Driven Adapter (Hexagonal Architecture).

Parses trade confirmation emails (HTML) and Daily Statement PDFs from ARR Gold Trading
(Aurora Trading Co., Ltd. / Paotang) into canonical domain TradeImportItem records.
"""
from decimal import Decimal
import hashlib
import io
import logging
import re
from typing import List, Optional, Tuple
from bs4 import BeautifulSoup
import pypdf

from tools.portfolio.domain.calculations import validate_reconciliation_invariant
from tools.portfolio.domain.models import (
    TradeImportItem,
    TradeFeeBreakdown,
    quantize_decimal,
    MONEY_QUANTUM,
    PRICE_QUANTUM,
    UNITS_QUANTUM,
)
from tools.portfolio.ports.arr_gold_port import ArrGoldParserPort

log = logging.getLogger(__name__)

GOLD_SYMBOL = "GOLD-99.99"
GOLD_ASSET_TYPE = "Gold"
GOLD_SOURCE = "ARR_GOLD"
GOLD_CURRENCY = "USD"


def _clean_str(val: str) -> str:
    return re.sub(r"\s+", " ", val).strip()


def _parse_decimal(val_str: str) -> Decimal:
    if not val_str:
        return Decimal("0.00")
    # Extract the first valid numeric token (handles glued PDF columns like 1.00-4,221.53)
    m = re.search(r"[-+]?[\d,]+(?:\.\d+)?", str(val_str))
    if not m:
        return Decimal("0.00")
    clean = m.group(0).replace(",", "")
    try:
        return Decimal(clean)
    except Exception:
        return Decimal("0.00")


def _compute_fingerprint(confirmation_no: str, trade_date: str, action: str, units: Decimal, price: Decimal) -> str:
    raw = f"{confirmation_no}|{trade_date}|{GOLD_SYMBOL}|{action}|{units}|{price}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _unpack_units_and_amount(glued_str: str, price: Decimal) -> Tuple[Decimal, Decimal]:
    """Unpack glued units and total amount from PDF table.

    In ARR Gold Statement PDFs, the columns 'Gold (Oz)' and 'Total Amount (USD)'
    are often extracted by PDF engines as a single contiguous token, e.g.:
    - '0.501,867.77' -> 0.50 Oz and 1,867.77 USD
    - '1.00-3,745.75' -> 1.00 Oz and 3,745.75 USD
    - '1.004,540.27' -> 1.00 Oz and 4,540.27 USD
    - '1.00-4,221.53' -> 1.00 Oz and 4,221.53 USD
    """
    if not glued_str:
        return Decimal("0.00"), Decimal("0.00")

    glued = glued_str.strip()
    best_units: Optional[Decimal] = None
    best_amount: Optional[Decimal] = None
    min_diff = Decimal("999999999")

    # Try all possible split positions between units and amount
    for i in range(1, len(glued)):
        u_str = glued[:i].replace(",", "").strip()
        a_str = glued[i:].replace(",", "").strip()
        if not u_str or not a_str or u_str in "+-" or a_str in "+-":
            continue
        try:
            u = Decimal(u_str)
            a = abs(Decimal(a_str))
            if u <= 0 or a <= 0:
                continue
            if price > 0:
                diff = abs(u * price - a)
                if diff < Decimal("0.05") and diff < min_diff:
                    min_diff = diff
                    best_units = u
                    best_amount = a
        except Exception:
            continue

    if best_units is not None and best_amount is not None:
        return best_units, best_amount

    # Fallback if split didn't find exact mathematical match
    # Check if there is a '-' or '+' dividing them
    m_sign = re.search(r"^([\d,\.]+)([+-][\d,\.]+)$", glued)
    if m_sign:
        u = _parse_decimal(m_sign.group(1))
        a = abs(_parse_decimal(m_sign.group(2)))
        return u, a

    num = _parse_decimal(glued)
    if num < Decimal("100"):
        return num, quantize_decimal(num * price, MONEY_QUANTUM)
    else:
        u = quantize_decimal(num / price, UNITS_QUANTUM) if price > 0 else Decimal("0.00")
        return u, num


class ArrGoldParserAdapter(ArrGoldParserPort):
    """Adapter parsing ARR Gold trade confirmations from HTML emails or PDF daily statements."""

    def parse_trade_html(self, html_content: str, email_subject: str = "") -> List[TradeImportItem]:
        """Parse raw HTML content of an ARR Gold trade execution email."""
        if not html_content:
            return []

        soup = BeautifulSoup(html_content, "html.parser")
        for s in soup(["script", "style"]):
            s.decompose()
        flat_text = " ".join(soup.get_text(separator=" ").split())

        # 1. Determine Action (BUY or SELL)
        action: Optional[str] = None
        if "ซื้อทองคำ" in flat_text or "Buy Gold" in email_subject or "Buy" in email_subject:
            action = "BUY"
        elif "ขายทองคำ" in flat_text or "Sell Gold" in email_subject or "Sell" in email_subject:
            action = "SELL"
        else:
            log.warning("Could not identify BUY/SELL action in ARR Gold email")
            return []

        # 2. Extract Confirmation No / Reference No
        ref_no = ""
        # Match from subject e.g. Buy Gold (B26061208401)
        subj_ref_match = re.search(r"\(([BS]\d+)\)", email_subject)
        if subj_ref_match:
            ref_no = subj_ref_match.group(1).strip()
        else:
            # Match from text e.g. เลขที่อ้างอิง: B25092602371 or เลขที่อ้างอิง B26061208401
            text_ref_match = re.search(r"เลขที่อ้างอิง[:\s]*([A-Za-z0-9]+)", flat_text)
            if text_ref_match:
                ref_no = text_ref_match.group(1).strip()

        if not ref_no:
            log.warning("No reference number found in ARR Gold email")
            return []

        # 3. Extract Trade Date (YYYY-MM-DD HH:MM:SS)
        trade_date = ""
        date_match = re.search(r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})", flat_text)
        if date_match:
            trade_date = date_match.group(1).strip()
        else:
            # Fallback to date only
            date_only = re.search(r"(\d{4}-\d{2}-\d{2})", flat_text)
            if date_only:
                trade_date = f"{date_only.group(1).strip()} 00:00:00"

        if not trade_date:
            log.warning("No trade date found in ARR Gold email %s", ref_no)
            return []

        # 4. Extract Price, Units, Gross/Net
        # Try table format first (Format A - 2026)
        # e.g. จำนวน 4,221.53 + 1 - 4,221.53 1.00
        units = Decimal("0.00")
        price = Decimal("0.00")
        net_amount = Decimal("0.00")

        table_match = re.search(
            r"จำนวน\s+([\d,\.]+)\s+([+-]?\s*[\d,\.]+)\s+([+-]?\s*[\d,\.]+)", flat_text
        )
        if table_match:
            price = _parse_decimal(table_match.group(1))
            units = _parse_decimal(table_match.group(2))
            net_amount = _parse_decimal(table_match.group(3))
        else:
            # Try vertical format (Format B - 2025)
            price_match = re.search(r"ราคา(?:ซื้อ|ขาย)หน่วยละ\s*([\d,\.]+)", flat_text)
            units_match = re.search(r"น้ำหนักทองคำ\s*[+-]?\s*([\d,\.]+)", flat_text)
            amt_match = re.search(r"คิดเป็นจำนวนเงิน\s*[+-]?\s*([\d,\.]+)", flat_text)

            if price_match:
                price = _parse_decimal(price_match.group(1))
            if units_match:
                units = _parse_decimal(units_match.group(1))
            if amt_match:
                net_amount = _parse_decimal(amt_match.group(1))

        if units <= 0 or price <= 0:
            log.warning("Invalid units (%s) or price (%s) in ARR Gold email %s", units, price, ref_no)
            return []

        if net_amount <= 0:
            net_amount = quantize_decimal(units * price, MONEY_QUANTUM)

        gross_amount = quantize_decimal(units * price, MONEY_QUANTUM)
        fees = TradeFeeBreakdown(fee_currency=GOLD_CURRENCY)

        # Invariant validation
        valid, reason = validate_reconciliation_invariant(
            units=units,
            price=price,
            gross_amount=gross_amount,
            fees=fees,
            net_amount=net_amount,
            action=action,
        )
        if not valid:
            log.warning("Reconciliation invariant warning for %s: %s", ref_no, reason)
            # Re-align gross/net to exact price * units
            gross_amount = net_amount

        item_id = f"ARR_{ref_no}"
        fingerprint = _compute_fingerprint(
            ref_no,
            trade_date[:10],
            action,
            quantize_decimal(units, UNITS_QUANTUM),
            quantize_decimal(price, PRICE_QUANTUM),
        )

        item = TradeImportItem(
            item_id=item_id,
            trade_date=trade_date[:10],
            settlement_date=trade_date[:10],
            symbol=GOLD_SYMBOL,
            action=action,  # type: ignore
            units=quantize_decimal(units, UNITS_QUANTUM),
            price=quantize_decimal(price, PRICE_QUANTUM),
            gross_amount=quantize_decimal(gross_amount, MONEY_QUANTUM),
            fees=fees,
            net_amount=quantize_decimal(net_amount, MONEY_QUANTUM),
            currency=GOLD_CURRENCY,
            confirmation_no=ref_no,
            order_id=ref_no,
            source=GOLD_SOURCE,
            fingerprint=fingerprint,
            cash_adjusted=False,  # Gold trading in Paotang uses USD wallet; preserve main cash
            asset_type=GOLD_ASSET_TYPE,
        )
        return [item]

    def parse_daily_statement_pdf(self, pdf_bytes: bytes) -> List[TradeImportItem]:
        """Parse raw PDF bytes of an ARR Gold Daily Report Statement."""
        if not pdf_bytes or not pdf_bytes.startswith(b"%PDF-"):
            return []

        try:
            reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
        except Exception as e:
            log.warning("Failed to parse ARR Gold PDF: %s", e)
            return []

        items: List[TradeImportItem] = []

        for page in reader.pages:
            text = page.extract_text()
            if not text:
                continue

            # Look for lines e.g.:
            # 12/06/2026 16:23:30 B26061208401 Buy -4,221.53 1.00-4,221.53
            # or 26/09/2025 10:47:04 B25092602371 Buy ...
            lines = text.splitlines()
            for line in lines:
                m = re.search(
                    r"(\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}:\d{2})\s+([A-Z0-9]+)\s+(Buy|Sell)\s+([+-]?[\d,\.]+)\s+(\S+)",
                    line,
                )
                if not m:
                    continue

                raw_date = m.group(1).strip()
                ref_no = m.group(2).strip()
                act_raw = m.group(3).strip().upper()
                price_raw = m.group(4)
                price = abs(_parse_decimal(price_raw))
                glued = m.group(5).strip()

                # Normalize date DD/MM/YYYY HH:MM:SS -> YYYY-MM-DD
                parts = raw_date.split()
                d_parts = parts[0].split("/")
                iso_date = f"{d_parts[2]}-{d_parts[1]}-{d_parts[0]}"

                action = "BUY" if act_raw == "BUY" else "SELL"
                units, amount = _unpack_units_and_amount(glued, price)
                if units <= 0 or price <= 0:
                    continue

                gross_amount = quantize_decimal(amount, MONEY_QUANTUM)
                net_amount = gross_amount
                fees = TradeFeeBreakdown(fee_currency=GOLD_CURRENCY)
                units_q = quantize_decimal(units, UNITS_QUANTUM)
                price_q = quantize_decimal(price, PRICE_QUANTUM)
                fingerprint = _compute_fingerprint(ref_no, iso_date, action, units_q, price_q)

                item = TradeImportItem(
                    item_id=f"ARR_{ref_no}",
                    trade_date=iso_date,
                    settlement_date=iso_date,
                    symbol=GOLD_SYMBOL,
                    action=action,  # type: ignore
                    units=units_q,
                    price=price_q,
                    gross_amount=gross_amount,
                    fees=fees,
                    net_amount=net_amount,
                    currency=GOLD_CURRENCY,
                    confirmation_no=ref_no,
                    order_id=ref_no,
                    source=GOLD_SOURCE,
                    fingerprint=fingerprint,
                    cash_adjusted=False,
                    asset_type=GOLD_ASSET_TYPE,
                )
                items.append(item)

        return items
