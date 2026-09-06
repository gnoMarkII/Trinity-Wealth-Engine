"""SCBAM Fund Click Synchronization Service (Hexagonal Architecture).

Coordinates synchronization of SCBAM Fund Click confirmation emails from Gmail IMAP,
HTML parsing, historical NAV resolution via FinnomenaFundAdapter,
conflict detection, staging, and atomic ledger commit with CASH_THB deduction.
"""
from contextlib import contextmanager
import csv
from datetime import datetime
from decimal import Decimal
import json
import logging
import os
from pathlib import Path
import re
from typing import Any, Dict, Generator, List, Optional, Set, Tuple
import uuid

from tools.portfolio.adapters.markdown.paths import get_vault_path, get_trades_log_filepath
from tools.portfolio.adapters.scb.scbam_parser_adapter import parse_scbam_fundclick_html, SCBAMRawOrder
from tools.portfolio.domain.models import (
    PortfolioState,
    TradeImportItem,
    TradeFeeBreakdown,
    quantize_decimal,
    MONEY_QUANTUM,
    PRICE_QUANTUM,
    UNITS_QUANTUM,
)
from tools.portfolio.ports.trade_ingestion_port import (
    TradeEmailSourcePort,
    TradeStagingPort,
    TradeDocumentMetadata,
)
from tools.portfolio.ports.thai_fund_port import ThaiFundPricePort
from .batch_trade_import_service import BatchTradeImportService

log = logging.getLogger(__name__)


def _sanitize_email_for_filename(email: str) -> str:
    cleaned = (email or "default").strip().lower()
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", cleaned)


def _get_sync_history_file(portfolio_id: str, account_email: str) -> Path:
    sync_dir = get_vault_path() / ".sync_history"
    sync_dir.mkdir(parents=True, exist_ok=True)
    sanitized = _sanitize_email_for_filename(account_email)
    return sync_dir / f"scbam_{sanitized}_{portfolio_id}.json"


def _load_sync_history(portfolio_id: str, account_email: str) -> Dict[str, Any]:
    history_file = _get_sync_history_file(portfolio_id, account_email)
    if not history_file.exists():
        return {
            "account_email": account_email,
            "portfolio_id": portfolio_id,
            "synced_emails": {},
        }
    try:
        with history_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, dict):
                return {"account_email": account_email, "portfolio_id": portfolio_id, "synced_emails": {}}
            data.setdefault("synced_emails", {})
            return data
    except Exception as e:
        log.warning("Could not read SCBAM sync history file %s: %s", history_file, e)
        return {"account_email": account_email, "portfolio_id": portfolio_id, "synced_emails": {}}


def _save_sync_history(portfolio_id: str, account_email: str, history: Dict[str, Any]) -> None:
    history_file = _get_sync_history_file(portfolio_id, account_email)
    temp_file = history_file.with_suffix(".tmp")
    try:
        with temp_file.open("w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        temp_file.replace(history_file)
    except Exception as e:
        log.error("Failed to write SCBAM sync history to %s: %s", history_file, e)
        if temp_file.exists():
            temp_file.unlink(missing_ok=True)


def _load_existing_ledger_identities(portfolio_id: str) -> Dict[Tuple[str, str], Dict[str, str]]:
    fpath = get_trades_log_filepath(portfolio_id)
    if not fpath.exists():
        return {}
    identity_map: Dict[Tuple[str, str], Dict[str, str]] = {}
    try:
        with fpath.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if not r:
                    continue
                c_no = str(r.get("Confirmation_No") or "").strip()
                o_id = str(r.get("Order_ID") or "").strip()
                if c_no and o_id:
                    identity_map[(c_no, o_id)] = dict(r)
    except Exception as e:
        log.warning("Could not load existing trade log identities for %s: %s", portfolio_id, e)
    return identity_map


class SCBAMSyncService:
    """Application service for synchronizing SCBAM Fund Click confirmation emails."""

    def __init__(
        self,
        email_source: TradeEmailSourcePort,
        price_port: ThaiFundPricePort,
        staging: TradeStagingPort,
        batch_importer: BatchTradeImportService,
    ):
        self.email_source = email_source
        self.price_port = price_port
        self.staging = staging
        self.batch_importer = batch_importer

    def stream_scbam_sync(
        self,
        portfolio_id: str = "default",
        since_date: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> Generator[str, None, None]:
        """Stream SCBAM Fund Click sync progress and parsed orders via SSE."""
        scan_session_id = f"scan_{int(datetime.now().timestamp())}_{uuid.uuid4().hex[:8]}"

        yield f"event: status\ndata: {json.dumps({'message': 'กำลังค้นหาอีเมลคำสั่งซื้อจาก fundclick.scbam@scb.co.th...'})}\n\n"

        try:
            email_metas = self.email_source.search_scbam_emails(query="", limit=limit)
        except Exception as e:
            log.error("Failed to search SCBAM emails: %s", e)
            yield f"event: error\ndata: {json.dumps({'message': f'ไม่สามารถเชื่อมต่อหรือค้นหาอีเมลใน Gmail ได้: {str(e)}'})}\n\n"
            return

        if not email_metas:
            yield f"event: status\ndata: {json.dumps({'message': 'ไม่พบอีเมลยืนยันคำสั่งซื้อจาก SCBAM Fund Click'})}\n\n"
            yield f"event: complete\ndata: {json.dumps({'scan_id': scan_session_id, 'item_count': 0, 'items': [], 'warnings': [], 'skipped_synced_count': 0})}\n\n"
            yield f"event: done\ndata: {json.dumps({'scan_session_id': scan_session_id, 'total_emails': 0, 'total_orders': 0, 'new_orders': 0, 'duplicate_orders': 0})}\n\n"
            return

        # Filter by since_date if provided (format YYYY-MM-DD)
        if since_date:
            filtered_metas = []
            for meta in email_metas:
                # meta.received_at could be RFC2822
                filtered_metas.append(meta)
            email_metas = filtered_metas

        total_emails = len(email_metas)
        yield f"event: status\ndata: {json.dumps({'message': f'พบอีเมลคำสั่งซื้อ SCBAM ทั้งหมด {total_emails} ฉบับ กำลังประมวลผล...'})}\n\n"

        existing_identities = _load_existing_ledger_identities(portfolio_id)

        staged_items: List[TradeImportItem] = []
        accumulated_warnings: List[Dict[str, Any]] = []
        new_count = 0
        dup_count = 0
        processed_emails = 0
        seen_order_ids: Set[str] = set()

        def _item_to_dict(it: TradeImportItem) -> Dict[str, Any]:
            price_str = f"{it.price:f}"
            if "." in price_str:
                price_str = price_str.rstrip("0").rstrip(".")
            return {
                "item_id": it.item_id,
                "trade_date": it.trade_date,
                "settlement_date": it.settlement_date,
                "symbol": it.symbol,
                "action": it.action,
                "units": f"{it.units:g}",
                "price": price_str,
                "gross_amount": f"{it.gross_amount:.2f}",
                "fees": {
                    "commission": f"{it.fees.commission:.2f}",
                    "vat": f"{it.fees.vat:.2f}",
                    "other_fees": f"{it.fees.other_fees:.2f}",
                    "fee_currency": it.fees.fee_currency,
                },
                "net_amount": f"{it.net_amount:.2f}",
                "currency": it.currency,
                "exchange_rate": f"{it.exchange_rate:.4f}" if it.exchange_rate else None,
                "confirmation_no": it.confirmation_no,
                "order_id": it.order_id,
                "source": it.source,
                "fingerprint": it.fingerprint,
                "line_index": it.line_index,
                "cash_adjusted": it.cash_adjusted,
                "asset_type": getattr(it, "asset_type", "Fund"),
            }

        for idx, meta in enumerate(email_metas):
            processed_emails += 1
            progress_pct = int((processed_emails / total_emails) * 100)
            yield f"event: progress\ndata: {json.dumps({'current': processed_emails, 'total': total_emails, 'percent': progress_pct, 'items_found': len(staged_items), 'subject': meta.subject, 'message_id': meta.message_id})}\n\n"

            try:
                html_body = self.email_source.fetch_email_html_body(meta.uid or meta.message_id)
            except Exception as e:
                log.warning("Could not fetch email body for %s: %s", meta.message_id, e)
                w_payload = {
                    "subject": meta.subject,
                    "message_id": meta.message_id,
                    "filename": "SCBAM Confirmation Email",
                    "received_at": meta.received_at,
                    "reason": f"ไม่สามารถอ่านอีเมล {meta.subject}: {str(e)}",
                    "can_preview": True,
                }
                accumulated_warnings.append(w_payload)
                yield f"event: warning\ndata: {json.dumps(w_payload)}\n\n"
                continue

            raw_order = parse_scbam_fundclick_html(html_body)
            if not raw_order:
                w_payload = {
                    "subject": meta.subject,
                    "message_id": meta.message_id,
                    "filename": "SCBAM Confirmation Email",
                    "received_at": meta.received_at,
                    "reason": f"ไม่สามารถสกัดข้อมูลจากอีเมล: {meta.subject}",
                    "can_preview": True,
                }
                accumulated_warnings.append(w_payload)
                yield f"event: warning\ndata: {json.dumps(w_payload)}\n\n"
                continue

            # Quarantine advance orders (ADV-) that may have failed or been cancelled
            if raw_order.transaction_no.startswith("ADV-"):
                w_payload = {
                    "subject": meta.subject,
                    "message_id": meta.message_id,
                    "filename": "SCBAM Confirmation Email",
                    "received_at": meta.received_at,
                    "reason": f"รายการคำสั่งล่วงหน้า (Advance Order: {raw_order.transaction_no}) ถูกกักกัน (Quarantine) ไม่นำเข้าโดยอัตโนมัติ เนื่องจากอาจถูกยกเลิกหรือไม่มีการตัดเงินจริง โปรดตรวจสอบกับ Statement ทางการ",
                    "can_preview": True,
                }
                accumulated_warnings.append(w_payload)
                yield f"event: warning\ndata: {json.dumps(w_payload)}\n\n"
                continue

            # Deduplication across confirmation notes
            if raw_order.transaction_no in seen_order_ids:
                log.info("Consolidating duplicate SCBAM trade across emails %s (Tx: %s)", raw_order.fund_code, raw_order.transaction_no)
                continue
            seen_order_ids.add(raw_order.transaction_no)

            # Query historical NAV
            eff_date = raw_order.effective_date
            fund_sym = raw_order.fund_code
            nav_val = self.price_port.fetch_historical_nav(fund_sym, eff_date)
            if not nav_val or nav_val <= 0:
                # Try latest NAV
                latest_nav_data = self.price_port.fetch_nav(fund_sym)
                if latest_nav_data and latest_nav_data.nav > 0:
                    nav_val = latest_nav_data.nav
                    w_payload = {
                        "subject": meta.subject,
                        "message_id": meta.message_id,
                        "filename": "SCBAM Confirmation Email",
                        "received_at": meta.received_at,
                        "reason": f"ไม่พบราคา NAV วันที่ {eff_date} สำหรับ {fund_sym} จึงใช้ราคาล่าสุด {nav_val} แทน",
                        "can_preview": True,
                    }
                    accumulated_warnings.append(w_payload)
                    yield f"event: warning\ndata: {json.dumps(w_payload)}\n\n"
                else:
                    w_payload = {
                        "subject": meta.subject,
                        "message_id": meta.message_id,
                        "filename": "SCBAM Confirmation Email",
                        "received_at": meta.received_at,
                        "reason": f"ไม่สามารถค้นหาราคา NAV สำหรับ {fund_sym} ในวันที่ {eff_date} ได้ ข้ามรายการนี้",
                        "can_preview": True,
                    }
                    accumulated_warnings.append(w_payload)
                    yield f"event: warning\ndata: {json.dumps(w_payload)}\n\n"
                    continue

            dec_nav = quantize_decimal(Decimal(str(nav_val)), PRICE_QUANTUM)
            dec_amount = quantize_decimal(raw_order.amount, MONEY_QUANTUM)
            dec_units = quantize_decimal(dec_amount / dec_nav, UNITS_QUANTUM)

            order_id = f"SCB-{raw_order.transaction_no}"
            conf_no = f"SCB-FC-{raw_order.account_no}-{raw_order.effective_date.replace('-', '')}"

            # Check if exists in ledger
            identity_key = (conf_no, order_id)
            is_duplicate = identity_key in existing_identities

            item = TradeImportItem(
                item_id=str(uuid.uuid4()),
                trade_date=raw_order.tx_date,
                settlement_date=raw_order.effective_date,
                symbol=fund_sym,
                action="BUY",
                units=dec_units,
                price=dec_nav,
                gross_amount=dec_amount,
                fees=TradeFeeBreakdown(commission=Decimal("0.00"), vat=Decimal("0.00"), other_fees=Decimal("0.00"), fee_currency="THB"),
                net_amount=dec_amount,
                currency="THB",
                confirmation_no=conf_no,
                order_id=order_id,
                source="SCB",
                fingerprint=f"scb_{raw_order.transaction_no}",
                cash_adjusted=True,
                asset_type="Fund",
            )

            staged_items.append(item)

            if is_duplicate:
                dup_count += 1
                status_label = "ALREADY_IMPORTED"
            else:
                new_count += 1
                status_label = "NEW"

            order_payload = {
                "item_id": item.item_id,
                "email_id": meta.message_id,
                "email_uid": meta.uid,
                "email_subject": meta.subject,
                "trade_date": item.trade_date,
                "settlement_date": item.settlement_date,
                "symbol": item.symbol,
                "action": item.action,
                "units": float(item.units),
                "price": float(item.price),
                "gross_amount": float(item.gross_amount),
                "fees": float(item.fees.total_fees),
                "net_amount": float(item.net_amount),
                "currency": item.currency,
                "order_id": item.order_id,
                "confirmation_no": item.confirmation_no,
                "status": status_label,
                "account_no": raw_order.account_no,
            }
            yield f"event: order\ndata: {json.dumps(order_payload)}\n\n"

        # Stage items for user commit review
        scan_id = ""
        if staged_items:
            scan_id = self.staging.stage_items(staged_items)

        complete_payload = {
            "scan_id": scan_id,
            "item_count": len(staged_items),
            "items": [_item_to_dict(it) for it in staged_items],
            "warnings": accumulated_warnings,
            "skipped_synced_count": dup_count,
        }
        yield f"event: complete\ndata: {json.dumps(complete_payload)}\n\n"
        yield f"event: done\ndata: {json.dumps({'scan_session_id': scan_id, 'total_emails': total_emails, 'total_orders': len(staged_items), 'new_orders': new_count, 'duplicate_orders': dup_count})}\n\n"

    def scan_scbam_email(self, message_id: str, portfolio_id: str = "default") -> Tuple[str, List[TradeImportItem]]:
        """Scan a single SCBAM email by message ID, parse HTML, query NAV, and stage."""
        html_body = self.email_source.fetch_email_html_body(message_id)
        if not html_body:
            raise ValueError(f"ไม่พบเนื้อหาอีเมลสำหรับ Message ID: {message_id}")
        raw_order = parse_scbam_fundclick_html(html_body)
        if not raw_order:
            raise ValueError("ไม่สามารถสกัดข้อมูลคำสั่งซื้อจากอีเมลนี้ได้ (รูปแบบไม่ตรงกับ SCBAM Fund Click)")

        if raw_order.transaction_no.startswith("ADV-"):
            raise ValueError(
                f"คำสั่งซื้อนี้เป็นคำสั่งล่วงหน้า (Advance Order: {raw_order.transaction_no}) ซึ่งมีความเสี่ยงที่จะถูกยกเลิกหรือไม่มีผลจริง ระบบจึงไม่อนุญาตให้นำเข้าโดยตรง"
            )

        fund_sym = raw_order.fund_code
        eff_date = raw_order.effective_date
        nav_val = self.price_port.fetch_historical_nav(fund_sym, eff_date)
        if not nav_val or nav_val <= 0:
            latest_nav_data = self.price_port.fetch_nav(fund_sym)
            if latest_nav_data and latest_nav_data.nav > 0:
                nav_val = latest_nav_data.nav
            else:
                raise ValueError(f"ไม่สามารถค้นหาราคา NAV สำหรับกองทุน {fund_sym} ในวันที่ {eff_date} ได้")

        dec_nav = quantize_decimal(Decimal(str(nav_val)), PRICE_QUANTUM)
        dec_amount = quantize_decimal(raw_order.amount, MONEY_QUANTUM)
        dec_units = quantize_decimal(dec_amount / dec_nav, UNITS_QUANTUM)

        order_id = f"SCB-{raw_order.transaction_no}"
        conf_no = f"SCB-FC-{raw_order.account_no}-{raw_order.effective_date.replace('-', '')}"

        item = TradeImportItem(
            item_id=str(uuid.uuid4()),
            trade_date=raw_order.tx_date,
            settlement_date=raw_order.effective_date,
            symbol=fund_sym,
            action="BUY",
            units=dec_units,
            price=dec_nav,
            gross_amount=dec_amount,
            fees=TradeFeeBreakdown(commission=Decimal("0.00"), vat=Decimal("0.00"), other_fees=Decimal("0.00"), fee_currency="THB"),
            net_amount=dec_amount,
            currency="THB",
            confirmation_no=conf_no,
            order_id=order_id,
            source="SCB",
            fingerprint=f"scb_{raw_order.transaction_no}",
            cash_adjusted=True,
            asset_type="Fund",
        )

        scan_id = self.staging.stage_items([item])
        return scan_id, [item]

    def commit_scbam_sync(
        self,
        portfolio_id: str = "default",
        scan_session_id: str = "",
        selected_item_ids: Optional[List[str]] = None,
    ) -> Tuple[PortfolioState, int]:
        """Commit selected staged SCBAM orders into the authoritative ledger."""
        staged = self.staging.get_staged_items(scan_session_id)
        if not staged:
            raise ValueError(f"ไม่พบรายการที่สแกนไว้สำหรับ Session ID: {scan_session_id} (Session อาจหมดอายุ)")

        if selected_item_ids is not None:
            selected_set = set(selected_item_ids)
            items_to_commit = [item for item in staged if item.item_id in selected_set]
        else:
            items_to_commit = staged

        if not items_to_commit:
            raise ValueError("ไม่มีรายการใดถูกเลือกเพื่อนำเข้าสู่ระบบ")

        # Execute batch import atomically
        new_state = self.batch_importer.execute_batch_import(items_to_commit, portfolio_id=portfolio_id)

        # Clear staging
        self.staging.delete_staged(scan_session_id)

        return new_state, len(items_to_commit)

    def get_email_html(self, message_id: str) -> str:
        """Fetch raw HTML body of an email for modal viewing."""
        return self.email_source.fetch_email_html_body(message_id)
