"""ARR Gold Trading Synchronization Service (Hexagonal Architecture).

Coordinates end-to-end synchronization of ARR Gold Trading confirmations from Gmail IMAP,
HTML/PDF parsing via ArrGoldParserPort, conflict checking, staging, and atomic ledger commit.
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
from tools.portfolio.domain.calculations import extract_active_ledger_identities
from tools.portfolio.domain.models import PortfolioState, TradeImportItem, quantize_decimal
from tools.portfolio.ports.trade_ingestion_port import (
    TradeEmailSourcePort,
    TradeStagingPort,
    TradeDocumentMetadata,
)
from tools.portfolio.ports.arr_gold_port import ArrGoldParserPort
from .batch_trade_import_service import BatchTradeImportService

log = logging.getLogger(__name__)


def _sanitize_email_for_filename(email: Any) -> str:
    cleaned = str(email or "default").strip().lower()
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", cleaned)


def _get_sync_history_file(portfolio_id: str, account_email: str) -> Path:
    sync_dir = get_vault_path() / ".sync_history"
    from tools.archivist.maintenance_guard import assert_write_allowed
    assert_write_allowed(sync_dir)
    sync_dir.mkdir(parents=True, exist_ok=True)
    sanitized = _sanitize_email_for_filename(account_email)
    return sync_dir / f"arr_gold_{sanitized}_{portfolio_id}.json"


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
        log.warning("Could not read ARR Gold sync history file %s: %s", history_file, e)
        return {"account_email": account_email, "portfolio_id": portfolio_id, "synced_emails": {}}


def _save_sync_history(portfolio_id: str, account_email: str, history: Dict[str, Any]) -> None:
    history_file = _get_sync_history_file(portfolio_id, account_email)
    temp_file = history_file.with_suffix(".tmp")
    try:
        with temp_file.open("w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        temp_file.replace(history_file)
    except Exception as e:
        log.warning("Could not write ARR Gold sync history file %s: %s", history_file, e)


def _load_existing_ledger_identities(
    portfolio_id: str, repo: Optional[Any] = None
) -> Dict[Tuple[str, str], Dict[str, str]]:
    """Load active, non-voided transactions keyed by (Confirmation_No, Order_ID)."""
    if repo is not None:
        try:
            with repo.unit_of_work(portfolio_id) as uow:
                rows = uow.read_trade_log_locked()
                return extract_active_ledger_identities(rows)
        except Exception as e:
            log.warning("Could not load identities from repository for %s: %s", portfolio_id, e)

    fpath = get_trades_log_filepath(portfolio_id)
    if not fpath.exists():
        return {}
    try:
        with fpath.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
            return extract_active_ledger_identities(rows)
    except Exception as e:
        log.warning("Could not load existing trade log identities for %s: %s", portfolio_id, e)
    return {}


def _check_conflict_with_ledger(item: TradeImportItem, ex: Dict[str, str]) -> Tuple[bool, str]:
    ex_sym = str(ex.get("Symbol") or "").strip().upper()
    ex_act = str(ex.get("Action") or "").strip().upper()
    ex_units = str(ex.get("Units") or "").strip()
    ex_price = str(ex.get("Price") or "").strip()
    ex_net = str(ex.get("Net_Amount") or "").strip()

    if ex_sym != item.symbol.strip().upper():
        return True, f"Symbol ขัดแย้งกัน ({ex_sym} vs {item.symbol})"
    if ex_act != item.action.strip().upper():
        return True, f"Action ขัดแย้งกัน ({ex_act} vs {item.action})"
    try:
        if ex_units and Decimal(ex_units) != item.units:
            return True, f"Units ขัดแย้งกัน ({ex_units} vs {item.units})"
    except Exception:
        if ex_units and ex_units != f"{item.units:g}":
            return True, f"Units ขัดแย้งกัน ({ex_units} vs {item.units})"
    try:
        if ex_price and Decimal(ex_price) != item.price:
            return True, f"Price ขัดแย้งกัน ({ex_price} vs {item.price})"
    except Exception:
        pass
    try:
        if ex_net and Decimal(ex_net) != item.net_amount:
            return True, f"Net Amount ขัดแย้งกัน ({ex_net} vs {item.net_amount})"
    except Exception:
        pass
    return False, ""


def _create_warning_info(
    m: TradeDocumentMetadata,
    reason: str,
) -> Dict[str, Any]:
    return {
        "message_id": m.message_id,
        "attachment_id": m.attachment_id,
        "subject": m.subject,
        "filename": m.filename or "arr_gold_trade.html",
        "received_at": m.received_at,
        "reason": reason,
        "can_preview": bool(m.attachment_id or m.message_id),
    }


def _item_to_preview_dict(it: TradeImportItem) -> Dict[str, Any]:
    price_str = f"{it.price:f}"
    if "." in price_str:
        price_str = price_str.rstrip("0").rstrip(".")
    return {
        "item_id": it.item_id,
        "trade_date": it.trade_date,
        "settlement_date": it.settlement_date or it.trade_date,
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
        "order_id": it.order_id or it.confirmation_no,
        "source": it.source,
        "fingerprint": it.fingerprint,
        "line_index": it.line_index,
        "cash_adjusted": it.cash_adjusted,
        "asset_type": getattr(it, "asset_type", "Gold") or "Gold",
    }


class ArrGoldSyncService:
    """Application Service coordinating synchronization of ARR Gold Trading confirmations."""

    def __init__(
        self,
        email_source: TradeEmailSourcePort,
        parser: ArrGoldParserPort,
        staging: TradeStagingPort,
        batch_import_service: BatchTradeImportService,
    ) -> None:
        self.email_source = email_source
        self.parser = parser
        self.staging = staging
        self.batch_import_service = batch_import_service
        self._scan_provenance: Dict[str, Dict[str, Any]] = {}

    def scan_emails(
        self,
        query: str = "",
        limit: Optional[int] = 50,
        since_date: Optional[str] = None,
    ) -> List[TradeDocumentMetadata]:
        """Fetch confirmation emails from Gmail IMAP."""
        return self.email_source.search_arr_gold_emails(query=query, limit=limit, since_date=since_date)

    def _parse_email_or_pdf(
        self,
        message_id: str,
        attachment_id: Optional[str] = None,
        subject: str = "",
    ) -> List[TradeImportItem]:
        """Parse trades from either statement PDF or confirmation email HTML."""
        is_pdf = bool(attachment_id and (attachment_id.endswith(".pdf") or "statement" in subject.lower() or "รายงาน" in subject))
        if is_pdf and attachment_id:
            try:
                pdf_bytes = self.email_source.fetch_pdf_attachment(message_id, attachment_id)
                return self.parser.parse_daily_statement_pdf(pdf_bytes)
            except Exception as e:
                log.warning("PDF parse fallback to HTML for %s: %s", message_id, e)
                try:
                    html = self.email_source.fetch_email_html_body(message_id)
                    return self.parser.parse_trade_html(html, email_subject=subject)
                except Exception:
                    raise e
        else:
            html = self.email_source.fetch_email_html_body(message_id)
            return self.parser.parse_trade_html(html, email_subject=subject)

    def scan_and_stage_email(
        self,
        message_id: str,
        attachment_id: str = "",
        subject: str = "",
        portfolio_id: str = "default",
        session_id: Optional[str] = None,
    ) -> Tuple[str, List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Fetch email/PDF content, parse items, check duplicate status, and stage."""
        items = self._parse_email_or_pdf(message_id, attachment_id, subject)
        if not items:
            return "", [], []

        scan_id = self.staging.stage_items(items, session_id=session_id)

        # Check existing ledger for duplicates
        existing_identities = _load_existing_ledger_identities(
            portfolio_id, repo=getattr(self.batch_import_service, "repo", None)
        )

        item_previews: List[Dict[str, Any]] = []
        warnings: List[Dict[str, Any]] = []

        for it in items:
            p = _item_to_preview_dict(it)
            key = (it.confirmation_no.strip(), (it.order_id or it.confirmation_no).strip())
            if key in existing_identities:
                p["is_duplicate"] = True
                warnings.append({
                    "type": "DUPLICATE_TRADE",
                    "item_id": it.item_id,
                    "confirmation_no": it.confirmation_no,
                    "message": f"รายการ {it.confirmation_no} มีอยู่ในพอร์ตแล้ว (บันทึกซ้ำ)",
                })
            else:
                p["is_duplicate"] = False
            item_previews.append(p)

        return scan_id, item_previews, warnings

    def get_staged(self, scan_id: str, session_id: Optional[str] = None) -> List[TradeImportItem]:
        """Retrieve previously staged items by scan_id."""
        return self.staging.get_staged_items(scan_id, session_id=session_id)

    def commit_staged(
        self,
        scan_id: str,
        portfolio_id: str = "default",
        selected_item_ids: Optional[List[str]] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Commit staged items to the transaction ledger atomically."""
        staged_items = self.staging.get_staged_items(scan_id, session_id=session_id)
        if not staged_items:
            raise ValueError(f"Scan ID '{scan_id}' ไม่พบในระบบ หรือหมดอายุแล้ว")

        if selected_item_ids:
            sel_set = set(selected_item_ids)
            items_to_commit = [it for it in staged_items if it.item_id in sel_set]
        else:
            items_to_commit = staged_items

        if not items_to_commit:
            raise ValueError("ไม่มีรายการที่ถูกเลือกเพื่อนำเข้า")

        state = self.batch_import_service.execute_batch_import(
            portfolio_id=portfolio_id,
            items=items_to_commit,
        )

        # Clean up staged items
        self.staging.delete_staged(scan_id, session_id=session_id)

        # Update sync history for emails whose trades were successfully committed
        provenance = self.staging.pop_provenance(scan_id=scan_id, session_id=session_id)
        if not provenance or not isinstance(provenance, dict):
            provenance = self._scan_provenance.pop(scan_id, None)

        if provenance and isinstance(provenance, dict):
            account_email = str(provenance.get("account_email") or "")
            history = _load_sync_history(portfolio_id=portfolio_id, account_email=account_email)
            synced_emails = history.setdefault("synced_emails", {})
            now_iso = datetime.now().isoformat()

            committed_order_ids = {it.order_id for it in items_to_commit if it.order_id}
            committed_confirmations = {it.confirmation_no for it in items_to_commit if it.confirmation_no}

            for msg_key, email_info in (provenance.get("emails") or {}).items():
                if isinstance(email_info, dict) and email_info.get("status") in ("staged", "already_synced") and not email_info.get("has_error"):
                    if selected_item_ids is not None:
                        email_conf = email_info.get("confirmation_no", "")
                        email_orders = set(email_info.get("order_ids", []))
                        is_committed = (email_conf in committed_confirmations) or bool(email_orders.intersection(committed_order_ids))
                        if not is_committed and email_info.get("status") != "already_synced":
                            continue

                    synced_emails[msg_key] = {
                        "synced_at": now_iso,
                        "subject": email_info.get("subject", ""),
                        "confirmation_no": email_info.get("confirmation_no", ""),
                        "order_ids": email_info.get("order_ids", []),
                    }

            _save_sync_history(portfolio_id=portfolio_id, account_email=account_email, history=history)

        return {
            "status": "success",
            "imported_count": len(items_to_commit),
            "state": state,
        }

    def stream_batch_sync(
        self,
        portfolio_id: str = "default",
        email_items: Optional[List[TradeDocumentMetadata]] = None,
        session_id: Optional[str] = None,
        force_rescan: bool = False,
    ) -> Generator[str, None, None]:
        """Batch process emails, check duplicates/conflicts, stage items, and yield SSE events for review."""
        account_email = os.getenv("GMAIL_IMAP_USER", "")
        sync_history = _load_sync_history(portfolio_id, account_email)
        synced_map = sync_history.get("synced_emails", {})

        if email_items is None:
            yield f"event: progress\ndata: {json.dumps({'message': 'กำลังค้นหาอีเมล ARR Gold ใน Gmail...', 'current': 0, 'total': 0, 'percent': 0})}\n\n"
            try:
                emails = self.scan_emails(limit=50)
            except Exception as e:
                yield f"event: error\ndata: {json.dumps({'message': f'ค้นหาอีเมลไม่สำเร็จ: {str(e)}'})}\n\n"
                return
        else:
            emails = email_items

        total_found = len(emails)
        if total_found == 0:
            yield f"event: complete\ndata: {json.dumps({'scan_id': '', 'item_count': 0, 'items': [], 'message': 'ไม่พบอีเมลใหม่ที่ต้องซิงค์', 'total': 0, 'imported': 0, 'warnings': [], 'skipped_synced_count': 0})}\n\n"
            return

        emails_to_process: List[TradeDocumentMetadata] = []
        skipped_count = 0
        for m in emails:
            msg_key = m.message_id
            if not force_rescan and msg_key in synced_map:
                skipped_count += 1
            else:
                emails_to_process.append(m)

        if not emails_to_process:
            yield f"event: complete\ndata: {json.dumps({'scan_id': '', 'item_count': 0, 'items': [], 'warnings': [], 'skipped_synced_count': skipped_count, 'already_in_portfolio_count': skipped_count, 'imported': 0, 'message': 'เอกสารทั้งหมดได้รับการประมวลผลแล้ว (ไม่มีรายการใหม่)'}, ensure_ascii=False)}\n\n"
            return

        existing_identities = _load_existing_ledger_identities(
            portfolio_id, repo=getattr(self.batch_import_service, "repo", None)
        )

        batch_identity_map: Dict[Tuple[str, str], TradeImportItem] = {}
        all_valid_items: List[TradeImportItem] = []
        email_status_map: Dict[str, Dict[str, Any]] = {}
        warnings: List[Dict[str, Any]] = []
        total_already_in_portfolio = 0

        total_to_process = len(emails_to_process)

        for idx, m in enumerate(emails_to_process):
            msg_key = m.message_id
            pct = round(((idx) / total_to_process) * 100, 1)

            yield f"event: progress\ndata: {json.dumps({'current': idx + 1, 'total': total_to_process, 'percent': pct, 'items_found': len(all_valid_items), 'subject': m.subject, 'message_id': m.message_id}, ensure_ascii=False)}\n\n"

            try:
                parsed_items = self._parse_email_or_pdf(m.message_id, m.attachment_id, m.subject)
            except Exception as e:
                warn_msg = f"ไม่สามารถประมวลผลอีเมล '{m.subject}': {e}"
                log.warning(warn_msg)
                warn_info = _create_warning_info(m, warn_msg)
                warnings.append(warn_info)
                email_status_map[msg_key] = {"status": "quarantined", "has_error": True, "reason": warn_msg}
                yield f"event: warning\ndata: {json.dumps(warn_info, ensure_ascii=False)}\n\n"
                continue

            email_conflict = False
            email_items_to_stage: List[TradeImportItem] = []

            for item in parsed_items:
                c_no = item.confirmation_no.strip()
                o_id = (item.order_id or c_no).strip()
                identity_key = (c_no, o_id)

                if identity_key in existing_identities:
                    ex_row = existing_identities[identity_key]
                    is_conflict, conflict_detail = _check_conflict_with_ledger(item, ex_row)
                    if is_conflict:
                        warn_msg = f"Conflict กับ Ledger เดิมในอีเมล '{m.subject}': {item.symbol} (Conf: {c_no}, Order: {o_id}) {conflict_detail}"
                        log.warning(warn_msg)
                        warn_info = _create_warning_info(m, warn_msg)
                        warnings.append(warn_info)
                        yield f"event: warning\ndata: {json.dumps(warn_info, ensure_ascii=False)}\n\n"
                        email_conflict = True
                        break
                    else:
                        log.info("Skipping ARR Gold trade %s already active in portfolio (Conf: %s, Order: %s)", item.symbol, c_no, o_id)
                        total_already_in_portfolio += 1
                        continue

                if identity_key in batch_identity_map:
                    prev_item = batch_identity_map[identity_key]
                    if (
                        prev_item.symbol.strip().upper() != item.symbol.strip().upper()
                        or prev_item.action.strip().upper() != item.action.strip().upper()
                        or prev_item.units != item.units
                        or prev_item.price != item.price
                        or prev_item.net_amount != item.net_amount
                    ):
                        warn_msg = f"Conflict ระหว่างเอกสาร: คำสั่งซื้อขาย ({c_no}, Order: {o_id}) ซ้ำกันแต่มียอดเงินขัดแย้งกัน"
                        log.warning(warn_msg)
                        warn_info = _create_warning_info(m, warn_msg)
                        warnings.append(warn_info)
                        yield f"event: warning\ndata: {json.dumps(warn_info, ensure_ascii=False)}\n\n"
                        email_conflict = True
                        break
                    else:
                        log.info("Consolidating duplicate ARR Gold trade across confirmation notes %s (Conf: %s, Order: %s)", item.symbol, c_no, o_id)
                        continue

                email_items_to_stage.append(item)

            if email_conflict:
                email_status_map[msg_key] = {
                    "status": "quarantined",
                    "has_error": True,
                    "subject": m.subject,
                }
            else:
                for item in email_items_to_stage:
                    batch_identity_map[(item.confirmation_no.strip(), (item.order_id or item.confirmation_no).strip())] = item
                    all_valid_items.append(item)

                email_status_map[msg_key] = {
                    "status": "staged" if email_items_to_stage else "already_synced",
                    "has_error": False,
                    "subject": m.subject,
                    "items": email_items_to_stage,
                    "confirmation_no": parsed_items[0].confirmation_no if parsed_items else "",
                    "order_ids": [it.order_id for it in parsed_items if it.order_id],
                }

        scan_id = ""
        if all_valid_items:
            scan_id = self.staging.stage_items(items=all_valid_items, session_id=session_id)
            prov = {
                "account_email": account_email,
                "portfolio_id": portfolio_id,
                "emails": email_status_map,
            }
            self.staging.stage_provenance(scan_id=scan_id, provenance=prov, session_id=session_id)
            self._scan_provenance[scan_id] = prov

        already_active_emails = {
            k: v for k, v in email_status_map.items()
            if v.get("status") == "already_synced" and not v.get("has_error")
        }
        if already_active_emails:
            now_iso = datetime.now().isoformat()
            synced_emails = sync_history.setdefault("synced_emails", {})
            for msg_key, em_info in already_active_emails.items():
                synced_emails[msg_key] = {
                    "synced_at": now_iso,
                    "subject": em_info.get("subject", ""),
                    "confirmation_no": em_info.get("confirmation_no", ""),
                    "order_ids": em_info.get("order_ids", []),
                }
            _save_sync_history(portfolio_id=portfolio_id, account_email=account_email, history=sync_history)

        dto_items = [_item_to_preview_dict(it) for it in all_valid_items]

        progress_payload = {
            "message": f"พบ {len(dto_items)} รายการใหม่ที่พร้อมนำเข้า" if dto_items else "การประมวลผลเสร็จสมบูรณ์",
            "current": total_to_process,
            "total": total_to_process,
            "percent": 100,
            "items_found": len(dto_items),
        }
        yield f"event: progress\ndata: {json.dumps(progress_payload, ensure_ascii=False)}\n\n"

        complete_payload = {
            "scan_id": scan_id,
            "item_count": len(dto_items),
            "items": dto_items,
            "warnings": warnings,
            "skipped_synced_count": skipped_count,
            "already_in_portfolio_count": total_already_in_portfolio,
            "imported": 0,
            "message": (
                f"พบ {len(dto_items)} รายการใหม่ที่พร้อมนำเข้า"
                if dto_items
                else f"เอกสารทั้งหมดได้รับการประมวลผลแล้ว (มีในพอร์ตแล้ว {len(already_active_emails) + skipped_count} ฉบับ)"
            ),
        }
        yield f"event: complete\ndata: {json.dumps(complete_payload, ensure_ascii=False)}\n\n"

