from contextlib import contextmanager
import csv
from datetime import datetime
from decimal import Decimal
import io
import json
import logging
import os
from pathlib import Path
import re
from typing import Any, Dict, Generator, List, Optional, Set, Tuple

import pypdf

from tools.portfolio.adapters.markdown.paths import get_vault_path, get_trades_log_filepath
from tools.portfolio.domain.calculations import extract_active_ledger_identities
from tools.portfolio.domain.models import PortfolioState, TradeImportItem
from tools.portfolio.ports.trade_ingestion_port import (
    TradeEmailSourcePort,
    TradeDocumentParserPort,
    TradeStagingPort,
    TradeDocumentMetadata,
)
from .batch_trade_import_service import BatchTradeImportService

log = logging.getLogger(__name__)


def _sanitize_email_for_filename(email: str) -> str:
    cleaned = (email or "default").strip().lower()
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", cleaned)


def _get_sync_history_file(portfolio_id: str, account_email: str) -> Path:
    sync_dir = get_vault_path() / ".sync_history"
    from tools.archivist.maintenance_guard import assert_write_allowed
    assert_write_allowed(sync_dir)
    sync_dir.mkdir(parents=True, exist_ok=True)
    sanitized = _sanitize_email_for_filename(account_email)
    return sync_dir / f"dime_{sanitized}_{portfolio_id}.json"



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
        log.warning("Could not read sync history file %s: %s", history_file, e)
        return {"account_email": account_email, "portfolio_id": portfolio_id, "synced_emails": {}}


def _save_sync_history(portfolio_id: str, account_email: str, history: Dict[str, Any]) -> None:
    history_file = _get_sync_history_file(portfolio_id, account_email)
    temp_file = history_file.with_suffix(".tmp")
    try:
        with temp_file.open("w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        temp_file.replace(history_file)
    except Exception as e:
        log.error("Failed to write sync history to %s: %s", history_file, e)
        if temp_file.exists():
            temp_file.unlink(missing_ok=True)


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
            log.warning("Could not load identities from repository unit of work for %s: %s", portfolio_id, e)

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


def _item_to_preview_dict(it: TradeImportItem) -> Dict[str, Any]:
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
        "asset_type": getattr(it, "asset_type", "Stock"),
    }


def _create_warning_info(
    m: TradeDocumentMetadata,
    reason: str,
) -> Dict[str, Any]:
    return {
        "message_id": m.message_id,
        "attachment_id": m.attachment_id,
        "subject": m.subject,
        "filename": m.filename or "confirmation_note.pdf",
        "received_at": m.received_at,
        "reason": reason,
        "can_preview": bool(m.attachment_id or m.message_id),
    }


class DimeSyncService:
    """Application service for end-to-end Dime synchronization."""

    def __init__(
        self,
        email_source: TradeEmailSourcePort,
        parser: TradeDocumentParserPort,
        staging: TradeStagingPort,
        batch_import_service: BatchTradeImportService,
    ):
        self.email_source = email_source
        self.parser = parser
        self.staging = staging
        self.batch_import_service = batch_import_service
        self._scan_provenance: Dict[str, Dict[str, Any]] = {}

    def scan_emails(self, query: str = "", limit: int = 500) -> List[TradeDocumentMetadata]:
        return self.email_source.search_dime_emails(query=query, limit=limit)

    def get_email_pdf(
        self,
        message_id: str,
        attachment_id: str,
        password: Optional[str] = None,
        decrypt: bool = True,
    ) -> Tuple[bytes, str]:
        """Fetch PDF bytes for an email attachment, optionally decrypting it."""
        pdf_bytes = self.email_source.fetch_pdf_attachment(message_id=message_id, attachment_id=attachment_id)
        filename = f"dime_{message_id}.pdf"

        if not decrypt:
            return pdf_bytes, filename

        effective_password = (password.strip() if password and password.strip() else None) or os.getenv("DIME_PDF_PASSWORD", "")
        try:
            reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
            if reader.is_encrypted:
                if effective_password:
                    decrypt_res = reader.decrypt(effective_password)
                    if decrypt_res == 0:
                        log.warning("PDF decryption failed: invalid password")
                        return pdf_bytes, filename
                else:
                    return pdf_bytes, filename

            writer = pypdf.PdfWriter()
            for page in reader.pages:
                writer.add_page(page)
            buf = io.BytesIO()
            writer.write(buf)
            return buf.getvalue(), filename
        except Exception as e:
            log.warning("Error decrypting PDF: %s", e)
            return pdf_bytes, filename

    def get_email_pdf_text(
        self,
        message_id: str,
        attachment_id: str,
        password: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Fetch PDF, decrypt, and extract raw text from each page."""
        pdf_bytes, filename = self.get_email_pdf(
            message_id=message_id,
            attachment_id=attachment_id,
            password=password,
            decrypt=True,
        )
        try:
            reader = pypdf.PdfReader(io.BytesIO(pdf_bytes))
            if reader.is_encrypted:
                effective_password = (password.strip() if password and password.strip() else None) or os.getenv("DIME_PDF_PASSWORD", "")
                if effective_password:
                    reader.decrypt(effective_password)

            pages = []
            for idx, page in enumerate(reader.pages):
                txt = page.extract_text() or ""
                pages.append({
                    "page_number": idx + 1,
                    "text": txt,
                })
            return {
                "message_id": message_id,
                "attachment_id": attachment_id,
                "filename": filename,
                "page_count": len(pages),
                "pages": pages,
            }
        except Exception as e:
            log.error("Error extracting text from PDF: %s", e)
            raise ValueError(f"ไม่สามารถอ่านข้อความจากไฟล์ PDF: {e}") from e

    def parse_and_stage_email(
        self,
        message_id: str,
        attachment_id: str,
        password: Optional[str] = None,
        session_id: Optional[str] = None,
        portfolio_id: str = "default",
    ) -> Tuple[str, List[TradeImportItem]]:
        pdf_bytes = self.email_source.fetch_pdf_attachment(message_id=message_id, attachment_id=attachment_id)
        items = self.parser.parse_confirmation_pdf(pdf_bytes=pdf_bytes, password=password)
        scan_id = self.staging.stage_items(items=items, session_id=session_id)

        # Stage provenance so commit_staged can persist to .sync_history across HTTP requests
        account_email = os.getenv("GMAIL_IMAP_USER", "")
        email_info = {
            "status": "staged",
            "has_error": False,
            "uid": "",
            "x_gm_msgid": message_id,
            "subject": "",
            "items": items,
            "confirmation_no": items[0].confirmation_no if items else "",
            "order_ids": [it.order_id for it in items if it.order_id],
        }
        prov = {
            "account_email": account_email,
            "portfolio_id": portfolio_id,
            "emails": {message_id: email_info},
        }
        self.staging.stage_provenance(scan_id=scan_id, provenance=prov, session_id=session_id)
        self._scan_provenance[scan_id] = prov
        return scan_id, items

    def parse_and_stage_upload(
        self,
        pdf_bytes: bytes,
        password: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Tuple[str, List[TradeImportItem]]:
        items = self.parser.parse_confirmation_pdf(pdf_bytes=pdf_bytes, password=password)
        scan_id = self.staging.stage_items(items=items, session_id=session_id)
        return scan_id, items

    def get_staged(self, scan_id: str, session_id: Optional[str] = None) -> List[TradeImportItem]:
        return self.staging.get_staged_items(scan_id=scan_id, session_id=session_id)

    def stream_batch_sync(
        self,
        password: Optional[str] = None,
        force_rescan: bool = False,
        portfolio_id: str = "default",
        session_id: Optional[str] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """Stream real-time discovery, parsing, validation, and staging of all Dime confirmation notes."""
        yield {
            "event": "start",
            "data": {
                "message": "กำลังค้นหาเอกสาร Confirmation Note ทั้งหมดจาก Dime...",
                "force_rescan": force_rescan,
            },
        }

        # 1. Search all confirmation emails
        emails = self.email_source.search_dime_emails(query="", limit=500)
        total_found = len(emails)
        account_email = os.getenv("GMAIL_IMAP_USER", "")
        if emails and emails[0].account_email:
            account_email = emails[0].account_email

        # 2. Check sync history with smart active ledger reconciliation
        history = _load_sync_history(portfolio_id=portfolio_id, account_email=account_email)
        synced_map = history.get("synced_emails", {})
        repo = getattr(self.batch_import_service, "repo", None)
        existing_identity_map = _load_existing_ledger_identities(portfolio_id=portfolio_id, repo=repo)

        emails_to_process: List[TradeDocumentMetadata] = []
        skipped_count = 0

        for m in emails:
            msg_key = m.x_gm_msgid or m.message_id
            rec = synced_map.get(msg_key)

            needs_reprocess = False
            if force_rescan or not rec:
                needs_reprocess = True
            else:
                # Reconcile: verify whether orders in this email are still active in the portfolio ledger
                order_ids = rec.get("order_ids", [])
                conf_no = rec.get("confirmation_no", "")
                if not order_ids:
                    needs_reprocess = True
                else:
                    for oid in order_ids:
                        if (conf_no, str(oid)) not in existing_identity_map:
                            log.info(
                                "Detected missing/voided trade (%s, %s) from email %s. Triggering re-sync.",
                                conf_no,
                                oid,
                                msg_key,
                            )
                            needs_reprocess = True
                            break

            if needs_reprocess:
                emails_to_process.append(m)
            else:
                skipped_count += 1

        yield {
            "event": "discovered",
            "data": {
                "total_found": total_found,
                "to_process": len(emails_to_process),
                "already_synced": skipped_count,
            },
        }

        if not emails_to_process:
            yield {
                "event": "complete",
                "data": {
                    "scan_id": "",
                    "item_count": 0,
                    "items": [],
                    "warnings": [],
                    "skipped_synced_count": skipped_count,
                    "already_in_portfolio_count": skipped_count,
                    "message": "เอกสารทั้งหมดได้รับการประมวลผลแล้ว (ไม่มีรายการใหม่)",
                },
            }
            return

        # 3. Prepare conflict checking structures
        batch_identity_map: Dict[Tuple[str, str], TradeImportItem] = {}
        all_valid_items: List[TradeImportItem] = []
        email_status_map: Dict[str, Dict[str, Any]] = {}
        warnings: List[Dict[str, str]] = []
        total_already_in_portfolio = 0

        total_to_process = len(emails_to_process)

        for idx, m in enumerate(emails_to_process):
            msg_key = m.x_gm_msgid or m.message_id
            pct = round(((idx) / total_to_process) * 100, 1)

            yield {
                "event": "progress",
                "data": {
                    "current": idx + 1,
                    "total": total_to_process,
                    "percent": pct,
                    "items_found": len(all_valid_items),
                    "subject": m.subject,
                    "message_id": m.message_id,
                },
            }

            # A. Fetch PDF
            try:
                pdf_bytes = self.email_source.fetch_pdf_attachment(
                    message_id=m.message_id,
                    attachment_id=m.attachment_id,
                )
            except Exception as e:
                warn_msg = f"ไม่สามารถดาวน์โหลดไฟล์แนบจากอีเมล '{m.subject}': {e}"
                log.warning(warn_msg)
                warn_info = _create_warning_info(m, warn_msg)
                warnings.append(warn_info)
                email_status_map[msg_key] = {"status": "quarantined", "has_error": True, "reason": warn_msg}
                yield {"event": "warning", "data": warn_info}
                continue

            # B. Parse PDF in isolated worker process
            try:
                parsed_items = self.parser.parse_confirmation_pdf(pdf_bytes=pdf_bytes, password=password)
            except Exception as e:
                warn_msg = f"ไม่สามารถอ่านข้อมูล PDF จากอีเมล '{m.subject}': {e}"
                log.warning(warn_msg)
                warn_info = _create_warning_info(m, warn_msg)
                warnings.append(warn_info)
                email_status_map[msg_key] = {"status": "quarantined", "has_error": True, "reason": warn_msg}
                yield {"event": "warning", "data": warn_info}
                continue

            if not parsed_items:
                warn_msg = f"อีเมล '{m.subject}' ไม่มีรายการคำสั่งซื้อขายที่รองรับ"
                log.info(warn_msg)
                warn_info = _create_warning_info(m, warn_msg)
                warnings.append(warn_info)
                email_status_map[msg_key] = {"status": "quarantined", "has_error": True, "reason": warn_msg}
                yield {"event": "warning", "data": warn_info}
                continue

            # C. Invariant Validation & Conflict Checking
            email_conflict = False
            email_items_to_stage: List[TradeImportItem] = []

            for item in parsed_items:
                c_no = item.confirmation_no.strip()
                o_id = (item.order_id or "").strip()

                if not o_id and item.source.upper() == "DIME":
                    warn_msg = f"รายการ {item.symbol} ในอีเมล '{m.subject}' ขาด Order ID (ห้ามใช้ fallback line_index)"
                    warn_info = _create_warning_info(m, warn_msg)
                    warnings.append(warn_info)
                    yield {"event": "warning", "data": warn_info}
                    email_conflict = True
                    break

                identity_key = (c_no, o_id)

                # Check conflict or match with active ledger rows
                if identity_key in existing_identity_map:
                    ex_row = existing_identity_map[identity_key]
                    is_conflict, conflict_detail = _check_conflict_with_ledger(item, ex_row)
                    if is_conflict:
                        warn_msg = f"Conflict กับ Ledger เดิมในอีเมล '{m.subject}': {item.symbol} (Conf: {c_no}, Order: {o_id}) {conflict_detail}"
                        log.warning(warn_msg)
                        warn_info = _create_warning_info(m, warn_msg)
                        warnings.append(warn_info)
                        yield {"event": "warning", "data": warn_info}
                        email_conflict = True
                        break
                    else:
                        # Exact match already active in portfolio ledger -> Skip it safely!
                        log.info("Skipping trade %s already active in portfolio (Conf: %s, Order: %s)", item.symbol, c_no, o_id)
                        total_already_in_portfolio += 1
                        continue

                # Check conflict with earlier emails in this batch
                if identity_key in batch_identity_map:
                    prev_item = batch_identity_map[identity_key]
                    # Check if figures differ
                    if (
                        prev_item.symbol.strip().upper() != item.symbol.strip().upper()
                        or prev_item.action.strip().upper() != item.action.strip().upper()
                        or prev_item.units != item.units
                        or prev_item.price != item.price
                        or prev_item.net_amount != item.net_amount
                        or prev_item.fees.commission != item.fees.commission
                    ):
                        warn_msg = f"Conflict ระหว่างเอกสาร: คำสั่งซื้อขาย ({c_no}, Order: {o_id}) ซ้ำกันแต่มียอดเงินขัดแย้งกัน"
                        log.warning(warn_msg)
                        warn_info = _create_warning_info(m, warn_msg)
                        warnings.append(warn_info)
                        yield {"event": "warning", "data": warn_info}
                        email_conflict = True
                        break
                    else:
                        # Exact duplicate across emails -> consolidate safely
                        log.info("Consolidating duplicate trade across confirmation notes %s (Conf: %s, Order: %s)", item.symbol, c_no, o_id)
                        continue

                email_items_to_stage.append(item)

            if email_conflict:
                email_status_map[msg_key] = {
                    "status": "quarantined",
                    "has_error": True,
                    "uid": m.uid,
                    "x_gm_msgid": m.x_gm_msgid,
                    "subject": m.subject,
                }
            else:
                for item in email_items_to_stage:
                    batch_identity_map[(item.confirmation_no.strip(), (item.order_id or "").strip())] = item
                    all_valid_items.append(item)

                email_status_map[msg_key] = {
                    "status": "staged" if email_items_to_stage else "already_synced",
                    "has_error": False,
                    "uid": m.uid,
                    "x_gm_msgid": m.x_gm_msgid,
                    "subject": m.subject,
                    "items": email_items_to_stage,
                    "confirmation_no": parsed_items[0].confirmation_no if parsed_items else "",
                    "order_ids": [it.order_id for it in parsed_items if it.order_id],
                }

        # 4. Stage valid items and persist provenance
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

        # 5. For emails verified to already have all items active in portfolio, record sync history right away
        already_active_emails = {
            k: v for k, v in email_status_map.items()
            if v.get("status") == "already_synced" and not v.get("has_error")
        }
        if already_active_emails:
            now_iso = datetime.now().isoformat()
            synced_emails = history.setdefault("synced_emails", {})
            for msg_key, em_info in already_active_emails.items():
                synced_emails[msg_key] = {
                    "uid": em_info.get("uid", ""),
                    "x_gm_msgid": em_info.get("x_gm_msgid", ""),
                    "synced_at": now_iso,
                    "item_count": len(em_info.get("order_ids", [])),
                    "confirmation_no": em_info.get("confirmation_no", ""),
                    "order_ids": em_info.get("order_ids", []),
                }
            _save_sync_history(portfolio_id=portfolio_id, account_email=account_email, history=history)

        dto_items = [_item_to_preview_dict(it) for it in all_valid_items]

        yield {
            "event": "complete",
            "data": {
                "scan_id": scan_id,
                "item_count": len(dto_items),
                "items": dto_items,
                "warnings": warnings,
                "skipped_synced_count": skipped_count,
                "already_in_portfolio_count": total_already_in_portfolio,
                "message": (
                    f"พบ {len(dto_items)} รายการใหม่/กู้คืนที่พร้อมนำเข้า"
                    if dto_items
                    else f"เอกสารทั้งหมดได้รับการประมวลผลแล้ว (มีในพอร์ตแล้ว {len(already_active_emails) + skipped_count} ฉบับ)"
                ),
            },
        }

    def commit_staged(
        self,
        scan_id: str,
        session_id: Optional[str] = None,
        portfolio_id: str = "default",
        selected_item_ids: Optional[List[str]] = None,
    ) -> PortfolioState:
        """Atomic commit under unit of work lock, updating sync history only upon success."""
        items = self.staging.get_staged_items(scan_id=scan_id, session_id=session_id)
        if selected_item_ids is not None:
            selected_set = set(selected_item_ids)
            items = [item for item in items if item.item_id in selected_set]

        if not items:
            raise ValueError("ไม่มีรายการใดถูกเลือกเพื่อนำเข้าสู่ระบบ")

        state = self.batch_import_service.execute_batch_import(items=items, portfolio_id=portfolio_id)
        self.staging.delete_staged(scan_id=scan_id, session_id=session_id)

        # Update sync history for emails whose trades were successfully committed
        provenance = self.staging.pop_provenance(scan_id=scan_id, session_id=session_id)
        if not provenance:
            provenance = self._scan_provenance.pop(scan_id, None)

        if provenance:
            account_email = provenance.get("account_email", "")
            history = _load_sync_history(portfolio_id=portfolio_id, account_email=account_email)
            synced_emails = history.setdefault("synced_emails", {})
            now_iso = datetime.now().isoformat()

            committed_order_ids = {it.order_id for it in items if it.order_id}
            committed_confirmations = {it.confirmation_no for it in items if it.confirmation_no}

            for msg_key, email_info in provenance.get("emails", {}).items():
                if email_info.get("status") in ("staged", "already_synced") and not email_info.get("has_error"):
                    if selected_item_ids is not None:
                        email_conf = email_info.get("confirmation_no", "")
                        email_orders = set(email_info.get("order_ids", []))
                        is_committed = (email_conf in committed_confirmations) or bool(email_orders.intersection(committed_order_ids))
                        if not is_committed and email_info.get("status") != "already_synced":
                            continue

                    synced_emails[msg_key] = {
                        "uid": email_info.get("uid", ""),
                        "x_gm_msgid": email_info.get("x_gm_msgid", ""),
                        "synced_at": now_iso,
                        "item_count": len(email_info.get("items", [])) or len(email_info.get("order_ids", [])),
                        "confirmation_no": email_info.get("confirmation_no", ""),
                        "order_ids": email_info.get("order_ids", []),
                    }

            _save_sync_history(portfolio_id=portfolio_id, account_email=account_email, history=history)

        return state
