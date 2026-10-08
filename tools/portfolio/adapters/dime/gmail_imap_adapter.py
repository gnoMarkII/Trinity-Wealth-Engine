"""Gmail IMAP Driven Adapter (Hexagonal Architecture).

Searches and retrieves trade confirmation emails and PDF attachments from Gmail via IMAP.
"""
import email
from email.header import decode_header
import imaplib
import os
import re
from typing import List, Optional

from core.logger import get_logger
from tools.portfolio.ports.trade_ingestion_port import (
    TradeEmailSourcePort,
    TradeDocumentMetadata,
)

log = get_logger(__name__)


def _decode_mime_words(s: Optional[str]) -> str:
    if not s:
        return ""
    parts = decode_header(s)
    decoded = []
    for part, enc in parts:
        if isinstance(part, bytes):
            decoded.append(part.decode(enc or "utf-8", errors="replace"))
        else:
            decoded.append(str(part))
    return "".join(decoded)


class GmailImapSourceAdapter(TradeEmailSourcePort):
    """IMAP client for retrieving Dime confirmation PDFs from Gmail."""

    def __init__(
        self,
        username: Optional[str] = None,
        password: Optional[str] = None,
        host: str = "imap.gmail.com",
        port: int = 993,
    ):
        self.username = username or os.getenv("GMAIL_IMAP_USER", "")
        self.password = password or os.getenv("GMAIL_IMAP_PASSWORD", "")
        self.host = host
        self.port = port

    def _get_connection(self) -> imaplib.IMAP4_SSL:
        username = self.username or os.getenv("GMAIL_IMAP_USER", "")
        password = self.password or os.getenv("GMAIL_IMAP_PASSWORD", "")
        clean_user = username.strip() if username else ""
        clean_pass = password.replace(" ", "").replace("-", "").strip() if password else ""
        if not clean_user or not clean_pass:
            raise ValueError("ยังไม่ได้กำหนดค่า GMAIL_IMAP_USER หรือ GMAIL_IMAP_PASSWORD ในไฟล์ .env")
        mail = imaplib.IMAP4_SSL(self.host, self.port)
        try:
            mail.login(clean_user, clean_pass)
        except imaplib.IMAP4.error as e:
            raise ValueError(f"เข้าสู่ระบบ Gmail IMAP ไม่สำเร็จ: {e} (กรุณาตรวจสอบ GMAIL_IMAP_USER และ App Password)") from e
        return mail

    def search_dime_emails(
        self,
        query: str = "",
        limit: Optional[int] = None,
        since_date: Optional[str] = None,
    ) -> List[TradeDocumentMetadata]:
        username = self.username or os.getenv("GMAIL_IMAP_USER", "")
        clean_user = username.strip() if username else ""
        if not self.username or not self.password:
            log.warning("Gmail credentials not provided; returning empty search results.")
            return []

        mail = self._get_connection()
        try:
            mail.select("INBOX", readonly=True)
            if not query:
                base_crit = '(FROM "dime.co.th" OR SUBJECT "Confirmation" SUBJECT "Confirmation Note")'
            else:
                base_crit = f'(FROM "dime.co.th" SUBJECT "{query}")'

            if since_date:
                try:
                    if len(since_date) == 10 and since_date[4] == "-" and since_date[7] == "-":
                        dt = datetime.strptime(since_date, "%Y-%m-%d")
                        imap_date = dt.strftime("%d-%b-%Y")
                    else:
                        imap_date = since_date
                    search_crit = f'({base_crit} SINCE {imap_date})'
                except Exception:
                    search_crit = base_crit
            else:
                search_crit = base_crit

            try:
                typ, data = mail.uid("SEARCH", "CHARSET", "UTF-8", search_crit)
            except Exception:
                typ, data = mail.uid("SEARCH", None, search_crit)

            if typ != "OK" or not data or not data[0]:
                return []

            uids = data[0].split()
            # Most recent first
            uids.reverse()

            if limit is not None and limit > 0:
                target_uids = uids[:limit]
            else:
                target_uids = uids

            results: List[TradeDocumentMetadata] = []

            # Batch fetch headers in chunks of 50 to optimize performance (<1.5s for all messages)
            chunk_size = 50
            for i in range(0, len(target_uids), chunk_size):
                chunk = target_uids[i : i + chunk_size]
                uid_set = b",".join(chunk)
                typ, fetch_res = mail.uid("FETCH", uid_set, "(UID X-GM-MSGID RFC822.HEADER)")
                if typ != "OK" or not fetch_res:
                    continue

                for item in fetch_res:
                    if isinstance(item, tuple):
                        header_blob = item[0]
                        body_bytes = item[1]
                        raw_meta = header_blob + b"\n" + (body_bytes if isinstance(body_bytes, bytes) else b"")
                        m_uid = re.search(rb"UID\s+(\d+)", raw_meta)
                        m_msgid = re.search(rb"X-GM-MSGID\s+(\d+)", raw_meta)

                        uid_str = m_uid.group(1).decode("ascii") if m_uid else ""
                        x_gm_msgid = m_msgid.group(1).decode("ascii") if m_msgid else uid_str

                        if not uid_str:
                            continue

                        msg = email.message_from_bytes(body_bytes)
                        subj = _decode_mime_words(msg.get("Subject"))
                        from_addr = _decode_mime_words(msg.get("From"))
                        date_str = msg.get("Date", "")

                        results.append(TradeDocumentMetadata(
                            message_id=x_gm_msgid,
                            attachment_id=f"{uid_str}_att0",
                            subject=subj,
                            sender=from_addr,
                            received_at=date_str,
                            filename=f"dime_confirmation_{x_gm_msgid}.pdf",
                            size_bytes=1024,
                            x_gm_msgid=x_gm_msgid,
                            uid=uid_str,
                            account_email=clean_user,
                        ))

            return results
        except imaplib.IMAP4.error as e:
            log.warning("IMAP error during search: %s", e)
            raise ValueError(f"เกิดข้อผิดพลาดจากระบบ Gmail IMAP: {e}") from e
        except Exception as e:
            if isinstance(e, ValueError):
                raise
            log.warning("Error during Gmail search: %s", e)
            raise ValueError(f"ไม่สามารถค้นหาอีเมลใน Gmail ได้: {e}") from e
        finally:
            try:
                mail.close()
            except Exception:
                pass
            try:
                mail.logout()
            except Exception:
                pass

    def search_wealthx_emails(
        self,
        query: str = "",
        limit: Optional[int] = None,
        since_date: Optional[str] = None,
    ) -> List[TradeDocumentMetadata]:
        username = self.username or os.getenv("GMAIL_IMAP_USER", "")
        clean_user = username.strip() if username else ""
        if not self.username or not self.password:
            log.warning("Gmail credentials not provided; returning empty search results.")
            return []

        mail = self._get_connection()
        try:
            mail.select("INBOX", readonly=True)
            base_crit = '(FROM "wealthx.co")'
            if since_date:
                try:
                    if len(since_date) == 10 and since_date[4] == "-" and since_date[7] == "-":
                        dt = datetime.strptime(since_date, "%Y-%m-%d")
                        imap_date = dt.strftime("%d-%b-%Y")
                    else:
                        imap_date = since_date
                    search_crit = f'({base_crit} SINCE {imap_date})'
                except Exception:
                    search_crit = base_crit
            else:
                search_crit = base_crit

            try:
                typ, data = mail.uid("SEARCH", "CHARSET", "UTF-8", search_crit)
            except Exception:
                typ, data = mail.uid("SEARCH", None, search_crit)

            if typ != "OK" or not data or not data[0]:
                return []

            uids = data[0].split()
            uids.reverse()

            results: List[TradeDocumentMetadata] = []
            chunk_size = 50
            for i in range(0, len(uids), chunk_size):
                chunk = uids[i : i + chunk_size]
                uid_set = b",".join(chunk)
                typ, fetch_res = mail.uid("FETCH", uid_set, "(UID X-GM-MSGID RFC822.HEADER)")
                if typ != "OK" or not fetch_res:
                    continue

                for item in fetch_res:
                    if isinstance(item, tuple):
                        header_blob = item[0]
                        body_bytes = item[1]
                        raw_meta = header_blob + b"\n" + (body_bytes if isinstance(body_bytes, bytes) else b"")
                        m_uid = re.search(rb"UID\s+(\d+)", raw_meta)
                        m_msgid = re.search(rb"X-GM-MSGID\s+(\d+)", raw_meta)

                        uid_str = m_uid.group(1).decode("ascii") if m_uid else ""
                        x_gm_msgid = m_msgid.group(1).decode("ascii") if m_msgid else uid_str

                        if not uid_str:
                            continue

                        msg = email.message_from_bytes(body_bytes)
                        subj = _decode_mime_words(msg.get("Subject"))
                        from_addr = _decode_mime_words(msg.get("From"))
                        date_str = msg.get("Date", "")

                        # Filter: Trade Confirmations ONLY (exclude monthly statements or other notifications)
                        is_confirmation = "ใบยืนยัน" in subj or "confirmation" in subj.lower()
                        is_statement = "statement" in subj.lower() or "รายงานยอด" in subj
                        if not is_confirmation or is_statement:
                            continue

                        # Optional query filter
                        if query and (query.lower() not in subj.lower()):
                            continue

                        results.append(TradeDocumentMetadata(
                            message_id=x_gm_msgid,
                            attachment_id=f"{uid_str}_att0",
                            subject=subj,
                            sender=from_addr,
                            received_at=date_str,
                            filename=f"wealthx_confirmation_{x_gm_msgid}.pdf",
                            size_bytes=1024,
                            x_gm_msgid=x_gm_msgid,
                            uid=uid_str,
                            account_email=clean_user,
                        ))

                        if limit is not None and limit > 0 and len(results) >= limit:
                            return results

            return results
        except imaplib.IMAP4.error as e:
            log.warning("IMAP error during WealthX search: %s", e)
            raise ValueError(f"เกิดข้อผิดพลาดจากระบบ Gmail IMAP: {e}") from e
        except Exception as e:
            if isinstance(e, ValueError):
                raise
            log.warning("Error during Gmail WealthX search: %s", e)
            raise ValueError(f"ไม่สามารถค้นหาอีเมล WealthX ใน Gmail ได้: {e}") from e
        finally:
            try:
                mail.close()
            except Exception:
                pass
            try:
                mail.logout()
            except Exception:
                pass

    def fetch_pdf_attachment(self, message_id: str, attachment_id: str) -> bytes:
        mail = self._get_connection()
        try:
            mail.select("INBOX", readonly=True)

            target_uid = None
            if attachment_id and "_" in attachment_id:
                candidate_uid = attachment_id.split("_")[0]
                if candidate_uid.isdigit():
                    target_uid = candidate_uid

            if not target_uid:
                typ, search_data = mail.uid("SEARCH", None, f"X-GM-MSGID {message_id}")
                if typ == "OK" and search_data and search_data[0]:
                    target_uid = search_data[0].split()[-1].decode("ascii")
                elif message_id.isdigit():
                    target_uid = message_id

            if not target_uid:
                raise ValueError(f"ไม่สามารถระบุ UID สำหรับข้อความ ID '{message_id}'")

            typ, msg_data = mail.uid("FETCH", target_uid, "(RFC822)")
            if typ != "OK" or not msg_data or not msg_data[0] or not isinstance(msg_data[0], tuple):
                raise ValueError(f"ไม่สามารถดาวน์โหลดข้อความ UID '{target_uid}' (Message ID: {message_id}) จาก Gmail ได้")

            raw_email = msg_data[0][1]
            msg = email.message_from_bytes(raw_email)

            for part in msg.walk():
                content_type = part.get_content_type()
                filename = part.get_filename()
                if filename and (content_type == "application/pdf" or filename.lower().endswith(".pdf")):
                    payload = part.get_payload(decode=True)
                    if payload and payload.startswith(b"%PDF-"):
                        return payload

            raise ValueError(f"ไม่พบไฟล์แนบ PDF ที่ถูกต้องในข้อความ UID '{target_uid}' (Message ID: {message_id})")
        except imaplib.IMAP4.error as e:
            log.warning("IMAP error during fetch: %s", e)
            raise ValueError(f"เกิดข้อผิดพลาดจากระบบ Gmail IMAP ขณะดาวน์โหลดไฟล์: {e}") from e
        except Exception as e:
            if isinstance(e, ValueError):
                raise
            log.warning("Error during Gmail attachment fetch: %s", e)
            raise ValueError(f"ไม่สามารถดาวน์โหลดไฟล์แนบจาก Gmail ได้: {e}") from e
        finally:
            try:
                mail.close()
            except Exception:
                pass
            try:
                mail.logout()
            except Exception:
                pass

    def search_scbam_emails(self, query: str = "", limit: Optional[int] = None) -> List[TradeDocumentMetadata]:
        username = self.username or os.getenv("GMAIL_IMAP_USER", "")
        clean_user = username.strip() if username else ""
        if not self.username or not self.password:
            log.warning("Gmail credentials not provided; returning empty search results.")
            return []

        mail = self._get_connection()
        try:
            mail.select("INBOX", readonly=True)
            search_crit = '(FROM "fundclick.scbam@scb.co.th")'

            try:
                typ, data = mail.uid("SEARCH", "CHARSET", "UTF-8", search_crit)
            except Exception:
                typ, data = mail.uid("SEARCH", None, search_crit)

            if typ != "OK" or not data or not data[0]:
                return []

            uids = data[0].split()
            uids.reverse()

            results: List[TradeDocumentMetadata] = []
            chunk_size = 50
            for i in range(0, len(uids), chunk_size):
                chunk = uids[i : i + chunk_size]
                uid_set = b",".join(chunk)
                typ, fetch_res = mail.uid("FETCH", uid_set, "(UID X-GM-MSGID RFC822.HEADER)")
                if typ != "OK" or not fetch_res:
                    continue

                for item in fetch_res:
                    if isinstance(item, tuple):
                        header_blob = item[0]
                        body_bytes = item[1]
                        raw_meta = header_blob + b"\n" + (body_bytes if isinstance(body_bytes, bytes) else b"")
                        m_uid = re.search(rb"UID\s+(\d+)", raw_meta)
                        m_msgid = re.search(rb"X-GM-MSGID\s+(\d+)", raw_meta)

                        uid_str = m_uid.group(1).decode("ascii") if m_uid else ""
                        x_gm_msgid = m_msgid.group(1).decode("ascii") if m_msgid else uid_str

                        if not uid_str:
                            continue

                        msg = email.message_from_bytes(body_bytes)
                        subj = _decode_mime_words(msg.get("Subject"))
                        from_addr = _decode_mime_words(msg.get("From"))
                        date_str = msg.get("Date", "")

                        # Filter: Purchase confirmations ONLY
                        if "ยืนยันการทำรายการซื้อกองทุน" not in subj:
                            continue

                        if query and (query.lower() not in subj.lower()):
                            continue

                        results.append(TradeDocumentMetadata(
                            message_id=x_gm_msgid,
                            attachment_id=f"{uid_str}_body",
                            subject=subj,
                            sender=from_addr,
                            received_at=date_str,
                            filename=f"scbam_order_{x_gm_msgid}.html",
                            size_bytes=2048,
                            x_gm_msgid=x_gm_msgid,
                            uid=uid_str,
                            account_email=clean_user,
                        ))

                        if limit is not None and limit > 0 and len(results) >= limit:
                            return results

            return results
        except imaplib.IMAP4.error as e:
            log.warning("IMAP error during SCBAM search: %s", e)
            raise ValueError(f"เกิดข้อผิดพลาดจากระบบ Gmail IMAP: {e}") from e
        except Exception as e:
            if isinstance(e, ValueError):
                raise
            log.warning("Error during Gmail SCBAM search: %s", e)
            raise ValueError(f"ไม่สามารถค้นหาอีเมล SCBAM ใน Gmail ได้: {e}") from e
        finally:
            try:
                mail.close()
            except Exception:
                pass
            try:
                mail.logout()
            except Exception:
                pass

    def fetch_email_html_body(self, message_id: str) -> str:
        mail = self._get_connection()
        try:
            mail.select("INBOX", readonly=True)

            target_uid = None
            if message_id and "_" in message_id:
                candidate_uid = message_id.split("_")[0]
                if candidate_uid.isdigit():
                    target_uid = candidate_uid

            if not target_uid:
                typ, search_data = mail.uid("SEARCH", None, f"X-GM-MSGID {message_id}")
                if typ == "OK" and search_data and search_data[0]:
                    target_uid = search_data[0].split()[-1].decode("ascii")
                elif message_id.isdigit():
                    target_uid = message_id

            if not target_uid:
                raise ValueError(f"ไม่สามารถระบุ UID สำหรับข้อความ ID '{message_id}'")

            typ, msg_data = mail.uid("FETCH", target_uid, "(RFC822)")
            if typ != "OK" or not msg_data or not msg_data[0] or not isinstance(msg_data[0], tuple):
                raise ValueError(f"ไม่สามารถดาวน์โหลดข้อความ UID '{target_uid}' (Message ID: {message_id}) จาก Gmail ได้")

            raw_email = msg_data[0][1]
            msg = email.message_from_bytes(raw_email)

            html_body = ""
            plain_body = ""
            for part in msg.walk():
                ctype = part.get_content_type()
                if ctype == "text/html":
                    payload = part.get_payload(decode=True)
                    if payload:
                        html_body = payload.decode("utf-8", errors="replace")
                elif ctype == "text/plain":
                    payload = part.get_payload(decode=True)
                    if payload:
                        plain_body = payload.decode("utf-8", errors="replace")

            body_to_return = html_body or plain_body
            if not body_to_return:
                raise ValueError(f"ไม่พบเนื้อหาอีเมลในข้อความ UID '{target_uid}' (Message ID: {message_id})")

            return body_to_return
        except imaplib.IMAP4.error as e:
            log.warning("IMAP error during body fetch: %s", e)
            raise ValueError(f"เกิดข้อผิดพลาดจากระบบ Gmail IMAP ขณะอ่านเนื้อหาอีเมล: {e}") from e
        except Exception as e:
            if isinstance(e, ValueError):
                raise
            log.warning("Error during Gmail body fetch: %s", e)
            raise ValueError(f"ไม่สามารถอ่านเนื้อหาอีเมลจาก Gmail ได้: {e}") from e
        finally:
            try:
                mail.close()
            except Exception:
                pass
            try:
                mail.logout()
            except Exception:
                pass

