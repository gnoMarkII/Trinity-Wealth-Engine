"""Application service for staged BBLAM PVD statement snapshot imports."""
from __future__ import annotations

import hashlib
import json
import uuid
from decimal import Decimal
from typing import Optional

from tools.portfolio.domain.calculations import recalc_all
from tools.portfolio.domain.ledger_change import LedgerChange
from tools.portfolio.domain.models import PortfolioState, TradeFeeBreakdown, TradeImportItem, _now_iso
from tools.portfolio.domain.pvd_calculations import MONEY, money, semantic_fingerprint, vested_amount
from tools.portfolio.domain.pvd_models import PvdStatement, PvdStatementMetadata
from tools.portfolio.domain.pvd_replay import apply_pvd_snapshot_rows
from tools.portfolio.domain.pvd_validation import validate_statement
from tools.portfolio.domain.validator import validate_portfolio_id
from tools.portfolio.ports.pvd_port import PvdCommitResult, PvdDocumentArchivePort, PvdParserPort, PvdScanResult
from tools.portfolio.ports.repository_port import PortfolioRepositoryPort
from tools.portfolio.ports.trade_ingestion_port import TradeStagingPort


MAX_PDF_BYTES = 10 * 1024 * 1024
PVD_BUCKET_ID = "retirement_pvd"


def _money(value: Decimal) -> str:
    return f"{money(value):.2f}"


def _json_model(model) -> str:
    return json.dumps(model.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _preview(statement: PvdStatement, vesting_percent: Decimal) -> tuple[list[dict], Decimal]:
    member_total = statement.summary.member_principal + statement.summary.member_benefit
    employer_total = statement.summary.employer_principal + statement.summary.employer_benefit
    vested = vested_amount(member_total, employer_total, vesting_percent)
    rows = []
    for holding in statement.holdings:
        valuation = holding.statement_market_value + holding.reconciliation_adjustment
        rows.append({
            "fund_code": holding.code,
            "name": holding.name,
            "units": str(holding.units),
            "statement_nav": str(holding.statement_nav),
            "statement_market_value": _money(holding.statement_market_value),
            "reconciliation_adjustment": _money(holding.reconciliation_adjustment),
            "adjusted_valuation": _money(valuation),
            "allocated_cost": _money(holding.allocated_cost),
            "unrealized_pnl": _money(valuation - holding.allocated_cost),
        })
    return rows, vested


class PvdSyncService:
    def __init__(
        self,
        *,
        repo: PortfolioRepositoryPort,
        parser: PvdParserPort,
        archive: PvdDocumentArchivePort,
        staging: TradeStagingPort,
    ) -> None:
        self.repo = repo
        self.parser = parser
        self.archive = archive
        self.staging = staging

    def analyze_upload(
        self,
        pdf_bytes: bytes,
        *,
        portfolio_id: str,
        session_id: str,
        password: Optional[str] = None,
        vesting_percent: Decimal = Decimal("100"),
    ) -> PvdScanResult:
        pid = validate_portfolio_id(portfolio_id)
        if not pdf_bytes or len(pdf_bytes) > MAX_PDF_BYTES:
            raise ValueError("ขนาดไฟล์ PDF ต้องไม่เกิน 10 MB")
        if not vesting_percent.is_finite() or not Decimal("0") <= vesting_percent <= Decimal("100"):
            raise ValueError("Vesting ต้องอยู่ระหว่าง 0 ถึง 100 เปอร์เซ็นต์")
        if not self.repo.portfolio_exists(pid):
            raise ValueError(f"ไม่พบพอร์ตไอดี '{pid}' ในระบบ")

        statement = self.parser.parse_statement(pdf_bytes, password=password)
        statement.source_pdf_sha256 = hashlib.sha256(pdf_bytes).hexdigest()
        # Keep only the masked display value and deterministic scope hash in
        # staged state; the raw member number never leaves this parse call.
        statement.member_id = "REDACTED"
        statement = validate_statement(statement)
        fingerprint = semantic_fingerprint(statement)
        state = self.repo.load_state(pid)
        recalc_all(state)

        current_nav = sum((Decimal(str(h.market_value_thb)) for h in state.holdings if h.asset_type == "PVD"), Decimal("0"))
        metadata = list(state.pvd_statements)
        if metadata:
            first = metadata[0]
            if first.fund_code != statement.fund_code or first.member_hash != statement.member_hash:
                return PvdScanResult(None, False, "scope_conflict", "พอร์ตนี้เชื่อมกับสมาชิกหรือกองทุน PVD รายอื่นแล้ว")
            for previous in metadata:
                if previous.semantic_fingerprint == fingerprint:
                    return PvdScanResult(None, False, "duplicate", "Statement นี้ถูกนำเข้าแล้ว", current_nav=_money(current_nav), new_nav=_money(statement.summary.total_nav), delta_nav="0.00")
            latest = max(metadata, key=lambda item: item.as_of_date)
            if statement.as_of_date < latest.as_of_date:
                return PvdScanResult(None, False, "backdated", "วันที่ Statement เก่ากว่างวดล่าสุดที่นำเข้าแล้ว")
            if statement.as_of_date == latest.as_of_date:
                return PvdScanResult(None, False, "date_conflict", "มี Statement ที่ต่างกันสำหรับวันที่เดียวกันอยู่แล้ว")

        current_pvd_codes = {
            h.symbol.removeprefix("PVD_").upper()
            for h in state.holdings
            if h.asset_type == "PVD" and h.symbol.upper().startswith("PVD_")
        }
        statement_codes = {row.code.upper() for row in statement.holdings}
        missing = sorted(current_pvd_codes - statement_codes)
        if missing:
            return PvdScanResult(None, False, "incomplete_statement", "งวดใหม่ไม่มีแถวของกองทุนเดิม จึงไม่สามารถสรุปได้ว่ากองทุนปิดแล้ว: " + ", ".join(missing))

        preview_rows, vested = _preview(statement, vesting_percent)
        items: list[TradeImportItem] = []
        for index, holding in enumerate(statement.holdings):
            item_id = f"pvd_{fingerprint[:16]}_{holding.code}"
            items.append(TradeImportItem(
                item_id=item_id,
                trade_date=statement.as_of_date.isoformat(),
                symbol=f"PVD_{holding.code}",
                action="PVD_SNAPSHOT",
                units=holding.units,
                price=holding.statement_nav,
                gross_amount=holding.statement_market_value,
                fees=TradeFeeBreakdown(),
                net_amount=holding.statement_market_value,
                currency="THB",
                confirmation_no=fingerprint,
                order_id=holding.code,
                source="BBLAM_PVD",
                fingerprint=hashlib.sha256(f"{fingerprint}|{holding.code}".encode()).hexdigest(),
                line_index=index,
                cash_adjusted=False,
                asset_type="PVD",
                bucket_id=PVD_BUCKET_ID,
                pvd_metadata_json=_json_model(statement),
            ))

        scan_id = self.staging.stage_items(items, ttl_seconds=1800, session_id=session_id)
        self.staging.stage_provenance(scan_id, {
            "kind": "BBLAM_PVD",
            "portfolio_id": pid,
            "epoch": state.pvd_import_epoch,
            "statement": statement.model_dump(mode="json"),
            "pdf_bytes": pdf_bytes,
            "fingerprint": fingerprint,
            "vesting_percent": str(vesting_percent),
        }, session_id=session_id)
        return PvdScanResult(
            scan_id,
            True,
            "ready",
            "ตรวจ Statement ผ่าน พร้อมนำเข้ายอด snapshot",
            statement={
                "fund_code": statement.fund_code,
                "masked_member_id": statement.masked_member_id,
                "as_of_date": statement.as_of_date.isoformat(),
                "period_start": statement.period_start.isoformat() if statement.period_start else None,
                "period_end": statement.period_end.isoformat() if statement.period_end else None,
                "member_principal": _money(statement.summary.member_principal),
                "member_benefit": _money(statement.summary.member_benefit),
                "employer_principal": _money(statement.summary.employer_principal),
                "employer_benefit": _money(statement.summary.employer_benefit),
                "total_contribution": _money(statement.summary.total_contribution),
                "total_nav": _money(statement.summary.total_nav),
                "member_vested_amount": _money(statement.summary.member_principal + statement.summary.member_benefit),
                "employer_vested_amount": _money(vested - statement.summary.member_principal - statement.summary.member_benefit),
                "semantic_fingerprint": fingerprint,
                "source_pdf_sha256": statement.source_pdf_sha256,
            },
            holdings=tuple(preview_rows),
            current_nav=_money(current_nav),
            new_nav=_money(statement.summary.total_nav),
            delta_nav=_money(statement.summary.total_nav - current_nav),
            vesting_percent=str(vesting_percent),
            vested_amount=_money(vested),
        )

    def commit(
        self,
        scan_id: str,
        *,
        portfolio_id: str,
        session_id: str,
        idempotency_key: str,
        vesting_percent: Decimal,
        selected_item_ids: Optional[list[str]] = None,
    ) -> PvdCommitResult:
        pid = validate_portfolio_id(portfolio_id)
        if not idempotency_key.strip() or len(idempotency_key) > 200:
            raise ValueError("idempotency_key ไม่ถูกต้อง")
        if not vesting_percent.is_finite() or not Decimal("0") <= vesting_percent <= Decimal("100"):
            raise ValueError("Vesting ต้องอยู่ระหว่าง 0 ถึง 100 เปอร์เซ็นต์")
        selected_identity = "ALL" if selected_item_ids is None else ",".join(sorted(set(selected_item_ids)))
        request_hash = hashlib.sha256(
            f"{pid}|{scan_id}|{vesting_percent}|{selected_identity}".encode("utf-8")
        ).hexdigest()
        store = getattr(self.repo, "store", None)
        if store is not None:
            prior_receipt = store.get_command_receipt(pid, "pvd_statement_commit", idempotency_key)
            if prior_receipt:
                if prior_receipt["request_hash"] != request_hash:
                    raise ValueError("idempotency_key นี้ถูกใช้กับ Statement หรือข้อมูล vesting อื่นแล้ว")
                current_state = self.repo.load_state(pid)
                return PvdCommitResult(current_state.model_dump(mode="json"), "duplicate", "คำสั่งนี้ถูก commit แล้ว; ส่งคืนสถานะพอร์ตปัจจุบัน", prior_receipt.get("result", {}).get("archive_id"))

        items = self.staging.get_staged_items(scan_id, session_id=session_id)
        provenance = self.staging.get_provenance(scan_id, session_id=session_id)
        if not provenance or provenance.get("kind") != "BBLAM_PVD" or provenance.get("portfolio_id") != pid:
            raise ValueError("ไม่พบข้อมูล PVD ที่ staged สำหรับพอร์ตนี้")
        item_ids = {item.item_id for item in items}
        if selected_item_ids is not None and set(selected_item_ids) != item_ids:
            raise ValueError("PVD statement ต้อง commit รายการกองทุนย่อยทั้งชุด")
        statement = PvdStatement.model_validate(provenance["statement"])
        fingerprint = str(provenance["fingerprint"])

        with self.repo.unit_of_work(pid) as uow:
            state = uow.load_state()
            current_epoch = state.pvd_import_epoch
            if current_epoch != provenance.get("epoch"):
                raise ValueError("พอร์ตถูก reset หรือมีการเริ่ม PVD import ใหม่หลังจากสแกน กรุณาสแกนเอกสารอีกครั้ง")
            if store is not None:
                receipt = store.get_command_receipt(pid, "pvd_statement_commit", idempotency_key)
                if receipt:
                    if receipt["request_hash"] != request_hash:
                        raise ValueError("idempotency_key นี้ถูกใช้กับ Statement หรือข้อมูล vesting อื่นแล้ว")
                    self.staging.delete_staged(scan_id, session_id=session_id)
                    return PvdCommitResult(state.model_dump(mode="json"), "duplicate", "คำสั่งนี้ถูก commit แล้ว; ส่งคืนสถานะพอร์ตปัจจุบัน")

            for previous in state.pvd_statements:
                if previous.semantic_fingerprint == fingerprint:
                    self.staging.delete_staged(scan_id, session_id=session_id)
                    return PvdCommitResult(state.model_dump(mode="json"), "duplicate", "Statement นี้ถูกนำเข้าแล้ว")
                if previous.member_hash != statement.member_hash or previous.fund_code != statement.fund_code:
                    raise ValueError("พอร์ตนี้เชื่อมกับสมาชิกหรือกองทุน PVD รายอื่นแล้ว")
            if state.pvd_statements:
                latest = max(state.pvd_statements, key=lambda metadata: metadata.as_of_date)
                if statement.as_of_date <= latest.as_of_date:
                    raise ValueError("Statement ต้องใหม่กว่างวดล่าสุดในพอร์ต")

            known_codes = {
                h.symbol.removeprefix("PVD_").upper()
                for h in state.holdings
                if h.asset_type == "PVD" and h.symbol.upper().startswith("PVD_")
            }
            current_codes = {holding.code.upper() for holding in statement.holdings}
            if known_codes - current_codes:
                raise ValueError("Statement ไม่มีแถวของกองทุนที่มีอยู่ กรุณาตรวจสอบว่าเอกสารเป็นงวดเต็มก่อน commit")

            archive_id = self.archive.archive(
                provenance["pdf_bytes"], portfolio_id=pid, statement=statement
            )
            member_total = statement.summary.member_principal + statement.summary.member_benefit
            employer_total = statement.summary.employer_principal + statement.summary.employer_benefit
            employer_vested = money(employer_total * vesting_percent / Decimal("100"))
            metadata = PvdStatementMetadata(
                fund_code=statement.fund_code,
                member_hash=statement.member_hash,
                masked_member_id=statement.masked_member_id,
                as_of_date=statement.as_of_date,
                period_start=statement.period_start,
                period_end=statement.period_end,
                total_contribution=statement.summary.total_contribution,
                total_nav=statement.summary.total_nav,
                member_principal=statement.summary.member_principal,
                member_benefit=statement.summary.member_benefit,
                employer_principal=statement.summary.employer_principal,
                employer_benefit=statement.summary.employer_benefit,
                employer_vesting_percent=vesting_percent,
                member_vested_amount=member_total,
                employer_vested_amount=employer_vested,
                source_pdf_sha256=statement.source_pdf_sha256,
                semantic_fingerprint=fingerprint,
                archive_id=archive_id,
                reconciliation_adjustment=statement.summary.total_nav - sum((h.statement_market_value for h in statement.holdings), Decimal("0")),
            )

            existing_rows = uow.read_trade_log_locked()
            timestamp = f"{statement.as_of_date.isoformat()} 00:00:00"
            existing_holding_map = {h.symbol.upper(): h.bucket_id for h in state.holdings if h.bucket_id}
            valid_target_ids = {t.bucket_id for t in state.allocation_targets}

            new_rows = []
            for holding in statement.holdings:
                row_id = f"pvd_{fingerprint[:20]}_{holding.code.lower()}"
                sym = f"PVD_{holding.code.upper()}"
                assigned_bucket = existing_holding_map.get(sym)
                if not assigned_bucket:
                    assigned_bucket = PVD_BUCKET_ID if PVD_BUCKET_ID in valid_target_ids else ""
                new_rows.append({
                    "Transaction_ID": row_id,
                    "Timestamp": timestamp,
                    "Symbol": sym,
                    "Action": "PVD_SNAPSHOT",
                    "Units": str(holding.units),
                    "Price": str(holding.statement_nav),
                    "Currency": "THB",
                    "FX_Rate": "",
                    "Cost_THB": _money(holding.allocated_cost),
                    "Realized_PnL_THB": "",
                    "Notes": f"BBLAM PVD statement {statement.as_of_date.isoformat()}",
                    "Gross_Amount": _money(holding.statement_market_value),
                    "Commission": "0.00",
                    "VAT": "0.00",
                    "Other_Fees": "0.00",
                    "Net_Amount": _money(holding.statement_market_value),
                    "Fee_Currency": "THB",
                    "Confirmation_No": fingerprint,
                    "Order_ID": holding.code.upper(),
                    "Settlement_Date": "",
                    "Source": "BBLAM_PVD",
                    "Fingerprint": hashlib.sha256(f"{fingerprint}|{holding.code.upper()}".encode()).hexdigest(),
                    "Cash_Adjusted": "NO",
                    "Related_Transaction_ID": "",
                    "Asset_Type": "PVD",
                    "Bucket_ID": assigned_bucket,
                    "PVD_Metadata_JSON": _json_model(metadata),
                    "Reconciliation_Adjustment": _money(holding.reconciliation_adjustment),
                })

            all_rows = [dict(row) for row in existing_rows] + new_rows
            apply_pvd_snapshot_rows(state.holdings, all_rows)
            state.pvd_statements.append(metadata)
            state.pvd_import_epoch = state.pvd_import_epoch or uuid.uuid4().hex
            if not state.allocation_targets:
                from tools.portfolio.domain.models import AllocationTarget
                state.allocation_targets.append(AllocationTarget(bucket_id=PVD_BUCKET_ID, name="Provident Fund", target_percent=0.0, color="#14B8A6"))
            recalc_all(state)
            state.last_updated = _now_iso()
            command_receipt = {
                "use_case": "pvd_statement_commit",
                "idempotency_key": idempotency_key,
                "request_hash": request_hash,
                "command_id": f"pvd_cmd_{uuid.uuid4().hex}",
                "checkpoint_state_hash": "",
                "result": {"status": "committed", "archive_id": archive_id, "as_of_date": statement.as_of_date.isoformat()},
            }
            uow.commit(state, LedgerChange(kind="replace_all", rows=all_rows), command_receipt=command_receipt)

        self.staging.delete_staged(scan_id, session_id=session_id)
        return PvdCommitResult(state.model_dump(mode="json"), "committed", "นำเข้า Statement PVD สำเร็จ", archive_id)

    def read_document(self, archive_id: str, *, portfolio_id: str) -> bytes:
        pid = validate_portfolio_id(portfolio_id)
        state = self.repo.load_state(pid)
        metadata = next((item for item in state.pvd_statements if item.archive_id == archive_id), None)
        if metadata is None:
            raise FileNotFoundError("ไม่พบเอกสาร PVD ที่ commit ในพอร์ตนี้")
        return self.archive.read(portfolio_id=pid, metadata=metadata.model_dump(mode="json"))
