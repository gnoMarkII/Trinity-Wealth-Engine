"""Knowledge publication adapter converting confirmed snapshots to typed KnowledgeWriteCommands.

Complies with:
- `schemas/vault/entities/investor-essence-v1.json`
- `schemas/vault/entities/investment-axis-v1.json`
- `schemas/vault/core/note-v2.json`
- `application.knowledge.write_models.KnowledgeWriteCommand`
"""
from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Dict, List, Optional

from application.knowledge.write_models import KnowledgeWriteCommand, KnowledgeWriteReceipt
from application.knowledge.write_ports import KnowledgeWritePort

logger = logging.getLogger(__name__)


def build_essence_markdown_body(statement: str, claims: List[Dict[str, Any]], confirmed_at: str) -> str:
    """Renders human-readable Markdown body for confirmed investor essence note."""
    lines = [
        "# ข้อตกลงแก่นแท้ในการลงทุน (Confirmed Investor Essence)",
        "",
        f"**ยืนยันเมื่อ:** {confirmed_at}",
        "",
        "## สรุปวิสัยทัศน์และแก่นแท้ (Core Statement)",
        "",
        statement or "ยังไม่มีข้อความสรุป",
        "",
        "## ข้อเท็จจริงที่ได้รับการยืนยัน (Accepted Principles)",
        "",
    ]
    for idx, c in enumerate(claims, start=1):
        text = c.get("effective_text") or c.get("text", "")
        source = c.get("source_kind", "direct")
        lines.append(f"{idx}. **{text}** *(ที่มา: {source})*")
        for ev in c.get("evidence_refs", []):
            quote = ev.get("quote", "")
            if quote:
                lines.append(f"   > หลักฐาน: \"{quote}\"")
        lines.append("")

    return "\n".join(lines).strip()


def build_axis_markdown_body(portfolio_id: str, snapshot: Dict[str, Any], confirmed_at: str) -> str:
    """Renders human-readable Markdown body for confirmed investment axis note."""
    lines = [
        f"# แกนหลักในการลงทุน (Investment Axis) — พอร์ต {portfolio_id}",
        "",
        f"**ยืนยันเมื่อ:** {confirmed_at}",
        "",
        "## 1. นโยบายพื้นฐาน (Basic Policy)",
        snapshot.get("basic_policy", ""),
        "",
        "## 2. ขีดจำกัดความเสี่ยง (Risk Limits)",
    ]
    for k, v in snapshot.get("risk_limits", {}).items():
        if isinstance(v, dict):
            val = v.get("value")
            unit = v.get("unit", "")
            lines.append(f"- **{k}:** {val} {unit}")
        else:
            lines.append(f"- **{k}:** {v}")

    lines.extend([
        "",
        "## 3. สินทรัพย์ที่ลงทุนและยกเว้น (Targets)",
        f"- **สินทรัพย์ที่ลงทุน:** {', '.join(snapshot.get('invest_targets', []))}",
        f"- **สินทรัพย์ที่ยกเว้น:** {', '.join(snapshot.get('exclude_targets', []))}",
        "",
        "## 4. วิธีการลงทุน (Methods)",
        f"- **วิธีหลัก:** {', '.join(snapshot.get('primary_methods', []))}",
        f"- **วิธีเสริม:** {', '.join(snapshot.get('secondary_methods', []))}",
        "",
        f"## 5. กรอบเวลาการลงทุน: {snapshot.get('investment_horizon', '')}",
        "",
        f"## 6. การจัดสรรสินทรัพย์ (Rebalance: {snapshot.get('rebalance_frequency', '')})",
    ])
    for r in snapshot.get("allocation_rows", []):
        cat = r.get("category_name") or r.get("category", "")
        pct = r.get("target_percent", "")
        role = r.get("role_description", "")
        lines.append(f"- **{cat}:** {pct}% — {role}")

    lines.extend([
        "",
        "## 7. นักลงทุนต้นแบบ (Role Models)",
        ", ".join(snapshot.get("role_models", [])) or "ไม่ระบุ",
        "",
        "## 8. สิ่งที่จะไม่ทำ (Non-Actions)",
    ])
    for na in snapshot.get("non_actions", []):
        lines.append(f"- ⛔ {na}")

    return "\n".join(lines).strip()


class KnowledgePublicationAdapter:
    """Adapter transforming confirmed snapshots into valid Vault KnowledgeWriteCommands."""

    def __init__(self, write_port: Optional[KnowledgeWritePort] = None) -> None:
        self._write_port = write_port

    def publish_essence(
        self,
        confirmation_id: str,
        snapshot_dict: Dict[str, Any],
        idempotency_key: str = "",
    ) -> Optional[Dict[str, str]]:
        if self._write_port is None:
            return None

        doc_key = f"investor-essence-{confirmation_id}"
        confirmed_at = snapshot_dict.get("confirmed_at_iso", "")
        statement = snapshot_dict.get("statement", "")
        claims = snapshot_dict.get("accepted_claims", [])
        unresolved = list(snapshot_dict.get("unresolved_topics", []))

        metadata = {
            "schema_version": 2,
            "note_id": confirmation_id,
            "document_key": doc_key,
            "entity_type": "investor_essence",
            "title": f"Investor Essence — {confirmation_id}",
            "scope": snapshot_dict.get("scope", "workspace"),
            "statement": statement or "Investor Essence Statement",
            "accepted_claims": [
                {
                    "claim_id": c.get("claim_id", ""),
                    "effective_text": c.get("effective_text") or c.get("text", ""),
                    "source_kind": c.get("source_kind", "direct"),
                    "evidence_refs": c.get("evidence_refs", []),
                }
                for c in claims
            ],
            "unresolved_topics": unresolved,
            "confirmed_at": confirmed_at,
            "content_hash": snapshot_dict.get("content_hash", ""),
        }
        body = build_essence_markdown_body(statement, claims, confirmed_at)

        payload_hash = hashlib.sha256(
            json.dumps(metadata, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()

        command = KnowledgeWriteCommand(
            operation="upsert_note",
            idempotency_key=idempotency_key or f"idem-essence-{confirmation_id}-{payload_hash[:12]}",
            document_key=doc_key,
            entity_type="investor_essence",
            producer="investor_essence_confirmation",
            producer_version="2.0",
            actor="system",
            payload={
                "metadata": metadata,
                "body": body,
            },
        )

        try:
            receipt: KnowledgeWriteReceipt = self._write_port.submit(command)
            if receipt and receipt.is_success:
                return {
                    "document_key": doc_key,
                    "note_id": receipt.note_id or confirmation_id,
                    "revision_id": str(receipt.revision_id or "1"),
                    "content_hash": receipt.content_hash or snapshot_dict.get("content_hash", ""),
                    "artifact_set_hash": receipt.artifact_set_hash or snapshot_dict.get("content_hash", ""),
                }
            else:
                logger.warning(
                    "Vault broker rejected essence publication (%s): %s",
                    confirmation_id,
                    getattr(receipt, "error_message", "unknown error"),
                )
        except Exception as exc:
            logger.warning("Vault broker submission error for essence %s: %s", confirmation_id, exc)

        return None

    def publish_axis(
        self,
        confirmation_id: str,
        portfolio_id: str,
        snapshot_dict: Dict[str, Any],
        idempotency_key: str = "",
    ) -> Optional[Dict[str, str]]:
        if self._write_port is None:
            return None

        doc_key = f"investment-axis-{portfolio_id}"
        confirmed_at = snapshot_dict.get("confirmed_at_iso", "")

        metadata = {
            "schema_version": 2,
            "note_id": confirmation_id,
            "document_key": doc_key,
            "entity_type": "investment_axis",
            "title": f"Investment Axis — {portfolio_id}",
            "portfolio_id": portfolio_id,
            "basic_policy": snapshot_dict.get("basic_policy", ""),
            "risk_limits": snapshot_dict.get("risk_limits", {}),
            "invest_targets": list(snapshot_dict.get("invest_targets", [])),
            "exclude_targets": list(snapshot_dict.get("exclude_targets", [])),
            "primary_methods": list(snapshot_dict.get("primary_methods", [])),
            "secondary_methods": list(snapshot_dict.get("secondary_methods", [])),
            "investment_horizon": snapshot_dict.get("investment_horizon", ""),
            "allocation_rows": list(snapshot_dict.get("allocation_rows", [])),
            "rebalance_frequency": snapshot_dict.get("rebalance_frequency", ""),
            "role_models": list(snapshot_dict.get("role_models", [])),
            "non_actions": list(snapshot_dict.get("non_actions", [])),
            "confirmed_at": confirmed_at,
            "content_hash": snapshot_dict.get("content_hash", ""),
        }
        body = build_axis_markdown_body(portfolio_id, snapshot_dict, confirmed_at)

        payload_hash = hashlib.sha256(
            json.dumps(metadata, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()

        command = KnowledgeWriteCommand(
            operation="upsert_note",
            idempotency_key=idempotency_key or f"idem-axis-{confirmation_id}-{payload_hash[:12]}",
            document_key=doc_key,
            entity_type="investment_axis",
            producer="investment_axis_confirmation",
            producer_version="2.0",
            actor="system",
            payload={
                "metadata": metadata,
                "body": body,
            },
        )

        try:
            receipt: KnowledgeWriteReceipt = self._write_port.submit(command)
            if receipt and receipt.is_success:
                return {
                    "document_key": doc_key,
                    "note_id": receipt.note_id or confirmation_id,
                    "revision_id": str(receipt.revision_id or "1"),
                    "content_hash": receipt.content_hash or snapshot_dict.get("content_hash", ""),
                    "artifact_set_hash": receipt.artifact_set_hash or snapshot_dict.get("content_hash", ""),
                }
            else:
                logger.warning(
                    "Vault broker rejected axis publication (%s): %s",
                    confirmation_id,
                    getattr(receipt, "error_message", "unknown error"),
                )
        except Exception as exc:
            logger.warning("Vault broker submission error for axis %s: %s", confirmation_id, exc)

        return None
