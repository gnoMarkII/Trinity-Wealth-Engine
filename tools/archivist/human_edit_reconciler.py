"""Detect and optionally import safe human edits from an Obsidian vault.

The default mode is shadow/read-only.  It never writes canonical files and
records evidence in external runtime storage.  Write-enabled mode submits a
new reconciliation command to the durable broker only after malformed YAML and
system-owned identity changes have been classified as conflicts.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from application.knowledge.write_models import KnowledgeWriteCommand, KnowledgeWriteReceipt
from application.knowledge.write_ports import KnowledgeWritePort
from tools.archivist.artifact_store import DurableArtifactStore, ArtifactError
from tools.archivist.metadata import parse_note, normalize_legacy_metadata
from tools.archivist.schema_registry import SchemaRegistry, load_default_registry
from tools.archivist.vault_policy import is_retired_note, is_searchable_note
from tools.archivist.runtime_layout import runtime_root_for
from tools.archivist.vault_paths import VaultPaths


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ReconciliationFinding:
    kind: str
    status: str
    relative_path: str
    note_id: Optional[str] = None
    document_key: Optional[str] = None
    baseline_revision_id: Optional[str] = None
    baseline_body_hash: Optional[str] = None
    current_body_hash: Optional[str] = None
    changed_fields: tuple[str, ...] = ()
    issues: tuple[str, ...] = ()
    receipt: Optional[dict[str, Any]] = None
    baseline_relative_path: Optional[str] = None
    grace_until: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "status": self.status,
            "relative_path": self.relative_path,
            "note_id": self.note_id,
            "document_key": self.document_key,
            "baseline_revision_id": self.baseline_revision_id,
            "baseline_body_hash": self.baseline_body_hash,
            "current_body_hash": self.current_body_hash,
            "changed_fields": list(self.changed_fields),
            "issues": list(self.issues),
            "receipt": self.receipt,
            "baseline_relative_path": self.baseline_relative_path,
            "grace_until": self.grace_until,
        }


@dataclass
class ReconciliationReport:
    scanned: int = 0
    findings: list[ReconciliationFinding] = field(default_factory=list)

    @property
    def counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for finding in self.findings:
            counts[finding.status] = counts.get(finding.status, 0) + 1
        return counts

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_version": 1,
            "scanned": self.scanned,
            "counts": self.counts,
            "findings": [finding.to_dict() for finding in self.findings],
        }


class HumanEditReconciler:
    """Periodic correctness scanner for canonical Markdown edits."""

    def __init__(
        self,
        *,
        vault_paths: Optional[VaultPaths] = None,
        broker: Optional[KnowledgeWritePort] = None,
        registry: Optional[SchemaRegistry] = None,
        runtime_root: Optional[str | Path] = None,
        runtime_base: Optional[str | Path] = None,
        delete_grace_seconds: int = 86_400,
    ) -> None:
        self.vault_paths = vault_paths or VaultPaths()
        self.registry = registry or load_default_registry()
        self.broker = broker
        if runtime_root is not None and runtime_base is not None:
            raise ValueError("provide either runtime_root (isolated override) or runtime_base (canonical base), not both")
        if runtime_root is None:
            self.runtime_root = runtime_root_for(self.vault_paths.root, runtime_base, create=True)
        else:
            self.runtime_root = Path(runtime_root).resolve()
        if self.runtime_root.is_relative_to(self.vault_paths.root):
            raise ValueError("reconciliation runtime must be outside the vault")
        self.event_path = self.runtime_root / "reconciliation" / "events.jsonl"
        self.artifacts = DurableArtifactStore(vault_paths=self.vault_paths)
        self.delete_grace_seconds = max(1, int(delete_grace_seconds))

    def scan(self) -> ReconciliationReport:
        report = ReconciliationReport()
        root = self.vault_paths.root
        if not root.is_dir():
            return report
        seen_note_ids: set[str] = set()
        for path in sorted(root.rglob("*.md")):
            if not is_searchable_note(path, vault_root=root):
                continue
            report.scanned += 1
            # Portfolio state, navigation, legacy backfills, and other
            # generated projections are intentionally outside the human-edit
            # reconciliation surface.
            # Their source of truth is a transactional/runtime or generated
            # adapter, so treating every projection as an editable canonical
            # note manufactures thousands of false conflicts.
            try:
                scoped_metadata, _scoped_body, scoped_issues = parse_note(path.read_text(encoding="utf-8"))
                if not scoped_issues and (
                    str(scoped_metadata.get("search_scope") or "").lower() == "excluded"
                    or scoped_metadata.get("production_eligible") is False
                ):
                    continue
            except (OSError, UnicodeDecodeError, ValueError):
                # Let _inspect classify malformed notes as conflicts below.
                pass
            findings = self._inspect(path)
            report.findings.extend(findings)
            for finding in findings:
                if finding.note_id:
                    seen_note_ids.add(finding.note_id)
            if not findings:
                try:
                    metadata, _, issues = parse_note(path.read_text(encoding="utf-8"))
                    if not issues and metadata.get("note_id"):
                        seen_note_ids.add(str(metadata["note_id"]))
                except (OSError, UnicodeDecodeError, ValueError):
                    pass
        report.findings.extend(self._missing_head_findings(seen_note_ids))
        return report

    def reconcile_once(self, *, write_enabled: bool = False) -> ReconciliationReport:
        report = self.scan()
        if write_enabled and self.broker is None:
            raise ValueError("write_enabled reconciliation requires a KnowledgeWritePort")
        if write_enabled:
            updated: list[ReconciliationFinding] = []
            for finding in report.findings:
                if finding.status != "manual_edit" or not finding.note_id:
                    updated.append(finding)
                    continue
                path = self.vault_paths.root / finding.relative_path
                try:
                    metadata, body, issues = parse_note(path.read_text(encoding="utf-8"))
                    if issues:
                        updated.append(finding)
                        continue
                    command = KnowledgeWriteCommand(
                        operation="upsert_note",
                        idempotency_key=f"reconcile:{finding.note_id}:{finding.current_body_hash}",
                        document_key=str(metadata.get("document_key") or finding.document_key or ""),
                        producer="human-edit-reconciler",
                        producer_version="r8",
                        actor="reconciliation",
                        expected_revision_id=finding.baseline_revision_id,
                        expected_content_hash=finding.current_body_hash,
                        payload={
                            "metadata": metadata,
                            "body": body,
                            **(
                                {"target_path": finding.relative_path}
                                if finding.kind == "rename_detected"
                                else {}
                            ),
                        },
                    )
                    receipt = self.broker.submit(command)
                    updated.append(
                        ReconciliationFinding(
                            **{
                                **finding.__dict__,
                                "status": "imported" if receipt.is_success else "conflict",
                                "receipt": receipt.to_dict(),
                            }
                        )
                    )
                except Exception as exc:  # noqa: BLE001 - evidence boundary
                    updated.append(
                        ReconciliationFinding(
                            **{**finding.__dict__, "status": "conflict", "issues": tuple(finding.issues) + (str(exc),)}
                        )
                    )
            report.findings = updated
        self._write_events(report)
        return report

    def _inspect(self, path: Path) -> list[ReconciliationFinding]:
        rel = path.relative_to(self.vault_paths.root).as_posix()
        try:
            raw_text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            return [ReconciliationFinding("read_error", "conflict", rel, issues=(str(exc),))]
        metadata, body, issues = parse_note(raw_text)
        if issues:
            return [ReconciliationFinding("malformed_yaml", "conflict", rel, issues=tuple(item.get("reason", "") for item in issues))]
        metadata, _ = normalize_legacy_metadata(metadata)
        note_id = str(metadata.get("note_id") or "").strip() or None
        document_key = str(metadata.get("document_key") or "").strip() or None
        if not note_id:
            if "00_Inbox" in path.parts:
                return []
            return [ReconciliationFinding("unidentified_create", "needs_identity", rel, document_key=document_key)]
        head_path = self.vault_paths.root / ".system" / "artifacts" / "heads" / f"{note_id}.json"
        if not head_path.is_file():
            return [ReconciliationFinding("unregistered_identity", "conflict", rel, note_id=note_id, document_key=document_key)]
        try:
            head = json.loads(head_path.read_text(encoding="utf-8"))
            revision_id = str(head["revision_id"])
            baseline = self.artifacts.get_revision_artifact(note_id, revision_id)
        except (OSError, ValueError, KeyError, ArtifactError) as exc:
            return [ReconciliationFinding("corrupt_baseline", "conflict", rel, note_id=note_id, document_key=document_key, issues=(str(exc),))]
        current_body_hash = _sha256(body)
        baseline_body_hash = _sha256(baseline.body)
        baseline_rel = str(baseline.manifest.get("projection_path") or "").strip() or None
        renamed = bool(baseline_rel and baseline_rel != rel and path.is_file())
        if current_body_hash == baseline_body_hash and metadata == baseline.metadata and not renamed:
            return []
        system_fields = set()
        for field in ("schema_version", "note_id", "document_key", "revision", "revision_id", "content_sha256", "artifact_set_hash", "current_revision_id", "registry_digest"):
            if metadata.get(field) != baseline.metadata.get(field):
                system_fields.add(field)
        if system_fields:
            return [ReconciliationFinding(
                "system_field_edit",
                "conflict",
                rel,
                note_id=note_id,
                document_key=document_key,
                baseline_revision_id=revision_id,
                baseline_body_hash=baseline_body_hash,
                current_body_hash=current_body_hash,
                changed_fields=tuple(sorted(system_fields)),
                issues=("system-owned frontmatter changed; automated overwrite blocked",),
                baseline_relative_path=baseline_rel,
            )]
        changed = tuple(sorted({key for key in set(metadata) | set(baseline.metadata) if metadata.get(key) != baseline.metadata.get(key)}))
        return [ReconciliationFinding(
            "rename_detected" if renamed else "manual_edit",
            "manual_edit",
            rel,
            note_id=note_id,
            document_key=document_key,
            baseline_revision_id=revision_id,
            baseline_body_hash=baseline_body_hash,
            current_body_hash=current_body_hash,
            changed_fields=changed,
            baseline_relative_path=baseline_rel,
            issues=("path rename detected; stable identity is preserved",) if renamed else (),
        )]

    def _missing_head_findings(self, seen_note_ids: set[str]) -> list[ReconciliationFinding]:
        """Report missing current projections without deleting anything."""
        heads_dir = self.vault_paths.root / ".system" / "artifacts" / "heads"
        if not heads_dir.is_dir():
            return []
        findings: list[ReconciliationFinding] = []
        grace_until = datetime.fromtimestamp(
            datetime.now(timezone.utc).timestamp() + self.delete_grace_seconds,
            timezone.utc,
        ).isoformat().replace("+00:00", "Z")
        for head_path in sorted(heads_dir.glob("*.json")):
            note_id = head_path.stem
            if note_id in seen_note_ids or is_retired_note(self.vault_paths.root, note_id=note_id):
                continue
            try:
                head = json.loads(head_path.read_text(encoding="utf-8"))
                revision_id = str(head["revision_id"])
                baseline = self.artifacts.get_revision_artifact(note_id, revision_id)
            except (OSError, UnicodeDecodeError, ValueError, KeyError, ArtifactError):
                continue
            baseline_rel = str(baseline.manifest.get("projection_path") or "").strip()
            # Pre-R8 manifests do not identify their current projection.  Do
            # not guess from the immutable revision filename and manufacture
            # deletion findings for legacy notes.
            if not baseline_rel or (self.vault_paths.root / baseline_rel).exists():
                continue
            if not self.registry.is_index_eligible(baseline.metadata, vector=False):
                continue
            findings.append(
                ReconciliationFinding(
                    "deleted_projection",
                    "delete_pending",
                    "",
                    note_id=note_id,
                    document_key=str(baseline.metadata.get("document_key") or "") or None,
                    baseline_revision_id=revision_id,
                    baseline_body_hash=_sha256(baseline.body),
                    baseline_relative_path=baseline_rel,
                    grace_until=grace_until,
                    issues=("current projection missing; deletion requires grace-period confirmation",),
                )
            )
        return findings

    def _write_events(self, report: ReconciliationReport) -> None:
        self.event_path.parent.mkdir(parents=True, exist_ok=True)
        with self.event_path.open("a", encoding="utf-8", newline="\n") as stream:
            for finding in report.findings:
                event = {"event_version": 1, "observed_at": _utc_now(), **finding.to_dict()}
                stream.write(json.dumps(event, ensure_ascii=False) + "\n")
