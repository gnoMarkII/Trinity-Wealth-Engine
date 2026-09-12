"""Assertion-derived acceptance evidence for Vault V2 rehearsals.

The report is a projection of recorded assertions.  It never infers PASS from
the existence of a plan or from a hard-coded checklist string.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

STATUSES = frozenset({"PASS", "FAIL", "NOT_RUN", "BLOCKED"})


@dataclass(frozen=True)
class AcceptanceRecord:
    case_id: str
    gate_ids: tuple[str, ...]
    run_id: str
    command: str
    phase: str
    actual: Any
    expected: Any
    status: str
    evidence_paths: tuple[str, ...] = ()
    evidence_hashes: tuple[str, ...] = ()
    reason: Optional[str] = None
    recorded_at: str = ""
    code_fingerprint: Optional[str] = None
    input_fingerprint: Optional[str] = None
    snapshot_fingerprint: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["gate_ids"] = list(self.gate_ids)
        value["evidence_paths"] = list(self.evidence_paths)
        value["evidence_hashes"] = list(self.evidence_hashes)
        return value


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(65536):
            digest.update(chunk)
    return digest.hexdigest()


def record_assertion(
    *,
    case_id: str,
    gate_ids: Iterable[str],
    run_id: str,
    command: str,
    phase: str,
    actual: Any,
    expected: Any,
    evidence_paths: Iterable[Path | str] = (),
    status: Optional[str] = None,
    reason: Optional[str] = None,
    code_fingerprint: Optional[str] = None,
    input_fingerprint: Optional[str] = None,
    snapshot_fingerprint: Optional[str] = None,
) -> AcceptanceRecord:
    paths = tuple(str(Path(item)) for item in evidence_paths)
    hashes = tuple(hash_file(Path(item)) for item in paths if Path(item).is_file())
    if status is None:
        status = "PASS" if actual == expected else "FAIL"
    status = str(status).upper()
    if status not in STATUSES:
        raise ValueError(f"Unknown acceptance status: {status}")
    if status == "PASS" and (not paths or len(hashes) != len(paths)):
        status = "NOT_RUN"
        reason = reason or "required evidence file is missing"
    return AcceptanceRecord(
        case_id=case_id,
        gate_ids=tuple(gate_ids),
        run_id=run_id,
        command=command,
        phase=phase,
        actual=actual,
        expected=expected,
        status=status,
        evidence_paths=paths,
        evidence_hashes=hashes,
        reason=reason,
        recorded_at=datetime.now(timezone.utc).isoformat(),
        code_fingerprint=code_fingerprint,
        input_fingerprint=input_fingerprint,
        snapshot_fingerprint=snapshot_fingerprint,
    )


def aggregate_acceptance_records(
    records: Iterable[AcceptanceRecord | Mapping[str, Any]],
    *,
    expected_run_id: Optional[str] = None,
    expected_code_fingerprint: Optional[str] = None,
    expected_input_fingerprint: Optional[str] = None,
    expected_snapshot_fingerprint: Optional[str] = None,
    mandatory_case_ids: Optional[Iterable[str]] = None,
) -> dict[str, Any]:
    """Validate assertion records and derive an honest aggregate status.

    The optional expected values let a caller bind a report to one code/input/
    snapshot. Missing evidence or fingerprints becomes ``NOT_RUN``; a present
    but mismatching hash is a ``FAIL``. Duplicate ``(run_id, case_id)`` rows
    are invalid because a later record must not silently replace an earlier
    observation.
    """
    normalized: list[dict[str, Any]] = []
    for item in records:
        value = item.to_dict() if isinstance(item, AcceptanceRecord) else dict(item)
        status = str(value.get("status", "NOT_RUN")).upper()
        if status not in STATUSES:
            status = "NOT_RUN"
            value["reason"] = value.get("reason") or "unknown assertion status"
        required_fields = ("case_id", "gate_ids", "run_id", "command", "phase", "actual", "expected")
        if any(field not in value for field in required_fields):
            status = "NOT_RUN"
            value["reason"] = value.get("reason") or "assertion record is missing required fields"
        elif not value.get("case_id") or not value.get("run_id") or not value.get("gate_ids"):
            status = "NOT_RUN"
            value["reason"] = value.get("reason") or "case_id, run_id, and gate_ids are required"
        elif expected_run_id and value.get("run_id") != expected_run_id:
            status = "NOT_RUN"
            value["reason"] = value.get("reason") or "record run_id does not match requested run"
        elif any(
            not isinstance(gate, str) or gate not in {f"A{i:02d}" for i in range(1, 31)}
            for gate in (value.get("gate_ids") or [])
        ):
            status = "NOT_RUN"
            value["reason"] = value.get("reason") or "record contains an unknown acceptance gate"
        if status == "PASS":
            paths = value.get("evidence_paths") or []
            hashes = value.get("evidence_hashes") or []
            if not paths or any(not Path(path).is_file() for path in paths):
                status = "NOT_RUN"
                value["reason"] = value.get("reason") or "missing evidence"
            elif not hashes or len(hashes) != len(paths) or any(not str(item) for item in hashes):
                status = "NOT_RUN"
                value["reason"] = value.get("reason") or "evidence hashes are required"
            else:
                current_hashes = [hash_file(Path(path)) for path in paths]
                if current_hashes != list(hashes):
                    status = "FAIL"
                    value["reason"] = value.get("reason") or "evidence changed after assertion"
        expected_fingerprints = {
            "code_fingerprint": expected_code_fingerprint,
            "input_fingerprint": expected_input_fingerprint,
            "snapshot_fingerprint": expected_snapshot_fingerprint,
        }
        if status == "PASS":
            for field_name, expected in expected_fingerprints.items():
                if expected is None:
                    continue
                observed = value.get(field_name)
                if not observed:
                    status = "NOT_RUN"
                    value["reason"] = value.get("reason") or f"missing {field_name}"
                    break
                if observed != expected:
                    status = "FAIL"
                    value["reason"] = value.get("reason") or f"{field_name} mismatch"
                    break
        value["status"] = status
        normalized.append(value)

    seen: dict[tuple[str, str], list[int]] = {}
    for idx, value in enumerate(normalized):
        key = (str(value.get("run_id", "")), str(value.get("case_id", "")))
        seen.setdefault(key, []).append(idx)
    for key, indices in seen.items():
        if key[0] and key[1] and len(indices) > 1:
            for idx in indices:
                normalized[idx]["status"] = "NOT_RUN"
                normalized[idx]["reason"] = normalized[idx].get("reason") or (
                    f"duplicate assertion case_id for run: {key[1]}"
                )

    mandatory = {str(case_id) for case_id in (mandatory_case_ids or ())}
    observed_cases = {str(item.get("case_id")) for item in normalized}
    missing_cases = sorted(mandatory - observed_cases)
    for case_id in missing_cases:
        normalized.append(
            {
                "case_id": case_id,
                "gate_ids": [],
                "run_id": expected_run_id or "",
                "command": "",
                "phase": "",
                "actual": None,
                "expected": None,
                "status": "NOT_RUN",
                "evidence_paths": [],
                "evidence_hashes": [],
                "reason": "mandatory assertion is missing",
            }
        )

    counts = {status: sum(1 for item in normalized if item["status"] == status) for status in STATUSES}
    if counts["FAIL"]:
        overall = "FAIL"
    elif counts["BLOCKED"]:
        overall = "BLOCKED"
    elif counts["NOT_RUN"]:
        overall = "NOT_RUN"
    elif normalized:
        overall = "PASS"
    else:
        overall = "NOT_RUN"
    return {
        "overall_status": overall,
        "counts": counts,
        "records": normalized,
        "missing_mandatory_cases": missing_cases,
    }


def write_acceptance_report(
    report: Mapping[str, Any],
    output_dir: Path | str,
    *,
    run_id: Optional[str] = None,
) -> tuple[Path, Path]:
    out = Path(output_dir).resolve()
    from tools.archivist.maintenance_guard import assert_write_allowed
    assert_write_allowed(out)
    out.mkdir(parents=True, exist_ok=True)
    payload = dict(report)
    payload["run_id"] = run_id or payload.get("run_id")
    payload["generated_at"] = datetime.now(timezone.utc).isoformat()
    json_path = out / "acceptance-report.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    lines = [
        "# Vault V2 Acceptance Report",
        "",
        f"**Overall status:** `{payload.get('overall_status', 'NOT_RUN')}`",
        "",
        "| Case | Gates | Phase | Status | Actual | Expected | Evidence |",
        "|---|---|---|---|---|---|---|",
    ]
    for item in payload.get("records", []):
        evidence = ", ".join(item.get("evidence_paths") or []) or "—"
        lines.append(
            f"| {item.get('case_id', 'unknown')} | {', '.join(item.get('gate_ids') or [])} | "
            f"{item.get('phase', '')} | **{item.get('status', 'NOT_RUN')}** | "
            f"{item.get('actual', '')!s} | {item.get('expected', '')!s} | {evidence} |"
        )
    md_path = out / "acceptance-report.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md_path, json_path
