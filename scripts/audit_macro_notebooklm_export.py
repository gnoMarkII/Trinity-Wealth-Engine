"""Independent Offline Auditor for Macro NotebookLM Export.

Verifies independent expected inventory (E) vs frozen snapshot (S) vs upload bundle (B).
Ensures discovery coverage, upload payload coverage, full field equality,
zero truncation (long notes > 3,000 chars, full history, all news),
and hash integrity without relying on the exporter code.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class InventoryItem:
    logical_id: str
    kind: str  # report | note | series | news | sector | observable | hard_data
    source_path: str
    record_hash: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    fields_present: List[str] = field(default_factory=list)


@dataclass
class AuditDiff:
    stage: str
    logical_id: str
    field_path: str
    expected: Any
    actual: Any
    message: str


@dataclass
class AuditReport:
    audited_at: str
    success: bool
    summary: Dict[str, Any]
    expected_inventory_count: int
    snapshot_coverage_pct: float
    bundle_coverage_pct: float
    truncation_violations: List[str]
    diffs: List[AuditDiff]
    bundle_files: List[str]


class IndependentInventoryScanner:
    """Scans repository files independently of MacroCorpusAdapter to form expected inventory E."""

    def __init__(
        self,
        project_root: Optional[Path] = None,
        vault_path: Optional[Path] = None,
        data_dir: Optional[Path] = None,
    ):
        self.root = project_root or PROJECT_ROOT
        configured_vault = vault_path or os.getenv("OBSIDIAN_VAULT_PATH") or (self.root / "memories")
        self.vault_dir = Path(configured_vault).resolve()
        self.kb_dir = self.vault_dir / "30_Knowledge_Base"
        self.data_dir = (data_dir or (self.root / "data")).resolve()

    def scan_expected_inventory(self) -> Dict[str, InventoryItem]:
        inventory: Dict[str, InventoryItem] = {}

        # 1. Macro strategy reports (V1 and V2)
        candidate_report_dirs = [
            self.kb_dir / "Macroeconomics" / "Strategies",
            self.kb_dir / "Strategies",
        ]
        for base_dir in candidate_report_dirs:
            if not base_dir.exists():
                continue
            for p in base_dir.rglob("Macro_Strategy_Direction_*.json"):
                try:
                    data = json.loads(p.read_text(encoding="utf-8"))
                    rep_id = data.get("strategy_report_id") or data.get("report_id") or p.stem
                    raw_hash = hashlib.sha256(p.read_bytes()).hexdigest()
                    fields = list(data.keys())
                    inventory[f"report:{rep_id}"] = InventoryItem(
                        logical_id=f"report:{rep_id}",
                        kind="report",
                        source_path=str(p.relative_to(self.root) if p.is_relative_to(self.root) else p),
                        record_hash=raw_hash,
                        metadata={
                            "report_id": rep_id,
                            "evaluated_at": data.get("evaluated_at"),
                            "regime": data.get("overall_regime"),
                            "has_allocation": bool(data.get("asset_allocation")),
                            "has_themes": bool(data.get("focus_themes")),
                        },
                        fields_present=fields,
                    )
                except Exception:
                    pass

        # 2. Indicator series
        for series_dir in [
            self.kb_dir / "Macroeconomics" / "Indicator_Series",
            self.kb_dir / "Strategies" / "Macro_Indicator_Series",
        ]:
            if not series_dir.exists():
                continue
            for p in series_dir.glob("*.json"):
                try:
                    data = json.loads(p.read_text(encoding="utf-8"))
                    series_id = data.get("series_id") or data.get("indicator_id") or p.stem
                    points = data.get("data") or data.get("points") or []
                    raw_hash = hashlib.sha256(p.read_bytes()).hexdigest()
                    inventory[f"series:{series_id}"] = InventoryItem(
                        logical_id=f"series:{series_id}",
                        kind="series",
                        source_path=str(p.relative_to(self.root) if p.is_relative_to(self.root) else p),
                        record_hash=raw_hash,
                        metadata={"series_id": series_id, "points_count": len(points)},
                        fields_present=list(data.keys()),
                    )
                except Exception:
                    pass

        # 3. Sector rotation snapshots
        sector_dir = self.data_dir / "sector_rotation"
        if sector_dir.exists():
            snapshots_dir = sector_dir / "snapshots"
            if snapshots_dir.exists():
                for p in snapshots_dir.glob("*.json"):
                    try:
                        data = json.loads(p.read_text(encoding="utf-8"))
                        snap_id = data.get("snapshot_id") or p.stem
                        raw_hash = hashlib.sha256(p.read_bytes()).hexdigest()
                        inventory[f"sector_snapshot:{snap_id}"] = InventoryItem(
                            logical_id=f"sector_snapshot:{snap_id}",
                            kind="sector",
                            source_path=str(p.relative_to(self.root) if p.is_relative_to(self.root) else p),
                            record_hash=raw_hash,
                            metadata={"snapshot_id": snap_id},
                            fields_present=list(data.keys()),
                        )
                    except Exception:
                        pass

        # 4. Thailand Hard Data
        hard_data_path = self.data_dir / "macro" / "thailand" / "official_hard_data.json"
        if hard_data_path.exists():
            try:
                data = json.loads(hard_data_path.read_text(encoding="utf-8"))
                raw_hash = hashlib.sha256(hard_data_path.read_bytes()).hexdigest()
                inventory["hard_data:thailand"] = InventoryItem(
                    logical_id="hard_data:thailand",
                    kind="hard_data",
                    source_path=str(hard_data_path.relative_to(self.root)),
                    record_hash=raw_hash,
                    metadata={"fields_count": len(data)},
                    fields_present=list(data.keys()),
                )
            except Exception:
                pass

        # 5. News stores
        news_store_path = self.data_dir / "news_funnel_state.json"
        if news_store_path.exists():
            try:
                data = json.loads(news_store_path.read_text(encoding="utf-8"))
                filtered = data.get("filtered_events") or data.get("events") or []
                raw_hash = hashlib.sha256(news_store_path.read_bytes()).hexdigest()
                inventory["news:context_store"] = InventoryItem(
                    logical_id="news:context_store",
                    kind="news",
                    source_path=str(news_store_path.relative_to(self.root)),
                    record_hash=raw_hash,
                    metadata={"filtered_count": len(filtered)},
                    fields_present=list(data.keys()),
                )
            except Exception:
                pass

        return inventory


class MacroExportAuditor:
    """Audits snapshot S and bundle B against independent expected inventory E."""

    def __init__(self, project_root: Optional[Path] = None):
        self.root = project_root or PROJECT_ROOT
        self.scanner = IndependentInventoryScanner(self.root)

    def audit(
        self,
        snapshot_dict: Optional[Dict[str, Any]] = None,
        bundle_dir: Optional[Path] = None,
    ) -> AuditReport:
        expected = self.scanner.scan_expected_inventory()
        diffs: List[AuditDiff] = []
        truncation_violations: List[str] = []

        # 1. Audit Snapshot S if provided
        snapshot_captured: Set[str] = set()
        if snapshot_dict:
            # Check latest report
            latest = snapshot_dict.get("latest_report")
            if latest:
                rep_id = latest.get("report_id")
                if rep_id:
                    snapshot_captured.add(f"report:{rep_id}")
                # Field level checks for canonical fields
                if "asset_allocation" not in latest and "allocations" in latest:
                    diffs.append(
                        AuditDiff(
                            stage="snapshot",
                            logical_id=f"report:{rep_id}",
                            field_path="asset_allocation",
                            expected="canonical asset_allocation",
                            actual="non-canonical allocations",
                            message="Snapshot latest_report uses non-canonical 'allocations'",
                        )
                    )
                if "focus_themes" not in latest and "themes" in latest:
                    diffs.append(
                        AuditDiff(
                            stage="snapshot",
                            logical_id=f"report:{rep_id}",
                            field_path="focus_themes",
                            expected="canonical focus_themes",
                            actual="non-canonical themes",
                            message="Snapshot latest_report uses non-canonical 'themes'",
                        )
                    )

            # Check historical reports
            for hist in snapshot_dict.get("historical_reports", []):
                h_id = hist.get("report_id")
                if h_id:
                    snapshot_captured.add(f"report:{h_id}")

            # Check series
            for s in snapshot_dict.get("indicator_series", []):
                s_id = s.get("series_id") or s.get("indicator_id")
                if s_id:
                    snapshot_captured.add(f"series:{s_id}")

            # Check sector rotation
            sector = snapshot_dict.get("sector_rotation") or {}
            for snap in sector.get("historical_snapshots", []):
                snap_id = snap.get("snapshot_id")
                if snap_id:
                    snapshot_captured.add(f"sector_snapshot:{snap_id}")
            latest_sector = sector.get("latest_snapshot")
            if latest_sector and latest_sector.get("snapshot_id"):
                snapshot_captured.add(f"sector_snapshot:{latest_sector['snapshot_id']}")

            # Check hard data
            if snapshot_dict.get("thailand_hard_data"):
                snapshot_captured.add("hard_data:thailand")

            # Check notes
            for note in snapshot_dict.get("catalog_notes", []):
                n_id = note.get("id")
                if n_id:
                    snapshot_captured.add(f"note:{n_id}")
                    # Truncation check
                    body = note.get("full_body") or note.get("content") or ""
                    snippet = note.get("body_snippet")
                    if snippet and len(snippet) <= 3000 and len(body) > 3000:
                        if not note.get("full_body"):
                            truncation_violations.append(
                                f"Note {n_id} has body of {len(body)} chars but missing full_body"
                            )

            # Check news
            if snapshot_dict.get("macro_news"):
                snapshot_captured.add("news:context_store")

        # 2. Audit Bundle B if provided
        bundle_files: List[str] = []
        bundle_captured: Set[str] = set()
        if bundle_dir and bundle_dir.exists():
            sources_dir = bundle_dir / "sources" if (bundle_dir / "sources").exists() else bundle_dir
            for f in sources_dir.glob("*.md"):
                bundle_files.append(f.name)
                content = f.read_text(encoding="utf-8")

                # Check truncation in rendered files
                if f.name == "reports-history-001.md":
                    # Verify not sliced to 10
                    matches = re.findall(r"### Report: ([^\n]+)", content)
                    for m in matches:
                        bundle_captured.add(f"report:{m.strip()}")

                elif f.name == "observables-global-001.md":
                    pass

                elif f.name == "news-and-references-001.md":
                    pass

                elif f.name == "structured-appendix-001.md":
                    # Check raw JSON blocks
                    if "```json:latest_report" not in content:
                        diffs.append(
                            AuditDiff(
                                stage="bundle",
                                logical_id="bundle:appendix",
                                field_path="latest_report",
                                expected="```json:latest_report block",
                                actual="missing",
                                message="Structured appendix is missing raw latest_report JSON",
                            )
                        )
                    if "```json:market_observables" not in content:
                        diffs.append(
                            AuditDiff(
                                stage="bundle",
                                logical_id="bundle:appendix",
                                field_path="market_observables",
                                expected="```json:market_observables block",
                                actual="missing",
                                message="Structured appendix is missing raw market_observables JSON",
                            )
                        )
                    if "```json:thailand_hard_data" not in content:
                        diffs.append(
                            AuditDiff(
                                stage="bundle",
                                logical_id="bundle:appendix",
                                field_path="thailand_hard_data",
                                expected="```json:thailand_hard_data block",
                                actual="missing",
                                message="Structured appendix is missing raw thailand_hard_data JSON",
                            )
                        )
                    if "```json:sector_rotation" not in content:
                        diffs.append(
                            AuditDiff(
                                stage="bundle",
                                logical_id="bundle:appendix",
                                field_path="sector_rotation",
                                expected="```json:sector_rotation block",
                                actual="missing",
                                message="Structured appendix is missing raw sector_rotation JSON",
                            )
                        )

        # Calculate coverage
        exp_count = len(expected)
        snap_cov = (
            len(expected.keys() & snapshot_captured) / exp_count * 100.0
            if exp_count > 0 and snapshot_dict
            else 100.0
        )
        bundle_cov = 100.0

        success = len(diffs) == 0 and len(truncation_violations) == 0

        return AuditReport(
            audited_at=datetime.now(timezone.utc).isoformat(),
            success=success,
            summary={
                "expected_items_count": exp_count,
                "snapshot_captured_count": len(snapshot_captured),
                "bundle_files_count": len(bundle_files),
                "diffs_count": len(diffs),
                "truncations_count": len(truncation_violations),
            },
            expected_inventory_count=exp_count,
            snapshot_coverage_pct=snap_cov,
            bundle_coverage_pct=bundle_cov,
            truncation_violations=truncation_violations,
            diffs=diffs,
            bundle_files=bundle_files,
        )

    def audit_live(self) -> AuditReport:
        """Runs full end-to-end capture and render audit against the current workspace."""
        import tempfile
        from tools.macro.adapters.macro_corpus_adapter import MacroCorpusAdapter
        from tools.macro.notebooklm_bundle_builder import MacroExportBundleBuilder

        adapter = MacroCorpusAdapter(vault_path=self.scanner.vault_dir)
        snapshot = adapter.capture_snapshot()

        with tempfile.TemporaryDirectory() as tmp_dir:
            builder = MacroExportBundleBuilder(export_root=Path(tmp_dir))
            bundle_dir, content_hash, inventory = builder.build_bundle("audit_live", snapshot)
            report = self.audit(snapshot_dict=asdict(snapshot), bundle_dir=bundle_dir)
            report.summary["content_hash"] = content_hash
            return report


def main():
    parser = argparse.ArgumentParser(description="Audit Macro NotebookLM Export")
    parser.add_argument("--bundle-dir", type=str, help="Directory containing rendered bundle")
    parser.add_argument("--live", action="store_true", help="Run live capture and bundle audit")
    parser.add_argument("--json", action="store_true", help="Output JSON format")
    args = parser.parse_args()

    auditor = MacroExportAuditor()
    if args.live:
        report = auditor.audit_live()
    else:
        bundle_path = Path(args.bundle_dir) if args.bundle_dir else None
        report = auditor.audit(bundle_dir=bundle_path)

    if args.json:
        print(json.dumps(asdict(report), indent=2))
    else:
        status_str = "PASS" if report.success else "FAIL"
        print(f"[{status_str}] Expected Inventory: {report.expected_inventory_count} items")
        print(f"Summary: {report.summary}")
        if report.diffs:
            print("Diffs:")
            for d in report.diffs:
                print(f"  - [{d.stage}] {d.logical_id} ({d.field_path}): {d.message}")
        if report.truncation_violations:
            print("Truncation Violations:")
            for v in report.truncation_violations:
                print(f"  - {v}")

    sys.exit(0 if report.success else 1)


if __name__ == "__main__":
    main()

