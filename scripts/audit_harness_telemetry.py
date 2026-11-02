"""Audit Harness Telemetry & Continuous Learning Report Generator.

Scans:
1. Active telemetry ledger (data/telemetry/probe_events.jsonl)
2. Empirical Macro strategy sidecars (memories/macro_strategy_direction/*.json)
3. Empirical Equity signals sidecars (memories/equity_signals/*.json)

Computes Pareto distributions, flag taxonomies, and generates actionable prompt hints.
"""
from collections import Counter
import glob
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.market.equity_flag_registry import categorize_equity_flag, EquityFlagCategory
from tools.telemetry.failure_ledger import read_probe_events, aggregate_probe_statistics


def analyze_macro_sidecars(memories_dir: Path) -> Dict[str, Any]:
    macro_files = glob.glob(str(memories_dir / "**" / "Macro_Strategy_Direction_*.json"), recursive=True)
    warnings_counter: Counter[str] = Counter()
    total_files = len(macro_files)

    for fpath in macro_files:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Strategy level warnings
            for w in data.get("validation_warnings", []) or []:
                w_str = str(w).split(":")[0].strip()
                warnings_counter[w_str] += 1
            # Asset level warnings
            for asset in data.get("asset_allocation", []) or []:
                for w in asset.get("validation_warnings", []) or []:
                    w_str = str(w).split(":")[0].strip()
                    warnings_counter[w_str] += 1
            # Pair trades
            for pt in data.get("pair_trades", []) or []:
                for w in pt.get("validation_warnings", []) or []:
                    w_str = str(w).split(":")[0].strip()
                    warnings_counter[w_str] += 1
        except Exception:
            continue

    return {
        "total_files": total_files,
        "warnings_counter": warnings_counter,
    }


def analyze_equity_sidecars(memories_dir: Path) -> Dict[str, Any]:
    equity_files = glob.glob(str(memories_dir / "30_Knowledge_Base" / "Stocks" / "**" / "Analysis" / "*latest.json"), recursive=True)
    flags_counter: Counter[str] = Counter()
    category_counter: Counter[str] = Counter()
    total_files = len(equity_files)

    for fpath in equity_files:
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            flags = data.get("data_quality_flags") or data.get("quant_signals", {}).get("data_quality_flags") or []
            for flag in flags:
                flags_counter[flag] += 1
                cat = categorize_equity_flag(flag).value
                category_counter[cat] += 1
        except Exception:
            continue

    return {
        "total_files": total_files,
        "flags_counter": flags_counter,
        "category_counter": category_counter,
    }


def generate_audit_report(output_path: Optional[Path] = None) -> str:
    memories_dir = PROJECT_ROOT / "memories"
    macro_res = analyze_macro_sidecars(memories_dir)
    equity_res = analyze_equity_sidecars(memories_dir)
    telemetry_events = read_probe_events()
    telemetry_stats = aggregate_probe_statistics(telemetry_events)

    lines = [
        "# 🛡️ Harness Telemetry & Continuous Learning Audit Report",
        f"Generated At: {Path(__file__).name} | Root: `{PROJECT_ROOT}`",
        "",
        "## 1. 📊 Macro Strategy Warnings (Pareto Analysis)",
        f"Total Macro Strategy Sidecars Audited: **{macro_res['total_files']}**",
        "",
        "| Warning Code | Occurrences | % of Sidecars | Severity / Recommended Action |",
        "|---|---|---|---|",
    ]

    for w_code, count in macro_res["warnings_counter"].most_common():
        pct = round(count / max(macro_res["total_files"], 1) * 100, 1)
        action = "Resolved via Auto-Backfill" if "SUPPORTING_DATA" in w_code else "Add Prompt Constraint"
        lines.append(f"| `{w_code}` | {count} | {pct}% | {action} |")

    lines.extend([
        "",
        "## 2. 📈 Equity Data Quality Flags (Taxonomy Breakdown)",
        f"Total Equity Sidecars Audited: **{equity_res['total_files']}**",
        "",
        "### Taxonomy Category Distribution",
    ])
    for cat, count in equity_res["category_counter"].most_common():
        lines.append(f"- **{cat}**: {count} instances")

    lines.extend([
        "",
        "### Top Individual Equity Flags",
        "| Flag Code | Category | Count | % of Sidecars | Interpretation |",
        "|---|---|---|---|---|",
    ])
    for flag, count in equity_res["flags_counter"].most_common(12):
        cat = categorize_equity_flag(flag).value
        pct = round(count / max(equity_res["total_files"], 1) * 100, 1)
        lines.append(f"| `{flag}` | `{cat}` | {count} | {pct}% | Classified in EquityFlagRegistry |")

    lines.extend([
        "",
        "## 3. 🎯 Live Probe Telemetry Ledger",
        f"Total Probe Events in Ledger: **{telemetry_stats['total_events']}**",
        f"Probe Resolution Rate: **{telemetry_stats['resolved_rate'] * 100}%**",
        "",
        "## 4. 💡 Actionable Prompt & Guardrail Feedback",
        "1. **Macro Dual-Source Enforcement:** Prompts must enforce at least 2 distinct `source_refs` per asset to eliminate `SINGLE_SOURCE_PENALTY`.",
        "2. **Deterministic Observable Grounding:** Macro allocator must directly reference numbers present in observable registry to prevent `SUPPORTING_DATA_MISMATCH`.",
        "3. **Equity Assumption Transparency:** Informational/Assumption flags (10b5-1, EPS proxy) must be presented as methodology notes rather than failure warnings.",
        "",
    ])

    report_content = "\n".join(lines)
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report_content)

    return report_content


if __name__ == "__main__":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    report = generate_audit_report()
    print(report)
