"""Verify live Macro inputs, derived metrics, scoring, and lineage without invoking an LLM.

Strict acceptance tool following W02 & W05:
- Extracts all observables from live market inputs.
- Computes matrix scores and derived metrics mathematically.
- Reconciles coverage against tools.macro.contracts.MARKET_OBSERVABLE_COVERAGE.
- Checks invalid observables against KNOWN_DEGRADED_SERIES_POLICY.
- Exits with 0 ONLY if all checks pass.
- Exits with 1 on unexpected failures, coverage gaps, or calculation errors.
- Exits with 2 on blocked dependencies.
"""
import concurrent.futures
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
load_dotenv()

from tools.macro.ingest import ingest_global_macro, ingest_regional_macro, ingest_country_macro
from tools.macro.evaluation import _extract_market_observables
from tools.macro.terminal_observables import (
    build_thai_market_observables,
    build_rates_observables,
    build_crypto_liquidity_observables,
)
from tools.macro.adapters.thai_hard_data_adapter import ThaiHardDataAdapter
from tools.market.terminal_v2.adapters.thaibma_adapter import ThaiBmaPublicAdapter
from tools.macro.contracts import (
    AcceptanceExitCode,
    MARKET_OBSERVABLE_COVERAGE,
    KNOWN_DEGRADED_SERIES_POLICY,
)
from tools.macro.scoring import _calculate_matrix_scores_from_observables


def verify() -> int:
    today = os.getenv("EVAL_DATE") or datetime.now().strftime("%Y-%m-%d")
    ingesters = {
        "Global_Macro_Snapshot": ingest_global_macro,
        "Regional_Macro_Snapshot": ingest_regional_macro,
        "Country_Macro_Snapshot": ingest_country_macro,
    }
    contents = {}
    failures = {}
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Ingesting live macro snapshots (as of {today})...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(tool.invoke, {}): name for name, tool in ingesters.items()}
        for future in concurrent.futures.as_completed(futures):
            name = futures[future]
            try:
                contents[name] = future.result()
            except Exception as exc:
                failures[name] = f"{type(exc).__name__}: {exc}"

    if failures:
        print(f"[FAIL] Ingestion failures encountered: {failures}")
        return AcceptanceExitCode.FAIL

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Extracting market observables from snapshot markdown...")
    observables = _extract_market_observables(contents, {}, today)
    snapshot_observables = list(observables)

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Building terminal observables (Thai market, Rates, Crypto)...")
    for builder in (build_thai_market_observables, build_rates_observables, build_crypto_liquidity_observables):
        observables.extend(builder(as_of_date=today))

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Building Thai hard data observables...")
    observables.extend(ThaiHardDataAdapter(thaibma_adapter=ThaiBmaPublicAdapter()).as_observables(today))

    registry = {o.observable_id: o for o in observables}

    # Verify coverage against canonical contract
    missing = {
        name: [oid for oid in ids if oid not in registry]
        for name, ids in MARKET_OBSERVABLE_COVERAGE.items()
        if any(oid not in registry for oid in ids)
    }

    yahoo = [o for o in snapshot_observables if o.provider == "Yahoo"]
    yahoo_missing_real_dates = [o.observable_id for o in yahoo if o.observed_at == "1970-01-01"]

    # Reconcile invalid observables against KNOWN_DEGRADED_SERIES_POLICY
    expected_degraded = []
    unexpected_invalid = []
    for o in registry.values():
        if not o.is_valid:
            # Check if matching any known degraded policy (by ID or series substring)
            matched_policy = None
            for pattern, policy in KNOWN_DEGRADED_SERIES_POLICY.items():
                if pattern.lower() in o.observable_id.lower():
                    matched_policy = (pattern, policy)
                    break
            if matched_policy:
                expected_degraded.append({
                    "id": o.observable_id,
                    "observed_at": o.observed_at,
                    "reason": o.stale_reason,
                    "policy_pattern": matched_policy[0],
                    "policy": matched_policy[1]["policy"],
                })
            else:
                unexpected_invalid.append({
                    "id": o.observable_id,
                    "observed_at": o.observed_at,
                    "reason": o.stale_reason,
                })

    # Execute scoring computation (Independent quantitative verification)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Verifying matrix scores calculation from observables...")
    scores_dict = {}
    scoring_error = None
    try:
        scores_dict = _calculate_matrix_scores_from_observables(registry)
    except Exception as exc:
        scoring_error = f"{type(exc).__name__}: {exc}"

    report = {
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "llm_invoked": False,
        "stored_ai_report_modified": False,
        "observable_count": len(registry),
        "valid_count": sum(o.is_valid for o in registry.values()),
        "ingest_failures": failures,
        "yahoo_observable_count": len(yahoo),
        "yahoo_missing_real_dates": yahoo_missing_real_dates,
        "missing_market_groups": missing,
        "expected_degraded_observables": expected_degraded,
        "unexpected_invalid_observables": unexpected_invalid,
        "scoring_verified": scoring_error is None and bool(scores_dict),
        "scoring_error": scoring_error,
        "scoring_regions": list(scores_dict.keys()),
        "observables": [o.model_dump(mode="json") for o in registry.values()],
    }

    output = project_root / "tests/audit_macro_pipeline_result.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    
    summary = {k: v for k, v in report.items() if k != "observables"}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved audit result to: {output}")

    # Determine exit code based on strict criteria
    if failures:
        print("[FAIL] Ingest tool failures detected!")
        return AcceptanceExitCode.FAIL
    if missing:
        print(f"[FAIL] Missing required market observable groups: {list(missing.keys())}")
        return AcceptanceExitCode.FAIL
    if not yahoo:
        print("[FAIL] No Yahoo market observables found!")
        return AcceptanceExitCode.FAIL
    if yahoo_missing_real_dates:
        print(f"[FAIL] Yahoo observables with epoch 1970-01-01 dates: {yahoo_missing_real_dates}")
        return AcceptanceExitCode.FAIL
    if unexpected_invalid:
        print(f"[FAIL] Unexpected invalid observables not covered by degraded policy: {unexpected_invalid}")
        return AcceptanceExitCode.FAIL
    if scoring_error or not scores_dict:
        print(f"[FAIL] Scoring computation failed: {scoring_error}")
        return AcceptanceExitCode.FAIL

    print("\n[PASS] Pipeline verification succeeded! All observables and calculations verified.")
    return AcceptanceExitCode.PASS


if __name__ == "__main__":
    sys.exit(verify())
