"""Audit all data on the Macro page: AI Analysis and Raw Market Endpoints.

Strict acceptance tool following W02:
- Exits with 0 ONLY if all checks pass completely.
- Exits with 1 on any data mismatch, missing snapshot, or endpoint error.
- Exits with 2 on missing prerequisites/blocked.
- Exits with 3 on uncaught script execution errors.
"""
import os
import sys
import json
from datetime import datetime, timezone
from pathlib import Path

project_root = Path(__file__).parent.parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

from fastapi.testclient import TestClient
from api.main import app
from api.config import get_webui_password
from tools.macro.contracts import AcceptanceExitCode, MARKET_OBSERVABLE_COVERAGE


def summarize_ai_integrity(dashboard: dict, snapshot: list | None) -> dict:
    """Compare persisted Quant evidence with the immutable report without invoking an LLM."""
    registry = dashboard.get("observable_registry") or {}
    
    if snapshot is None:
        return {
            "snapshot_found": False,
            "snapshot_observables_count": 0,
            "report_observables_count": len(registry),
            "missing_from_report": ["ALL_SNAPSHOT_OBSERVABLES_MISSING (No snapshot file found)"],
            "changed_evidence": [],
            "invalid_or_unknown_citations": [],
            "invalid_observables_count": sum(o.get("is_valid") is False for o in registry.values()),
            "market_groups_missing_from_report": {
                name: [oid for oid in ids if oid not in registry]
                for name, ids in MARKET_OBSERVABLE_COVERAGE.items()
                if any(oid not in registry for oid in ids)
            },
            "limitations": [
                "CRITICAL: Snapshot file could not be found in the strategy vault.",
            ],
        }

    missing = [o["observable_id"] for o in snapshot if o.get("observable_id") not in registry]
    changed = [
        o["observable_id"]
        for o in snapshot
        if o.get("observable_id") in registry
        and any(
            o.get(field) != registry[o["observable_id"]].get(field)
            for field in ("value", "unit", "observed_at", "is_valid")
        )
    ]
    bad_refs = set()
    for section in ("regime_evidence", "asset_allocation", "pair_trades"):
        for item in dashboard.get(section) or []:
            for ref in item.get("observable_refs") or []:
                if ref not in registry or registry[ref].get("is_valid") is False:
                    bad_refs.add(ref)

    return {
        "snapshot_found": True,
        "snapshot_observables_count": len(snapshot),
        "report_observables_count": len(registry),
        "missing_from_report": sorted(missing),
        "changed_evidence": sorted(changed),
        "invalid_or_unknown_citations": sorted(bad_refs),
        "invalid_observables_count": sum(o.get("is_valid") is False for o in registry.values()),
        "market_groups_missing_from_report": {
            name: [oid for oid in ids if oid not in registry]
            for name, ids in MARKET_OBSERVABLE_COVERAGE.items()
            if any(oid not in registry for oid in ids)
        },
        "limitations": [
            "This audit reads the stored report; it does not generate a new AI analysis.",
            "Thai GDP/CPI/MPI are read from local official_hard_data.json; live authority provenance is not verified here.",
            "Market refreshes may be newer than the evidence pinned to the AI report.",
        ],
    }


def audit() -> int:
    failures: list[str] = []
    
    try:
        client = TestClient(app, raise_server_exceptions=False)
    except Exception as exc:
        print(f"[FATAL] Failed to initialize TestClient: {exc}")
        return AcceptanceExitCode.ERROR

    pw = get_webui_password()
    login_res = client.post("/api/auth/login", json={"password": pw})
    if login_res.status_code != 200:
        print(f"[BLOCKED] Login failed: {login_res.status_code} {login_res.text}")
        return AcceptanceExitCode.BLOCKED

    report = {"audited_at": datetime.now(timezone.utc).isoformat()}

    # 1. AI Dashboard (/api/macro/dashboard)
    print("Auditing /api/macro/dashboard...")
    r = client.get("/api/macro/dashboard")
    if r.status_code != 200:
        report["ai_dashboard"] = {"status": "error", "code": r.status_code, "detail": r.text}
        failures.append(f"/api/macro/dashboard returned HTTP {r.status_code}")
    else:
        d = r.json()
        report["ai_dashboard"] = {
            "status": "ok",
            "evaluated_at": d.get("evaluated_at"),
            "strategy_report_id": d.get("strategy_report_id"),
            "overall_regime": d.get("overall_regime"),
            "time_horizon": d.get("time_horizon"),
            "conviction_level": d.get("conviction_level"),
            "key_assumptions_count": len(d.get("key_assumptions", [])),
            "focus_themes_count": len(d.get("focus_themes", [])),
            "asset_allocation_count": len(d.get("asset_allocation", [])),
            "asset_allocation_items": [
                {
                    "class": a.get("asset_class"),
                    "region": a.get("region"),
                    "stance": a.get("stance"),
                    "confidence": a.get("confidence"),
                }
                for a in d.get("asset_allocation", [])
            ],
            "pair_trades_count": len(d.get("pair_trades", [])),
            "risk_scenarios_count": len(d.get("risk_scenarios", [])),
            "dashboard_indicators_count": len(d.get("dashboard_indicators", [])),
            "observable_registry_count": len(d.get("observable_registry", {}) or {}),
            "regional_assessments": d.get("regional_assessments"),
            "thailand_market_stance": d.get("thailand_market_stance"),
            "sector_snapshot_id": d.get("sector_snapshot_id"),
            "sector_analysis_present": d.get("sector_analysis") is not None,
            "warnings_count": len(d.get("warnings", [])),
        }
        
        # Check snapshot binding
        eval_date = str(d.get("evaluated_at", ""))[:10]
        from tools.macro.adapters.strategy_vault_adapter import StrategyVaultAdapter
        vault_adapter = StrategyVaultAdapter()
        snapshots_dir = vault_adapter.vault_path / "30_Knowledge_Base/Macroeconomics/Daily_Snapshots"
        snapshot_files = list(snapshots_dir.rglob(f"Macro_Observables_Snapshot_{eval_date}*.json"))
        
        snapshot = None
        if snapshot_files:
            latest_snapshot_file = max(snapshot_files, key=lambda p: p.stat().st_mtime)
            try:
                snapshot = json.loads(latest_snapshot_file.read_text(encoding="utf-8"))
            except Exception as exc:
                failures.append(f"Corrupt snapshot file {latest_snapshot_file}: {exc}")
        else:
            failures.append(f"No snapshot file found for evaluated_at date '{eval_date}' in {snapshots_dir}")

        integrity = summarize_ai_integrity(d, snapshot)
        report["ai_data_integrity"] = integrity
        
        if not integrity["snapshot_found"]:
            failures.append("Snapshot was not found for AI report")
        if integrity["missing_from_report"]:
            failures.append(f"{len(integrity['missing_from_report'])} snapshot observables missing from report")
        if integrity["changed_evidence"]:
            failures.append(f"{len(integrity['changed_evidence'])} observables changed between snapshot and report")
        if integrity["invalid_or_unknown_citations"]:
            failures.append(f"{len(integrity['invalid_or_unknown_citations'])} invalid or unknown citations in report")
        if integrity["market_groups_missing_from_report"]:
            failures.append(f"Market groups missing observables: {list(integrity['market_groups_missing_from_report'].keys())}")

    # 2. Market Raw Endpoints
    market_endpoints = {
        "us_yield_curve": "/api/v2/market/macro/treasury/yield-curve",
        "financial_stress": "/api/v2/market/macro/financial-stress",
        "metals_cot_gold": "/api/v2/market/commodities/metals/cot?commodity=gold",
        "global_policy_rates": "/api/v2/market/macro/global-policy-rates",
        "commodity_volatility": "/api/v2/market/commodities/volatility",
        "auction_demand_note_10y": "/api/v2/market/macro/treasury/auction-demand?security_type=Note&security_term=10-Year",
        "auction_demand_bill_13w": "/api/v2/market/macro/treasury/auction-demand?security_type=Bill&security_term=13-Week",
        "us_national_debt": "/api/v2/market/macro/treasury/debt?limit=30",
        "crypto_liquidity": "/api/v2/market/macro/crypto-liquidity",
        "th_investor_flow": "/api/v2/market/thailand/flow?market=SET",
        "th_retail_gold": "/api/v2/market/thailand/gold",
        "th_market_valuation": "/api/v2/market/thailand/valuation?market=SET",
        "th_market_breadth": "/api/v2/market/thailand/breadth?market=SET",
        "sector_rotation_latest": "/api/macro/sector-rotation/latest?timeframe=weekly&tail=12",
    }

    report["raw_market_endpoints"] = {}
    for name, path in market_endpoints.items():
        print(f"Auditing {path}...")
        resp = client.get(path)
        if resp.status_code == 200:
            data = resp.json()
            summary = {"status": "ok", "code": 200}
            if isinstance(data, list):
                summary["items_count"] = len(data)
                summary["sample"] = data[0] if data else None
                summary["stale_items_count"] = sum(bool(item.get("is_stale")) for item in data if isinstance(item, dict))
                if not data:
                    summary["status"] = "empty"
                    failures.append(f"Endpoint {name} returned empty list")
            elif isinstance(data, dict):
                summary["keys"] = list(data.keys())
                if "points" in data:
                    summary["points_count"] = len(data["points"])
                if "rates" in data:
                    summary["rates_count"] = len(data["rates"])
                if "rows" in data:
                    summary["rows_count"] = len(data["rows"])
                summary["sample"] = {k: v for k, v in list(data.items())[:5]}
                summary["is_stale"] = data.get("is_stale", False)
                summary["stale_reason"] = data.get("stale_reason")
                summary["observed_at"] = next(
                    (data[k] for k in ("observation_date", "as_of_date", "as_of", "announced_at", "latest_auction_date") if data.get(k)),
                    None,
                )
                if "rates" in data:
                    summary["stale_rate_items_count"] = sum(bool(item.get("is_stale")) for item in data["rates"])
            report["raw_market_endpoints"][name] = summary
        else:
            report["raw_market_endpoints"][name] = {
                "status": "error",
                "code": resp.status_code,
                "detail": resp.text[:200],
            }
            failures.append(f"Endpoint {name} failed with HTTP {resp.status_code}")

    # 3. Indicators Series Check (from dashboard_indicators)
    report["indicators_sample_series"] = {}
    if report["ai_dashboard"].get("status") == "ok":
        indicators = d.get("dashboard_indicators", [])
        for ind in indicators:
            ind_id = ind.get("indicator_id")
            s_res = client.get(f"/api/macro/indicators/{ind_id}/series?range=3m")
            report["indicators_sample_series"][ind_id] = {
                "code": s_res.status_code,
                "points_count": len(s_res.json().get("points", [])) if s_res.status_code == 200 else 0,
            }
            if s_res.status_code != 200:
                failures.append(f"Indicator series {ind_id} failed with HTTP {s_res.status_code}")

    report["failures"] = failures
    report["success"] = len(failures) == 0

    # Save output to audit file
    audit_path = project_root / "tests/audit_macro_page_result.json"
    audit_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nAudit completed! Report saved to {audit_path}")
    
    if failures:
        print(f"\n[FAIL] Audit failed with {len(failures)} discrepancies:")
        for f in failures:
            print(f"  - {f}")
        return AcceptanceExitCode.FAIL

    print("\n[PASS] All Macro Page and API endpoints audited successfully with 0 failures!")
    return AcceptanceExitCode.PASS


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(audit())
