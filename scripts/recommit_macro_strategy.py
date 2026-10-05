"""Recommit macro strategy report with fresh Thai hard data observables."""
import os
import sys
import json
import uuid
from pathlib import Path
from datetime import datetime, timezone

project_root = Path(__file__).parent.parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

from schemas.macro_schemas import MacroStrategyDirection, MarketObservable, QuantScore
from tools.macro.evaluation import evaluate_macro_matrix
from tools.macro.report_formatter import write_strategy_json_sidecar
from tools.macro.adapters.strategy_vault_adapter import StrategyVaultAdapter

def recommit():
    today_str = os.environ.get("EVAL_DATE", datetime.now().strftime("%Y-%m-%d"))
    vault_base = Path(os.getenv("OBSIDIAN_VAULT_PATH", "./memories")).resolve()
    
    # 1. Evaluate fresh macro matrix (includes Thai hard data adapter)
    print("Evaluating fresh macro matrix...")
    matrix_res = evaluate_macro_matrix.invoke({})
    quant_score = QuantScore.model_validate_json(matrix_res)
    print(f"QuantScore evaluated. Thailand state: {quant_score.regions.get('Thailand').economic_state}")
    print(f"Thailand data gaps: {quant_score.regions.get('Thailand').data_gaps}")
    print(f"Thailand confidence: {quant_score.regions.get('Thailand').confidence}")

    # 2. Load existing strategy direction
    direction_path = vault_base / "30_Knowledge_Base" / "Macroeconomics" / "Strategies" / today_str[:4] / today_str[5:7] / f"Macro_Strategy_Direction_{today_str}.json"
    if not direction_path.exists():
        direction_path = vault_base / "30_Knowledge_Base" / "Strategies" / f"Macro_Strategy_Direction_{today_str}.json"
    
    if not direction_path.exists():
        print(f"Error: {direction_path} does not exist")
        return 1

    with open(direction_path, "r", encoding="utf-8") as f:
        existing_data = json.load(f)

    # Convert existing_data to MacroStrategyDirection
    direction = MacroStrategyDirection.model_validate(existing_data)

    # Ensure direction.thailand_market_stance has fresh microstructure and a complete institutional rationale
    th_quant = quant_score.regions.get("Thailand").market_stance or {}
    current_stance = dict(direction.thailand_market_stance or {})
    for k, v in th_quant.items():
        if k not in current_stance or not current_stance[k]:
            current_stance[k] = v
    if not current_stance.get("rationale"):
        from tools.macro.terminal_observables import synthesize_thai_market_stance_narrative
        thai_assets = [
            a for a in getattr(direction, "asset_allocation", []) or []
            if getattr(a, "region", "") == "Thailand" or "THB" in getattr(a, "asset_class", "")
        ]
        current_stance["rationale"] = synthesize_thai_market_stance_narrative(current_stance, thai_assets)
    direction.thailand_market_stance = current_stance

    # 3. Build observable registry from fresh matrix observables
    obs_registry = {}
    for o in quant_score.market_observables:
        obs_registry[o.observable_id] = o

    # Preserve any existing observables from existing_data
    for k, v in existing_data.get("observable_registry", {}).items():
        if k not in obs_registry:
            try:
                obs_registry[k] = MarketObservable.model_validate(v)
            except Exception:
                pass

    # 4. Prepare regional assessments mapping
    regional_assessments = {
        r_name: r_metrics.model_dump(mode="json")
        for r_name, r_metrics in quant_score.regions.items()
    }

    # 5. Commit new report via write_strategy_json_sidecar
    new_run_id = f"macro_sync_{today_str}_{uuid.uuid4().hex[:12]}"
    run_started_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    print(f"Committing new macro report with run_id={new_run_id}...")
    sidecar_path, canonical_payload = write_strategy_json_sidecar(
        direction=direction,
        evaluated_date=today_str,
        observable_registry=obs_registry,
        report_references=existing_data.get("report_references", []),
        regional_assessments=regional_assessments,
        evaluated_sources=existing_data.get("evaluated_sources", []),
        run_id=new_run_id,
        job_id=new_run_id,
        run_started_at=run_started_at,
        sector_snapshot_id=existing_data.get("sector_snapshot_id"),
        sector_analysis=existing_data.get("sector_analysis"),
        return_canonical_payload=True,
    )
    print(f"Committed sidecar to: {sidecar_path}")
    print(f"New strategy_report_id: {canonical_payload.get('strategy_report_id')}")

    # 6. Verify with StrategyVaultAdapter().latest()
    latest_strategy = StrategyVaultAdapter().latest()
    th_assessment = latest_strategy.get("regional_assessments", {}).get("Thailand", {})
    print("Verification of StrategyVaultAdapter.latest():")
    print(f"  Thailand State: {th_assessment.get('economic_state')}")
    print(f"  Thailand Confidence: {th_assessment.get('confidence')}")
    print(f"  Thailand Coverage: {th_assessment.get('coverage')}")
    print(f"  Thailand Data Gaps: {th_assessment.get('data_gaps')}")
    print(f"  Thailand Fiscal Health: {th_assessment.get('fiscal_health')}")
    
    assert th_assessment.get("data_gaps") == [], f"Expected empty data_gaps, got {th_assessment.get('data_gaps')}"
    assert th_assessment.get("confidence") == 0.9, f"Expected 0.9 confidence, got {th_assessment.get('confidence')}"
    print("SUCCESS: Strategy report committed and verified successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(recommit())
