import pytest
import json
from datetime import datetime
from pathlib import Path
from tools.macro.evaluation import evaluate_macro_matrix
from schemas.macro_schemas import QuantScore

def test_evaluation_quant_valid_json(tmp_vault):
    today_str = datetime.now().strftime("%Y-%m-%d")
    snapshots_dir = tmp_vault / "30_Knowledge_Base" / "Macroeconomics" / "Daily_Snapshots" / today_str
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    us_content = """# 🇺🇸 United States
| ดัชนี | ค่าล่าสุด |
|-------|----------|
| **Real GDP (YoY %)** | 3.0% |
| **CPI (YoY %)** | 3.5% |
| **Fed Funds Rate** | 5.25% |"""

    global_content = """| ดัชนี | ค่าล่าสุด |
|-------|----------|
| **VIX Index** | 15.00 |"""

    (snapshots_dir / f"Country_Macro_Snapshot_{today_str}.md").write_text(us_content, encoding="utf-8")
    (snapshots_dir / f"Global_Macro_Snapshot_{today_str}.md").write_text(global_content, encoding="utf-8")
    (snapshots_dir / f"Regional_Macro_Snapshot_{today_str}.md").write_text("", encoding="utf-8")

    result = evaluate_macro_matrix.invoke({})

    # 1. Parse JSON
    parsed = json.loads(result)

    # 2. Validate Schema
    quant = QuantScore.model_validate(parsed)

    # 3. Check values
    assert quant.global_geopolitics_score == 1.0
    assert "United States" in quant.regions
    assert quant.regions["United States"].economic_state.value in ["Goldilocks", "Reflation", "Stagflation", "Recession"]

def test_evaluation_quant_missing_file(tmp_vault):
    # Don't create any files
    result = evaluate_macro_matrix.invoke({})
    assert "Error:" in result


def test_evaluation_quant_structured_observables_and_stale_cutoff(tmp_vault):
    today_str = datetime.now().strftime("%Y-%m-%d")
    snapshots_dir = tmp_vault / "30_Knowledge_Base" / "Macroeconomics" / "Daily_Snapshots" / today_str
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    country_content = f"""# United States
| ดัชนี | ค่าล่าสุด | ก่อนหน้า | MA ย้อนหลัง | วันที่ |
|-------|----------|----------|----------|----------|
| **Real GDP (YoY %)** | 3.0% | 2.8% | 2.7% | {today_str} |
| **CPI (YoY %)** | 3.5% | 3.4% | 3.2% | {today_str} |
| **Fed Funds Rate (`FEDFUNDS`)** | 5.25% | 5.25% | 5.25% | {today_str} |
| **10Y-2Y Spread (`T10Y2Y`)** | 0.35% | 0.30% | 0.10% | {today_str} |

# Thailand
| ดัชนี | ค่าล่าสุด | ก่อนหน้า | MA ย้อนหลัง | วันที่ |
|-------|----------|----------|----------|----------|
| **Policy Rate** | 2.50% | 2.50% | 2.50% | {today_str} |
| **Thailand 10Y Gov Bond Yield [StaticProxy] (`TH10Y`)** | 2.65% | 2.65% | 2.65% | {today_str} |
"""
    global_content = f"""| ดัชนี | ค่าล่าสุด | ก่อนหน้า | MA ย้อนหลัง | วันที่ |
|-------|----------|----------|----------|----------|
| **VIX Index (`^VIX`)** | 15.00 | 16.00 | 17.00 | {today_str} |
| **High Yield Bond ETF (`HYG`)** | 80.00 | 79.00 | 78.00 | {today_str} |
| **Investment Grade Bond ETF (`LQD`)** | 100.00 | 101.00 | 102.00 | {today_str} |
| **WTI Crude Oil (`CL=F`)** | 70.00 | 69.00 | 68.00 | 2026-06-01 |
"""
    regional_content = f"""# United States
| ดัชนี | ค่าล่าสุด | ก่อนหน้า | MA ย้อนหลัง | วันที่ |
|-------|----------|----------|----------|----------|
| **S&P 500 (`^GSPC`)** | 6000.00 | 5900.00 | 5800.00 | {today_str} |
"""
    (snapshots_dir / f"Country_Macro_Snapshot_{today_str}.md").write_text(country_content, encoding="utf-8")
    (snapshots_dir / f"Global_Macro_Snapshot_{today_str}.md").write_text(global_content, encoding="utf-8")
    (snapshots_dir / f"Regional_Macro_Snapshot_{today_str}.md").write_text(regional_content, encoding="utf-8")

    result = evaluate_macro_matrix.invoke({})
    quant = QuantScore.model_validate(json.loads(result))

    assert quant.market_observables
    assert any(obs.observable_id == "obs_spread_us_10y_2y" for obs in quant.market_observables)
    assert any(obs.observable_id == "obs_diff_us_th_policy_rate" for obs in quant.market_observables)
    assert any(obs.observable_id == "obs_ratio_hyg_lqd" for obs in quant.market_observables)
    assert next(obs for obs in quant.market_observables if obs.provider == "StaticProxy").is_valid is False
    assert next(obs for obs in quant.market_observables if "WTI" in obs.indicator).is_valid is False
