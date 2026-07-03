import pytest
from schemas.macro_schemas import (
    GeographicScope, Region, EconomicIndicator, EconomicState,
    TrendDirection, CellMetrics, RegionStateEvaluation, MacroEconomicMatrix, IndicatorComponent
)
from tools.macro.parsers import _parse_float_from_str, _parse_markdown_table_rows

def test_macro_schemas_validation():
    # Verify we can construct and validate the Pydantic models correctly
    comp = IndicatorComponent(
        symbol_or_id="GDPC1",
        name="Real GDP YoY",
        value=2.5,
        unit="% YoY",
        date="2026-06-16",
        change_pct=0.1
    )
    assert comp.value == 2.5
    assert comp.symbol_or_id == "GDPC1"

    cell = CellMetrics(
        score=0.5,
        trend=TrendDirection.UP,
        status_label="Expansion",
        components=[comp]
    )
    assert cell.score == 0.5
    assert len(cell.components) == 1

    eval_result = RegionStateEvaluation(
        region=Region.USA,
        scope=GeographicScope.COUNTRY,
        evaluated_state=EconomicState.GOLDILOCKS,
        confidence_score=1.0,
        recommended_assets=["Growth Stocks"],
        rationale="Strong growth and low inflation",
        cells={EconomicIndicator.ECONOMIC_GROWTH: cell}
    )
    assert eval_result.evaluated_state == EconomicState.GOLDILOCKS

    matrix = MacroEconomicMatrix(
        evaluations={Region.USA: eval_result}
    )
    assert len(matrix.evaluations) == 1
    assert matrix.evaluations[Region.USA].region == Region.USA

def test_parse_float_from_str():
    assert _parse_float_from_str("2.50%") == 2.5
    assert _parse_float_from_str("▲0.50%") == 0.5
    assert _parse_float_from_str("▼-1.25%") == -1.25
    assert _parse_float_from_str("1,588.05") == 1588.05
    assert _parse_float_from_str("32.4700 THB") == 32.47
    assert _parse_float_from_str(None) is None
    assert _parse_float_from_str("") is None
    assert _parse_float_from_str("invalid") is None

def test_parse_markdown_table_rows():
    md = """
| ตัวชี้วัด | ค่าล่าสุด | เปลี่ยนแปลง |
|-----------|----------|-------------|
| **อัตราดอกเบี้ยนโยบาย (Policy Rate)** | 2.50% | - |
| **อัตราแลกเปลี่ยน (USD/THB)** | 32.4700 THB | - |
"""
    rows = _parse_markdown_table_rows(md)
    assert len(rows) == 2
    assert rows[0]["ตัวชี้วัด"] == "**อัตราดอกเบี้ยนโยบาย (Policy Rate)**"
    assert rows[0]["ค่าล่าสุด"] == "2.50%"
