from datetime import datetime, timezone, timedelta
import json
import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.state_db import get_connection, record_dcf_evaluation

client = TestClient(app)


@pytest.fixture(autouse=True)
def override_require_session():
    from api.auth import require_session
    app.dependency_overrides[require_session] = lambda: {"user_id": "mock_user"}
    yield
    app.dependency_overrides = {}


def test_valuation_targets_empty_ticker_returns_unavailable():
    res = client.get("/api/equity/NONEXISTENT_TICKER_XYZ/valuation-targets")
    assert res.status_code == 200
    data = res.json()
    assert data["ticker"] == "NONEXISTENT_TICKER_XYZ"
    assert data["status"] == "unavailable"
    assert data["comparability_status"] == "unknown"
    assert len(data["scenarios"]) == 0


def test_valuation_targets_canonical_ledger_seed_and_read():
    conn = get_connection()
    eval_id = "eval_AAPL_2026-08-20_001"
    evaluated_at = "2026-08-20T10:00:00Z"
    scenarios = {
        "base": {"target_price": 220.0, "upside_pct": 10.0, "margin_of_safety_pct": 9.1},
        "bull": {"target_price": 260.0, "upside_pct": 30.0, "margin_of_safety_pct": 23.0},
        "bear": {"target_price": 180.0, "upside_pct": -10.0, "margin_of_safety_pct": -12.5},
    }
    record_dcf_evaluation(
        conn=conn,
        evaluation_id=eval_id,
        ticker="AAPL",
        market="US",
        evaluated_at=evaluated_at,
        scenarios=scenarios,
        model_version="dcf_v1.0",
        valuation_price_basis="split_adjusted_only",
        current_price_at_eval=200.0,
        wacc_pct=8.5,
        valuation_verdict="undervalued",
        corporate_action_evidence=[],
        input_snapshot={"observable_refs": ["fred:DGS10", "damodaran:erp"]},
    )

    res = client.get("/api/equity/AAPL/valuation-targets")
    assert res.status_code == 200
    data = res.json()
    assert data["evaluation_id"] == eval_id
    assert data["ticker"] == "AAPL"
    assert data["market"] == "US"
    assert data["currency"] == "USD"
    assert data["valuation_verdict"] == "undervalued"
    assert data["wacc_pct"] == 8.5
    assert data["comparability_status"] == "comparable"
    assert data["scenario_order_valid"] is True
    assert len(data["scenarios"]) == 3
    base_sc = next(s for s in data["scenarios"] if s["scenario_name"] == "base")
    assert base_sc["target_price"] == 220.0
    assert base_sc["label"] == "DCF Base"
    assert base_sc["color"] == "emerald"


def test_valuation_targets_comparability_post_eval_split_detection():
    conn = get_connection()
    eval_id = "eval_NVDA_2026-06-01_001"
    evaluated_at = "2026-06-01T10:00:00Z"
    scenarios = {
        "base": {"target_price": 1200.0, "upside_pct": 20.0, "margin_of_safety_pct": 16.6},
        "bull": {"target_price": 1500.0, "upside_pct": 50.0, "margin_of_safety_pct": 33.3},
        "bear": {"target_price": 900.0, "upside_pct": -10.0, "margin_of_safety_pct": -11.1},
    }
    # Stock split occurred on 2026-06-10 (AFTER evaluation date 2026-06-01)
    corp_evidence = [
        {"event_type": "split", "effective_date": "2026-06-10", "ratio": 10.0}
    ]
    record_dcf_evaluation(
        conn=conn,
        evaluation_id=eval_id,
        ticker="NVDA",
        market="US",
        evaluated_at=evaluated_at,
        scenarios=scenarios,
        model_version="dcf_v1.0",
        valuation_price_basis="split_adjusted_only",
        current_price_at_eval=1000.0,
        wacc_pct=9.2,
        valuation_verdict="undervalued",
        corporate_action_evidence=corp_evidence,
    )

    res = client.get("/api/equity/NVDA/valuation-targets")
    assert res.status_code == 200
    data = res.json()
    assert data["comparability_status"] == "not_comparable"
    assert len(data["comparability_reasons"]) > 0
    assert "Unadjusted stock split" in data["comparability_reasons"][0]


def test_valuation_targets_monotonicity_validation():
    conn = get_connection()
    eval_id = "eval_BROKEN_2026-08-20_001"
    evaluated_at = "2026-08-20T10:00:00Z"
    # Invalid order: Bear is higher than Base!
    scenarios = {
        "base": {"target_price": 100.0, "upside_pct": 0.0},
        "bull": {"target_price": 150.0, "upside_pct": 50.0},
        "bear": {"target_price": 120.0, "upside_pct": 20.0},
    }
    record_dcf_evaluation(
        conn=conn,
        evaluation_id=eval_id,
        ticker="BROKEN",
        market="US",
        evaluated_at=evaluated_at,
        scenarios=scenarios,
    )

    res = client.get("/api/equity/BROKEN/valuation-targets")
    assert res.status_code == 200
    data = res.json()
    assert data["scenario_order_valid"] is False
    assert data["valuation_verdict"] == "unknown"


def test_valuation_targets_stale_status_detection():
    conn = get_connection()
    eval_id = "eval_OLD_2025-01-01_001"
    evaluated_at = "2025-01-01T10:00:00Z"
    scenarios = {
        "base": {"target_price": 50.0, "upside_pct": 5.0},
    }
    record_dcf_evaluation(
        conn=conn,
        evaluation_id=eval_id,
        ticker="OLD",
        market="US",
        evaluated_at=evaluated_at,
        scenarios=scenarios,
    )

    res = client.get("/api/equity/OLD/valuation-targets")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "stale"
