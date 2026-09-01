import json
import math
from pathlib import Path
import pytest
import pandas as pd

from schemas.micro_quant_schemas import AnalysisEvidenceSnapshot
from tools.market.evidence_manifest_builder import (
    build_analysis_evidence_snapshot,
    replay_analysis_from_manifest,
)
from tools.market.quant_engine import create_atomic_market_snapshot
from tools.market.technical import compute_tactical_setup
from tools.market.dcf_valuation import compute_institutional_reverse_dcf


from datetime import datetime
def test_manifest_building_and_offline_replay(tmp_path):
    # 1. Create a synthetic dataset
    ticker = "FTNT"
    today_str = datetime.now().strftime("%Y-%m-%d")
    dates = pd.date_range(end=today_str, periods=250, freq="B")
    prices = [150.0 + i * 0.1 for i in range(250)]
    prices[-1] = 172.78
    df_1y = pd.DataFrame({
        "Date": dates.strftime("%Y-%m-%d"),
        "Open": prices,
        "High": [p + 2.0 for p in prices],
        "Low": [p - 2.0 for p in prices],
        "Close": prices,
        "Volume": [1000000] * 250,
    })

    market_data_payload = {
        "symbol": "FTNT",
        "currentPrice": 172.78,
        "sharesOutstanding": 733713653,
        "marketCap": 126771044965,
        "totalRevenue": 6799600000,
        "ebit": 2302400000,
        "totalCash": 3000000000,
        "totalDebt": 1000000000,
        "beta": 1.15,
    }

    val_params_payload = {
        "base_revenue": 6799600000,
        "base_ebit_margin_pct": 33.86,
        "total_cash": 3000000000,
        "total_debt": 1000000000,
        "beta": 1.15,
    }

    macro_payload = {
        "risk_free_rate_pct": 4.25,
        "erp_pct": -0.22,
    }

    raw_payloads = {
        "market_data": market_data_payload,
        "technical_ohlcv_1y": df_1y.to_dict(orient="records"),
        "valuation_parameters": val_params_payload,
        "macro_valuation": macro_payload,
    }

    # 2. Build snapshot with CAS storage
    cas_dir = tmp_path / "cas"
    cas_dir.mkdir(parents=True, exist_ok=True)

    # Patch CAS directory for testing
    import tools.market.evidence_store as ev_store
    orig_dir = ev_store._DEFAULT_CAS_DIR
    ev_store._DEFAULT_CAS_DIR = cas_dir

    try:
        snapshot = build_analysis_evidence_snapshot(
            ticker="FTNT",
            market="US",
            as_of_date="2026-08-27",
            raw_payloads=raw_payloads,
            derived_features={
                "analysis_price": 172.78,
                "price_sync_status": "synced",
                "breakout_planned_rr": 2.00,
                "reverse_dcf_is_actionable": False,
            },
        )

        assert snapshot.metadata.coverage_pct == 100.0
        assert snapshot.metadata.snapshot_sha256 is not None

        # 3. Replay offline from CAS storage
        replayed, is_reconciled, diffs = replay_analysis_from_manifest(
            snapshot,
            base_dir=cas_dir,
        )

        assert is_reconciled is True, f"Replay reconciliation failed with diffs: {diffs}"
        assert replayed["analysis_price"] == 172.78
        assert replayed["reverse_dcf_is_actionable"] is False
        assert replayed["breakout_planned_rr"] == 2.00
    finally:
        ev_store._DEFAULT_CAS_DIR = orig_dir
