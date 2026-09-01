"""Evidence Manifest Builder (Phase 1 & v3.1).

Responsible for assembling the AnalysisEvidenceSnapshot manifest, saving raw payloads
to the Content-Addressed Store (.evidence_cache/{sha256}.json), and calculating dynamic coverage.
"""
from datetime import datetime, timezone
import hashlib
import json
import uuid
from typing import Any, Dict, List, Optional, Tuple

from schemas.micro_quant_schemas import (
    AnalysisEvidenceSnapshot,
    CorporateActionsEvidence,
    EvidenceItemMetadata,
    EvidenceManifestItem,
    SnapshotMetadata,
)
from tools.market.evidence_store import save_evidence_payload


def generate_analysis_run_id(ticker: str) -> str:
    """Generate a sortable timestamp UUIDv7-style run ID."""
    now_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    rand_suffix = uuid.uuid4().hex[:8]
    clean_sym = ticker.replace(".", "_").upper()
    return f"run_{clean_sym}_{now_str}_{rand_suffix}"


def build_analysis_evidence_snapshot(
    ticker: str,
    market: str,
    as_of_date: str,
    raw_payloads: Dict[str, Any],
    corporate_actions_info: Optional[Dict[str, Any]] = None,
    derived_features: Optional[Dict[str, Any]] = None,
    data_quality_flags: Optional[List[str]] = None,
) -> AnalysisEvidenceSnapshot:
    """Assemble an immutable Evidence Manifest and store raw payloads to CAS."""
    now_iso = datetime.now(timezone.utc).isoformat()
    manifest_items: Dict[str, EvidenceManifestItem] = {}
    flags = list(data_quality_flags or [])
    
    total_applicable_categories = max(1, len(raw_payloads))
    available_categories = 0

    for category, payload in raw_payloads.items():
        if payload is None or (isinstance(payload, dict) and not payload):
            # Not fetched or empty -> mark explicitly as unavailable
            manifest_items[category] = EvidenceManifestItem(
                item_id=f"{ticker}_{category}",
                metadata=EvidenceItemMetadata(
                    source_as_of=as_of_date,
                    retrieved_at=now_iso,
                    source_uri=f"provider:///{ticker}/{category}",
                    payload_hash=hashlib.sha256(b"").hexdigest(),
                    status="unavailable",
                    stale_reason=f"No {category} payload available at runtime",
                ),
            )
            continue

        # Valid payload -> store to CAS
        storage_ref, payload_hash = save_evidence_payload(
            payload,
            metadata={
                "ticker": ticker,
                "category": category,
                "as_of_date": as_of_date,
                "retrieved_at": now_iso,
            },
        )
        manifest_items[category] = EvidenceManifestItem(
            item_id=f"{ticker}_{category}",
            metadata=EvidenceItemMetadata(
                source_as_of=as_of_date,
                retrieved_at=now_iso,
                source_uri=f"yfinance:///{ticker}/{category}",
                payload_hash=payload_hash,
                status="available",
            ),
            storage_ref=storage_ref,
        )
        available_categories += 1

    # Corporate actions evidence
    corp_meta = EvidenceItemMetadata(
        source_as_of=as_of_date,
        retrieved_at=now_iso,
        source_uri=f"yfinance:///{ticker}/corporate_actions",
        payload_hash=hashlib.sha256(json.dumps(corporate_actions_info or {}, sort_keys=True).encode()).hexdigest(),
        status="available" if corporate_actions_info else "unavailable",
    )
    corp_evidence = CorporateActionsEvidence(
        metadata=corp_meta,
        chart_price_basis="split_and_dividend_adjusted",
        valuation_price_basis="unadjusted_close",
    )

    # Calculate true dynamic coverage % clamped strictly to [0.0, 100.0]
    raw_coverage = (available_categories / total_applicable_categories) * 100.0
    coverage_pct = min(100.0, max(0.0, round(raw_coverage, 1)))
    if coverage_pct < 50.0:
        flags.append("sparse_evidence_coverage")

    # Assemble manifest-level metadata
    manifest_bytes = json.dumps(
        {k: v.model_dump() for k, v in manifest_items.items()},
        sort_keys=True,
    ).encode("utf-8")
    snapshot_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    run_id = generate_analysis_run_id(ticker)

    snap_metadata = SnapshotMetadata(
        analysis_run_id=run_id,
        schema_version="3.1",
        as_of_date=as_of_date,
        generated_at=now_iso,
        snapshot_sha256=snapshot_sha256,
        data_quality_flags=flags,
        coverage_pct=coverage_pct,
    )

    return AnalysisEvidenceSnapshot(
        metadata=snap_metadata,
        manifest_items=manifest_items,
        corporate_actions=corp_evidence,
        derived_features=derived_features or {},
    )


def replay_analysis_from_manifest(
    snapshot: AnalysisEvidenceSnapshot,
    base_dir: Optional[Any] = None,
) -> Tuple[Dict[str, Any], bool, Dict[str, Any]]:
    """Replays deterministic analysis metrics entirely offline from CAS payloads."""
    import math
    import pandas as pd
    from tools.market.evidence_store import load_evidence_payload
    from tools.market.quant_engine import create_atomic_market_snapshot
    from tools.market.technical import compute_tactical_setup
    from tools.market.dcf_valuation import compute_institutional_reverse_dcf

    loaded_payloads: Dict[str, Any] = {}
    for cat, item in snapshot.manifest_items.items():
        if item.storage_ref:
            try:
                loaded_payloads[cat] = load_evidence_payload(
                    item.storage_ref,
                    expected_hash=item.metadata.payload_hash,
                    base_dir=base_dir,
                )
            except Exception:
                loaded_payloads[cat] = None
        else:
            loaded_payloads[cat] = None

    market_info = loaded_payloads.get("market_data") or {}
    ohlcv_raw = loaded_payloads.get("technical_ohlcv") or loaded_payloads.get("technical_ohlcv_1y")
    if isinstance(ohlcv_raw, dict) and "bars" in ohlcv_raw:
        df_1y = pd.DataFrame(ohlcv_raw["bars"])
    elif isinstance(ohlcv_raw, list):
        df_1y = pd.DataFrame(ohlcv_raw)
    else:
        df_1y = pd.DataFrame()

    if not df_1y.empty and "Date" in df_1y.columns:
        df_1y["Date"] = pd.to_datetime(df_1y["Date"])
        df_1y.set_index("Date", inplace=True)

    # 1. Replay Atomic Market Snapshot
    snapshot_meta = snapshot.metadata
    as_of_date = snapshot_meta.as_of_date
    ticker = snapshot_meta.analysis_run_id.split("_")[1] if "_" in snapshot_meta.analysis_run_id else "UNKNOWN"
    market = "TH" if ticker.endswith(".BK") or ticker.endswith("_BK") else "US"

    atomic_snap, _ = create_atomic_market_snapshot(
        provider_symbol=ticker,
        df_1y=df_1y,
        info=market_info if isinstance(market_info, dict) else {},
        market=market,
    )

    # 2. Replay Tactical Setup
    tactical_setup, _ = compute_tactical_setup(
        ticker=ticker,
        market=market,
        price_history_df=df_1y,
        current_price=atomic_snap.analysis_price,
    )

    # 3. Replay Reverse DCF
    val_params = loaded_payloads.get("valuation_parameters") or {}
    macro_val = loaded_payloads.get("macro_valuation") or {}
    rf_pct = macro_val.get("risk_free_rate_pct", 2.75 if market == "TH" else 4.25)
    erp_pct = macro_val.get("erp_pct", 5.50)

    rev = val_params.get("base_revenue") or (market_info.get("totalRevenue") if isinstance(market_info, dict) else 0.0) or 0.0
    ebit_margin = val_params.get("base_ebit_margin_pct") or 0.0
    shares = atomic_snap.shares_outstanding or (market_info.get("sharesOutstanding") if isinstance(market_info, dict) else 1.0) or 1.0
    cash = val_params.get("total_cash") or (market_info.get("totalCash") if isinstance(market_info, dict) else 0.0) or 0.0
    debt = val_params.get("total_debt") or (market_info.get("totalDebt") if isinstance(market_info, dict) else 0.0) or 0.0
    beta = val_params.get("beta") or (market_info.get("beta") if isinstance(market_info, dict) else 1.0) or 1.0

    reverse_dcf = None
    if rev > 0 and ebit_margin > 0:
        from schemas.macro_schemas import MarketObservable
        erp_obs_key = "obs_erp_gspc" if market == "US" else "obs_th_erp"
        rf_obs_key = "obs_dgs10" if market == "US" else "obs_th_10y_yield"
        macro_reg = {
            rf_obs_key: MarketObservable(
                observable_id=rf_obs_key,
                asset_bucket="fixed_income",
                region=market,
                indicator="DGS10" if market == "US" else "TH_GOV_10Y",
                value=str(rf_pct),
                unit="%",
                observed_at=as_of_date,
                source_file="manifest_replay",
                provider="FRED" if market == "US" else "ThaiBMA",
                is_valid=True,
            ),
            erp_obs_key: MarketObservable(
                observable_id=erp_obs_key,
                asset_bucket="equities",
                region=market,
                indicator="ERP_GSPC" if market == "US" else "ERP_SET",
                value=str(erp_pct),
                unit="%",
                observed_at=as_of_date,
                source_file="manifest_replay",
                provider="MacroEngine",
                is_valid=True,
            ),
        }
        reverse_dcf, _ = compute_institutional_reverse_dcf(
            ticker=ticker,
            market=market,
            current_price=atomic_snap.analysis_price,
            shares_outstanding=float(shares),
            base_revenue=float(rev),
            base_ebit_margin_pct=float(ebit_margin),
            cash_and_equivalents=float(cash),
            total_debt=float(debt),
            beta=float(beta),
            macro_registry=macro_reg,
        )

    # 4. Replay Quant Metrics from OHLCV and market data
    volatility_pct: Optional[float] = None
    adtv_val: Optional[float] = None
    mdd_pct: Optional[float] = None
    price_pctle: Optional[float] = None
    price_z: Optional[float] = None

    if not df_1y.empty and "Close" in df_1y.columns and len(df_1y["Close"].dropna()) >= 20:
        c_1y = df_1y["Close"].dropna()
        ret = c_1y.pct_change().dropna()
        if len(ret) >= 20:
            volatility_pct = round(float(ret.std() * math.sqrt(252) * 100.0), 2)

        if "Volume" in df_1y.columns:
            v_1y = df_1y["Volume"].dropna()
            val_series = c_1y * v_1y
            adtv_val = round(float(val_series.tail(20).mean()), 2)

        # 3Y / 5Y series if available
        ohlcv_5y = loaded_payloads.get("price_context_5y")
        if ohlcv_5y:
            df_5y = pd.DataFrame(ohlcv_5y)
            if not df_5y.empty and "Close" in df_5y.columns:
                c_5y = df_5y["Close"].dropna()
                if len(c_5y) >= 20:
                    curr = atomic_snap.analysis_price
                    price_pctle = round(float((c_5y < curr).mean() * 100.0), 2)
                    mean_5y = float(c_5y.mean())
                    std_5y = float(c_5y.std())
                    price_z = round(float((curr - mean_5y) / std_5y), 2) if std_5y > 0 else 0.0

        # MDD 3Y
        ohlcv_3y = loaded_payloads.get("mdd_3y")
        if ohlcv_3y:
            df_3y = pd.DataFrame(ohlcv_3y)
            if not df_3y.empty and "Close" in df_3y.columns:
                c_3y = df_3y["Close"].dropna()
                peak = c_3y.cummax()
                dd = (c_3y - peak) / peak
                mdd_pct = round(float(dd.min() * 100.0), 2)

    # 5. Analyst Upside & Peer metrics
    analyst_payload = loaded_payloads.get("analyst_targets") or {}
    analyst_mean = analyst_payload.get("targetMean") or analyst_payload.get("mean")
    analyst_upside_pct: Optional[float] = None
    if analyst_mean and atomic_snap.analysis_price > 0:
        analyst_upside_pct = round(((float(analyst_mean) - atomic_snap.analysis_price) / atomic_snap.analysis_price) * 100.0, 2)

    # 6. Compare with derived_features
    replayed_features = {
        "analysis_price": atomic_snap.analysis_price,
        "market_cap": atomic_snap.market_cap,
        "price_sync_status": atomic_snap.price_sync_status,
        "sma_50": tactical_setup.sma_50,
        "sma_200": tactical_setup.sma_200,
        "atr_14": tactical_setup.atr_14,
        "key_support": tactical_setup.key_support_level,
        "key_resistance": tactical_setup.key_resistance_level,
        "pullback_rr": tactical_setup.tactical_risk_reward_ratio,
        "breakout_planned_rr": tactical_setup.breakout_planned_rr,
        "breakout_current_rr": tactical_setup.breakout_current_rr,
        "breakout_entry_status": tactical_setup.breakout_entry_status,
        "breakout_entry_eligible": tactical_setup.breakout_entry_eligible,
        "reverse_dcf_target_12m": reverse_dcf.target_price_12m if reverse_dcf else None,
        "reverse_dcf_upside_pct": reverse_dcf.upside_12m_pct if reverse_dcf else None,
        "reverse_dcf_is_actionable": reverse_dcf.is_actionable if reverse_dcf else None,
        "volatility_pct": volatility_pct,
        "adtv_local_currency": adtv_val,
        "mdd_pct": mdd_pct,
        "price_percentile_5y": price_pctle,
        "price_zscore_5y": price_z,
        "analyst_upside_pct": analyst_upside_pct,
    }

    # Verify reconciliation with tolerance
    diffs: Dict[str, Any] = {}
    is_reconciled = True
    recorded_derived = snapshot.derived_features or {}
    for k, replayed_v in replayed_features.items():
        if k in recorded_derived:
            rec_v = recorded_derived[k]
            if replayed_v is None and rec_v is None:
                continue
            if isinstance(replayed_v, (int, float)) and isinstance(rec_v, (int, float)):
                if not math.isclose(replayed_v, rec_v, rel_tol=1e-3, abs_tol=1e-2):
                    diffs[k] = {"replayed": replayed_v, "recorded": rec_v}
                    is_reconciled = False
            elif replayed_v != rec_v:
                diffs[k] = {"replayed": replayed_v, "recorded": rec_v}
                is_reconciled = False

    return replayed_features, is_reconciled, diffs
