"""Risk Correlation Analytics (Pillar 4) with quality guardrails (>= 45 overlapping trading days)."""
from datetime import datetime
from typing import Any, Optional
from schemas.macro_schemas import MarketObservable


DEFAULT_CORRELATION_PAIRS = [
    {"id": "obs_corr_spy_tlt_60d", "asset_1": "SPY", "asset_2": "TLT", "indicator": "60-Day Rolling Correlation (SPY vs TLT)", "bucket": "equities", "breakdown_threshold": 0.30},
    {"id": "obs_corr_spy_gld_60d", "asset_1": "SPY", "asset_2": "GLD", "indicator": "60-Day Rolling Correlation (SPY vs GLD)", "bucket": "equities", "breakdown_threshold": 0.50},
    {"id": "obs_corr_spy_uup_60d", "asset_1": "SPY", "asset_2": "UUP", "indicator": "60-Day Rolling Correlation (SPY vs UUP)", "bucket": "equities", "breakdown_threshold": 0.40},
]


def _default_correlation_calculator(a1: str, a2: str, window: int = 60) -> Optional[dict[str, Any]]:
    try:
        import yfinance as yf
        t1 = yf.Ticker(a1)
        t2 = yf.Ticker(a2)
        h1 = t1.history(period="1y")
        h2 = t2.history(period="1y")
        if h1.empty or h2.empty or "Close" not in h1.columns or "Close" not in h2.columns:
            return None
        s1 = h1["Close"].pct_change().dropna()
        s2 = h2["Close"].pct_change().dropna()
        combined = s1.to_frame(name="s1").join(s2.to_frame(name="s2"), how="inner").tail(window)
        if len(combined) < 45:
            return {"correlation": None, "overlapping_days": len(combined)}
        corr = combined["s1"].corr(combined["s2"])
        return {"correlation": float(corr) if corr is not None else None, "overlapping_days": len(combined)}
    except Exception:
        return None


def build_risk_correlation_observables(
    correlation_calculator: Optional[Any] = None,
    pair_configs: Optional[list[dict[str, Any]]] = None,
    today_str: Optional[str] = None,
    use_mock_fallback: bool = False
) -> list[MarketObservable]:
    """
    Build rolling 60-day correlation observables with data quality guardrails.
    Requires >= 45 overlapping trading days to be marked as valid.
    """
    if correlation_calculator is None and not use_mock_fallback:
        correlation_calculator = _default_correlation_calculator
    if today_str is None:
        today_str = datetime.now().strftime("%Y-%m-%d")
    if pair_configs is None:
        pair_configs = DEFAULT_CORRELATION_PAIRS

    results: list[MarketObservable] = []
    for cfg in pair_configs:
        obs_id = cfg["id"]
        a1 = cfg["asset_1"]
        a2 = cfg["asset_2"]
        indicator = cfg["indicator"]
        bucket = cfg["bucket"]
        thresh = cfg.get("breakdown_threshold", 0.30)

        corr_val: Optional[float] = None
        overlapping_days = 0
        is_valid = False
        stale_reason = ""

        if correlation_calculator is not None:
            try:
                res = correlation_calculator(a1, a2, window=60)
                if isinstance(res, dict):
                    corr_val = res.get("correlation")
                    overlapping_days = res.get("overlapping_days", 0)
                elif isinstance(res, (int, float)):
                    corr_val = float(res)
                    overlapping_days = 60  # Default assume full window if raw float returned
            except Exception as e:
                stale_reason = f"Calculation error: {str(e)}"

        # Default mock fallback for robust offline testing/evaluation if no calculator provided
        if corr_val is None and not stale_reason and use_mock_fallback:
            if "tlt" in a2.lower():
                corr_val = -0.15
                overlapping_days = 58
            elif "gld" in a2.lower():
                corr_val = 0.05
                overlapping_days = 59
            elif "uup" in a2.lower():
                corr_val = -0.30
                overlapping_days = 55

        if corr_val is None or overlapping_days < 45:
            is_valid = False
            if not stale_reason:
                stale_reason = f"Insufficient overlapping trading days ({overlapping_days} < 45 required for 60-day window)"
            val_str = "N/A"
            metadata = {
                "asset_1": a1,
                "asset_2": a2,
                "window_days": 60,
                "overlapping_days": overlapping_days,
                "correlation_value": None,
                "is_valid": False,
                "is_breakdown": False
            }
        else:
            is_valid = True
            val_str = f"{corr_val:.2f}"
            is_breakdown = bool(corr_val > thresh)
            metadata = {
                "asset_1": a1,
                "asset_2": a2,
                "window_days": 60,
                "overlapping_days": overlapping_days,
                "correlation_value": round(float(corr_val), 4),
                "breakdown_threshold": thresh,
                "is_breakdown": is_breakdown,
                "is_valid": True
            }

        results.append(MarketObservable(
            observable_id=obs_id,
            asset_bucket=bucket,  # type: ignore
            region="Global",
            indicator=indicator,
            value=val_str,
            unit="correlation",
            observed_at=today_str,
            source_file="risk_analytics.py",
            is_valid=is_valid,
            stale_reason=stale_reason if not is_valid else "",
            metadata=metadata
        ))

    return results
