"""Derived Pair Trade Ratios with structured machine-readable metadata (Pillar 3)."""
from datetime import datetime
from typing import Any, Optional
from schemas.macro_schemas import MarketObservable


DEFAULT_PAIR_CONFIGS = [
    {"id": "obs_pair_qqq_spy", "long": "QQQ", "short": "SPY", "indicator": "US Tech vs US Market Ratio (QQQ/SPY)", "bucket": "equities"},
    {"id": "obs_pair_spy_vgk", "long": "SPY", "short": "VGK", "indicator": "US vs Europe Equities Ratio (SPY/VGK)", "bucket": "equities"},
    {"id": "obs_pair_gld_spy", "long": "GLD", "short": "SPY", "indicator": "Gold vs US Equities Ratio (GLD/SPY)", "bucket": "commodities"},
]


def _default_price_getter(symbol: str) -> Optional[float]:
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="5d")
        if not hist.empty and "Close" in hist.columns:
            return float(hist["Close"].iloc[-1])
        info = ticker.info if isinstance(ticker.info, dict) else {}
        for k in ["regularMarketPrice", "currentPrice", "previousClose"]:
            if k in info and info[k]:
                return float(info[k])
    except Exception:
        pass
    return None


def _get_historical_ratio_stats(long_sym: str, short_sym: str) -> Optional[dict[str, float]]:
    try:
        import yfinance as yf
        t1 = yf.Ticker(long_sym)
        t2 = yf.Ticker(short_sym)
        h1 = t1.history(period="1y")
        h2 = t2.history(period="1y")
        if h1.empty or h2.empty or "Close" not in h1.columns or "Close" not in h2.columns:
            return None
        s1 = h1["Close"].dropna()
        s2 = h2["Close"].dropna()
        df = s1.to_frame(name="long").join(s2.to_frame(name="short"), how="inner")
        if len(df) < 45:
            return None
        ratios = df["long"] / df["short"]
        mean_val = float(ratios.mean())
        std_val = float(ratios.std())
        curr_ratio = float(ratios.iloc[-1])
        z = (curr_ratio - mean_val) / std_val if std_val > 0 else 0.0
        return {"mean": mean_val, "std": std_val, "z_score": z}
    except Exception:
        return None


def build_derived_pair_observables(
    existing_observables: Optional[list[MarketObservable]] = None,
    price_getter: Optional[Any] = None,
    pair_configs: Optional[list[dict[str, str]]] = None,
    today_str: Optional[str] = None,
    use_mock_fallback: bool = False,
    historical_ratio_calculator: Optional[Any] = None
) -> list[MarketObservable]:
    """
    Build derived pair trade ratio observables from existing observables or price getter.
    Includes structured machine-readable metadata for divergence checking.
    """
    if price_getter is None and not use_mock_fallback:
        price_getter = _default_price_getter
    if today_str is None:
        today_str = datetime.now().strftime("%Y-%m-%d")
    if pair_configs is None:
        pair_configs = DEFAULT_PAIR_CONFIGS

    # Create a quick lookup for prices from existing observables if available
    price_map: dict[str, float] = {}
    if existing_observables:
        for obs in existing_observables:
            id_lower = getattr(obs, "observable_id", "").lower()
            val_str = getattr(obs, "value", "")
            try:
                val_num = float(str(val_str).replace(",", ""))
                if "qqq" in id_lower:
                    price_map["QQQ"] = val_num
                elif "spy" in id_lower or "gspc" in id_lower:
                    price_map["SPY"] = val_num
                elif "vgk" in id_lower:
                    price_map["VGK"] = val_num
                elif "gld" in id_lower or "gold" in id_lower:
                    price_map["GLD"] = val_num
            except (ValueError, TypeError):
                continue

    results: list[MarketObservable] = []
    for cfg in pair_configs:
        long_sym = cfg["long"]
        short_sym = cfg["short"]
        
        long_price = price_map.get(long_sym)
        short_price = price_map.get(short_sym)

        if price_getter is not None and (long_price is None or short_price is None):
            try:
                if long_price is None:
                    long_price = price_getter(long_sym)
                if short_price is None:
                    short_price = price_getter(short_sym)
            except Exception:
                pass

        obs_id = cfg["id"]
        indicator = cfg["indicator"]
        bucket = cfg["bucket"]

        if long_price is None or short_price is None or short_price == 0:
            results.append(MarketObservable(
                observable_id=obs_id,
                asset_bucket=bucket,  # type: ignore
                region="Global",
                indicator=indicator,
                value="N/A",
                unit="ratio",
                observed_at=today_str,
                source_file="derived_ratios.py",
                is_valid=False,
                stale_reason=f"Missing price for leg: {'long ('+long_sym+')' if long_price is None else ''} {'short ('+short_sym+')' if short_price is None else ''}".strip(),
                metadata={
                    "long_ticker": long_sym,
                    "short_ticker": short_sym,
                    "ratio": None,
                    "is_valid": False
                }
            ))
            continue

        ratio_val = long_price / short_price
        stats = historical_ratio_calculator(long_sym, short_sym) if historical_ratio_calculator is not None else (_get_historical_ratio_stats(long_sym, short_sym) if not use_mock_fallback else None)
        if stats:
            mean_est = stats["mean"]
            std_est = stats["std"]
            z_score = stats["z_score"]
        elif use_mock_fallback:
            mean_est = ratio_val * 0.98
            std_est = ratio_val * 0.05
            z_score = (ratio_val - mean_est) / std_est if std_est > 0 else 0.0
        else:
            results.append(MarketObservable(
                observable_id=obs_id,
                asset_bucket=bucket,  # type: ignore
                region="Global",
                indicator=indicator,
                value="N/A",
                unit="ratio",
                observed_at=today_str,
                source_file="derived_ratios.py",
                is_valid=False,
                stale_reason="Missing real market historical price series to calculate Z-score",
                metadata={
                    "long_ticker": long_sym,
                    "short_ticker": short_sym,
                    "ratio": round(float(ratio_val), 4),
                    "is_valid": False
                }
            ))
            continue

        metadata = {
            "long_ticker": long_sym,
            "short_ticker": short_sym,
            "long_price": round(float(long_price), 4),
            "short_price": round(float(short_price), 4),
            "ratio": round(float(ratio_val), 4),
            "mean_ratio_1y": round(float(mean_est), 4),
            "std_ratio_1y": round(float(std_est), 4),
            "z_score_1y": round(float(z_score), 2),
            "pair_type": "equity_relative_value" if bucket == "equities" else "cross_asset",
            "is_valid": True
        }

        results.append(MarketObservable(
            observable_id=obs_id,
            asset_bucket=bucket,  # type: ignore
            region="Global",
            indicator=indicator,
            value=f"{ratio_val:.4f}",
            unit="ratio",
            observed_at=today_str,
            source_file="derived_ratios.py",
            is_valid=True,
            metadata=metadata
        ))

    return results
