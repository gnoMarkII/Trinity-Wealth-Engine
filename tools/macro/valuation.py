import os
import re
from datetime import datetime
from typing import Optional, Any, Callable
import yfinance as yf
from fredapi import Fred

from core.logger import get_logger
from schemas.macro_schemas import MarketObservable
from .ticker_config import VALUATION_RICH_ERP_THRESHOLD, CREDIT_SPREAD_DANGER_THRESHOLD, CREDIT_SPREAD_WIDENING_3M_BPS

log = get_logger(__name__)


def _default_ticker_info_getter(symbol: str) -> dict[str, Any]:
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        return info if isinstance(info, dict) else {}
    except Exception as e:
        log.warning(f"Failed to fetch info for symbol {symbol}: {e}")
        return {}


def _parse_val_to_float(val_str: Any) -> Optional[float]:
    if isinstance(val_str, (int, float)):
        return float(val_str)
    if not isinstance(val_str, str) or not val_str.strip() or val_str.strip() == "-" or val_str.strip().upper() == "N/A":
        return None
    match = re.search(r"[-+]?\d*\.\d+|\d+", str(val_str))
    if match:
        try:
            return float(match.group(0))
        except ValueError:
            return None
    return None


def _find_dgs10_in_observables(observables: list[MarketObservable]) -> Optional[float]:
    for obs in observables:
        if not obs.is_valid:
            continue
        ind_low = obs.indicator.lower()
        id_low = obs.observable_id.lower()
        # Look for 10Y Treasury Yield
        if "dgs10" in id_low or "t10y" in id_low or "tnx" in id_low or "10y" in id_low or "10-year treasury" in ind_low or "10-year" in ind_low:
            if "spread" in ind_low or "breakeven" in ind_low or "t10y2y" in id_low or "t10yie" in id_low or "dfii10" in id_low:
                continue
            val = _parse_val_to_float(obs.value)
            if val is not None and val > 0:
                # If from ^TNX, Yahoo usually gives yield directly, e.g. 4.35
                return val
    return None


def build_valuation_observables(
    existing_observables: Optional[list[MarketObservable]] = None,
    ticker_info_getter: Optional[Callable[[str], dict[str, Any]]] = None,
    dgs10_value: Optional[float] = None,
) -> list[MarketObservable]:
    """
    Build Equity Risk Premium (ERP), Forward Earnings Yield, and isolation for Trailing P/E.
    Strict lookup order: ^GSPC -> SPY -> VOO.
    Only forwardPE is used for ERP and Forward Earnings Yield.
    If forwardPE is missing, sets ERP.is_valid = False and fallback to trailingPE as a separate observable.
    """
    if existing_observables is None:
        existing_observables = []
    if ticker_info_getter is None:
        ticker_info_getter = _default_ticker_info_getter

    today_str = datetime.now().strftime("%Y-%m-%d")
    results: list[MarketObservable] = []

    # 1. Resolve 10Y Treasury Yield (DGS10)
    resolved_dgs10 = dgs10_value
    if resolved_dgs10 is None:
        resolved_dgs10 = _find_dgs10_in_observables(existing_observables)
    if resolved_dgs10 is None:
        # Try fetching from FRED or fallback default
        try:
            api_key = os.getenv("FRED_API_KEY")
            if api_key:
                fred = Fred(api_key=api_key)
                s = fred.get_series("DGS10").dropna()
                if not s.empty:
                    resolved_dgs10 = float(s.iloc[-1])
        except Exception as e:
            log.debug(f"Could not fetch DGS10 from FRED: {e}")

    if resolved_dgs10 is None:
        resolved_dgs10 = 4.25  # Reasonable fallback if offline
        log.info(f"Using default DGS10 fallback: {resolved_dgs10}%")

    # 2. Strict lookup order for Forward P/E
    lookup_order = ["^GSPC", "SPY", "VOO"]
    forward_pe: Optional[float] = None
    symbol_used: str = ""
    trailing_pe: Optional[float] = None
    trailing_symbol_used: str = ""

    for sym in lookup_order:
        info = ticker_info_getter(sym)
        val_fpe = info.get("forwardPE")
        if val_fpe is None:
            val_fpe = info.get("forward_pe")
        val_fpe_num = _parse_val_to_float(val_fpe)
        if val_fpe_num is not None and val_fpe_num > 0:
            forward_pe = val_fpe_num
            symbol_used = sym
            break
        # Track trailing P/E as potential fallback isolation
        if trailing_pe is None:
            val_tpe = info.get("trailingPE")
            if val_tpe is None:
                val_tpe = info.get("trailing_pe")
            val_tpe_num = _parse_val_to_float(val_tpe)
            if val_tpe_num is not None and val_tpe_num > 0:
                trailing_pe = val_tpe_num
                trailing_symbol_used = sym

    # 3. Create Observables based on rules
    if forward_pe is not None and forward_pe > 0:
        ey_decimal = 1.0 / forward_pe
        ey_pct = ey_decimal * 100.0
        dgs10_decimal = resolved_dgs10 / 100.0
        erp_decimal = ey_decimal - dgs10_decimal
        erp_pct = erp_decimal * 100.0

        # Forward Earnings Yield Observable
        obs_ey = MarketObservable(
            observable_id="obs_ey_gspc",
            asset_bucket="equities",
            region="US",
            indicator="S&P 500 Forward Earnings Yield",
            value=f"{ey_pct:.2f}",
            unit="%",
            observed_at=today_str,
            source_file="valuation.py",
            provider=f"Yahoo Finance ({symbol_used})",
            confidence="high",
            is_valid=True,
            observable_type="valuation",
            calculation_method="1 / ForwardPE",
            metadata={
                "forward_pe": round(forward_pe, 2),
                "earnings_yield_decimal": round(ey_decimal, 6),
                "earnings_yield_pct": round(ey_pct, 2),
                "symbol_used": symbol_used,
            }
        )
        results.append(obs_ey)

        # Equity Risk Premium (ERP) Observable
        obs_erp = MarketObservable(
            observable_id="obs_erp_gspc",
            asset_bucket="equities",
            region="US",
            indicator="S&P 500 Equity Risk Premium (ERP)",
            value=f"{erp_pct:.2f}",
            unit="%",
            observed_at=today_str,
            source_file="valuation.py",
            provider=f"Yahoo Finance ({symbol_used}) & FRED (DGS10)",
            confidence="high",
            is_valid=True,
            observable_type="valuation",
            calculation_method="(1 / ForwardPE) - 10Y Treasury Yield",
            input_observable_ids=["obs_ey_gspc", "obs_dgs10"],
            metadata={
                "forward_pe": round(forward_pe, 2),
                "dgs10": round(resolved_dgs10, 2),
                "erp_decimal": round(erp_decimal, 6),
                "erp_pct": round(erp_pct, 2),
                "threshold_decimal": VALUATION_RICH_ERP_THRESHOLD,
                "is_rich": erp_decimal < VALUATION_RICH_ERP_THRESHOLD,
                "symbol_used": symbol_used,
            }
        )
        results.append(obs_erp)
    else:
        # Safe Fallback State when Forward P/E is missing
        log.warning("Forward P/E missing across ^GSPC, SPY, VOO. Setting ERP.is_valid = False.")
        obs_erp_invalid = MarketObservable(
            observable_id="obs_erp_gspc",
            asset_bucket="equities",
            region="US",
            indicator="S&P 500 Equity Risk Premium (ERP)",
            value="N/A",
            unit="%",
            observed_at=today_str,
            source_file="valuation.py",
            provider="Yahoo Finance",
            confidence="low",
            is_valid=False,
            stale_reason="Missing forwardPE from all benchmark symbols (^GSPC, SPY, VOO)",
            observable_type="valuation",
            calculation_method="(1 / ForwardPE) - 10Y Treasury Yield",
            metadata={"threshold_decimal": VALUATION_RICH_ERP_THRESHOLD}
        )
        results.append(obs_erp_invalid)

        # If Trailing P/E exists, isolate it as a separate observable without using it for ERP
        if trailing_pe is not None and trailing_pe > 0:
            obs_tpe = MarketObservable(
                observable_id="obs_trailing_pe_gspc",
                asset_bucket="equities",
                region="US",
                indicator="S&P 500 Trailing P/E Ratio",
                value=f"{trailing_pe:.2f}",
                unit="ratio",
                observed_at=today_str,
                source_file="valuation.py",
                provider=f"Yahoo Finance ({trailing_symbol_used})",
                confidence="medium",
                is_valid=True,
                observable_type="valuation",
                calculation_method="Trailing P/E from Yahoo Finance",
                metadata={
                    "trailing_pe": round(trailing_pe, 2),
                    "symbol_used": trailing_symbol_used,
                    "note": "Isolated trailing P/E; NOT used for ERP or forward valuation calculations."
                }
            )
            results.append(obs_tpe)

    return results


def build_credit_spread_observable(
    existing_observables: Optional[list[MarketObservable]] = None,
    fred_getter: Optional[Callable[[str], Optional[float]]] = None,
) -> Optional[MarketObservable]:
    """
    Build or enhance High Yield Credit Spread observable (BAMLH0A0HYM2).
    Tool produces data with machine-readable metadata; validator checks warning conditions.
    """
    if existing_observables is None:
        existing_observables = []
    today_str = datetime.now().strftime("%Y-%m-%d")

    # Check if BAMLH0A0HYM2 is already in existing observables
    for obs in existing_observables:
        if obs.is_valid and ("bamlh0a0hym2" in obs.observable_id.lower() or "high yield" in obs.indicator.lower()):
            val_num = _parse_val_to_float(obs.value)
            if val_num is not None:
                obs.observable_type = "valuation"
                obs.metadata.update({
                    "hy_spread_pct": val_num,
                    "danger_threshold": CREDIT_SPREAD_DANGER_THRESHOLD,
                    "widening_3m_bps_threshold": CREDIT_SPREAD_WIDENING_3M_BPS,
                })
                return obs

    # If not present, try fetching or using getter
    val: Optional[float] = None
    if fred_getter is not None:
        val = fred_getter("BAMLH0A0HYM2")
    else:
        try:
            api_key = os.getenv("FRED_API_KEY")
            if api_key:
                fred = Fred(api_key=api_key)
                s = fred.get_series("BAMLH0A0HYM2").dropna()
                if not s.empty:
                    val = float(s.iloc[-1])
        except Exception as e:
            log.debug(f"Could not fetch BAMLH0A0HYM2 from FRED: {e}")

    if val is not None and val > 0:
        obs_hy = MarketObservable(
            observable_id="obs_hy_spread",
            asset_bucket="fixed_income",
            region="US",
            indicator="High Yield Bond Spread (ICE BofA)",
            value=f"{val:.2f}",
            unit="% pts",
            observed_at=today_str,
            source_file="valuation.py",
            provider="FRED (BAMLH0A0HYM2)",
            confidence="high",
            is_valid=True,
            observable_type="valuation",
            metadata={
                "hy_spread_pct": round(val, 2),
                "danger_threshold": CREDIT_SPREAD_DANGER_THRESHOLD,
                "widening_3m_bps_threshold": CREDIT_SPREAD_WIDENING_3M_BPS,
            }
        )
        return obs_hy

    return None
