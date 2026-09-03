"""Unit tests for Macro Publication Calendar & Cadence Freshness (SIFMA 5-day publication calendar & ERP cadence)."""
import pytest
from datetime import datetime, timezone, timedelta
from schemas.macro_schemas import MarketObservable
from tools.market.dcf_valuation import validate_macro_observables_freshness


def _create_obs(obs_id: str, indicator: str, value: str, observed_at: str) -> MarketObservable:
    return MarketObservable(
        observable_id=obs_id,
        asset_bucket="fixed_income",
        region="USA",
        indicator=indicator,
        value=value,
        unit="%",
        observed_at=observed_at,
        source_file="test_macro.md",
        is_valid=True,
    )


def test_macro_freshness_within_calendar_window():
    now_dt = datetime.now(timezone.utc)
    recent_date_str = (now_dt - timedelta(days=2)).strftime("%Y-%m-%d")

    macro_reg = {
        "obs_dgs10": _create_obs("obs_dgs10", "DGS10", "4.25", recent_date_str),
        "obs_erp_gspc": _create_obs("obs_erp_gspc", "ERP", "4.50", recent_date_str),
    }

    is_fresh, flags = validate_macro_observables_freshness(macro_reg, as_of_dt=now_dt)
    assert is_fresh is True
    assert len(flags) == 0


def test_macro_freshness_stale_detection():
    now_dt = datetime.now(timezone.utc)
    stale_dgs10_date = (now_dt - timedelta(days=12)).strftime("%Y-%m-%d")
    stale_erp_date = (now_dt - timedelta(days=45)).strftime("%Y-%m-%d")

    macro_reg = {
        "obs_dgs10": _create_obs("obs_dgs10", "DGS10", "4.25", stale_dgs10_date),
        "obs_erp_gspc": _create_obs("obs_erp_gspc", "ERP", "4.50", stale_erp_date),
    }

    is_fresh, flags = validate_macro_observables_freshness(macro_reg, as_of_dt=now_dt)
    assert is_fresh is False
    assert "macro_stale_publication:us_10y" in flags
    assert "macro_stale_publication:erp" in flags
