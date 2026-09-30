from datetime import datetime

from schemas.macro_schemas import (
    EconomicState,
    MacroStrategyDirection,
    MarketObservable,
    RegimeEvidenceComponent,
)
from tools.macro.dashboard import (
    build_dashboard_indicators,
    load_indicator_series,
    persist_indicator_series,
)


def _direction() -> MacroStrategyDirection:
    return MacroStrategyDirection(
        evaluated_at=datetime.now().isoformat(),
        overall_regime=EconomicState.GOLDILOCKS,
        asset_allocation=[],
        focus_themes=[],
        conviction_level="medium",
        conviction_rationale="Macro data remains constructive.",
        quant_narrative_alignment="aligned",
        regime_evidence=[
            RegimeEvidenceComponent(
                dimension="Rates",
                signal="Stable",
                evidence="US 10Y Yield = 4.25%",
                observable_refs=["obs_us10y_20260710"],
            )
        ],
    )


def _observable(observed_at: str, value: str) -> MarketObservable:
    return MarketObservable(
        observable_id="obs_us10y_20260710",
        asset_bucket="fixed_income",
        region="United States",
        indicator="US 10Y Yield",

        value=value,
        unit="%",
        observed_at=observed_at,
        source_file="Global_Macro_Snapshot.md",
        provider="FRED",
    )


def test_dashboard_indicators_include_only_cited_observables(tmp_path):
    # Without >= 2 points in series_dir, chart_available must be False (fail-closed)
    indicators = build_dashboard_indicators(
        _direction(),
        {
            "obs_us10y_20260710": _observable("2026-07-10", "4.25%"),
            "obs_unused": _observable("2026-07-10", "4.20%"),
        },
        vault_path=tmp_path,
    )

    assert len(indicators) == 1
    assert indicators[0]["indicator_id"] == "obs_us10y_20260710"
    assert indicators[0]["series_key"] == "obs_us10y"
    assert indicators[0]["value"] == 4.25
    assert indicators[0]["chart_available"] is False
    assert indicators[0]["region"] == "United States"
    assert indicators[0]["source_type"] == "provider"
    assert indicators[0]["status"] == "verified"

    # When series file exists with >= 2 distinct dates, chart_available must be True
    persist_indicator_series(tmp_path, [
        {"series_key": "obs_us10y", "label": "US 10Y Yield", "unit": "%", "observed_at": "2026-07-09", "value": 4.2},
        {"series_key": "obs_us10y", "label": "US 10Y Yield", "unit": "%", "observed_at": "2026-07-10", "value": 4.25},
    ])
    indicators_with_chart = build_dashboard_indicators(
        _direction(),
        {
            "obs_us10y_20260710": _observable("2026-07-10", "4.25%"),
        },
        vault_path=tmp_path,
    )
    assert indicators_with_chart[0]["chart_available"] is True



def test_dashboard_indicators_includes_thailand_stance_refs():
    dir_with_stance = _direction()
    dir_with_stance.thailand_market_stance = {
        "status": "calculated",
        "observable_refs": ["obs_th_breadth_ad"],
    }
    registry = {
        "obs_us10y_20260710": _observable("2026-07-10", "4.25%"),
        "obs_th_breadth_ad": MarketObservable(
            observable_id="obs_th_breadth_ad",
            asset_bucket="equities",
            region="Thailand",
            indicator="SET Advance/Decline Ratio",
            value="1.25",
            unit="ratio",
            observed_at="2026-09-27",
            source_file="Terminal_V2_Thai_Market.md",
            provider="SET",
        ),
    }
    indicators = build_dashboard_indicators(dir_with_stance, registry)
    indicator_ids = [ind["indicator_id"] for ind in indicators]
    assert "obs_th_breadth_ad" in indicator_ids
    th_ind = next(ind for ind in indicators if ind["indicator_id"] == "obs_th_breadth_ad")
    assert th_ind["region"] == "Thailand"



def test_indicator_series_replaces_same_day_snapshot_and_filters_range(tmp_path):
    first = {
        "series_key": "obs_us10y",
        "label": "US 10Y Yield",
        "unit": "%",
        "observed_at": "2025-07-10",
        "value": 4.1,
    }
    second = {**first, "observed_at": "2026-07-09", "value": 4.2}
    latest = {**second, "observed_at": "2026-07-10", "value": 4.25}
    replacement = {**latest, "value": 4.3}

    persist_indicator_series(tmp_path, [first, second, latest, replacement])

    assert load_indicator_series(tmp_path, "obs_us10y", "1m") == [
        {"observed_at": "2026-07-09", "value": 4.2},
        {"observed_at": "2026-07-10", "value": 4.3},
    ]
