"""Unit tests for Macro NotebookLM Bundle Builder, Formatters, and Corpus Snapshot."""
import json
import tempfile
from pathlib import Path

import pytest

from application.macro.notebooklm_export_ports import MacroCorpusSnapshot
from tools.macro.notebooklm_bundle_formatter import (
    format_research_guide,
    format_current_macro_report,
    format_historical_reports,
    format_us_market_observables,
    format_thailand_market_observables,
    format_global_and_catalog_notes,
    format_sector_rotation,
    format_news_and_references,
    format_structured_appendix,
)
from tools.macro.notebooklm_bundle_builder import MacroExportBundleBuilder


@pytest.fixture
def sample_snapshot() -> MacroCorpusSnapshot:
    return MacroCorpusSnapshot(
        snapshot_at="2026-10-05T12:00:00Z",
        strategy_report_id="macro_strategy_2026-10-05",
        latest_report={
            "report_id": "macro_strategy_2026-10-05",
            "title": "Macro Strategy Report 2026-10-05",
            "content_md": "# Daily Macro Strategy Report\n\nRegime: Risk-On\nInflation: 2.3%",
            "overall_regime": "Risk-On",
            "time_horizon": "3-6 Months",
            "conviction_level": "High",
            "conviction_rationale": "Liquidity expanding across major central banks.",
            "evaluated_at": "2026-10-05T09:30:00Z",
            "asset_allocation": [
                {
                    "asset_class": "US Equities",
                    "region": "US",
                    "stance": "Overweight",
                    "confidence": "high",
                    "rationale": "Earnings growth resilient",
                }
            ],
            "pair_trades": [],
            "risk_scenarios": [],
            "focus_themes": ["Tech Resilience", "Steepening Curve"],
            "warnings": [],
        },
        historical_reports=[
            {
                "report_id": "macro_strategy_2026-10-04",
                "title": "Macro Strategy Report 2026-10-04",
                "evaluated_at": "2026-10-04T09:30:00Z",
                "overall_regime": "Neutral",
                "conviction_rationale": "Gold resilience continuing, dollar softening.",
            }
        ],
        catalog_notes=[
            {
                "id": "note_001",
                "title": "Fed Speech Analysis",
                "timestamp": "2026-10-05T08:00:00Z",
                "speaker": "Jerome Powell",
                "summary": "Fed remains data-dependent with balanced risks.",
                "tags": ["fed", "rates", "us"],
                "content": "Speech on monetary policy normalization.",
            }
        ],
        indicator_series=[
            {"indicator_id": "us_cpi", "as_of": "2026-09-01", "value": 2.4}
        ],
        market_observables={
            "yield_curve": {
                "observation_date": "2026-10-05",
                "yields": [
                    {"maturity": "2Y", "yield_percent": 3.95},
                    {"maturity": "10Y", "yield_percent": 4.12},
                ],
                "spread_10y_2y_bps": 17.0,
            },
            "financial_stress": {
                "latest_value": -0.85,
                "latest_date": "2026-10-04",
                "regime_label": "Normal/Low Stress",
            },
            "flow_set": {
                "market": "SET",
                "as_of": "2026-10-05",
                "investors": [
                    {"investor_type": "Foreign", "net_value": 1500000000},
                ],
            },
        },
        thailand_hard_data={
            "headline_cpi": 0.88,
            "core_cpi": 0.75,
            "gdp_growth_yoy": 2.8,
            "public_debt_to_gdp_pct": 63.85,
            "as_of_period": "2026-Q2",
        },
        sector_rotation={
            "leaders": ["ICT", "HELTH"],
            "laggards": ["ENERG", "COMM"],
            "summary": "Defensives and tech leading SET momentum.",
        },
        news_funnel={
            "pending": [
                {
                    "title": "Fed rate cut expectations firm up",
                    "publisher": "Bloomberg",
                    "cluster_label": "Monetary Policy",
                    "url": "https://example.com/fed",
                }
            ],
            "filtered": [],
        },
        metadata={
            "counts": {
                "historical_reports": 1,
                "catalog_notes": 1,
                "market_observables_cached": 3,
                "market_observables_total": 13,
            },
            "warnings": [],
        },
    )


def test_bundle_formatters(sample_snapshot):
    guide = format_research_guide(sample_snapshot)
    assert "# Macro Research Companion Guide & Evidence Catalog" in guide
    assert "System Role & Scope Invariant" in guide
    assert "Recommended Research Questions" in guide

    curr_rep = format_current_macro_report(sample_snapshot)
    assert "# Current Macro Strategy Report" in curr_rep
    assert "Risk-On" in curr_rep

    rep_hist = format_historical_reports(sample_snapshot)
    assert "Historical Macro Strategy Reports" in rep_hist
    assert "macro_strategy_2026-10-04" in rep_hist

    obs_us = format_us_market_observables(sample_snapshot)
    assert "US Macro Market Observables" in obs_us

    obs_th = format_thailand_market_observables(sample_snapshot)
    assert "Thailand Macro & Market Telemetry" in obs_th
    assert "63.85" in obs_th

    obs_global = format_global_and_catalog_notes(sample_snapshot)
    assert "Global & Regional Macro Telemetry" in obs_global

    sec_rot = format_sector_rotation(sample_snapshot)
    assert "Sector Rotation & Industry Relative Strength" in sec_rot

    news_ref = format_news_and_references(sample_snapshot)
    assert "Macro News Funnel & Content References" in news_ref
    assert "Fed rate cut expectations firm up" in news_ref

    appendix = format_structured_appendix(sample_snapshot)
    assert "Structured Appendix & Lineage Mappings" in appendix
    assert "Lineage Mappings" in appendix


def test_bundle_builder_generates_sources_and_deterministic_hash(sample_snapshot):
    with tempfile.TemporaryDirectory() as tmpdir:
        builder = MacroExportBundleBuilder(export_root=Path(tmpdir))
        bundle_dir, content_hash, inventory = builder.build_bundle("export_test_001", sample_snapshot)

        assert bundle_dir.exists()
        assert (bundle_dir / "sources").exists()
        assert len(inventory["sources"]) == 9
        assert content_hash != ""

        # Verify all source files exist
        for s in inventory["sources"]:
            p = bundle_dir / s["relative_path"]
            assert p.exists()
            assert p.is_file()
            assert p.stat().st_size > 0

        # Verify manifest files
        corpus_path = bundle_dir / "corpus.json"
        assert corpus_path.exists()
        with open(corpus_path, "r", encoding="utf-8") as f:
            corpus_data = json.load(f)
            assert corpus_data["strategy_report_id"] == "macro_strategy_2026-10-05"

        inventory_path = bundle_dir / "inventory.json"
        assert inventory_path.exists()
        with open(inventory_path, "r", encoding="utf-8") as f:
            inv_data = json.load(f)
            assert inv_data["total_sources"] == 9
            assert inv_data["content_hash"] == content_hash

        # Idempotency / deterministic content_hash
        bundle_dir2, content_hash2, _ = builder.build_bundle("export_test_002", sample_snapshot)
        assert content_hash == content_hash2


def test_bundle_hash_changes_on_snapshot_difference(sample_snapshot):
    with tempfile.TemporaryDirectory() as tmpdir:
        builder = MacroExportBundleBuilder(export_root=Path(tmpdir))
        _, content_hash1, _ = builder.build_bundle("exp1", sample_snapshot)

        # Mutate sample_snapshot
        mutated_snapshot = MacroCorpusSnapshot(
            snapshot_at="2026-10-05T12:00:00Z",
            strategy_report_id="macro_strategy_2026-10-05_diff",
            latest_report={
                "report_id": "macro_strategy_2026-10-05_diff",
                "title": "Macro Strategy Report 2026-10-05 Modified",
                "content_md": "# Daily Macro Strategy Report\n\nRegime: Defensive\nInflation: 3.5%",
                "overall_regime": "Defensive",
                "evaluated_at": "2026-10-05T09:30:00Z",
            },
            historical_reports=[],
            catalog_notes=[],
            indicator_series=[],
            market_observables={},
            thailand_hard_data=sample_snapshot.thailand_hard_data,
            sector_rotation=sample_snapshot.sector_rotation,
            news_funnel=sample_snapshot.news_funnel,
            metadata={"counts": {}, "warnings": []},
        )
        _, content_hash2, _ = builder.build_bundle("exp2", mutated_snapshot)
        assert content_hash1 != content_hash2


def test_canonical_report_formatting_and_dict_observables():
    snapshot = MacroCorpusSnapshot(
        snapshot_at="2026-10-05T12:00:00Z",
        strategy_report_id="macro_strategy_canonical_001",
        latest_report={
            "strategy_report_id": "macro_strategy_canonical_001",
            "evaluated_at": "2026-10-05T09:00:00Z",
            "overall_regime": "Reflation",
            "time_horizon": "6-12 Months",
            "conviction_level": "high",
            "quant_narrative_alignment": "aligned",
            "conviction_rationale": "High conviction based on dual easing.",
            "asset_allocation": [
                {
                    "asset_class": "Commodities",
                    "stance": "Overweight",
                    "delta": "+5%",
                    "confidence": "high",
                    "rationale": "Broad commodity cycle bottoming.",
                    "supporting_data": ["BCOM trend positive"],
                },
                {
                    "asset_class": "Cash",
                    "stance": "Underweight",
                    "delta": "-5%",
                    "confidence": "medium",
                    "rationale": "Negative real rate environment.",
                },
            ],
            "focus_themes": ["Commodity Supercycle", "Emerging Market Carry"],
            "regime_probabilities": {"Reflation": "65%", "Goldilocks": "25%", "Stagflation": "10%"},
            "regime_evidence": [
                {
                    "dimension": "Monetary Policy",
                    "signal": "Dovish",
                    "evidence": "Global rate cuts accelerating",
                    "conflict": None,
                    "confidence": "high",
                }
            ],
            "observable_registry": {
                "bcom_index": {
                    "provider": "Bloomberg",
                    "indicator": "Commodity Spot",
                    "value": 105.4,
                    "unit": "Index",
                    "observed_at": "2026-10-05",
                    "status": "ok",
                },
                "fed_target_upper": {
                    "provider": "FRED",
                    "indicator": "Federal Funds Target",
                    "value": 4.5,
                    "unit": "%",
                    "observed_at": "2026-10-04",
                    "status": "ok",
                },
            },
        },
        historical_reports=[],
        catalog_notes=[],
        indicator_series=[],
        market_observables={},
        thailand_hard_data=None,
        sector_rotation=None,
        news_funnel={"pending": [], "filtered": []},
        metadata={"counts": {}, "warnings": []},
    )

    rendered = format_current_macro_report(snapshot)
    assert "Commodities" in rendered
    assert "Overweight" in rendered
    assert "Commodity Supercycle" in rendered
    assert "Emerging Market Carry" in rendered
    assert "Reflation" in rendered
    assert "bcom_index" in rendered
    assert "fed_target_upper" in rendered
    assert "Bloomberg" in rendered


def test_no_truncation_of_history_or_filtered_news():
    history = [
        {
            "strategy_report_id": f"hist_rep_{i}",
            "evaluated_at": f"2026-09-{i:02d}T00:00:00Z",
            "overall_regime": "Reflation",
            "conviction_level": "medium",
            "summary": f"Historical report summary for run {i}",
        }
        for i in range(1, 16)
    ]
    filtered_news = [
        {
            "id": f"news_{i}",
            "title": f"Filtered macro event {i}",
            "filter_reason": "Low institutional impact",
            "published_at": "2026-10-01",
        }
        for i in range(1, 26)
    ]

    snapshot = MacroCorpusSnapshot(
        snapshot_at="2026-10-05T12:00:00Z",
        strategy_report_id="test_no_truncation",
        latest_report=None,
        historical_reports=history,
        catalog_notes=[],
        indicator_series=[],
        market_observables={},
        thailand_hard_data=None,
        sector_rotation=None,
        news_funnel={"pending": [], "filtered": filtered_news},
        metadata={"counts": {}, "warnings": []},
    )

    hist_rendered = format_historical_reports(snapshot)
    for i in range(1, 16):
        assert f"hist_rep_{i}" in hist_rendered

    news_rendered = format_news_and_references(snapshot)
    for i in range(1, 26):
        assert f"news_{i}" in news_rendered


def test_market_observables_provider_cache_keys(tmp_path):
    from tools.market.terminal_v2.application.cache import ThreadSafeTTLCache
    from tools.macro.adapters.macro_corpus_adapter import MacroCorpusAdapter

    cache = ThreadSafeTTLCache(default_ttl_seconds=3600)
    cache.set("ofr:fsi:latest", {"fsi_value": -0.45, "date": "2026-10-04"})
    cache.set("bis:policy_rates:latest", {"US": 4.5, "TH": 2.25})
    cache.set("cftc:cot:disagg:088691", {"commercial_net": -210000, "report_date": "2026-09-30"})
    cache.set("cboe:commodity_vol:OVX", {"symbol": "OVX", "value": 31.2})
    cache.set("treasury:debt:30", {"total_public_debt_mil": 36100000})
    cache.set("settrade:flow:SET", {"foreign_net_mb": 1420.5})
    cache.set("goldtraders:retail:quote", {"bar_sell": 43500})
    cache.set("settrade:stats:SET", {"pe_ratio": 16.2})
    cache.set("crypto:benchmark:btc", {"price_usd": 68500})

    adapter = MacroCorpusAdapter(vault_path=tmp_path, shared_cache=cache)
    obs = adapter._capture_market_observables()

    assert obs["ofr_financial_stress"]["status"] == "cached"
    assert obs["ofr_financial_stress"]["data"]["fsi_value"] == -0.45
    assert obs["global_policy_rates"]["status"] == "cached"
    assert obs["gold_cot"]["status"] == "cached"
    assert obs["commodity_volatility"]["status"] == "cached"
    assert obs["us_national_debt"]["status"] == "cached"
    assert obs["thai_investor_flow"]["status"] == "cached"
    assert obs["thai_retail_gold"]["status"] == "cached"
    assert obs["thai_market_valuation"]["status"] == "cached"
    assert obs["crypto_macro_liquidity"]["status"] == "cached"

