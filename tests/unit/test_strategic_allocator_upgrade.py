import pytest
from schemas.macro_schemas import (
    _has_hard_data_numbers,
    AssetAllocationView,
    AssetStance,
    PairTradeStrategy,
    RiskMitigationScenario,
    MacroStrategyDirection,
    EconomicState,
    RegimeEvidenceComponent,
    MarketObservable,
)
from tools.macro.report_formatter import format_macro_strategy_report

def test_regex_negative_cases():
    """Negative cases: strings without actual numbers + financial context should fail."""
    assert not _has_hard_data_numbers(["I have 2 concerns"])
    assert not _has_hard_data_numbers(["yield rising"])
    assert not _has_hard_data_numbers(["score improved"])
    assert not _has_hard_data_numbers(["strong economic growth"])
    assert not _has_hard_data_numbers(["rate pressure is high"])
    assert not _has_hard_data_numbers(["2026-07-02 date observation"])
    assert not _has_hard_data_numbers([])

def test_regex_positive_cases():
    """Positive cases: strings with numbers + financial context or unit should pass."""
    assert _has_hard_data_numbers(["PMI 45.2"])
    assert _has_hard_data_numbers(["10Y yield at 4.40%"])
    assert _has_hard_data_numbers(["+0.85 score"])
    assert _has_hard_data_numbers(["150 bps spread"])
    assert _has_hard_data_numbers(["1.2x ratio"])
    assert _has_hard_data_numbers(["2026-07-02 level 4500 points"])
    assert _has_hard_data_numbers(["10Y/2Y curve inversion"])

def test_asset_allocation_defensive_degradation():
    """Test that lack of hard data automatically downgrades confidence to low with warnings."""
    view = AssetAllocationView(
        asset_class="US Equities (Tech Growth)",
        asset_bucket="equities",
        stance=AssetStance.OVERWEIGHT,
        rationale="Strong growth expected",
        confidence="high",
        supporting_data=["yield rising without numbers"]
    )
    assert view.confidence == "low"
    assert view.stance == AssetStance.NEUTRAL
    assert len(view.validation_warnings) >= 2
    assert any("DEFENSIVE_LOW_SUPPORTING_DATA" in str(w) or "Downgraded confidence to LOW" in str(w) for w in view.validation_warnings)
    assert any("ACTIVE_ALLOC_GUARDRAIL" in str(w) or "Active Allocation Guardrail" in str(w) for w in view.validation_warnings)

def test_pair_trade_gradual_risk_budget_guardrail():
    """Test gradual risk budget guardrail after pair trade execution quality checks."""
    # Case 1: medium conf without executable pair-trade controls -> low/small risk budget
    pt_med = PairTradeStrategy(
        long_leg="US Tech",
        short_leg="EU Industrials",
        thesis="AI capex",
        catalyst="Q2 earnings",
        risk="Hawkish Fed",
        time_horizon="1-3 Months",
        confidence="medium",
        sizing_guidance="high_risk_budget",
        supporting_data=["Quant Score = +0.85"]
    )
    assert pt_med.confidence == "low"
    assert pt_med.sizing_guidance == "small_risk_budget"
    assert any("PT_EXECUTION_GUARDRAIL" in str(w) or "Pair Trade Execution Guardrail" in str(w) for w in pt_med.validation_warnings)

    # Case 2: low conf (due to missing hard data) -> small risk budget
    pt_low = PairTradeStrategy(
        long_leg="US Tech",
        short_leg="EU Industrials",
        thesis="AI capex",
        catalyst="Q2 earnings",
        risk="Hawkish Fed",
        time_horizon="1-3 Months",
        confidence="high",
        sizing_guidance="high_risk_budget",
        supporting_data=["no numbers here"]
    )
    assert pt_low.confidence == "low"
    assert pt_low.sizing_guidance == "small_risk_budget"
    assert any("PT_RISK_BUDGET" in str(w) or "small_risk_budget" in str(w) for w in pt_low.validation_warnings)

def test_risk_mitigation_quality_gate():
    """Test that risk mitigation requires hard data, warning indicators, and hedge instruments."""
    rs = RiskMitigationScenario(
        tail_risk="US Recession",
        probability="medium",
        impact="severe",
        early_warning_indicators=[],
        hedge_instruments=["Long TLT"],
        trigger_to_activate="VIX > 28",
        cost_or_tradeoff="1.5% carry",
        supporting_data=["10Y Yield = 4.40%"],
        confidence="high"
    )
    assert rs.confidence == "low"
    assert len(rs.validation_warnings) == 1

def test_50_percent_portfolio_degradation_rule():
    """Test that portfolio conviction downgrades if >=50% of asset views (with len>=3) lack hard data."""
    direction = MacroStrategyDirection(
        evaluated_at="2026-07-02T10:00:00Z",
        overall_regime=EconomicState.REFLATION,
        conviction_level="high",
        conviction_rationale="Strong macro fundamentals",
        quant_narrative_alignment="aligned",
        focus_themes=["AI Capex"],
        asset_allocation=[
            AssetAllocationView(
                asset_class="US Equities", asset_bucket="equities", stance=AssetStance.OVERWEIGHT, rationale="Good",
                confidence="high", supporting_data=["Quant Score = +0.85"]
            ),
            AssetAllocationView(
                asset_class="EU Equities", asset_bucket="equities", stance=AssetStance.UNDERWEIGHT, rationale="Weak",
                confidence="high", supporting_data=["no numbers"]  # becomes low
            ),
            AssetAllocationView(
                asset_class="EM Equities", asset_bucket="equities", stance=AssetStance.NEUTRAL, rationale="Neutral",
                confidence="low", supporting_data=["PMI = 48.5"]   # already low
            ),
            AssetAllocationView(
                asset_class="US Treasuries", asset_bucket="fixed_income", stance=AssetStance.OVERWEIGHT, rationale="Yields high",
                confidence="high", supporting_data=["10Y Yield = 4.40%"]
            )
        ]
    )
    # Backfilled missing core classes also count as low-confidence coverage gaps.
    assert direction.conviction_level == "low"
    assert any("PORTFOLIO_DEFENSIVE_LOW" in str(w) or ">=50%" in str(w) for w in direction.validation_warnings)

def test_formatter_collapsible_callouts_and_empty_sections():
    """Test report formatter renders callouts and suppresses empty sections."""
    direction = MacroStrategyDirection(
        evaluated_at="2026-07-02T10:00:00Z",
        overall_regime=EconomicState.REFLATION,
        conviction_level="high",
        conviction_rationale="Strong macro fundamentals",
        quant_narrative_alignment="aligned",
        focus_themes=["AI Capex"],
        asset_allocation=[
            AssetAllocationView(
                asset_class="US Equities (AI/Tech Growth)", asset_bucket="equities", stance=AssetStance.OVERWEIGHT, rationale="Good",
                confidence="high", supporting_data=["Quant Score = +0.85"]
            )
        ],
        pair_trades=[
            PairTradeStrategy(
                long_leg="US Tech", short_leg="EU Industrials", thesis="AI vs manufacturing",
                catalyst="Q2 earnings", risk="Fed hawkish", time_horizon="1-3 Months",
                confidence="high", sizing_guidance="small_risk_budget", implementation_idea="Overweight Tech",
                supporting_data=["Score = +0.85"]
            )
        ],
        risk_scenarios=[]  # empty
    )
    report = format_macro_strategy_report(direction)

    # Pair trade without executable controls is suppressed.
    assert "> [!tip]- **Pair Trade:** Long US Tech / Short EU Industrials" not in report
    assert "Implementation Idea:** Overweight Tech" not in report

    # Check empty risk scenarios section is suppressed
    assert "## 🛡️ Portfolio Risk Mitigation & Hedging" not in report

    # Check formatter-controlled compliance disclaimer is present
    assert "> [!CAUTION] ข้อสงวนสิทธิ์และคำชี้แจงการใช้งาน" in report
    assert "ไม่ถือเป็นคำแนะนำการลงทุนรายบุคคล" in report

def test_backward_compatibility():
    """Test backward compatibility with minimal old data."""
    direction = MacroStrategyDirection(
        evaluated_at="2026-07-02T10:00:00Z",
        overall_regime=EconomicState.STAGFLATION,
        conviction_level="medium",
        conviction_rationale="Stagflation risks",
        quant_narrative_alignment="divergent",
        focus_themes=["Defensive"],
        asset_allocation=[
            AssetAllocationView(
                asset_class="Gold", asset_bucket="commodities", stance=AssetStance.OVERWEIGHT, rationale="Inflation hedge"
            )
        ]
    )
    assert direction.pair_trades == []
    assert direction.risk_scenarios == []
    report = format_macro_strategy_report(direction)
    assert "## ⚖️ Relative Value & Pair Trades" not in report
    assert "> [!CAUTION] ข้อสงวนสิทธิ์และคำชี้แจงการใช้งาน" in report


def test_utf8_boundary_raw_bytes_roundtrip(tmp_path):
    """Test raw UTF-8 byte writing and reading back without any mojibake literals."""
    import frontmatter as fm
    test_file = tmp_path / "test_mojibake.md"
    content = "---\ntitle: กลยุทธ์ลงทุน 2026 🧭\ntags:\n  - หุ้นไทย\n---\n# รายงานภาวะตลาด\nหุ้นไทย SET ทดสอบ 1,550 จุด 🚀"

    # Write using UTF-8 explicitly
    test_file.write_text(content, encoding="utf-8")

    # Read back raw bytes and assert UTF-8 encoding
    raw_bytes = test_file.read_bytes()
    assert b"\xe0\xb8" in raw_bytes  # Thai UTF-8 prefix bytes present
    assert b"\xf0\x9f" in raw_bytes  # Emoji UTF-8 prefix bytes present

    # Read back text and verify no mojibake literals
    read_text = test_file.read_text(encoding="utf-8")
    assert "ðŸ" not in read_text
    assert "à¸" not in read_text
    assert "กลยุทธ์ลงทุน 2026 🧭" in read_text
    assert "หุ้นไทย SET ทดสอบ 1,550 จุด 🚀" in read_text

    # Verify frontmatter loading via explicit UTF-8 string load
    post = fm.loads(read_text)
    assert post.metadata["title"] == "กลยุทธ์ลงทุน 2026 🧭"
    assert "หุ้นไทย" in post.metadata["tags"]


def test_defensive_low_when_supporting_data_empty():
    """Test that when supporting_data is empty, models downgrade confidence to low without auto-extraction."""
    # Test RiskMitigationScenario
    rs = RiskMitigationScenario(
        tail_risk="Geopolitical conflict",
        probability="medium",
        impact="severe",
        early_warning_indicators=["Gold breaking $2,500"],
        hedge_instruments=["Long Oil ETF"],
        trigger_to_activate="VIX index spike above 28 pts",
        trigger_type="Daily close breakout",
        volume_threshold="Options volume > 100M USD",
        unwind_or_cover_condition="VIX falls below 22 pts",
        cost_or_tradeoff="Option premium",
        supporting_data=[]
    )
    assert rs.confidence == "low"
    assert any("DEFENSIVE_LOW" in str(w) for w in rs.validation_warnings)
    assert len(rs.supporting_data) == 0

    # Test PairTradeStrategy
    pt = PairTradeStrategy(
        long_leg="Gold",
        short_leg="Equities",
        thesis="Safe haven demand pushing Gold above $2,400",
        catalyst="Q3 Inflation CPI > 3.0%",
        risk="Fed rate cut 50 bps",
        time_horizon="3 Months",
        confidence="medium",
        supporting_data=[]
    )
    assert pt.confidence == "low"
    assert any("DEFENSIVE_LOW" in str(w) for w in pt.validation_warnings)
    assert len(pt.supporting_data) == 0


def test_graceful_drop_with_summary_warning():
    """Test that risk scenarios and pair trades without hard data are dropped and add summary warnings."""
    direction = MacroStrategyDirection(
        evaluated_at="2026-07-02T10:00:00Z",
        overall_regime=EconomicState.GOLDILOCKS,
        conviction_level="high",
        conviction_rationale="Solid fundamentals",
        quant_narrative_alignment="aligned",
        focus_themes=["Growth"],
        asset_allocation=[
            AssetAllocationView(asset_class="US Equities", asset_bucket="equities", stance=AssetStance.OVERWEIGHT, rationale="Tech boom", supporting_data=["10Y Yield = 4.20%"]),
            AssetAllocationView(asset_class="Global Bonds", asset_bucket="fixed_income", stance=AssetStance.NEUTRAL, rationale="Stable rates", supporting_data=["Spread 120 bps"]),
            AssetAllocationView(asset_class="Gold", asset_bucket="commodities", stance=AssetStance.UNDERWEIGHT, rationale="Risk-on", supporting_data=["Gold $2,300"]),
            AssetAllocationView(asset_class="USD Currency", asset_bucket="fx", stance=AssetStance.NEUTRAL, rationale="DXY steady", supporting_data=["DXY 102.5"]),
            AssetAllocationView(asset_class="Cash T-Bills", asset_bucket="cash", stance=AssetStance.UNDERWEIGHT, rationale="Low yield", supporting_data=["T-Bill yield 4.5%"])
        ],
        pair_trades=[
            # Invalid pair trade with hard data but incomplete executable controls
            PairTradeStrategy(long_leg="US Tech", short_leg="EU Industrials", thesis="Growth divergence", catalyst="CPI < 2.5%", risk="Spike in rates", time_horizon="3 Months", confidence="high", supporting_data=["Spread 200 bps"]),
            # Invalid pair trade with NO numbers or observables anywhere
            PairTradeStrategy(long_leg="A", short_leg="B", thesis="just general feeling", catalyst="some event", risk="some risk", time_horizon="1 Month", confidence="medium", supporting_data=[])
        ],
        risk_scenarios=[
            # Invalid risk scenario with NO numbers or observables
            RiskMitigationScenario(tail_risk="Unknown risk", probability="low", impact="manageable", early_warning_indicators=["bad news"], hedge_instruments=["cash"], trigger_to_activate="market crash", cost_or_tradeoff="none", supporting_data=[])
        ]
    )
    # Assert graceful drop happened
    assert len(direction.pair_trades) == 0
    assert len(direction.risk_scenarios) == 0

    # Assert summary warnings were recorded
    assert any("GRACEFUL_DROP_PAIR_TRADES" in str(w) or "Graceful Drop: 2 pair trade(s) omitted" in str(w) for w in direction.validation_warnings)
    assert any("GRACEFUL_DROP_RISK_SCENARIOS" in str(w) or "Graceful Drop: 1 risk scenario(s) omitted" in str(w) for w in direction.validation_warnings)


def test_proxy_observables_recognized():
    """Test that proxy observables like Gold, Oil, CDS, VIX, freight are recognized as hard data."""
    assert _has_hard_data_numbers(["Brent oil surge above $85"])
    assert _has_hard_data_numbers(["CDS spread widening 45 bps"])
    assert _has_hard_data_numbers(["Shipping freight rate index up 15%"])
    assert _has_hard_data_numbers(["Defense ETF inflows +$500M"])


def test_5_asset_classes_coverage_target_warning():
    """Test that asset allocation with fewer than 5 classes generates a coverage warning without backfilling."""
    direction = MacroStrategyDirection(
        evaluated_at="2026-07-02T10:00:00Z",
        overall_regime=EconomicState.STAGFLATION,
        conviction_level="medium",
        conviction_rationale="Inflation persists",
        quant_narrative_alignment="aligned",
        focus_themes=["Defensive"],
        asset_allocation=[
            AssetAllocationView(asset_class="US Equities", asset_bucket="equities", stance=AssetStance.UNDERWEIGHT, rationale="High valuation", supporting_data=["P/E 24x"]),
            AssetAllocationView(asset_class="Gold", asset_bucket="commodities", stance=AssetStance.OVERWEIGHT, rationale="Inflation hedge", supporting_data=["Gold $2,400"]),
            AssetAllocationView(asset_class="Cash", asset_bucket="cash", stance=AssetStance.OVERWEIGHT, rationale="Safety", supporting_data=["Yield 5.0%"])
        ]
    )
    assert len(direction.asset_allocation) == 3
    assert any("COVERAGE_WARNING_INCOMPLETE" in str(w) or "Coverage Warning: Asset allocation contains only 3 classes" in str(w) for w in direction.validation_warnings)


def test_contradiction_guardrails_and_executable_fields():
    """Test contradiction checks (Equities+Bonds Overweight, Regime vs Sticky Inflation) and new executable fields."""
    direction = MacroStrategyDirection(
        evaluated_at="2026-07-02T10:00:00Z",
        overall_regime=EconomicState.REFLATION,
        conviction_level="high",
        conviction_rationale="Bullish across assets",
        quant_narrative_alignment="aligned",
        focus_themes=["Growth"],
        asset_allocation=[
            AssetAllocationView(
                asset_class="US Growth Equities",
                asset_bucket="equities",
                stance=AssetStance.OVERWEIGHT,
                rationale="AI boom but consumer sentiment ต่ำ",
                confidence="high",
                supporting_data=["Score = 0.93", "CPI > 3.0%", "Housing starts ลดลงเหลือ 1,177k"]
            ),
            AssetAllocationView(
                asset_class="US Treasuries Bonds",
                asset_bucket="fixed_income",
                stance=AssetStance.OVERWEIGHT,
                rationale="Rates falling",
                confidence="high",
                supporting_data=["10Y Yield = 4.30%"]
            ),
            AssetAllocationView(asset_class="Gold", asset_bucket="commodities", stance=AssetStance.NEUTRAL, rationale="Stable", supporting_data=["Gold $2,300"]),
            AssetAllocationView(asset_class="USD Currency", asset_bucket="fx", stance=AssetStance.NEUTRAL, rationale="Stable", supporting_data=["DXY 102.5"]),
            AssetAllocationView(asset_class="Cash", asset_bucket="cash", stance=AssetStance.NEUTRAL, rationale="Stable", supporting_data=["Yield 4.5%"])
        ],
        pair_trades=[
            PairTradeStrategy(
                long_leg="US Growth Equities",
                short_leg="Thai SET Index",
                thesis="Divergence in earnings",
                catalyst="Q2 tech earnings > 10%",
                risk="SET rebound",
                time_horizon="3 Months",
                confidence="medium",
                sizing_guidance="medium_risk_budget",
                instrument_proxy="Long QQQ ETF / Short SET50 Index Futures",
                hedge_ratio="1.0 : 0.8 Beta-adjusted",
                fx_handling="Unhedged USD/THB",
                stop_loss_trigger="SET Index ทะลุ 1,620 จุด หรือ VIX > 28",
                    target_gain_or_rebalance="Take profit +8%",
                    max_drawdown_limit="Max Drawdown 5%",
                    supporting_data=["Score divergence +0.80 vs -0.20", "VIX confirmation spread = 12 pts"],
                    observable_refs=["obs_score_divergence", "obs_vix_confirmation_spread"]
                )
            ],
        risk_scenarios=[
            RiskMitigationScenario(
                tail_risk="Thai bond refinancing crash",
                probability="medium",
                impact="moderate",
                early_warning_indicators=["Spread widening 50 bps"],
                hedge_instruments=["Short SET50 Futures"],
                trigger_to_activate="SET หลุด 1,550 จุด",
                volume_threshold="มูลค่าซื้อขาย > 50,000 ล้านบาท",
                trigger_type="Daily Close Breakout (ปิดสิ้นวัน)",
                unwind_or_cover_condition="Cover short เมื่อ SET ยืนเหนือ 1,570 จุด",
                cost_or_tradeoff="Carry cost 2%",
                supporting_data=["Bond maturity 100,000 MB"]
            )
            ],
            observable_registry={
                "obs_score_divergence": MarketObservable(
                    observable_id="obs_score_divergence",
                    asset_bucket="risk",
                    region="Global",
                    indicator="US Growth vs Thai SET Score Divergence",
                    value="1.00",
                    unit="spread",
                    observed_at="2026-07-02",
                    source_file="macro_input_20260702.json",
                    provider="Derived",
                    is_valid=True,
                ),
                "obs_vix_confirmation_spread": MarketObservable(
                    observable_id="obs_vix_confirmation_spread",
                    asset_bucket="risk",
                    region="Global",
                    indicator="VIX and SET Stress Confirmation Spread",
                    value="12",
                    unit="pts",
                    observed_at="2026-07-02",
                    source_file="macro_input_20260702.json",
                    provider="Derived",
                    is_valid=True,
                ),
            }
        )

    # Check that both Equities and Bonds Overweight without "barbell"/"hedge" reconciliation triggered warning
    assert any("CONVICTION_CONTRADICTION" in str(w) or "Contradiction Warning: Portfolio recommends Overweight on both Equities Growth and Long Treasuries" in str(w) for w in direction.validation_warnings)
    # Check that Regime Reflation with CPI > 3% triggered warning
    assert any("REGIME_CONTRADICTION" in str(w) or "Regime Contradiction Warning" in str(w) for w in direction.validation_warnings)
    # Check that conviction and confidence were downgraded from high to medium
    assert direction.conviction_level == "medium"
    assert direction.asset_allocation[0].confidence == "medium"
    assert direction.asset_allocation[1].confidence == "medium"

    # Verify executable fields are present and provenanced
    pt = direction.pair_trades[0]
    assert pt.instrument_proxy == "Long QQQ ETF / Short SET50 Index Futures"
    assert not any("[From stop_loss_trigger]" in item for item in pt.supporting_data)

    rs = direction.risk_scenarios[0]
    assert rs.volume_threshold == "มูลค่าซื้อขาย > 50,000 ล้านบาท"
    assert not any("[From volume_threshold]" in item for item in rs.supporting_data)

    report = format_macro_strategy_report(direction)
    assert "Instrument Proxy / เครื่องมือจริง:** Long QQQ ETF / Short SET50 Index Futures" in report
    assert "Stop Loss Trigger / จุดตัดขาดทุน:** SET Index ทะลุ 1,620 จุด หรือ VIX > 28" in report
    assert "Volume Threshold / ปริมาณซื้อขายยืนยัน:** มูลค่าซื้อขาย > 50,000 ล้านบาท" in report
    assert "Trigger Type / ประเภทจุดตัด:** Daily Close Breakout (ปิดสิ้นวัน)" in report


def test_4_layer_institutional_models_and_guardrails():
    """Test 4-Layer institutional model features: Single-Source Penalty and Stale Data Degradation."""
    # 1. Test Single-Source Penalty in AssetAllocationView
    asset = AssetAllocationView(
        asset_class="US Treasuries 10Y",
        asset_bucket="fixed_income",
        stance=AssetStance.OVERWEIGHT,
        rationale="Attractive real yields",
        confidence="high",
        data_confidence="high",
        signal_confidence="high",
        implementation_confidence="high",
        supporting_data=["10Y Yield at 4.25%"],
        source_refs=["single_report.md"]  # Only 1 source -> should trigger penalty
    )
    assert asset.confidence == "medium"
    assert any("SINGLE_SOURCE_PENALTY" in str(w) or "Single-Source Penalty" in str(w) for w in asset.validation_warnings)

    # 2. Test Stale Data Degradation in MacroStrategyDirection
    direction = MacroStrategyDirection(
        evaluated_at="2026-07-02T10:00:00Z",
        overall_regime=EconomicState.REFLATION,
        conviction_level="high",
        conviction_rationale="Strong growth and easing liquidity",
        quant_narrative_alignment="aligned",
        focus_themes=["AI Expansion", "Rate Cuts"],
        stale_data_warnings=["CPI report is 2 months stale"],  # Stale warning -> should degrade conviction
        asset_allocation=[
            AssetAllocationView(
                asset_class="US Equities", asset_bucket="equities", stance=AssetStance.OVERWEIGHT, rationale="Growth",
                confidence="high", supporting_data=["S&P at 5500 pts"], source_refs=["src1", "src2"]
            ),
            AssetAllocationView(
                asset_class="US Fixed Income", asset_bucket="fixed_income", stance=AssetStance.NEUTRAL, rationale="Rates balanced",
                confidence="high", supporting_data=["10Y Yield = 4.25%"], source_refs=["src1", "src2"]
            ),
            AssetAllocationView(
                asset_class="Commodities Gold", asset_bucket="commodities", stance=AssetStance.NEUTRAL, rationale="Real rates offset hedge demand",
                confidence="high", supporting_data=["Gold = 2400 USD/oz"], source_refs=["src1", "src2"]
            ),
            AssetAllocationView(
                asset_class="FX / Currencies", asset_bucket="fx", stance=AssetStance.NEUTRAL, rationale="USD stable",
                confidence="high", supporting_data=["DXY = 102.5"], source_refs=["src1", "src2"]
            ),
            AssetAllocationView(
                asset_class="Cash / T-Bills", asset_bucket="cash", stance=AssetStance.NEUTRAL, rationale="Cash yield remains positive",
                confidence="high", supporting_data=["13-Week T-Bill Yield = 4.8%"], source_refs=["src1", "src2"]
            ),
        ]
    )
    assert direction.conviction_level == "medium"
    assert any("STALE_DATA_DEGRADATION" in str(w) or "Stale Data Degradation" in str(w) for w in direction.validation_warnings)
    assert not any("COVERAGE_BACKFILL" in str(w) or "Coverage Backfill" in str(w) for w in direction.validation_warnings)


def test_7_section_institutional_report_formatter():
    """Test that format_macro_strategy_report renders all 7 Institutional Dashboard sections."""
    direction = MacroStrategyDirection(
        evaluated_at="2026-07-02T10:00:00Z",
        overall_regime=EconomicState.REFLATION,
        conviction_level="high",
        conviction_rationale="Macro fundamentals aligned",
        quant_narrative_alignment="aligned",
        focus_themes=["Tech Capex", "EM Flows"],
        regime_probabilities={"Reflation": "50%", "Sticky Inflation": "30%", "Soft Landing": "20%"},
        regime_evidence=[
            RegimeEvidenceComponent(
                dimension="Growth", signal="Resilient", evidence="GDP +2.8%",
                conflict="Manufacturing PMI soft at 49.5 pts", confidence="high"
            )
        ],
        asset_allocation=[
            AssetAllocationView(
                asset_class="Equities", asset_bucket="equities", stance=AssetStance.OVERWEIGHT, rationale="Earnings expansion",
                confidence="high", allocation_delta="+5% vs benchmark", benchmark_ref="MSCI World Index",
                time_horizon="Strategic (6-12 Months)", supporting_data=["EPS growth +10%"],
                source_refs=["Goldman Sachs Q3 Report", "Morgan Stanley Outlook"]
            )
        ],
        pair_trades=[
            PairTradeStrategy(
                long_leg="US Tech", short_leg="EU Industrials", thesis="Growth divergence",
                catalyst="AI adoption accelerating", risk="Valuation compression", time_horizon="3-6 Months",
                confidence="high", sizing_guidance="medium_risk_budget", instrument_proxy="Long QQQ / Short FEZ",
                hedge_ratio="1.0 : 0.8 Beta-adjusted", fx_handling="Unhedged USD",
                    entry_trigger="Breakout above MA50 at 450 pts", stop_loss_trigger="QQQ drops below 430 pts",
                    target_gain_or_rebalance="Take profit +10%", max_drawdown_limit="Max DD 4%",
                    review_frequency="Weekly review", supporting_data=["Tech EPS +15% vs EU -2%", "Tech valuation spread 4 pts"],
                    observable_refs=["obs_tech_eu_eps_spread", "obs_tech_eu_valuation_spread"]
                )
            ],
        risk_scenarios=[
            RiskMitigationScenario(
                tail_risk="Geopolitical escalation in Middle East",
                probability="medium", impact="severe", mitigation_strategy="Long Oil & Gold",
                early_warning_indicators=["Brent crude breaking $85"], hedge_instruments=["Brent Futures", "GLD ETF"],
                trigger_to_activate="Brent > $85 or Gold > $2,400", volume_threshold="Daily volume > 100M USD",
                trigger_type="Daily Close Breakout", unwind_or_cover_condition="Brent falls below $80",
                cost_or_tradeoff="Carry cost 150 bps p.a.", hedge_size="10% of portfolio value",
                hedge_purpose="portfolio_hedge", confidence="high",
                supporting_data=["VIX option skew rising 5 pts"]
            )
        ],
        source_files=["macro_input_20260702.json"],
        data_timestamp_notes=["All asset class prices as of July 1, 2026"],
        observable_registry={
            "obs_tech_eu_eps_spread": MarketObservable(
                observable_id="obs_tech_eu_eps_spread",
                asset_bucket="equities",
                region="Global",
                indicator="US Tech vs EU Industrials EPS Spread",
                value="17",
                unit="% pts",
                observed_at="2026-07-02",
                source_file="macro_input_20260702.json",
                provider="Derived",
                is_valid=True,
            ),
            "obs_tech_eu_valuation_spread": MarketObservable(
                observable_id="obs_tech_eu_valuation_spread",
                asset_bucket="equities",
                region="Global",
                indicator="US Tech vs EU Industrials Valuation Spread",
                value="4",
                unit="pts",
                observed_at="2026-07-02",
                source_file="macro_input_20260702.json",
                provider="Derived",
                is_valid=True,
            ),
        }
    )

    report = format_macro_strategy_report(direction)

    # Check 7 Section Headers
    assert "# 🧭 1. Executive View" in report
    assert "### 🎲 ความน่าจะเป็นของสภาวะเศรษฐกิจ (Regime Probabilities)" in report
    assert "## 📊 2. Evidence Dashboard" in report
    assert "## 📈 3. Cross-Asset Allocation Summary" in report
    assert "## ⚡ 4. Key Contradictions & Quant-Narrative Divergence" in report
    assert "## ⚖️ 5. Relative Value & Pair Trades (Trade Ideas)" in report
    assert "## 🛡️ 6. Portfolio Risk Mitigation & Hedging (Hedging Plan)" in report
    assert "## 📋 7. หมายเหตุด้านคุณภาพข้อมูลและระบบ" in report

    # Check key fields rendered
    assert "| **Reflation** | 50% |" in report
    assert "| **Soft Landing** | 20% |" in report
    assert "| **Growth** | Resilient | GDP +2.8% | Manufacturing PMI soft at 49.5 pts | HIGH |" in report
    assert "+5% vs benchmark" in report
    assert "MSCI World Index" in report
    assert "Entry Trigger / จุดเข้าเทรด:** Breakout above MA50 at 450 pts" in report
    assert "Review Frequency / ความถี่ทบทวน:** Weekly review" in report
    assert "Hedge Size / ขนาดป้องกันความเสี่ยง:** 10% of portfolio value" in report
    assert "macro_input_20260702.json" in report


def test_gold_and_us_growth_contradiction_guardrails():
    """Test Gold rationale guardrail (#5) and US Growth conflicting signals check (#3)."""
    direction = MacroStrategyDirection(
        evaluated_at="2026-07-02T10:00:00Z",
        overall_regime=EconomicState.REFLATION,
        conviction_level="high",
        quant_narrative_alignment="aligned",
        key_assumptions=["US Economy growing, but consumer sentiment low and housing starts weak"],
        focus_themes=["Tech Growth", "Inflation"],
        conviction_rationale="Tested rationale",
        regime_probabilities={"Reflation": "60%", "Recession": "40%"},
        asset_allocation=[
            AssetAllocationView(
                asset_class="US Equities (AI/Tech Growth)",
                asset_bucket="equities",
                stance=AssetStance.OVERWEIGHT,
                allocation_delta="+5%",
                benchmark_ref="S&P 500",
                time_horizon="3 Months",
                rationale="AI growth thesis strong despite rising yields and weak housing starts",
                confidence="high",
                why_not_high="None",
                supporting_data=["US Growth score 1.0, 10Y yields rising to 4.5%"]
            ),
            AssetAllocationView(
                asset_class="Precious Metals (Gold)",
                asset_bucket="commodities",
                stance=AssetStance.OVERWEIGHT,
                allocation_delta="+3%",
                benchmark_ref="Commodity Index",
                time_horizon="6 Months",
                rationale="Overweight Gold due to geopolitical tensions and war in Middle East",
                confidence="high",
                why_not_high="None",
                supporting_data=["Gold price above $2,300 due to war tensions"]
            ),
            AssetAllocationView(
                asset_class="Cash USD",
                asset_bucket="cash",
                stance=AssetStance.NEUTRAL,
                allocation_delta="0%",
                benchmark_ref="Cash",
                time_horizon="1 Month",
                rationale="Holding cash for liquidity at 5.25% yield",
                confidence="high",
                why_not_high="None",
                supporting_data=["Yield 5.25%"]
            )
        ]
    )

    # Check Gold downgraded from HIGH to MEDIUM due to geopolitics-only rationale without real yield/fed anchoring
    gold = next(a for a in direction.asset_allocation if "Gold" in a.asset_class)
    assert gold.confidence == "medium"
    assert any("GOLD_CONTRADICTION" in str(w) or "Contradiction Degradation" in str(w) for w in gold.validation_warnings)
    assert any("GOLD_RATIONALE_WARNING" in str(w) or "Gold Rationale Warning" in str(w) for w in direction.validation_warnings)

    # Check US Equities downgraded from HIGH to MEDIUM due to conflicting rising yields/housing starts
    us_eq = next(a for a in direction.asset_allocation if "US Equities" in a.asset_class)
    assert us_eq.confidence == "medium"
    assert any("US_EQUITY_CONTRADICTION" in str(w) or "Contradiction Degradation" in str(w) for w in us_eq.validation_warnings)
