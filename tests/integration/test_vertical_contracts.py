"""Phase 5 Vertical Integration Contract Tests (Non-Networked Producer-to-Consumer).

Exercises real multi-module pipelines without mocking internal domain/application transforms:
1. Flow 1: Sidecar (dict & model, market aliases) -> evaluate_portfolio_advisory
   -> _convert_report_to_dto + build_advisory_llm_payload -> JSON serialization.
2. Flow 2: Real PortfolioState (float FX=35.5) + Dated Trades_Log.csv (via
   PortfolioTradingService) + serialized dict Dividend Schedules
   -> PortfolioPerformanceService.record_performance_snapshot
   -> MarkdownPerformanceAdapter write & read_history / raw CSV round-trip.
3. Flow 3: FinancialAutopsySnapshot -> compute_equity_quant_signals
   (real dupont_accrual, quant_scoring waterfall, signal_audit, equity_rules_engine)
   -> QuantSignals JSON contract.
"""
from __future__ import annotations

import csv
from decimal import Decimal
import json
from typing import Literal, Optional
import pandas as pd
import pytest

from application.portfolio.advisory_service import _convert_report_to_dto
from core.portfolio_advisory.engine import evaluate_portfolio_advisory
from core.portfolio_advisory.llm_payload_builder import build_advisory_llm_payload
from schemas.micro_quant_schemas import (
    DeterministicScorecard,
    InsiderConviction,
    QuantSignals,
    TacticalSetup,
)
from schemas.portfolio_advisory_schemas import PortfolioAdvisoryResponseDTO
from tools.market.asset_resolver import AssetClass, ResolvedAsset
from tools.market.financial_autopsy import (
    FinancialAutopsyFetchResult,
    FinancialAutopsyPeriod,
    FinancialAutopsySnapshot,
)
from tools.market.quant_engine import PriceSeriesQuality
from tools.portfolio.domain.models import AllocationTarget, Holding, PortfolioState, Summary
from tools.portfolio.ports.dividend_port import DividendHistoryPort
from tools.portfolio.ports.price_port import MarketPricePort


class _NoOpPricePort(MarketPricePort):
    """Non-networked price port for offline vertical integration tests."""

    def fetch_price(self, symbol: str, currency: Literal["THB", "USD"]) -> Optional[float]:
        return None

    def fetch_fx_rate(
        self, date_str: Optional[str] = None, fallback_rate: Optional[float] = None
    ) -> tuple[float, Literal["historical", "live", "fallback"]]:
        return float(fallback_rate or 36.5), "fallback"

    def refresh_portfolio_prices(self, state: PortfolioState) -> dict[str, str]:
        return {}

    def fetch_fundamentals(self, state: PortfolioState, force: bool = False) -> dict[str, str]:
        return {"updated": "0", "skipped": str(len(state.holdings))}


class _StaticDividendPort(DividendHistoryPort):
    requires_network: bool = False

    def __init__(self, schedules_by_sym: dict[str, list[dict]] | None = None) -> None:
        self._data = schedules_by_sym or {}

    def fetch_dividend_history(self, symbols: list[str]) -> dict[str, list[dict]]:
        return {sym: self._data.get(sym, []) for sym in symbols}


def test_flow1_advisory_sidecar_to_dto_and_llm_payload_contract():
    """Sidecar quant_signals survive engine -> DTO mapper -> LLM payload -> JSON."""
    state = PortfolioState(
        last_updated="2026-11-01T10:00:00",
        fx_rates={"USDTHB": 35.0},
        summary=Summary(
            total_value_thb=160_000.0,
            cash_balance_thb=100_000.0,
            usd_thb_rate=35.0,
        ),
        allocation_targets=[
            AllocationTarget(bucket_id="core", name="Core Equities", target_percent=90.0),
            AllocationTarget(bucket_id="cash", name="Cash Reserve", target_percent=10.0),
        ],
        holdings=[
            Holding(
                symbol="PTT",
                asset_type="Stock",
                market="TH",
                units=1000.0,
                current_price_thb=30.0,
                market_value_thb=30_000.0,
                bucket_id="core",
            ),
            Holding(
                symbol="SCB",
                asset_type="Stock",
                market="TH",
                units=300.0,
                current_price_thb=100.0,
                market_value_thb=30_000.0,
                bucket_id="core",
            ),
            Holding(
                symbol="CASH_THB",
                asset_type="Cash",
                units=100_000.0,
                market_value_thb=100_000.0,
                bucket_id="cash",
            ),
        ],
    )

    ptt_signals = QuantSignals(
        ticker="PTT",
        market="TH",
        volatility_pct=10.0,
        adtv_local_currency=15_000.0,  # Shallow ADTV triggers clear positive market impact
        beta=0.80,
        composite_score=74.0,
        evaluated_at="2026-11-01T09:00:00+00:00",
        tactical_setup=TacticalSetup(
            price_stage="STAGE_2_MARKUP",
            current_price=30.0,
            buy_zone_min=28.0,
            buy_zone_max=32.0,
            invalidation_stop_loss=26.0,
        ),
        deterministic_scorecard=DeterministicScorecard(
            core_conviction_score=8.5,
            business_conviction_score=8.5,
            action_stance="ACCUMULATE_NOW",
        ),
    )

    # Supply wrong-market "PTT" US distractor alongside valid "PTT.BK" TH sidecar + dict "SCB"
    equity_sidecars = {
        "PTT": {
            "ticker": "PTT",
            "market": "US",
            "analysis_date": "2026-11-01",
            "quant_signals": {
                "ticker": "PTT",
                "market": "US",
                "volatility_pct": 99.0,
                "adtv_local_currency": 999_999_999.0,
            },
        },
        "PTT.BK": {
            "ticker": "PTT",
            "market": "TH",
            "analysis_date": "2026-11-01",
            "quant_signals": ptt_signals,
        },
        "SCB": {
            "ticker": "SCB",
            "market": "TH",
            "analysis_date": "2026-11-01",
            "quant_signals": {
                "ticker": "SCB",
                "market": "TH",
                "volatility_pct": 40.0,
                "adtv_local_currency": 500_000_000.0,
                "beta": 1.20,
            },
        },
    }

    report = evaluate_portfolio_advisory(
        portfolio_state=state,
        equity_sidecars=equity_sidecars,
    )

    # 1. Domain engine extracted nested signals (not 50/50 fallback from outer defaults)
    ptt_rc = report.risk_radar.risk_contributions_pct["PTT"]
    scb_rc = report.risk_radar.risk_contributions_pct["SCB"]
    assert float(ptt_rc) < 20.0
    assert float(scb_rc) > 80.0
    assert report.risk_radar.volatility_estimated_symbols == []

    ptt_item = next(h for h in report.holdings_advisory if h.symbol == "PTT")
    assert ptt_item.volatility_pct == Decimal("10.00")
    assert ptt_item.volatility_source == "observed"
    assert ptt_item.is_volatility_estimated is False
    assert ptt_item.lot_sizing is not None and ptt_item.lot_sizing.amount_native > Decimal("0")
    assert ptt_item.estimated_friction_bps is not None
    # TH minimum fee / commission + positive market impact from shallow ADTV=15,000 THB
    assert ptt_item.estimated_friction_bps > Decimal("20.0")

    # 2. Application DTO mapper & FastAPI JSON schema round-trip
    dto = _convert_report_to_dto(report)
    dto_json = dto.model_dump(mode="json")
    validated_dto = PortfolioAdvisoryResponseDTO.model_validate(dto_json)

    assert validated_dto.risk_radar.risk_contributions_pct["PTT"] == pytest.approx(float(ptt_rc), abs=1e-4)
    assert validated_dto.risk_radar.risk_contributions_pct["SCB"] == pytest.approx(float(scb_rc), abs=1e-4)
    assert validated_dto.risk_radar.total_portfolio_volatility_pct is not None

    dto_ptt = next(h for h in validated_dto.holdings_advisory if h.symbol == "PTT")
    assert dto_ptt.risk_contribution_pct == pytest.approx(float(ptt_rc), abs=1e-4)
    assert dto_ptt.estimated_friction_bps == pytest.approx(float(ptt_item.estimated_friction_bps), abs=1e-4)
    assert dto_ptt.volatility_pct == pytest.approx(10.0, abs=1e-4)
    assert dto_ptt.volatility_source == "observed"
    assert dto_ptt.is_volatility_estimated is False

    # 3. LLM Synthesis Payload Builder & JSON serialization
    llm_payload = json.loads(json.dumps(build_advisory_llm_payload(report), ensure_ascii=False))
    assert llm_payload["risk_radar"]["risk_contributions_pct"]["PTT"] == pytest.approx(float(ptt_rc), abs=1e-4)
    assert llm_payload["risk_radar"]["risk_contributions_pct"]["SCB"] == pytest.approx(float(scb_rc), abs=1e-4)
    llm_ptt = next(h for h in llm_payload["holdings_advisory"] if h["symbol"] == "PTT")
    assert llm_ptt["risk_contribution_pct"] == pytest.approx(float(ptt_rc), abs=1e-4)
    assert llm_ptt["estimated_friction_bps"] == pytest.approx(float(ptt_item.estimated_friction_bps), abs=1e-4)
    assert llm_ptt["volatility_pct"] == pytest.approx(10.0, abs=1e-4)
    assert llm_ptt["volatility_source"] == "observed"
    assert llm_ptt["is_volatility_estimated"] is False


def test_flow2_performance_service_fx_entitlement_and_csv_roundtrip(isolated_portfolio, tmp_vault):
    """Dated trades + float USDTHB FX + dict dividend schedule persist Economic NAV via real Markdown adapter."""
    from tools.portfolio.adapters.markdown.journal_vault_adapter import JournalVaultAdapter
    from tools.portfolio.adapters.markdown.paths import get_performance_filepath
    from tools.portfolio.adapters.markdown.performance_adapter import MarkdownPerformanceAdapter
    from tools.portfolio.adapters.markdown.repository_adapter import MarkdownVaultRepositoryAdapter
    from tools.portfolio.domain.ledger_change import LedgerChange
    from tools.portfolio.services.performance_service import PortfolioPerformanceService
    from tools.portfolio.services.trading_service import PortfolioTradingService

    repo = MarkdownVaultRepositoryAdapter()
    perf_repo = MarkdownPerformanceAdapter()
    price_port = _NoOpPricePort()
    journal_port = JournalVaultAdapter()

    # Seed portfolio cash pots & float FX rate (35.5 matches the historical crash repro)
    with repo.unit_of_work("default") as uow:
        state = uow.load_state()
        state.fx_rates["USDTHB"] = 35.5
        for h in state.holdings:
            if h.symbol == "CASH_THB":
                h.units = 100_000.0
                h.market_value_thb = 100_000.0
            elif h.symbol == "CASH_USD":
                h.units = 5_000.0
                h.market_value_thb = 5_000.0 * 35.5
        uow.commit(state, LedgerChange(kind="unchanged"))

    trading_svc = PortfolioTradingService(
        repo=repo,
        price_provider=price_port,
        journal_provider=journal_port,
    )

    # Pre-XD buys (entitled) + Post-XD buy (excluded) + Post-XD full sell (retained claim)
    trading_svc.structured_execute_trade(
        symbol="AAPL",
        asset_type="Stock",
        action="buy",
        units=10.0,
        price=100.0,
        currency="USD",
        exchange_rate=35.5,
        date="2026-04-01",
    )
    trading_svc.structured_execute_trade(
        symbol="AAPL",
        asset_type="Stock",
        action="buy",
        units=5.0,
        price=100.0,
        currency="USD",
        exchange_rate=35.5,
        date="2026-04-12",
    )
    trading_svc.structured_execute_trade(
        symbol="PTT",
        asset_type="Stock",
        action="buy",
        units=200.0,
        price=30.0,
        currency="THB",
        date="2026-04-01",
    )
    trading_svc.structured_execute_trade(
        symbol="PTT",
        asset_type="Stock",
        action="sell",
        units=200.0,
        price=30.0,
        currency="THB",
        date="2026-04-13",
    )

    loaded_after_trades = repo.load_state("default")
    assert isinstance(loaded_after_trades.fx_rates["USDTHB"], float)
    assert loaded_after_trades.fx_rates["USDTHB"] == pytest.approx(35.5)

    # Supply serialized dict dividend schedules spanning USD and TH stocks
    dict_schedules = [
        {
            "ticker": "AAPL",
            "market": "US",
            "currency": "USD",
            "ex_date": "2026-04-10",
            "pay_date": "2026-04-25",
            "dps_native": "2.00",
            "withholding_tax_pct": 15.0,
        },
        {
            "ticker": "PTT",
            "market": "TH",
            "currency": "THB",
            "ex_date": "2026-04-10",
            "pay_date": "2026-04-25",
            "dps_native": 1.0,
            "withholding_tax_pct": "10",
        },
    ]

    perf_svc = PortfolioPerformanceService(
        repo=repo,
        perf_repo=perf_repo,
        price_provider=price_port,
        dividend_provider=_StaticDividendPort(),
    )

    msg = perf_svc.record_performance_snapshot(
        refresh_prices=False,
        portfolio_id="default",
        dividend_schedules=dict_schedules,
        as_of_date="2026-04-15",
    )
    assert msg.startswith("[PERF SNAPSHOT]")
    assert not msg.startswith("Error:")

    # Expected accrued receivable:
    # - AAPL: 10 pre-XD shares * $2.00 * 0.85 = $17.00 USD * 35.5 = 603.50 THB
    # - PTT: 200 pre-XD shares (retained despite full post-XD sell) * 1.00 * 0.90 = 180.00 THB
    # - Total Accrued = 783.50 THB
    history = perf_repo.read_history(portfolio_id="default")
    assert len(history) == 1
    row = history[0]
    assert Decimal(row["Accrued_Dividend_Receivable_THB"]) == Decimal("783.50")
    assert Decimal(row["Economic_NAV_THB"]) == (
        Decimal(row["Total_NAV_THB"]) + Decimal("783.50")
    ).quantize(Decimal("0.01"))

    # Verify raw CSV header & columns on disk
    csv_path = get_performance_filepath("default")
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames is not None
        assert "Accrued_Dividend_Receivable_THB" in reader.fieldnames
        assert "Economic_NAV_THB" in reader.fieldnames
        raw_rows = list(reader)
    assert len(raw_rows) == 1
    assert raw_rows[0]["Accrued_Dividend_Receivable_THB"] == "783.50"
    assert raw_rows[0]["Economic_NAV_THB"] == row["Economic_NAV_THB"]

    # Settlement on pay_date clears receivable with zero double counting
    msg_paid = perf_svc.record_performance_snapshot(
        refresh_prices=False,
        portfolio_id="default",
        dividend_schedules=dict_schedules,
        as_of_date="2026-04-25",
    )
    assert msg_paid.startswith("[PERF SNAPSHOT]")
    history_after = perf_repo.read_history(portfolio_id="default")
    assert len(history_after) == 2
    assert Decimal(history_after[-1]["Accrued_Dividend_Receivable_THB"]) == Decimal("0.00")
    assert Decimal(history_after[-1]["Economic_NAV_THB"]) == Decimal(history_after[-1]["Total_NAV_THB"])


def test_flow3_autopsy_to_dupont_sloan_waterfall_and_quant_signals_json(monkeypatch, tmp_vault):
    """FinancialAutopsySnapshot flows through real dupont_accrual + quant_scoring + QuantSignals JSON."""
    from tools.market import equity_quant_tool as eqt

    eqt._INFO_CACHE.clear()
    eqt._INFO_ERROR_CACHE.clear()

    resolved = ResolvedAsset(
        raw_symbol="ACME",
        provider_symbol="ACME",
        asset_class=AssetClass.STOCK_US,
        market="US",
        confidence="high",
        eligible_for_financial_autopsy=True,
    )
    periods = [
        FinancialAutopsyPeriod(
            fiscal_period_end="2025-12-31",
            total_revenue=1000.0,
            net_income=150.0,
            # (150 - 49.96) / ((1100 + 900)/2) = 100.04 / 1000 = 0.10004 (> 0.10 even though 4dp=0.1000)
            operating_cash_flow=49.96,
            free_cash_flow=40.0,
            total_assets=1100.0,
            stockholders_equity=500.0,
            operating_income=200.0,
            ebit=200.0,
            gross_profit=450.0,
            current_assets=400.0,
            current_liabilities=200.0,
            total_debt=250.0,
            long_term_debt=150.0,
            interest_expense=10.0,
            tax_expense=40.0,
            income_before_tax=190.0,
            shares_outstanding=100.0,
            payout_ratio_pct=20.0,
            source_tier="filing_authoritative",
            period_type="annual",
        ),
        FinancialAutopsyPeriod(
            fiscal_period_end="2024-12-31",
            total_revenue=850.0,
            net_income=120.0,
            operating_cash_flow=130.0,
            free_cash_flow=100.0,
            total_assets=900.0,
            stockholders_equity=450.0,
            operating_income=160.0,
            ebit=160.0,
            gross_profit=370.0,
            current_assets=350.0,
            current_liabilities=190.0,
            total_debt=260.0,
            long_term_debt=160.0,
            shares_outstanding=100.0,
            source_tier="filing_authoritative",
            period_type="annual",
        ),
    ]
    autopsy_snap = FinancialAutopsySnapshot(
        ticker="ACME",
        provider_symbol="ACME",
        market="US",
        currency="USD",
        unit="raw",
        retrieval_timestamp="2026-11-01 12:00:00",
        periods=periods,
        current_pe=18.0,
        source_tier="filing_authoritative",
    )
    autopsy_res = FinancialAutopsyFetchResult(asset=resolved, status="success", snapshot=autopsy_snap)

    dates = pd.date_range("2026-09-25", periods=25, freq="B")
    df_1y = pd.DataFrame(
        {
            "Open": [24.0 + i * 0.1 for i in range(len(dates))],
            "High": [24.5 + i * 0.1 for i in range(len(dates))],
            "Low": [23.5 + i * 0.1 for i in range(len(dates))],
            "Close": [24.0 + i * 0.1 for i in range(len(dates))],
            "Volume": [1_500_000.0 for _ in range(len(dates))],
        },
        index=dates,
    )
    valid_q = PriceSeriesQuality(trading_days=120, is_valid=True)

    # Monkeypatch ONLY external I/O fetchers; leave dupont_accrual, quant_scoring,
    # signal_audit, evidence_manifest_builder, equity_rules_engine completely unmocked.
    monkeypatch.setattr(eqt, "resolve_asset", lambda *args, **kwargs: resolved)
    monkeypatch.setattr(eqt, "get_financial_autopsy", lambda *args, **kwargs: autopsy_res)
    monkeypatch.setattr(
        eqt,
        "_yf_info",
        lambda *args, **kwargs: {
            "shortName": "Acme Corp",
            "sector": "Technology",
            "currentPrice": 26.4,
            "trailingPE": 18.0,
            "priceToBook": 5.2,
            "enterpriseToEbitda": 12.0,
            "returnOnEquity": 0.30,
            "profitMargins": 0.15,
            "trailingAnnualDividendYield": 0.015,
            "debtToEquity": 50.0,
            "currentRatio": 2.0,
            "targetMeanPrice": 32.0,
            "targetHighPrice": 35.0,
            "targetLowPrice": 28.0,
            "averageVolume": 1_500_000,
            "averageVolume10Day": 1_400_000,
            "fiftyDayAverage": 25.0,
            "twoHundredDayAverage": 22.0,
            "sharesOutstanding": 100.0,
            "numberOfAnalystOpinions": 10,
            "totalCash": 100.0,
            "totalDebt": 250.0,
            "totalRevenue": 1000.0,
            "operatingMargins": 0.20,
            "bookValue": 5.0,
        },
    )
    monkeypatch.setattr(eqt, "_get_price_history", lambda *args, **kwargs: df_1y)
    monkeypatch.setattr(eqt, "compute_beta", lambda *args, **kwargs: (1.15, valid_q))
    monkeypatch.setattr(eqt, "compute_volatility", lambda *args, **kwargs: (22.5, valid_q))
    monkeypatch.setattr(eqt, "compute_mdd", lambda *args, **kwargs: (-18.0, valid_q))
    monkeypatch.setattr(
        eqt,
        "compute_technical_indicators",
        lambda *args, **kwargs: ({"rsi_14": 58.0, "macd_signal": "bullish"}, valid_q),
    )
    monkeypatch.setattr(eqt, "compute_price_percentile", lambda *args, **kwargs: (62.0, 0.5, valid_q))
    monkeypatch.setattr(eqt, "fetch_peer_metrics", lambda *args, **kwargs: [])
    monkeypatch.setattr(eqt, "fetch_earnings_revision_data", lambda *args, **kwargs: (None, None))
    monkeypatch.setattr(eqt, "load_latest_macro_observables", lambda: {})
    monkeypatch.setattr(
        eqt,
        "compute_canonical_insider_conviction",
        lambda *args, **kwargs: (
            InsiderConviction(
                open_market_p_count_90d=2,
                unflagged_s_count_90d=0,
                status="moderate_buying",
                data_status="available",
            ),
            [],
        ),
    )
    monkeypatch.setattr(eqt, "load_latest_macro_regime_context", lambda: None)
    monkeypatch.setattr(
        "tools.market.guidance_engine.extract_verified_earnings_guidance",
        lambda *args, **kwargs: (None, []),
    )
    # Fail loudly if any hidden yfinance network call escapes isolation
    monkeypatch.setattr(
        "yfinance.Ticker",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected yfinance.Ticker call")),
    )

    raw_json = eqt.compute_equity_quant_signals.func(ticker="ACME", market="US")
    signals = QuantSignals.model_validate_json(raw_json)

    assert signals.ticker == "ACME"
    assert signals.market == "US"

    # Real DuPont 3-factor precision & honest published factor reconstruction error
    dp = signals.dupont_breakdown
    assert dp is not None
    assert dp["net_profit_margin_pct"] == pytest.approx(15.0)
    assert dp["asset_turnover"] == pytest.approx(0.9091, abs=1e-4)
    assert dp["equity_multiplier"] == pytest.approx(2.2)
    assert dp["roe_pct"] == pytest.approx(30.0)
    assert dp["published_roe_reconstructed_pct"] == pytest.approx(30.0003, abs=1e-4)
    assert dp["closure_error"] == pytest.approx(0.0003, abs=1e-5)
    assert dp["internal_closure_error"] == pytest.approx(0.0, abs=1e-8)

    # Real Sloan unrounded threshold check (+0.10004 > +0.10 triggers manipulation warning despite 0.1000 display)
    sl = signals.sloan_accrual
    assert sl is not None
    assert sl["assets_basis"] == "two_period_average"
    assert sl["sloan_accrual_ratio"] == pytest.approx(0.1000, abs=1e-4)
    assert sl["raw_sloan_accrual_ratio"] == pytest.approx(0.10004, abs=1e-6)
    assert sl["earnings_quality_rating"] == "EARNINGS_MANIPULATION_WARNING"

    # Real waterfall weights & composite score reconciliation
    wf = signals.composite_score_waterfall
    assert wf is not None
    assert signals.composite_score is not None
    assert wf["composite_score"] == pytest.approx(signals.composite_score, abs=1e-4)
    assert sum(wf["base_weight_normalized_pct"].values()) == pytest.approx(100.0, abs=0.05)
    assert (wf["pillar_contributions_sum"] + wf["rounding_reconciliation_delta"]) == pytest.approx(
        wf["composite_score"], abs=1e-4
    )

    # Real SmartMoney + scorecard propagated through full QuantSignals JSON
    assert signals.smart_money_flags is not None
    assert signals.smart_money_flags.insider_buy_count_90d == 2
    assert signals.deterministic_scorecard is not None
    assert 1.0 <= signals.deterministic_scorecard.core_conviction_score <= 10.0
    assert signals.metric_basis["dupont_3factor"].startswith("annual:")
    assert signals.metric_basis["sloan_accrual"].startswith("two_period_average:")

