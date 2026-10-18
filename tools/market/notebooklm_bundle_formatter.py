"""Deterministic Markdown Formatter for Equity NotebookLM Research Export.

Produces structured Markdown documents optimized for NotebookLM's
grounding, indexing, and citation search engine without losing numeric or lineage precision.
Follows deterministic Python math invariants and preserves all institutional metrics.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from application.equity.notebooklm_export_ports import EquityCorpusSnapshot


def _fmt_num(
    val: Any,
    fmt: str = ",.2f",
    prefix: str = "",
    suffix: str = "",
    default: str = "N/A",
) -> str:
    """Format numeric value safely without raising TypeError on None or unsupported types."""
    if val is None or val == "N/A" or val == "":
        return default
    try:
        fval = float(val)
        formatted = f"{fval:{fmt}}"
        return f"{prefix}{formatted}{suffix}"
    except (ValueError, TypeError):
        return f"{prefix}{val}{suffix}"


def format_research_guide(snapshot: EquityCorpusSnapshot) -> str:
    """Part 00: Research Guide, Corpus Inventory, and Citation Handbook."""
    counts = snapshot.metadata.get("counts", {})
    warnings = snapshot.metadata.get("warnings", [])

    lines = [
        f"# {snapshot.ticker} Equity Research Companion Guide & Evidence Catalog",
        "",
        f"- **Ticker:** `{snapshot.ticker}`",
        f"- **Market:** `{snapshot.market}`",
        f"- **Snapshot Timestamp (UTC):** `{snapshot.snapshot_at}`",
        f"- **Historical Sidecars Retained:** `{counts.get('historical_analyses', len(snapshot.historical_analyses))}`",
        f"- **Earnings Call Transcripts/Notes:** `{counts.get('earnings_calls', len(snapshot.earnings_calls))}`",
        f"- **News & Curated References:** `{counts.get('news_items', len(snapshot.news))}`",
        f"- **User & Vault Notes:** `{counts.get('user_notes', len(snapshot.notes))}`",
        "",
        "---",
        "",
        "## 1. System Role & Scope Invariant",
        f"This Notebook serves exclusively as a **Research Companion** for {snapshot.ticker}.",
        "- **What this Notebook does:** Deep-dives into company fundamentals, audits valuation & reverse DCF growth assumptions, interrogates insider buying/selling, evaluates macro headwinds, and provides grounded citations against SEC filings and earnings notes.",
        "- **What this Notebook NEVER does:** It does not trigger automated buy/sell orders, generate podcast audio, broadcast alerts to Discord, or modify the live conviction score of the Trinity Wealth Engine.",
        "",
        "## 2. Recommended Research Questions",
        f"1. *'What are the core drivers behind the current investment stance for {snapshot.ticker}?'*",
        "2. *'What implied revenue growth and operating margin is the current stock price discounting according to the Reverse DCF?'*",
        "3. *'Are there recent Form 4 insider transactions, and what is the net insider conviction score?'*",
        "4. *'How do the company operating KPIs compare against sector peers and historical medians?'*",
        "5. *'What specific falsifiers or invalidation triggers would break our core thesis?'*",
        "6. *'What did management emphasize in the latest earnings call regarding guidance and capex?'*",
        "",
        "## 3. Data Inventory & Evidence Sources",
        "| Document Part | Description | Status |",
        "| :--- | :--- | :--- |",
        f"| **01-latest-analysis.md** | 4-Pillar Scorecard, Reverse DCF, Tactical Setup, Thesis | {'Available' if snapshot.latest_analysis else 'Missing'} |",
        f"| **02-analysis-history.md** | Chronological analysis sidecars and rating evolutions | {len(snapshot.historical_analyses)} records |",
        f"| **03-financials.md** | Financial statements, SEC 10-K/10-Q, Capital Quality | {'Available' if snapshot.sec_financials or snapshot.financials or (snapshot.latest_analysis and snapshot.latest_analysis.get('quant_signals')) else 'Unavailable'} |",
        f"| **04-insider-ownership.md** | Form 4 transactions, insider conviction, ownership structure | {'Available' if snapshot.sec_insider_trades or (snapshot.latest_analysis and snapshot.latest_analysis.get('quant_signals')) else 'Unavailable'} |",
        f"| **05-earnings-calls.md** | Earnings call notes and guidance takeaways | {len(snapshot.earnings_calls)} records |",
        f"| **06-news.md** | Curated news funnel and regulatory headlines | {len(snapshot.news)} items |",
        f"| **07-market-derivatives.md** | Nasdaq consensus, Cboe options chain, Max Pain, IV | {'Available' if snapshot.options_chain or snapshot.consensus or (snapshot.latest_analysis and snapshot.latest_analysis.get('quant_signals')) else 'Unavailable'} |",
        f"| **08-price-and-notes.md** | OHLCV market snapshot, portfolio position context, user notes | {'Available' if snapshot.ohlcv_summary or snapshot.portfolio_holding or snapshot.latest_analysis else 'Empty'} |",
        "| **09-structured-appendix.md** | Machine-readable JSON data payload | Loaded |",
        "",
    ]

    if warnings:
        lines.append("## 4. Warnings & Data Gaps")
        for w in warnings:
            lines.append(f"- ⚠️ {w}")
        lines.append("")

    return "\n".join(lines)


def format_latest_analysis(snapshot: EquityCorpusSnapshot) -> str:
    """Part 01: Latest Executive Investment Thesis, Scorecard, and Valuation."""
    latest = snapshot.latest_analysis or {}
    quant = latest.get("quant_signals", {})

    # Robust multi-schema compatibility
    scorecard = quant.get("deterministic_scorecard") or quant.get("scorecard") or {}
    rev_dcf = quant.get("reverse_dcf_result") or {}
    dcf_res = quant.get("dcf_result") or {}
    legacy_dcf = quant.get("dcf_valuation") or {}
    tactical = quant.get("tactical_setup") or {}
    kpis = quant.get("sector_kpis") or quant.get("sector_specific_kpis") or {}
    piotroski = quant.get("piotroski_breakdown") or {}
    port_context = (
        snapshot.portfolio_holding
        or latest.get("portfolio_context")
        or quant.get("portfolio_context")
        or {}
    )
    thesis_delta = latest.get("thesis_delta") or {}
    falsifiers = quant.get("thesis_falsifiers") or latest.get("thesis_falsifiers") or []

    # Scorecard fields
    action_stance = scorecard.get("action_stance") or "UNKNOWN"
    stance_reason = scorecard.get("action_stance_reason") or scorecard.get("stance_reason") or "N/A"
    core_conviction = scorecard.get("core_conviction_score", "N/A")
    biz_conviction = scorecard.get("business_conviction_score", core_conviction)
    invest_conviction = scorecard.get("investment_conviction_score", core_conviction)
    exec_readiness = scorecard.get("execution_readiness_score", "N/A")
    macro_alignment = scorecard.get("macro_alignment_status") or "neutral"
    macro_penalty = scorecard.get("macro_headwind_penalty", 0) or 0
    macro_note = scorecard.get("macro_alignment_note", "")

    # Pillar scores
    pillar1_score = scorecard.get("fundamental_quality_score") or scorecard.get("fundamental_score") or "N/A"
    pillar2_score = scorecard.get("valuation_margin_score") or scorecard.get("valuation_score") or "N/A"
    pillar3_score = (
        scorecard.get("guidance_expectation_score")
        or scorecard.get("analyst_expectations_score")
        or scorecard.get("expectations_score")
        or "N/A"
    )
    pillar4_score = exec_readiness

    # Fundamental metrics
    f_score = piotroski.get("f_score") or quant.get("piotroski_f_score") or "N/A"
    roic = quant.get("roic_pct", "N/A")
    wacc = (
        rev_dcf.get("fixed_parameters", {}).get("wacc_pct")
        or dcf_res.get("wacc_pct")
        or legacy_dcf.get("wacc_pct")
        or quant.get("wacc_pct", "N/A")
    )
    spread = (
        quant.get("economic_spread_pct")
        or (
            round(float(roic) - float(wacc), 2)
            if isinstance(roic, (int, float)) and isinstance(wacc, (int, float))
            else "N/A"
        )
    )
    gross_margin_stab = (
        "Improved" if piotroski.get("delta_gross_margin_improved")
        else quant.get("gross_margin_stability", "Stable")
    )

    # Valuation & Yield metrics
    adj_value = quant.get("peer_relative_score") or quant.get("sector_adjusted_value_score") or quant.get("value_score") or "N/A"
    net_yield = quant.get("net_shareholder_yield_pct") or quant.get("fcf_yield_pct") or "N/A"
    buyback_yield = quant.get("net_buyback_yield_pct", "N/A")
    div_yield = quant.get("dividend_yield_pct", 0.0)

    # Guidance & Revisions
    beat_freq = quant.get("earnings_beat_frequency", "N/A")
    revision_trend = (
        f"{quant.get('eps_revision_net_30d', 0)} net revisions (30D change: {quant.get('eps_estimate_change_30d_pct', 0)}%)"
        if "eps_revision_net_30d" in quant
        else quant.get("guidance_revision_trend", "N/A")
    )

    # Tactical setup
    stage = tactical.get("price_stage") or tactical.get("stage") or "N/A"
    setup_status = (
        tactical.get("pullback_entry_status")
        or tactical.get("breakout_entry_status")
        or tactical.get("setup")
        or "N/A"
    )
    setup_reason = tactical.get("setup_reason", "")
    current_price = (
        quant.get("raw_analysis_price")
        or tactical.get("current_price")
        or legacy_dcf.get("current_price")
        or "N/A"
    )

    # Reverse DCF details
    target_intrinsic = (
        rev_dcf.get("intrinsic_value_today")
        or legacy_dcf.get("intrinsic_value")
        or dcf_res.get("scenarios", {}).get("base", {}).get("target_price")
        or "N/A"
    )
    target_12m = (
        rev_dcf.get("target_price_12m")
        or dcf_res.get("scenarios", {}).get("base", {}).get("target_price")
        or "N/A"
    )
    upside_pct = (
        rev_dcf.get("upside_12m_pct")
        or legacy_dcf.get("upside_pct")
        or dcf_res.get("scenarios", {}).get("base", {}).get("upside_pct")
        or "N/A"
    )
    implied_growth = (
        rev_dcf.get("market_implied_growth_pct")
        or legacy_dcf.get("implied_revenue_growth_5y_pct")
        or "N/A"
    )
    implied_margin = (
        rev_dcf.get("market_implied_margin_pct")
        or legacy_dcf.get("implied_operating_margin_pct")
        or "N/A"
    )
    terminal_growth = (
        rev_dcf.get("fixed_parameters", {}).get("terminal_growth_pct")
        or legacy_dcf.get("terminal_growth_pct")
        or "N/A"
    )
    consensus_price = rev_dcf.get("consensus_target_price", "N/A")
    consensus_low = rev_dcf.get("consensus_target_low")
    consensus_high = rev_dcf.get("consensus_target_high")
    analyst_count = rev_dcf.get("analyst_count", "N/A")

    lines = [
        f"# {snapshot.ticker} Latest Equity Investment Thesis & Scorecard",
        "",
        f"- **Company Name:** {latest.get('company_name', snapshot.ticker)}",
        f"- **Analysis Date:** {latest.get('analysis_date', 'N/A')}",
        f"- **Evaluated At (UTC):** {latest.get('evaluated_at', 'N/A')}",
        f"- **Generated By:** `{latest.get('generated_by', 'equity_intel')}`",
        "",
        "---",
        "",
        "## 1. Executive Action Stance & 4-Pillar Scorecard",
        f"- **Action Stance:** `{action_stance}`",
        f"- **Stance Reason:** {stance_reason}",
        f"- **Core Business Conviction:** `{biz_conviction} / 10`",
        f"- **Investment Conviction:** `{invest_conviction} / 10`",
        f"- **Execution Readiness:** `{exec_readiness} / 10`",
        f"- **Macro Alignment:** `{macro_alignment}` (Penalty: -{macro_penalty} pts)",
    ]
    if macro_note:
        lines.append(f"  - *Note:* {macro_note}")

    lines.extend([
        "",
        "### 4-Pillar Score Breakdown",
        f"1. **Fundamental Quality (Pillar 1):** `{pillar1_score} / 100`",
        f"   - Piotroski F-Score: `{f_score} / 9`",
        f"   - ROIC: `{roic}%` (WACC: `{wacc}%` | Spread: `{spread}%`)",
        f"   - Gross Margin Stability: `{gross_margin_stab}`",
        f"2. **Valuation Attractiveness (Pillar 2):** `{pillar2_score} / 100`",
        f"   - Sector Adjusted Value Score: `{adj_value} / 100`",
        f"   - Net Shareholder Yield: `{net_yield}%` (Buyback: `{buyback_yield}%` | Div: `{div_yield}%`)",
        f"3. **Guidance & Market Expectations (Pillar 3):** `{pillar3_score} / 100`",
        f"   - Earnings Beat Frequency: `{beat_freq}`",
        f"   - Guidance Revision Trend: `{revision_trend}`",
        f"4. **Execution & Technical Momentum (Pillar 4):** `{pillar4_score} / 100`",
        f"   - Market Stage: `{stage}`",
        f"   - Tactical Setup: `{setup_status}` {f'({setup_reason})' if setup_reason else ''}",
        f"   - Key Levels: Support `${tactical.get('key_support_level', 'N/A')}` | Resistance `${tactical.get('key_resistance_level', 'N/A')}` | Invalidation Stop `${tactical.get('invalidation_stop_loss', 'N/A')}`",
        f"   - Buy Zone: `${tactical.get('buy_zone_min', 'N/A')} - ${tactical.get('buy_zone_max', 'N/A')}` (R/R: `{tactical.get('tactical_risk_reward_ratio', 'N/A')}`)",
        "",
        "---",
        "",
        "## 2. Reverse DCF & Intrinsic Value Forensic",
        f"- **Target Intrinsic Price Today:** `${target_intrinsic}`",
        f"- **12-Month Target Price:** `${target_12m}`",
        f"- **DCF Upside / Downside (12M):** `{upside_pct}%`",
        f"- **Market Price Discounted:** `${current_price}`",
        f"- **Implied Revenue Growth (Next 5Y):** `{implied_growth}%`",
        f"- **Implied Operating Margin:** `{implied_margin}%`",
        f"- **WACC Applied:** `{wacc}%` | **Terminal Growth:** `{terminal_growth}%`",
    ])

    if consensus_price != "N/A":
        lines.append(
            f"- **Wall Street Consensus Target:** `${consensus_price}` "
            f"(Range: `${consensus_low}` - `${consensus_high}`, Analysts: `{analyst_count}`)"
        )
    lines.append("")

    if kpis:
        sector_label = kpis.get("sector_name") or kpis.get("sector") or quant.get("peer_sector", "N/A")
        lines.extend([
            "---",
            "",
            "## 3. Sector-Specific Operating KPIs",
            f"- **Sector Classification:** `{sector_label}`",
        ])
        if kpis.get("sector_health_score") is not None:
            lines.append(f"- **Sector Health Score:** `{kpis.get('sector_health_score')} / 100`")
        kpi_items = kpis.get("kpis")
        if isinstance(kpi_items, list):
            for item in kpi_items:
                if isinstance(item, dict):
                    name = item.get("name", "KPI")
                    val = item.get("value", "N/A")
                    unit = item.get("unit", "")
                    unit_str = "%" if unit == "percent" else (f" {unit}" if unit and unit != "%" else unit)
                    bench = item.get("benchmark")
                    bench_str = f" (Benchmark: `{bench}`)" if bench else ""
                    lines.append(f"- **{name}:** `{val}{unit_str}`{bench_str}")
        else:
            for k, v in kpis.items():
                if k not in ("sector", "sector_name", "as_of_date", "status", "flags", "sector_health_score"):
                    lines.append(f"- **{k.replace('_', ' ').title()}:** `{v}`")
        lines.append("")

    if port_context:
        is_held = bool(port_context.get("is_held"))
        hint = port_context.get("position_action_hint", "N/A")
        units_str = _fmt_num(port_context.get("units"), fmt=",.4f", default="0.0")
        avg_cost_str = _fmt_num(port_context.get("avg_cost"), fmt=",.4f", prefix="$", default="N/A")
        weight_str = _fmt_num(port_context.get("current_weight_pct"), fmt=",.2f", suffix="%", default="0.0%")
        pnl_str = _fmt_num(port_context.get("unrealized_pnl_pct"), fmt=",.2f", suffix="%", default="N/A")

        lines.extend([
            "---",
            "",
            "## 4. Portfolio Position Context",
            f"- **Holding Status:** `{'Currently Held' if is_held else 'Not Held'}`",
            f"- **Position Action Hint:** `{hint}`",
            f"- **Units Held:** `{units_str}` @ `{avg_cost_str}`",
            f"- **Portfolio Weight:** `{weight_str}`",
            f"- **Unrealized PnL:** `{pnl_str}`",
            "",
        ])

    if thesis_delta:
        lines.extend([
            "---",
            "",
            "## 5. Thesis Evolution Delta",
            f"- **Previous Analysis Date:** `{thesis_delta.get('previous_analysis_date', 'N/A')}`",
            f"- **Stance Change:** `{thesis_delta.get('previous_action_stance', 'N/A')}` ➔ `{thesis_delta.get('current_action_stance', action_stance)}`",
            f"- **Price Change Since Previous:** `{thesis_delta.get('price_change_pct', 0)}%`",
            f"- **Conviction Score Delta:** `{thesis_delta.get('conviction_score_delta', 0)}`",
            f"- **Target Price Delta:** `{thesis_delta.get('target_price_change_pct', 0)}%`",
            f"- **Breached Falsifiers:** {len(thesis_delta.get('breached_falsifiers', []))}",
            "",
        ])

    if falsifiers:
        lines.extend([
            "---",
            "",
            "## 6. Thesis Falsifiers & Invalidation Triggers",
        ])
        for f in falsifiers:
            name = f.get("metric_name") or f.get("falsifier_id", "Falsifier")
            cond = f.get("condition", "N/A")
            expl = f.get("narrative_explanation") or f.get("explanation", "")
            lines.append(f"- **{name}:** `{cond}`")
            if expl:
                lines.append(f"  - *Impact:* {expl}")
        lines.append("")

    lines.extend([
        "---",
        "",
        "## 7. Narrative Investment Thesis",
        "### Base Case Summary",
        f"{latest.get('base_case_summary', 'No executive summary provided.')}",
        "",
        "### Detailed Narrative Analysis",
        f"{latest.get('narrative_analysis', 'No detailed narrative analysis provided.')}",
        "",
    ])

    return "\n".join(lines)


def format_historical_analyses(snapshot: EquityCorpusSnapshot) -> str:
    """Part 02: Historical Analysis Sidecars."""
    lines = [
        f"# {snapshot.ticker} Analysis History & Thesis Evolution",
        "",
        f"Total retained sidecars: {len(snapshot.historical_analyses)}",
        "",
    ]

    if not snapshot.historical_analyses:
        lines.append("No historical analysis records available beyond the latest report.")
        return "\n".join(lines)

    for idx, item in enumerate(snapshot.historical_analyses, start=1):
        quant = item.get("quant_signals", {})
        scorecard = quant.get("deterministic_scorecard") or quant.get("scorecard") or {}
        piotroski = quant.get("piotroski_breakdown") or {}
        stance = scorecard.get("action_stance") or item.get("market_sentiment", "N/A")
        conviction = (
            scorecard.get("core_conviction_score")
            or scorecard.get("business_conviction_score")
            or item.get("composite_score")
            or "N/A"
        )
        f_score = piotroski.get("f_score") or quant.get("piotroski_f_score", "N/A")
        roic = quant.get("roic_pct", "N/A")

        lines.extend([
            f"## Record {idx}: {item.get('analysis_date', 'N/A')} ({str(item.get('evaluated_at', 'N/A'))[:16]})",
            f"- **Stance:** `{stance}`",
            f"- **Conviction:** `{conviction} / 10`",
            f"- **F-Score:** `{f_score} / 9` | **ROIC:** `{roic}%`",
            f"- **Base Case Summary:** {item.get('base_case_summary', 'N/A')}",
            "",
            "---",
            "",
        ])

    return "\n".join(lines)


def format_financials(snapshot: EquityCorpusSnapshot) -> str:
    """Part 03: Financial Statements & SEC EDGAR Filings."""
    sec_fin = snapshot.sec_financials
    fin = snapshot.financials
    quant = (snapshot.latest_analysis or {}).get("quant_signals", {})
    piotroski = quant.get("piotroski_breakdown") or {}

    lines = [
        f"# {snapshot.ticker} Financial Statements & Regulatory Filings",
        "",
    ]

    if sec_fin:
        lines.extend([
            "## 1. SEC EDGAR 10-K / 10-Q Financial Statements",
            f"- **Source:** SEC EDGAR (CIK: `{sec_fin.get('cik', 'N/A')}`)",
            f"- **Entity Name:** `{sec_fin.get('entity_name', snapshot.ticker)}`",
            f"- **Latest Filing / As-of Date:** `{sec_fin.get('as_of_date') or sec_fin.get('latest_filing_date', 'N/A')}`",
            "",
        ])

        # Summary Metrics if present
        if sec_fin.get("revenue_usd") is not None or sec_fin.get("operating_cash_flow_usd") is not None:
            lines.extend([
                "### Key Reported Summary Metrics (USD)",
                f"- **Revenue:** `{_fmt_num(sec_fin.get('revenue_usd'), prefix='$', default='N/A')}`",
                f"- **Operating Cash Flow:** `{_fmt_num(sec_fin.get('operating_cash_flow_usd'), prefix='$', default='N/A')}`",
                f"- **Capital Expenditures:** `{_fmt_num(sec_fin.get('capex_usd'), prefix='$', default='N/A')}`",
                f"- **Free Cash Flow:** `{_fmt_num(sec_fin.get('free_cash_flow_usd'), prefix='$', default='N/A')}` (Margin: `{_fmt_num(sec_fin.get('free_cash_flow_margin'), suffix='%', default='N/A')}`)",
                f"- **Long-Term Debt:** `{_fmt_num(sec_fin.get('long_term_debt_usd'), prefix='$', default='N/A')}`",
                f"- **Debt to OCF Ratio:** `{_fmt_num(sec_fin.get('debt_to_ocf_ratio'), suffix='x', default='N/A')}`",
                f"- **Shares Outstanding:** `{_fmt_num(sec_fin.get('shares_outstanding'), fmt=',.0f', default='N/A')}`",
                "",
            ])

        statements = sec_fin.get("statements", {})
        if statements:
            for name, data in statements.items():
                lines.append(f"### {name.title().replace('_', ' ')}")
                if isinstance(data, dict):
                    lines.append("| Metric | Value | Currency |")
                    lines.append("| :--- | :--- | :--- |")
                    for k, v in data.items():
                        lines.append(f"| {k} | {v} | USD |")
                    lines.append("")

    elif fin:
        lines.extend([
            "## 1. Standard Financial Statements",
            f"- **Fiscal Year / Period:** `{fin.get('period', 'N/A')}`",
            "",
        ])
        for statement_type in ("income_statement", "balance_sheet", "cash_flow"):
            data = fin.get(statement_type)
            if data and isinstance(data, dict):
                lines.append(f"### {statement_type.replace('_', ' ').title()}")
                lines.append("| Line Item | Value |")
                lines.append("| :--- | :--- |")
                for k, v in data.items():
                    lines.append(f"| {k} | {v} |")
                lines.append("")

    # Fallback to rich quantitative financial statement tables from quant_signals
    if quant:
        lines.extend([
            "## Quantitative Financial Performance & Capital Health",
            "",
            "### Operating Performance & Cash Flow Quality",
            "| Metric | Value | Institutional Interpretation |",
            "| :--- | :--- | :--- |",
            f"| **Revenue Growth (YoY)** | `{quant.get('revenue_growth_yoy_pct', 'N/A')}%` | Top-line expansion rate |",
            f"| **Net Income Growth (YoY)** | `{quant.get('net_income_growth_yoy_pct', 'N/A')}%` | GAAP bottom-line earnings growth |",
            f"| **ROIC (Return on Invested Capital)** | `{quant.get('roic_pct', 'N/A')}%` | Capital deployment efficiency |",
            f"| **Free Cash Flow Margin** | `{quant.get('fcf_margin_pct', 'N/A')}%` | FCF as % of revenues |",
            f"| **Free Cash Flow Yield** | `{quant.get('fcf_yield_pct', 'N/A')}%` | Cash generated per market cap |",
            f"| **FCF 3-Year CAGR** | `{quant.get('fcf_cagr_3y', 'N/A')}%` | Multi-year cash generation trajectory |",
            f"| **OCF to Net Income Ratio** | `{quant.get('ocf_to_net_income', 'N/A')}` | Earnings quality / accrual check |",
            "",
            "### Solvency & Balance Sheet Strength",
            "| Solvency Metric | Value | Benchmark / Health |",
            "| :--- | :--- | :--- |",
            f"| **Net Debt / EBITDA** | `{quant.get('net_debt_ebitda', 'N/A')}x` | Leverage multiple (< 2.5x healthy) |",
            f"| **Interest Coverage** | `{quant.get('interest_coverage', 'N/A')}x` | Operating profit to interest expense |",
            f"| **Debt to Equity (D/E)** | `{quant.get('de_ratio_pct', 'N/A')}%` | Capital structure ratio |",
            f"| **Current Ratio** | `{quant.get('current_ratio', 'N/A')}x` | Short-term liquidity buffer |",
            f"| **Debt Quality Score** | `{quant.get('debt_quality_score', 'N/A')} / 100` | Proprietary debt distress model |",
            f"| **Solvency Score** | `{quant.get('solvency_score', 'N/A')} / 100` | Aggregate balance sheet score |",
            "",
        ])

        if piotroski:
            lines.extend([
                f"### Piotroski F-Score Structural Forensic (`{piotroski.get('f_score', 'N/A')} / 9`)",
                "| Audit Dimension | Test Criteria | Result |",
                "| :--- | :--- | :--- |",
                f"| **Profitability** | Positive Return on Assets (ROA > 0) | `{'PASS' if piotroski.get('roa_positive') else 'FAIL'}` |",
                f"| **Profitability** | Positive Operating Cash Flow (CFO > 0) | `{'PASS' if piotroski.get('cfo_positive') else 'FAIL'}` |",
                f"| **Profitability** | Improving ROA YoY | `{'PASS' if piotroski.get('delta_roa_positive') else 'FAIL'}` |",
                f"| **Cash Quality** | Accrual Quality (CFO > ROA) | `{'PASS' if piotroski.get('accrual_quality') else 'FAIL'}` |",
                f"| **Leverage** | Decreased Long-Term Leverage YoY | `{'PASS' if piotroski.get('delta_leverage_improved') else 'FAIL'}` |",
                f"| **Liquidity** | Improved Current Ratio YoY | `{'PASS' if piotroski.get('delta_liquidity_improved') else 'FAIL'}` |",
                f"| **Dilution** | No Ordinary Share Dilution | `{'PASS' if piotroski.get('no_share_dilution') else 'FAIL'}` |",
                f"| **Operating Efficiency** | Gross Margin Expansion YoY | `{'PASS' if piotroski.get('delta_gross_margin_improved') else 'FAIL'}` |",
                f"| **Operating Efficiency** | Asset Turnover Ratio Expansion YoY | `{'PASS' if piotroski.get('delta_asset_turnover_improved') else 'FAIL'}` |",
                "",
            ])

    if not sec_fin and not fin and not quant:
        lines.append("No stored financial statement records available for this stock.")

    return "\n".join(lines)


def format_insider_ownership(snapshot: EquityCorpusSnapshot) -> str:
    """Part 04: Form 4 Insider Trades & Conviction."""
    sec_insider = snapshot.sec_insider_trades
    quant = (snapshot.latest_analysis or {}).get("quant_signals", {})
    insider_conv = quant.get("insider_conviction") or {}
    smart_money = quant.get("smart_money_flags") or {}

    lines = [
        f"# {snapshot.ticker} Insider Transactions & Management Conviction",
        "",
    ]

    if sec_insider:
        lines.extend([
            "## 1. SEC Form 4 Filings Summary",
            f"- **Net Insider Sentiment:** `{sec_insider.get('sentiment', 'NEUTRAL')}`",
            f"- **Net Buy Ratio (90D):** `{sec_insider.get('net_buy_ratio_90d', 'N/A')}`",
            f"- **Total Open Market Purchases (90D):** `{_fmt_num(sec_insider.get('p_notional_sum_90d'), prefix='$', default='$0.00')}`",
            f"- **Total Open Market Sales (90D):** `{_fmt_num(sec_insider.get('s_notional_sum_90d'), prefix='$', default='$0.00')}`",
            "",
            "### Form 4 Filings Log",
            "| Filing Date | Insider Name | Title | Transaction | Shares | Price | Value |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ])

        trades = sec_insider.get("transactions", [])
        if isinstance(trades, list):
            for t in trades[:30]:
                date_val = t.get("filing_date") or t.get("transaction_date", "N/A")
                name_val = t.get("insider_name") or t.get("reporting_owner", "N/A")
                title_val = t.get("title") or t.get("officer_title", "N/A")
                type_val = t.get("transaction_type") or t.get("transaction_code", "N/A")
                shares_val = t.get("shares", "N/A")
                price_val = t.get("price") or t.get("price_per_share", 0) or 0
                notional_val = t.get("value") or t.get("notional_usd", 0) or 0
                lines.append(
                    f"| {date_val} | {name_val} | {title_val} | {type_val} | "
                    f"{shares_val} | ${float(price_val):.2f} | ${float(notional_val):,.0f} |"
                )
        lines.append("")

    # Fallback to quant engine smart money & insider conviction
    if insider_conv or smart_money:
        lines.extend([
            "## Ownership Structure & Smart Money Conviction",
            "",
            "### Institutional & Insider Ownership Breakdown",
            f"- **Institutional Ownership:** `{smart_money.get('institutional_ownership_pct', 'N/A')}%`",
            f"- **Insider Ownership:** `{smart_money.get('insider_ownership_pct', 'N/A')}%`",
            f"- **Short Interest:** `{smart_money.get('short_interest_pct', 'N/A')}%` (Short Squeeze Risk: `{'YES' if smart_money.get('short_squeeze_risk') else 'NO'}`)",
            f"- **Overall Smart Money Signal:** `{smart_money.get('overall_smart_money_flag') or smart_money.get('insider_signal') or 'neutral'}`",
            "",
            "### 90-Day Insider Trading Activity (Quant Audited)",
            f"- **Open Market Purchases (Code P):** `{insider_conv.get('open_market_p_count_90d', 0)}` transactions (Total: `${insider_conv.get('open_market_p_value_usd_str', '0.00')}`)",
            f"- **Open Market Sales (Code S):** `{insider_conv.get('open_market_s_count_90d', 0)}` transactions (Total: `${insider_conv.get('open_market_s_value_usd_str', '0.00')}`)",
            f"- **Rule 10b5-1 Planned Sales:** `{insider_conv.get('rule_10b5_1_s_count_90d', 0)}` transactions (Total: `${insider_conv.get('rule_10b5_1_s_value_usd_str', '0.00')}`)",
            f"- **Signal Status:** `{insider_conv.get('status', 'neutral_no_signal')}` (Confidence: `{insider_conv.get('signal_confidence', 'moderate')}`)",
            "",
        ])

    if not sec_insider and not insider_conv and not smart_money:
        lines.append("No SEC Form 4 insider transaction records available.")

    return "\n".join(lines)


def format_earnings_calls(snapshot: EquityCorpusSnapshot) -> str:
    """Part 05: Earnings Call Transcripts & Summaries."""
    lines = [
        f"# {snapshot.ticker} Earnings Call Transcripts & Management Notes",
        "",
        f"Total earnings call notes: {len(snapshot.earnings_calls)}",
        "",
    ]

    if not snapshot.earnings_calls:
        lines.append("No retained earnings call notes found for this company.")
        return "\n".join(lines)

    for call in snapshot.earnings_calls:
        lines.extend([
            f"## {call.get('title', 'Earnings Call')}",
            f"- **Filename:** `{call.get('filename', 'N/A')}`",
            f"- **Recorded Date:** `{call.get('modified_at', 'N/A')}`",
            "",
            f"{call.get('content', '')}",
            "",
            "---",
            "",
        ])

    return "\n".join(lines)


def format_news_and_references(snapshot: EquityCorpusSnapshot) -> str:
    """Part 06: Curated News Funnel & Regulatory Headlines."""
    lines = [
        f"# {snapshot.ticker} Curated News & Narrative Evidence",
        "",
        f"Total news items: {len(snapshot.news)}",
        "",
    ]

    if not snapshot.news:
        lines.append("No news items stored in vault or cache.")
        return "\n".join(lines)

    for idx, item in enumerate(snapshot.news[:50], start=1):
        headline = item.get("headline") or item.get("title", "News Headline")
        published = item.get("published_at") or item.get("date", "N/A")
        source = item.get("publisher") or item.get("source", "N/A")
        url = item.get("article_url") or item.get("link") or item.get("url", "N/A")
        summary = item.get("summary") or item.get("snippet", "No summary provided.")

        lines.extend([
            f"### {idx}. {headline}",
            f"- **Published:** `{published}`",
            f"- **Source:** `{source}`",
            f"- **URL:** {url}",
            f"- **Summary:** {summary}",
            "",
        ])

    return "\n".join(lines)


def format_market_derivatives(snapshot: EquityCorpusSnapshot) -> str:
    """Part 07: Market Derivatives, Cboe Options & Consensus."""
    lines = [
        f"# {snapshot.ticker} Derivatives & Consensus Telemetry",
        "",
    ]

    cons = snapshot.consensus or {}
    opt = snapshot.options_chain or {}
    mp = snapshot.max_pain or {}
    quant = (snapshot.latest_analysis or {}).get("quant_signals", {})
    rev_dcf = quant.get("reverse_dcf_result") or {}

    # 1. Consensus Resolution
    ratings_raw = cons.get("ratings")
    consensus_stance = "N/A"
    analyst_count = "N/A"
    if isinstance(ratings_raw, dict):
        consensus_stance = ratings_raw.get("consensus", "N/A")
        analyst_count = ratings_raw.get("analyst_count", "N/A")
    elif isinstance(ratings_raw, str):
        m_stance = re.search(r"consensus=['\"]?([^,'\"\)]+)", ratings_raw)
        if m_stance:
            consensus_stance = m_stance.group(1)
        m_count = re.search(r"analyst_count=(\d+)", ratings_raw)
        if m_count:
            analyst_count = m_count.group(1)

    mean_target = (
        cons.get("target_mean")
        or rev_dcf.get("consensus_target_price")
        or "N/A"
    )
    high_target = (
        cons.get("target_high")
        or rev_dcf.get("consensus_target_high")
        or "N/A"
    )
    low_target = (
        cons.get("target_low")
        or rev_dcf.get("consensus_target_low")
        or "N/A"
    )
    if analyst_count == "N/A":
        analyst_count = cons.get("num_analysts") or rev_dcf.get("analyst_count") or "N/A"

    if mean_target != "N/A" or consensus_stance != "N/A":
        lines.extend([
            "## 1. Nasdaq / Wall Street Consensus",
            f"- **Consensus Stance:** `{consensus_stance}`",
            f"- **Mean Target Price:** `${mean_target}`",
            f"- **High Target Price:** `${high_target}`",
            f"- **Low Target Price:** `${low_target}`",
            f"- **Total Analysts:** `{analyst_count}`",
            "",
        ])

    if mp:
        lines.extend([
            "## 2. Cboe Max Pain Invariant",
            f"- **Max Pain Strike:** `${mp.get('max_pain_strike', 'N/A')}`",
            f"- **Total Call OI:** `{mp.get('total_call_oi', 'N/A'):,}`" if isinstance(mp.get('total_call_oi'), (int, float)) else f"- **Total Call OI:** `{mp.get('total_call_oi', 'N/A')}`",
            f"- **Total Put OI:** `{mp.get('total_put_oi', 'N/A'):,}`" if isinstance(mp.get('total_put_oi'), (int, float)) else f"- **Total Put OI:** `{mp.get('total_put_oi', 'N/A')}`",
            f"- **Put/Call Ratio:** `{mp.get('pcr', 'N/A')}`",
            "",
        ])

    if opt:
        lines.extend([
            "## 3. Cboe Delayed Options Chain Summary",
            f"- **Expiration Date:** `{opt.get('expiration_date', 'N/A')}`",
            f"- **IV 30-Day Metric:** `{opt.get('iv30_decimal', 'N/A')}`",
            f"- **Total Contracts Listed:** `{len(opt.get('contracts', []))}`",
            "",
        ])

    if not lines or len(lines) <= 2:
        lines.append("No market derivatives or consensus records available.")

    return "\n".join(lines)


def format_price_and_notes(snapshot: EquityCorpusSnapshot) -> str:
    """Part 08: OHLCV Telemetry & User Notes."""
    lines = [
        f"# {snapshot.ticker} Market Price Telemetry & User Vault Notes",
        "",
    ]

    ohlcv = snapshot.ohlcv_summary
    quant = (snapshot.latest_analysis or {}).get("quant_signals", {})
    tactical = quant.get("tactical_setup") or {}

    lines.append("## 1. Market Price Telemetry")
    if ohlcv:
        lines.extend([
            f"- **Analysis Close Price:** `${ohlcv.get('analysis_price', 'N/A')}` (as of `{ohlcv.get('analysis_price_as_of', 'N/A')}`)",
            f"- **Price Source:** `{ohlcv.get('price_source', 'N/A')}`",
            f"- **Latest OHLCV Date:** `{ohlcv.get('latest_ohlcv_date', 'N/A')}`",
            f"- **Session Status:** `{ohlcv.get('market_session_status', 'N/A')}`",
            f"- **Sync Status:** `{ohlcv.get('price_sync_status', 'fresh')}`",
            "",
        ])
    elif quant:
        price = quant.get("raw_analysis_price") or tactical.get("current_price", "N/A")
        lines.extend([
            f"- **Analysis Price:** `${price}`",
            f"- **50-Day Moving Average (SMA 50):** `${tactical.get('sma_50', 'N/A')}`",
            f"- **200-Day Moving Average (SMA 200):** `${tactical.get('sma_200', 'N/A')}`",
            f"- **14-Day ATR:** `${tactical.get('atr_14', 'N/A')}`",
            f"- **Average Daily Trading Volume (ADTV):** `{_fmt_num(quant.get('adtv_local_currency'), prefix='$', default='N/A')}`",
            f"- **5-Year Price Percentile:** `{quant.get('price_percentile_5y', 'N/A')}%` (Z-score: `{quant.get('price_zscore_5y', 'N/A')}`)",
            f"- **Maximum Drawdown (MDD):** `{quant.get('mdd_pct', 'N/A')}%`",
            f"- **Beta:** `{quant.get('beta', 'N/A')}` | **Volatility:** `{quant.get('volatility_pct', 'N/A')}%`",
            "",
        ])

    lines.extend([
        "## 2. Vault User Notes & Concept References",
        f"Total user notes: {len(snapshot.notes)}",
        "",
    ])

    if not snapshot.notes:
        lines.append("No user notes linked to this symbol in the Vault.")
    else:
        for n in snapshot.notes[:30]:
            lines.extend([
                f"- **{n.get('title', 'Note')}** (`{n.get('folder', 'Vault')}`)",
                f"  - Modified: `{n.get('modified_at', 'N/A')}`",
                f"  - Snippet: {n.get('snippet', '')}",
                "",
            ])

    return "\n".join(lines)


def format_structured_appendix(snapshot: EquityCorpusSnapshot) -> str:
    """Part 09: Structured Mathematical Appendix (JSON)."""
    quant = (snapshot.latest_analysis or {}).get("quant_signals", {})
    payload = {
        "ticker": snapshot.ticker,
        "market": snapshot.market,
        "snapshot_at": snapshot.snapshot_at,
        "metadata": snapshot.metadata,
        "latest_scorecard": quant.get("deterministic_scorecard") or quant.get("scorecard"),
        "latest_dcf": quant.get("reverse_dcf_result") or quant.get("dcf_result") or quant.get("dcf_valuation"),
        "latest_tactical": quant.get("tactical_setup"),
        "portfolio_holding": snapshot.portfolio_holding,
    }
    return (
        f"# {snapshot.ticker} Structured Appendix\n\n"
        "```json\n"
        f"{json.dumps(payload, indent=2, default=str)}\n"
        "```\n"
    )
