"""Single @tool ที่รวบรวม Equity Quant Signals ทั้งหมดแบบ deterministic — ผูกกับ equity_quant agent
ตัวเดียว มิเรอร์ evaluate_macro_matrix (tools/macro/evaluation.py) ที่เป็น tool เดียวของ macro_quant

ไม่มี field ตัวเลขไหนใน output มาจากการตัดสินใจของ LLM — LLM ที่ผูกกับ tool นี้มีหน้าที่แค่ตีความ
ticker/market จาก instruction แล้วส่งต่อผลลัพธ์ดิบกลับไปเท่านั้น (ดู prompts/skills/equity_quant/SKILL.md)
"""
import math
import os
import time
from decimal import Decimal, InvalidOperation
from datetime import datetime, timezone
from typing import Literal, Optional

from langchain_core.tools import tool

from core.logger import get_logger
from schemas.micro_quant_schemas import QuantSignals
from tools.market.asset_resolver import resolve_asset
from tools.market.financial_autopsy import get_financial_autopsy, calculate_piotroski_f_score
from tools.market.quant_engine import (
    compute_beta,
    compute_volatility,
    compute_mdd,
    compute_technical_indicators,
    compute_growth_rates,
    compute_price_percentile,
    create_atomic_market_snapshot,
    _get_price_history,
    compute_amihud_illiquidity_20d_with_metadata,
)
from tools.market.quant_scoring import (
    compute_value_score,
    compute_quality_score,
    compute_momentum_score,
    compute_price_target_outlook,
    compute_growth_score,
    compute_dividend_score,
    compute_solvency_score,
    compute_trading_liquidity,
    compute_composite_score,
    compute_composite_score_waterfall,
    compute_fcf_quality_score,
    compute_debt_quality_score,
    compute_net_shareholder_yield,
)
from core.equity_analytics.dupont_accrual import (
    compute_dupont_3factor,
    compute_sloan_accrual_ratio,
)
from core.equity_analytics.signal_audit import audit_signal_maturity
from tools.market.consensus_engine import compute_horizon_matched_expectation_gap
from tools.market.pit_fundamentals_ledger import compute_pit_multiple_bands
from tools.market.peer_valuation import fetch_peer_metrics, compute_peer_relative_score
from tools.market.earnings_momentum import fetch_earnings_revision_data, compute_earnings_revision_score
from tools.macro.evaluation import load_latest_macro_observables
from tools.macro.valuation import _find_dgs10_in_observables
from schemas.macro_schemas import MarketObservable
from tools.market.dcf_valuation import compute_dcf_valuation, compute_institutional_reverse_dcf
from tools.market.ownership import compute_smart_money_flags, compute_canonical_insider_conviction
from tools.market.technical import compute_tactical_setup
from tools.market.sector_kpi_engine import compute_sector_kpis
from tools.market.equity_rules_engine import compute_deterministic_scorecard, load_latest_macro_regime_context
from .core import _yf_info

log = get_logger(__name__)

# TTL cache local ต่อไฟล์นี้ (ไม่แตะ tools/market/core.py ที่ fundamentals/technical/consensus/news
# ใช้ร่วมกัน เพื่อไม่ให้กระทบ test suite ของไฟล์เหล่านั้น) — ลดการยิง .info ซ้ำเมื่อวิเคราะห์ ticker
# เดิมซ้ำในช่วงเวลาสั้นๆ เช่น re-run วิเคราะห์หุ้นตัวเดิมในหลาย turn ติดกัน
_INFO_CACHE: dict[str, tuple[dict, float]] = {}
_INFO_ERROR_CACHE: dict[str, float] = {}
_INFO_SUCCESS_TTL_SECONDS = 6 * 3600
_INFO_ERROR_TTL_SECONDS = 60.0


def _get_info_cached(provider_symbol: str) -> dict:
    now = time.time()
    if provider_symbol in _INFO_CACHE:
        info, ts = _INFO_CACHE[provider_symbol]
        if now - ts < _INFO_SUCCESS_TTL_SECONDS:
            return info
        del _INFO_CACHE[provider_symbol]

    if provider_symbol in _INFO_ERROR_CACHE:
        ts = _INFO_ERROR_CACHE[provider_symbol]
        if now - ts < _INFO_ERROR_TTL_SECONDS:
            raise RuntimeError(f"cached failure for {provider_symbol}")
        del _INFO_ERROR_CACHE[provider_symbol]

    try:
        info = _yf_info(provider_symbol)
    except Exception:
        _INFO_ERROR_CACHE[provider_symbol] = now
        raise

    _INFO_CACHE[provider_symbol] = (info, now)
    return info


def _ma_cross_label(ma50: Optional[float], ma200: Optional[float]) -> Optional[str]:
    if ma50 is None or ma200 is None:
        return None
    return "golden_cross" if ma50 > ma200 else "death_cross"


@tool
def compute_equity_quant_signals(ticker: str, market: Literal["TH", "US"] = "US") -> str:
    """คำนวณ Quant Signals ของหุ้นรายตัวแบบ deterministic ทั้งหมด — 5 Pillars: Value/Quality/Growth/
    Momentum/Dividend (รวมเป็น Composite Score เดียว) บวก Solvency (Risk Gate แยก) และ Trading Liquidity

    [Usage/When to use]
    ใช้เมื่อต้องการวิเคราะห์เชิงปริมาณของหุ้นรายตัว 1 ตัวแบบสถาบัน — ครอบคลุมการประเมินมูลค่า (Value),
    คุณภาพกิจการ (Quality), การเติบโต (Growth), โมเมนตัมราคา (Momentum), ปันผล (Dividend),
    ความแข็งแรงทางการเงิน (Solvency), สภาพคล่องการซื้อขาย (Trading Liquidity), ความผันผวน
    (Beta/Volatility/MDD) และเป้าหมายราคานักวิเคราะห์ (Upside/Downside) — ทุกค่าคำนวณจาก Python
    ล้วนๆ ไม่มีการประเมินเชิงอัตวิสัย

    [Caution]
    - ค่าที่เป็น None หมายถึงข้อมูลไม่พอ/ไม่มีจริงจาก Yahoo Finance (พบบ่อยกับหุ้นไทยที่ไม่มี
      analyst target price) หรือหุ้นเพิ่ง IPO ที่มีราคาย้อนหลังไม่ถึง 45 วันทำการ — ห้ามเดาแทนค่า None
    - momentum_score วัด 'ความแรงของโมเมนตัมขาขึ้นเชิงเทคนิค' (momentum-chasing) ไม่ใช่คำแนะนำซื้อ
    - dividend_score สูงอาจเป็น 'Value Trap' (ราคาร่วงหนักทำให้ yield ดูสูงลวงตา) ต้องดู payout_ratio ประกอบ
    - solvency_score เป็น Risk Gate แยก ไม่ถูกรวมใน composite_score — ต้องพิจารณาเป็นตัวชี้วัดความเสี่ยง
      leverage แยกต่างหากเสมอ ไม่ใช่ตัวชี้วัดผลตอบแทน
    - adtv_local_currency เป็นสกุลเงินท้องถิ่น (THB สำหรับ TH, USD สำหรับ US) ห้ามเทียบข้าม market ตรงๆ
    - Beta ของหุ้นไทยเทียบกับ ^SET.BK (SET Index) ส่วนหุ้นสหรัฐฯ เทียบกับ ^GSPC (S&P 500)
    - peer_relative_score เทียบ P/E กับ peer ใน sector เดียวกัน (US เท่านั้นตอนนี้) เป็น Contextual
      ไม่รวมใน composite_score — ต้องมี peer ที่ดึงข้อมูลสำเร็จ ≥2 ตัวถึงจะคำนวณได้
    - price_percentile_5y เป็น percentile ของ 'ราคา' ไม่ใช่ Valuation Multiple — Contextual เท่านั้น
    - earnings_momentum_score มาจากการปรับประมาณการกำไรของนักวิเคราะห์ 30 วันล่าสุด — Contextual เท่านั้น
    - คืนค่าเป็น JSON string ของ QuantSignals — ส่งต่อกลับไปตรงๆ ห้ามแก้ไขตัวเลขในนั้น

    Args:
        ticker (str): Ticker symbol เช่น 'AAPL', 'PTT' (ห้ามมี .BK suffix — ระบบจะเติมให้)
        market (Literal["TH","US"]): 'TH' สำหรับหุ้นไทย (SET) หรือ 'US' สำหรับหุ้นอเมริกา (default)
    """
    try:
        resolved = resolve_asset(ticker, market_hint=market)
        provider_symbol = resolved.provider_symbol or ticker.strip().upper()

        autopsy_result = get_financial_autopsy(resolved)
        autopsy = autopsy_result.snapshot

        try:
            info = _get_info_cached(provider_symbol)
        except Exception as e:
            log.warning("compute_equity_quant_signals: _yf_info failed for %s: %s", provider_symbol, e)
            info = {}

        # 0. Atomic Market Snapshot — Single Source of Truth for Price & Market Cap
        df_1y = None
        try:
            df_1y = _get_price_history(provider_symbol, "1y")
        except Exception as e:
            log.warning("compute_equity_quant_signals: _get_price_history failed for %s: %s", provider_symbol, e)

        atomic_snapshot, snapshot_flags = create_atomic_market_snapshot(
            provider_symbol=provider_symbol,
            df_1y=df_1y,
            info=info,
            market=market,
        )
        analysis_price = atomic_snapshot.analysis_price
        analysis_mcap = atomic_snapshot.market_cap
        shares_out = atomic_snapshot.shares_outstanding or info.get("sharesOutstanding")

        benchmark = "^SET.BK" if market == "TH" else "^GSPC"
        beta, beta_q = compute_beta(provider_symbol, benchmark=benchmark)
        volatility_pct, vol_q = compute_volatility(provider_symbol)
        mdd_pct, mdd_q = compute_mdd(provider_symbol)
        tech, tech_q = compute_technical_indicators(provider_symbol)
        amihud_meta = compute_amihud_illiquidity_20d_with_metadata(
            df_1y,
            currency="THB" if market == "TH" else "USD",
        )
        amihud_illiquidity_20d = amihud_meta["value"]

        company_name = info.get("shortName")
        peer_sector = info.get("sector")

        pe = autopsy.current_pe if (autopsy and autopsy.current_pe is not None) else info.get("trailingPE")
        pb = info.get("priceToBook")
        ev_ebitda = info.get("enterpriseToEbitda")
        value_score, value_flag = compute_value_score(pe, pb, ev_ebitda, sector=peer_sector)

        roe_raw = info.get("returnOnEquity")
        roe_pct = roe_raw * 100 if roe_raw is not None else None
        margin_raw = info.get("profitMargins")
        profit_margin_pct = margin_raw * 100 if margin_raw is not None else None
        fcf_debt_ratio = None
        if autopsy and autopsy.periods:
            latest = autopsy.periods[0]
            if latest.free_cash_flow is not None and latest.total_debt not in (None, 0):
                fcf_debt_ratio = latest.free_cash_flow / latest.total_debt
        quality_score, quality_flag = compute_quality_score(roe_pct, profit_margin_pct, fcf_debt_ratio)

        growth_rates, growth_flags = compute_growth_rates(autopsy.periods if autopsy else [])
        growth_score, growth_score_flag = compute_growth_score(
            growth_rates["revenue_growth_yoy_pct"], growth_rates["net_income_growth_yoy_pct"]
        )

        if tech_q is not None and not tech_q.is_valid:
            rsi_14 = None
            macd_signal = None
            ma_cross = None
        else:
            rsi_14 = tech.get("rsi_14") if tech else None
            macd_signal = tech.get("macd_signal") if tech else None
            ma_cross = _ma_cross_label(info.get("fiftyDayAverage"), info.get("twoHundredDayAverage"))
        momentum_score, momentum_flag = compute_momentum_score(rsi_14, macd_signal, ma_cross)

        # trailingAnnualDividendYield ต้อง *100 (decimal จริง) — ต่างจาก dividendYield ที่ percent-scaled
        # อยู่แล้ว (0.37 = 0.37%) ดู tools/market/fundamentals.py สำหรับ precedent ของ gotcha นี้
        dividend_yield_raw = info.get("trailingAnnualDividendYield")
        dividend_yield_pct = dividend_yield_raw * 100 if dividend_yield_raw is not None else None
        payout_ratio_pct = autopsy.periods[0].payout_ratio_pct if (autopsy and autopsy.periods) else None
        if payout_ratio_pct is None and (dividend_yield_pct == 0.0 or dividend_yield_raw == 0.0):
            payout_ratio_pct = 0.0
        dividend_score, dividend_flag = compute_dividend_score(dividend_yield_pct, payout_ratio_pct)

        # debtToEquity จาก yfinance เป็นสเกล % อยู่แล้ว (150.0 = 1.5x) ใช้ตรงๆ ไม่แปลงหน่วยเพิ่ม
        de_ratio_pct = info.get("debtToEquity")
        current_ratio = info.get("currentRatio")
        solvency_score, solvency_flag = compute_solvency_score(de_ratio_pct, current_ratio)

        upside_pct, downside_pct = compute_price_target_outlook(
            analysis_price,
            target_mean=info.get("targetMeanPrice"),
            target_high=info.get("targetHighPrice"),
            target_low=info.get("targetLowPrice"),
        )

        adtv_local_currency, liquidity_flag = compute_trading_liquidity(
            info.get("averageVolume"), info.get("averageVolume10Day"), analysis_price, market
        )

        composite_score, composite_flag = compute_composite_score(
            value_score, quality_score, growth_score, momentum_score, dividend_score
        )
        composite_score_waterfall = compute_composite_score_waterfall(
            value_score, quality_score, growth_score, momentum_score, dividend_score
        )

        # Peer/Sector Relative Valuation — Contextual เท่านั้น ไม่รวมใน composite_score
        peer_metrics = fetch_peer_metrics(peer_sector, exclude_ticker=resolved.raw_symbol, market=market)
        peer_relative_score, pe_vs_peer_avg_pct, peer_count, peer_flag = compute_peer_relative_score(pe, peer_metrics)

        # Historical Price Context — Contextual เท่านั้น ไม่รวมใน composite_score (ดู docstring
        # compute_price_percentile: เป็น percentile ของราคา ไม่ใช่ Valuation Multiple)
        price_percentile_5y, price_zscore_5y, price_pctile_q = compute_price_percentile(provider_symbol)

        # Earnings Momentum & Revisions — Contextual เท่านั้น ไม่รวมใน composite_score
        revision_data, revision_fetch_flag = fetch_earnings_revision_data(provider_symbol)
        eps_revision_net_30d, eps_estimate_change_30d_pct, earnings_momentum_score, revision_score_flag = (
            compute_earnings_revision_score(revision_data)
        )

        # Cash Flow & Capital Quality
        latest_p = autopsy.periods[0] if (autopsy and autopsy.periods) else None
        ttm_p = autopsy.ttm_period if (autopsy and hasattr(autopsy, "ttm_period") and autopsy.ttm_period) else None

        # Prioritize rolling 4-quarter TTM FCF over stale vendor attributes
        fcf_raw = (ttm_p.free_cash_flow if (ttm_p and ttm_p.free_cash_flow is not None) else None) or info.get("freeCashflow") or (latest_p.free_cash_flow if latest_p else None)
        fcf_yield_pct = round((fcf_raw / analysis_mcap) * 100.0, 2) if (fcf_raw is not None and analysis_mcap is not None and analysis_mcap > 0) else None
        
        # FCF Margin & OCF/NI: Prioritize consistent TTM period from autopsy
        if ttm_p and ttm_p.free_cash_flow is not None and ttm_p.total_revenue and ttm_p.total_revenue > 0:
            fcf_margin_pct = round((ttm_p.free_cash_flow / ttm_p.total_revenue) * 100.0, 2)
        else:
            fcf_period_val = latest_p.free_cash_flow if (latest_p and latest_p.free_cash_flow is not None) else fcf_raw
            rev_period_val = latest_p.total_revenue if (latest_p and latest_p.total_revenue is not None) else info.get("totalRevenue")
            fcf_margin_pct = round((fcf_period_val / rev_period_val) * 100.0, 2) if (fcf_period_val is not None and rev_period_val is not None and rev_period_val > 0) else None

        # FCF CAGR 3Y
        fcf_cagr_3y = None
        if autopsy and autopsy.periods and len(autopsy.periods) >= 4:
            fcf_latest = autopsy.periods[0].free_cash_flow
            fcf_3y_ago = autopsy.periods[3].free_cash_flow
            if fcf_latest is not None and fcf_3y_ago is not None and fcf_latest > 0 and fcf_3y_ago > 0:
                fcf_cagr_3y = round((((fcf_latest / fcf_3y_ago) ** (1.0 / 3.0)) - 1.0) * 100.0, 2)

        ocf_raw = (ttm_p.operating_cash_flow if (ttm_p and ttm_p.operating_cash_flow is not None) else None) or info.get("operatingCashflow") or (latest_p.operating_cash_flow if latest_p else None)
        if ttm_p and ttm_p.operating_cash_flow is not None and ttm_p.net_income and ttm_p.net_income > 0:
            ocf_to_net_income = round(ttm_p.operating_cash_flow / ttm_p.net_income, 2)
        else:
            ocf_period_val = latest_p.operating_cash_flow if (latest_p and latest_p.operating_cash_flow is not None) else ocf_raw
            net_inc_period_val = latest_p.net_income if (latest_p and latest_p.net_income is not None) else None
            ocf_to_net_income = round(ocf_period_val / net_inc_period_val, 2) if (ocf_period_val is not None and net_inc_period_val is not None and net_inc_period_val > 0) else None

        ebitda_raw = info.get("ebitda")
        total_debt_raw = info.get("totalDebt") or (autopsy.periods[0].total_debt if (autopsy and autopsy.periods) else None)
        cash_raw = info.get("totalCash")
        net_debt_ebitda = round((total_debt_raw - cash_raw) / ebitda_raw, 2) if (total_debt_raw is not None and cash_raw is not None and ebitda_raw is not None and ebitda_raw > 0) else None

        # GAAP Operating Income (prioritized over vendor EBIT)
        op_income_val = (latest_p.operating_income if (latest_p and latest_p.operating_income is not None) else None) or (latest_p.ebit if latest_p else None)
        ebit_val = latest_p.ebit if latest_p else None
        interest_exp = latest_p.interest_expense if latest_p else info.get("interestExpense")
        tax_exp = latest_p.tax_expense if latest_p else None
        pretax_inc = latest_p.income_before_tax if latest_p else None

        interest_coverage = round(op_income_val / interest_exp, 2) if (op_income_val is not None and interest_exp is not None and interest_exp > 0) else None

        # Effective Tax Rate Formula
        if tax_exp is not None and pretax_inc is not None and pretax_inc > 0:
            tax_rate = max(0.0, min(0.40, tax_exp / pretax_inc))
        else:
            tax_rate = 0.21 if market == "US" else 0.20

        # ROIC Formula: NOPAT / Invested Capital using GAAP Operating Income
        roic_pct = None
        tot_eq = (latest_p.stockholders_equity if latest_p else None) or info.get("totalStockholderEquity") or info.get("stockholderEquity")
        if op_income_val is not None:
            nopat = op_income_val * (1.0 - tax_rate)
            if total_debt_raw is not None and tot_eq is not None and cash_raw is not None:
                inv_cap = total_debt_raw + tot_eq - cash_raw
                if inv_cap > 0:
                    roic_pct = round((nopat / inv_cap) * 100.0, 2)
                elif latest_p and latest_p.total_assets and latest_p.current_liabilities:
                    # Net Cash Tech Company: Use Operating Invested Capital (Total Assets - Current Liabilities)
                    op_inv_cap = latest_p.total_assets - latest_p.current_liabilities
                    if op_inv_cap > 0:
                        roic_pct = round((nopat / op_inv_cap) * 100.0, 2)
            elif tot_eq is not None and tot_eq > 0:
                roic_pct = round((nopat / tot_eq) * 100.0, 2)

        fcf_quality_score, fcf_q_flag = compute_fcf_quality_score(fcf_yield_pct, ocf_to_net_income)
        debt_quality_score, debt_q_flag = compute_debt_quality_score(interest_coverage, net_debt_ebitda)

        # DCF Valuation Engine
        fcf_per_share = (fcf_raw / shares_out) if (fcf_raw is not None and shares_out is not None and shares_out > 0) else 0.0
        macro_registry = load_latest_macro_observables()

        # Ensure fresh 10Y yield for US equities if vault snapshot is stale (non-mock, non-test environment)
        if market == "US" and "PYTEST_CURRENT_TEST" not in os.environ and isinstance(macro_registry, dict) and "MagicMock" not in type(macro_registry).__name__ and macro_registry:
            now_dt = datetime.now(timezone.utc)
            dgs10_val, dgs10_id = _find_dgs10_in_observables(list(macro_registry.values()))
            obs_item = macro_registry.get(dgs10_id) if dgs10_id else None
            obs_date_str = getattr(obs_item, "observed_at", None) if obs_item else None
            is_dgs10_stale = True
            if obs_date_str and isinstance(obs_date_str, str):
                try:
                    obs_d = datetime.strptime(obs_date_str[:10], "%Y-%m-%d").date()
                    if (now_dt.date() - obs_d).days <= 7:
                        is_dgs10_stale = False
                except Exception:
                    pass
            if is_dgs10_stale and obs_item and getattr(obs_item, "source_file", "") != "mock" and "MagicMock" not in type(obs_item).__name__:
                try:
                    import yfinance as yf
                    tnx_ticker = yf.Ticker("^TNX")
                    live_tnx = tnx_ticker.fast_info.get("lastPrice") or tnx_ticker.info.get("regularMarketPrice")
                    if live_tnx and float(live_tnx) > 0:
                        today_str = now_dt.strftime("%Y-%m-%d")
                        macro_registry["obs_dgs10_live"] = MarketObservable(
                            observable_id="obs_dgs10_live",
                            asset_bucket="fixed_income",
                            region="US",
                            indicator="10-Year Treasury Constant Maturity Rate",
                            value=f"{float(live_tnx):.2f}",
                            unit="%",
                            observed_at=today_str,
                            source_file="live_market",
                            provider="Yahoo Finance (^TNX)",
                            confidence="high",
                            is_valid=True,
                            observable_type="economic_indicator",
                        )
                except Exception as tnx_err:
                    log.warning("Could not fetch live ^TNX for fresh DCF: %s", tnx_err)

        dcf_result, dcf_flags = compute_dcf_valuation(
            ticker=resolved.raw_symbol.strip().upper(),
            market=market,
            current_price=analysis_price or 0.0,
            beta=beta,
            fcf_per_share=fcf_per_share,
            market_cap=analysis_mcap or 0.0,
            total_debt=total_debt_raw or 0.0,
            interest_expense=interest_exp,
            tax_rate=tax_rate,
            fcf_cagr_3y=fcf_cagr_3y,
            macro_registry=macro_registry,
            forward_eps=info.get("forwardEps"),
            trailing_eps=info.get("trailingEps"),
            cash_and_equivalents=cash_raw or 0.0,
        )

        # Canonical SEC Form 4 Insider Conviction & Smart Money Integration
        insider_conviction, ins_flags = compute_canonical_insider_conviction(provider_symbol, market=market)
        has_filtered_form4 = bool(
            insider_conviction
            and insider_conviction.status not in ("unavailable", "not_applicable")
            and insider_conviction.data_status == "available"
        )
        if has_filtered_form4:
            smart_money_flags, ownership_flags = compute_smart_money_flags(
                provider_symbol,
                info,
                has_10b51_filtered_data=True,
                discretionary_buy_count=insider_conviction.open_market_p_count_90d,
                discretionary_sell_count=insider_conviction.unflagged_s_count_90d,
            )
        else:
            smart_money_flags, ownership_flags = compute_smart_money_flags(provider_symbol, info)

        # Net Shareholder Yield (Dividend + Buyback - Dilution)
        latest_sh = (autopsy.periods[0].shares_outstanding if (autopsy and autopsy.periods) else None) or shares_out
        prev_sh = (autopsy.periods[1].shares_outstanding if (autopsy and len(autopsy.periods) >= 2) else None)
        net_buyback_yield_pct, net_shareholder_yield_pct, dilution_flag = compute_net_shareholder_yield(
            dividend_yield_pct, latest_sh, prev_sh
        )

        # Declarative Sector KPIs
        sector_kpis_summary = compute_sector_kpis(peer_sector or "", autopsy.periods if autopsy else [])
        sector_kpis_dict = sector_kpis_summary.model_dump() if sector_kpis_summary else None

        # DuPont 3-Factor & Sloan Accrual Analysis (period-consistent & non-silent error reporting)
        dupont_dict = None
        sloan_dict = None
        dupont_sloan_flags: list[str] = []
        dupont_basis = "unavailable_missing_inputs"
        sloan_basis = "unavailable_missing_inputs"
        if autopsy and autopsy.periods:
            p0 = autopsy.periods[0]
            p0_type = getattr(p0, "period_type", None)
            p0_end = str(getattr(p0, "fiscal_period_end", "") or "")
            p1 = None
            for cand in autopsy.periods[1:]:
                c_type = getattr(cand, "period_type", None)
                c_end = str(getattr(cand, "fiscal_period_end", "") or "")
                if (p0_type or c_type) and c_type != p0_type:
                    continue
                if p0_end and c_end and c_end >= p0_end:
                    continue
                p1 = cand
                break

            ni = getattr(p0, "net_income", None)
            rev = getattr(p0, "total_revenue", None)
            ta0 = getattr(p0, "total_assets", None)
            eq0 = getattr(p0, "stockholders_equity", None)
            ocf = getattr(p0, "operating_cash_flow", None)
            ta1 = getattr(p1, "total_assets", None) if p1 else None

            if ni is not None and rev is not None and ta0 is not None and eq0 is not None:
                try:
                    dp = compute_dupont_3factor(
                        net_income=Decimal(str(ni)),
                        revenue=Decimal(str(rev)),
                        total_assets=Decimal(str(ta0)),
                        stockholders_equity=Decimal(str(eq0)),
                    )
                    if dp is None:
                        dupont_sloan_flags.append("dupont_unavailable:non_positive_denominator")
                        dupont_basis = "unavailable_non_positive_denominator"
                    else:
                        dupont_dict = {
                            "net_profit_margin_pct": float(dp.net_profit_margin_pct),
                            "asset_turnover": float(dp.asset_turnover),
                            "equity_multiplier": float(dp.equity_multiplier),
                            "roe_pct": float(dp.roe_pct),
                            "closure_error": float(dp.closure_error),
                            "internal_closure_error": float(dp.internal_closure_error),
                            "published_roe_reconstructed_pct": float(dp.published_roe_reconstructed_pct),
                            "is_published_reconciled": dp.is_published_reconciled,
                            "fiscal_period_end": p0_end or None,
                            "period_type": p0_type or None,
                        }
                        dupont_basis = f"{p0_type or 'unknown'}:{p0_end or 'unknown'}"
                        if not dp.is_published_reconciled:
                            dupont_sloan_flags.append("dupont_display_reconciliation_gap:quality")
                except (InvalidOperation, ValueError, TypeError) as exc:
                    log.warning("DuPont calculation failed for %s: %s", provider_symbol, exc)
                    dupont_sloan_flags.append("dupont_calculation_failed:quality")
                    dupont_basis = "calculation_failed"

            if ni is not None and ocf is not None and ta0 is not None:
                try:
                    sl = compute_sloan_accrual_ratio(
                        net_income=Decimal(str(ni)),
                        operating_cash_flow=Decimal(str(ocf)),
                        total_assets_current=Decimal(str(ta0)),
                        total_assets_prior=Decimal(str(ta1)) if ta1 is not None else None,
                    )
                    if sl is None:
                        dupont_sloan_flags.append("sloan_unavailable:non_positive_assets")
                        sloan_basis = "unavailable_non_positive_assets"
                    else:
                        sloan_dict = {
                            "sloan_accrual_ratio": float(sl.sloan_accrual_ratio),
                            "raw_sloan_accrual_ratio": (
                                float(sl.raw_sloan_accrual_ratio)
                                if sl.raw_sloan_accrual_ratio is not None
                                else float(sl.sloan_accrual_ratio)
                            ),
                            "earnings_quality_rating": sl.earnings_quality_rating,
                            "assets_basis": sl.assets_basis,
                            "interpretation": sl.interpretation,
                            "fiscal_period_end": p0_end or None,
                            "period_type": p0_type or None,
                            "prior_fiscal_period_end": str(getattr(p1, "fiscal_period_end", "") or "") or None if p1 else None,
                        }
                        sloan_basis = f"{sl.assets_basis}:{p0_type or 'unknown'}:{p0_end or 'unknown'}"
                except (InvalidOperation, ValueError, TypeError) as exc:
                    log.warning("Sloan accrual calculation failed for %s: %s", provider_symbol, exc)
                    dupont_sloan_flags.append("sloan_calculation_failed:quality")
                    sloan_basis = "calculation_failed"

        # -------------------------------------------------------------
        # Institutional 4-Pillar Integration
        # -------------------------------------------------------------
        # 1. Piotroski F-Score
        sector_name = peer_sector or info.get("sector")
        piotroski_breakdown = calculate_piotroski_f_score(autopsy.periods if autopsy else [], sector=sector_name)

        # 2. Reported EBIT Margin & Base Revenue Extraction (Harmonized to TTM Run-rate)
        margin_flags: List[str] = []

        if ttm_p and ttm_p.total_revenue and ttm_p.total_revenue > 0:
            rev_base = float(ttm_p.total_revenue)
            base_revenue_period_type = "ttm"
        else:
            ttm_rev = info.get("totalRevenue")
            ann_rev = latest_p.total_revenue if (latest_p and latest_p.total_revenue) else None
            if ttm_rev is not None and ttm_rev > 0:
                rev_base = float(ttm_rev)
                base_revenue_period_type = "ttm"
            elif ann_rev is not None and ann_rev > 0:
                rev_base = float(ann_rev)
                base_revenue_period_type = "annual"
            else:
                rev_base = 0.0
                base_revenue_period_type = "unknown"

        # Prioritize true SEC GAAP Operating Margin from ttm_p (32.44%)
        if ttm_p and ttm_p.operating_income is not None and ttm_p.total_revenue and ttm_p.total_revenue > 0:
            reported_ebit_margin_pct = round((ttm_p.operating_income / ttm_p.total_revenue) * 100.0, 2)
            margin_source_tier = "filing_authoritative"
            margin_period_type = "ttm"
            margin_fiscal_period = "TTM (GAAP 10-Q)"
        elif base_revenue_period_type == "ttm" and info.get("operatingMargins") is not None:
            reported_ebit_margin_pct = round(info["operatingMargins"] * 100.0, 2)
            margin_source_tier = "primary_best_effort"
            margin_period_type = "ttm"
            margin_fiscal_period = "TTM"
        elif latest_p and latest_p.operating_income is not None and latest_p.total_revenue and latest_p.total_revenue > 0:
            reported_ebit_margin_pct = round((latest_p.operating_income / latest_p.total_revenue) * 100.0, 2)
            margin_source_tier = latest_p.source_tier or (autopsy.source_tier if autopsy else None) or "unknown"
            margin_period_type = latest_p.period_type or "unknown"
            margin_fiscal_period = latest_p.fiscal_period_end or "FY"
        elif latest_p and latest_p.ebit is not None and latest_p.total_revenue and latest_p.total_revenue > 0:
            reported_ebit_margin_pct = round((latest_p.ebit / latest_p.total_revenue) * 100.0, 2)
            margin_source_tier = latest_p.source_tier or (autopsy.source_tier if autopsy else None) or "unknown"
            margin_period_type = latest_p.period_type or "unknown"
            margin_fiscal_period = latest_p.fiscal_period_end or "FY"
            if margin_source_tier == "unknown":
                margin_flags.append("unknown_financial_source_tier:dcf")
            if margin_period_type == "unknown":
                margin_flags.append("unknown_financial_period_type:dcf")
        elif info.get("operatingMargins") is not None:
            reported_ebit_margin_pct = round(info["operatingMargins"] * 100.0, 2)
            margin_source_tier = "fallback"
            margin_period_type = "ttm"
            margin_fiscal_period = "TTM"
            margin_flags.append("operating_margin_proxy:dcf")
        else:
            reported_ebit_margin_pct = None
            margin_source_tier = None
            margin_period_type = None
            margin_fiscal_period = None
            margin_flags.append("missing_operating_margin:dcf")

        # 3. 5-Year Explicit DCF & True Reverse DCF Solver
        from schemas.micro_quant_schemas import HistoricalFinancialItem
        historical_3y: list[HistoricalFinancialItem] = []
        if autopsy and autopsy.periods:
            ann_periods = [p for p in autopsy.periods if getattr(p, "period_type", "") == "annual"]
            recent_ann = sorted(ann_periods[:3], key=lambda x: str(x.fiscal_period_end or ""))
            for idx, p in enumerate(recent_ann):
                r = float(p.total_revenue or 0.0)
                raw_eb = p.operating_income if p.operating_income is not None else p.ebit
                if raw_eb is None:
                    continue
                eb = float(raw_eb)
                m = round((eb / r * 100.0), 2) if r > 0 else 0.0
                fy = (p.fiscal_period_end[:4] + "A") if p.fiscal_period_end else f"Y-{len(recent_ann)-idx}A"
                gw = None
                if idx > 0 and recent_ann[idx - 1].total_revenue and recent_ann[idx - 1].total_revenue > 0:
                    pr = float(recent_ann[idx - 1].total_revenue)
                    gw = round(((r / pr) - 1.0) * 100.0, 2)
                gp = float(p.gross_profit) if getattr(p, "gross_profit", None) is not None else None
                gp_m = round((gp / r * 100.0), 2) if (gp is not None and r > 0) else None
                rd = float(p.research_and_development) if getattr(p, "research_and_development", None) is not None else None
                rd_m = round((rd / r * 100.0), 2) if (rd is not None and r > 0) else None
                sga = float(p.selling_general_admin) if getattr(p, "selling_general_admin", None) is not None else None
                sga_m = round((sga / r * 100.0), 2) if (sga is not None and r > 0) else None
                deprec = float(p.depreciation) if getattr(p, "depreciation", None) is not None else None
                raw_capex = getattr(p, "capital_expenditure", None) if getattr(p, "capital_expenditure", None) is not None else getattr(p, "capital_expenditures", None)
                capex = float(raw_capex) if raw_capex is not None else None
                capex_ratio = None
                lifecycle_phase = "unavailable"
                if capex is not None and deprec is not None and deprec > 0:
                    capex_ratio = round(abs(capex) / abs(deprec), 2)
                    if capex_ratio >= 1.5:
                        lifecycle_phase = "aggressive_expansion"
                    elif capex_ratio >= 0.85:
                        lifecycle_phase = "sustainable_maintenance"
                    else:
                        lifecycle_phase = "underinvestment_warning"
                historical_3y.append(
                    HistoricalFinancialItem(
                        fiscal_year=fy,
                        revenue=r,
                        revenue_growth_pct=gw,
                        ebit=eb,
                        ebit_margin_pct=m,
                        depreciation=deprec,
                        capex=capex,
                        fcf=float(p.free_cash_flow) if getattr(p, "free_cash_flow", None) is not None else None,
                        gross_profit=gp,
                        gross_margin_pct=gp_m,
                        research_and_development=rd,
                        rd_pct_of_rev=rd_m,
                        selling_general_admin=sga,
                        sga_pct_of_rev=sga_m,
                        capex_to_depreciation_ratio=capex_ratio,
                        reinvestment_lifecycle_phase=lifecycle_phase,
                    )
                )

        reinvest_pct = 10.0
        # Blended Forward Growth: Fundamental (Reinvestment Rate x ROIC) & Wall Street Consensus
        raw_hist_growth = growth_rates.get("revenue_growth_yoy_pct")
        street_growth_raw = info.get("revenueGrowth")
        street_growth_pct = round(street_growth_raw * 100.0, 2) if (street_growth_raw is not None and math.isfinite(street_growth_raw)) else None

        fundamental_growth_pct = None
        if roic_pct is not None and roic_pct > 0:
            fundamental_growth_pct = round((reinvest_pct / 100.0) * roic_pct, 2)
        elif roe_pct is not None and roe_pct > 0:
            fundamental_growth_pct = round((reinvest_pct / 100.0) * roe_pct, 2)

        forecast_rev_growth = raw_hist_growth
        if street_growth_pct is not None and raw_hist_growth is not None:
            bounded_street = max(min(street_growth_pct, 45.0), -20.0)
            bounded_hist = max(min(raw_hist_growth, 45.0), -20.0)
            forecast_rev_growth = round(0.5 * bounded_hist + 0.5 * bounded_street, 2)
        elif street_growth_pct is not None:
            forecast_rev_growth = round(max(min(street_growth_pct, 45.0), -20.0), 2)
        elif fundamental_growth_pct is not None and raw_hist_growth is not None:
            forecast_rev_growth = round(0.5 * raw_hist_growth + 0.5 * fundamental_growth_pct, 2)

        # Analyst consensus count: enrich with Nasdaq Equity Intelligence if missing from vendor
        analyst_opinions = info.get("numberOfAnalystOpinions")
        if (analyst_opinions is None or analyst_opinions == 0) and market == "US":
            try:
                from tools.market.terminal_v2.bootstrap import get_terminal_data_service
                nasdaq_snap = get_terminal_data_service().get_nasdaq_consensus(provider_symbol)
                if nasdaq_snap and nasdaq_snap.ratings:
                    analyst_opinions = nasdaq_snap.ratings.analyst_count
            except Exception:
                pass

        reverse_dcf_result, rdcf_flags = compute_institutional_reverse_dcf(
            ticker=resolved.raw_symbol.strip().upper(),
            market=market,
            current_price=analysis_price or 0.0,
            shares_outstanding=shares_out or 0.0,
            base_revenue=rev_base,
            base_ebit_margin_pct=reported_ebit_margin_pct,
            tax_rate=tax_rate,
            reinvestment_rate_pct=reinvest_pct,
            beta=beta,
            total_debt=total_debt_raw or 0.0,
            cash_and_equivalents=cash_raw or 0.0,
            interest_expense=interest_exp,
            macro_registry=macro_registry,
            sector=sector_name,
            forecast_revenue_growth_pct=forecast_rev_growth,
            expected_dividend_per_share=(dividend_yield_raw or 0.0) * (analysis_price or 0.0),
            ebit_margin_fiscal_period=margin_fiscal_period,
            ebit_margin_period_type=margin_period_type,
            ebit_margin_source_tier=margin_source_tier,
            consensus_target_price=info.get("targetMeanPrice"),
            consensus_target_high=info.get("targetHighPrice"),
            consensus_target_low=info.get("targetLowPrice"),
            analyst_count=analyst_opinions,
            base_revenue_period_type=base_revenue_period_type,
            roe_pct=roe_pct,
            book_value_per_share=info.get("bookValue"),
            dividend_yield_pct=dividend_yield_pct,
            historical_financials=historical_3y if historical_3y else None,
        )

        # 4. Tactical Setup (S/R, ATR, Stage, Tactical R:R) with shared price history
        tactical_setup, tac_flags = compute_tactical_setup(
            provider_symbol, market=market, current_price=analysis_price, price_history_df=df_1y
        )

        # 5. Canonical SEC Form 4 Insider Conviction (computed above with Smart Money)
        now_iso = datetime.now(timezone.utc).isoformat()
        as_of_date = atomic_snapshot.latest_ohlcv_date or now_iso[:10]

        # 5b. Point-in-Time Signal Maturity Guard (embargo buffer check)
        signal_maturity_basis = "not_evaluated"
        signal_maturity_passed: Optional[bool] = None
        signal_audit_flags: list[str] = []
        if autopsy and autopsy.periods:
            latest_disclosure = (
                getattr(autopsy.periods[0], "reported_at", None)
                or getattr(autopsy.periods[0], "fiscal_period_end", None)
            )
            if latest_disclosure:
                is_mature = audit_signal_maturity(
                    as_of_date=as_of_date,
                    filing_date=str(latest_disclosure),
                    embargo_days=2,
                )
                signal_maturity_passed = is_mature
                signal_maturity_basis = "mature" if is_mature else "immature_disclosure"
                if not is_mature:
                    signal_audit_flags.append("immature_financial_disclosure:signal_audit")

        # 5c. Horizon-Matched Consensus Expectation Gap & Historical PIT Bands
        consensus_payload: Optional[dict] = None
        if revision_data is not None:
            consensus_payload = dict(revision_data)
            if "trend_0y" not in consensus_payload:
                consensus_payload["trend_0y"] = {
                    "current_estimate": revision_data.get("estimate_current"),
                    "estimate_30d_ago": revision_data.get("estimate_30d_ago"),
                    "change_30d_pct": eps_estimate_change_30d_pct,
                }
        expectation_gap_pct, consensus_eps_momentum, consensus_status, consensus_flags = (
            compute_horizon_matched_expectation_gap(
                consensus_data=consensus_payload,
                market_implied_growth_pct=reverse_dcf_result.market_implied_growth_pct if reverse_dcf_result else None,
            )
        )

        pit_bands_dict: Optional[dict] = None
        pit_bands_status = "unavailable"
        pit_observations_count = 0
        try:
            pit_bands = compute_pit_multiple_bands(
                resolved.raw_symbol.strip().upper(),
                current_pe=pe,
            )
            pit_bands_status = pit_bands.status
            pit_observations_count = pit_bands.observations_count
            pit_bands_dict = pit_bands.model_dump()
        except Exception as pit_err:
            log.debug("compute_pit_multiple_bands skipped for %s: %s", provider_symbol, pit_err)

        flags = [
            f for f in [
                *snapshot_flags, value_flag, quality_flag, momentum_flag, *growth_flags, growth_score_flag,
                dividend_flag, solvency_flag, liquidity_flag, composite_flag, peer_flag,
                revision_fetch_flag or revision_score_flag, fcf_q_flag, debt_q_flag,
                *dcf_flags, *ownership_flags, *rdcf_flags, *tac_flags, *ins_flags, *margin_flags,
                dilution_flag, *dupont_sloan_flags, *signal_audit_flags,
            ] if f
        ]
        for name, q in [
            ("beta", beta_q), ("volatility", vol_q), ("mdd", mdd_q), ("technical_indicators", tech_q),
            ("price_percentile", price_pctile_q),
        ]:
            if not q.is_valid:
                flags.append(f"{q.stale_reason}:{name}")

        # Metric Basis Attribution (Section 7)
        ebit_basis = "annual_reported" if margin_period_type == "annual" else ("ttm" if margin_period_type == "ttm" else "fallback")
        metric_basis = {
            "de_ratio": "provider_point_in_time",
            "current_ratio": "provider_point_in_time",
            "fcf_margin": "annual_reported" if (latest_p and latest_p.free_cash_flow is not None) else ("ttm" if fcf_raw is not None else "fallback"),
            "fcf_yield": "ttm" if fcf_raw is not None else "fallback",
            "ebit_margin": ebit_basis,
            "amihud_return_price_basis": str(amihud_meta["return_price_basis"]),
            "amihud_sample_window_days": str(amihud_meta["window_days"]),
            "amihud_valid_days": str(amihud_meta["valid_days"]),
            "amihud_quality_status": str(amihud_meta["status"]),
            "amihud_currency": str(amihud_meta["currency"]),
            "dupont_3factor": dupont_basis,
            "sloan_accrual": sloan_basis,
            "signal_maturity": signal_maturity_basis,
            "consensus_expectation": consensus_status,
            "pit_multiple_bands": pit_bands_status,
            "pit_observations_count": str(pit_observations_count),
        }

        signals = QuantSignals(
            ticker=resolved.raw_symbol.strip().upper(),
            market=market,
            company_name=company_name,
            value_score=value_score,
            quality_score=quality_score,
            momentum_score=momentum_score,
            beta=beta,
            volatility_pct=volatility_pct,
            mdd_pct=mdd_pct,
            upside_pct=upside_pct,
            downside_pct=downside_pct,
            revenue_growth_yoy_pct=growth_rates["revenue_growth_yoy_pct"],
            net_income_growth_yoy_pct=growth_rates["net_income_growth_yoy_pct"],
            growth_score=growth_score,
            dividend_yield_pct=dividend_yield_pct,
            payout_ratio_pct=payout_ratio_pct,
            dividend_score=dividend_score,
            de_ratio_pct=de_ratio_pct,
            current_ratio=current_ratio,
            solvency_score=solvency_score,
            fcf_yield_pct=fcf_yield_pct,
            fcf_margin_pct=fcf_margin_pct,
            fcf_cagr_3y=fcf_cagr_3y,
            interest_coverage=interest_coverage,
            net_debt_ebitda=net_debt_ebitda,
            roic_pct=roic_pct,
            ocf_to_net_income=ocf_to_net_income,
            fcf_quality_score=fcf_quality_score,
            debt_quality_score=debt_quality_score,
            adtv_local_currency=adtv_local_currency,
            amihud_illiquidity_20d=amihud_illiquidity_20d,
            composite_score=composite_score,
            peer_sector=peer_sector,
            peer_count=peer_count,
            pe_vs_peer_avg_pct=pe_vs_peer_avg_pct,
            peer_relative_score=peer_relative_score,
            price_percentile_5y=price_percentile_5y,
            price_zscore_5y=price_zscore_5y,
            eps_revision_net_30d=eps_revision_net_30d,
            eps_estimate_change_30d_pct=eps_estimate_change_30d_pct,
            earnings_momentum_score=earnings_momentum_score,
            dcf_result=dcf_result,
            smart_money_flags=smart_money_flags,
            evaluated_at=datetime.now(timezone.utc).isoformat(),
            data_quality_flags=flags,
            piotroski_breakdown=piotroski_breakdown,
            reverse_dcf_result=reverse_dcf_result,
            tactical_setup=tactical_setup,
            insider_conviction=insider_conviction,
            atomic_market_snapshot=atomic_snapshot,
            raw_analysis_price=atomic_snapshot.latest_ohlcv_close,
            raw_analysis_price_str=atomic_snapshot.raw_analysis_price_str,
            metric_basis=metric_basis,
            sector_kpis=sector_kpis_dict,
            net_buyback_yield_pct=net_buyback_yield_pct,
            net_shareholder_yield_pct=net_shareholder_yield_pct,
            dupont_breakdown=dupont_dict,
            sloan_accrual=sloan_dict,
            composite_score_waterfall=composite_score_waterfall,
        )

        # 6. Guidance Extraction & Verification (Phase 1 & v3.1)
        autopsy_periods = autopsy.periods if autopsy and autopsy.periods else []

        from tools.market.guidance_engine import extract_verified_earnings_guidance
        guidance_context, guidance_flags = extract_verified_earnings_guidance(
            ticker=resolved.raw_symbol.strip().upper(),
            target_fiscal_period=autopsy_periods[0].fiscal_period_end if autopsy_periods else None,
        )
        for gf in guidance_flags:
            if gf not in flags:
                flags.append(gf)
        signals.earnings_guidance_context = guidance_context

        # 3-Tier Margin Metrics & Provenance (P1.6)
        # 4-Tier Margin Metrics & Provenance
        from schemas.micro_quant_schemas import MarginMetricItem
        gaap_margin_item = None
        if ttm_p and ttm_p.operating_income is not None and ttm_p.total_revenue and ttm_p.total_revenue > 0:
            gaap_margin_item = MarginMetricItem(
                value_pct=round((ttm_p.operating_income / ttm_p.total_revenue) * 100.0, 2),
                period_end=ttm_p.fiscal_period_end,
                period_type="ttm",
                definition="Standardized Rolling 4-Quarter TTM GAAP Operating Margin",
                source_provenance="SEC 10-Q/10-K Filings Standalone TTM",
            )
        elif latest_p and latest_p.operating_income is not None and latest_p.total_revenue and latest_p.total_revenue > 0:
            gaap_margin_item = MarginMetricItem(
                value_pct=round((latest_p.operating_income / latest_p.total_revenue) * 100.0, 2),
                period_end=latest_p.fiscal_period_end,
                period_type=latest_p.period_type or "annual",
                definition="Standardized Annual GAAP Operating Margin",
                source_provenance="SEC 10-K Filings Annual",
            )
        non_gaap_margin_item = None
        if guidance_context and guidance_context.verified_claims:
            for claim in guidance_context.verified_claims:
                if claim.metric_name == "non_gaap_operating_margin_midpoint_pct":
                    non_gaap_margin_item = MarginMetricItem(
                        value_pct=claim.numeric_value,
                        period_end=guidance_context.fiscal_quarter,
                        period_type="guidance_forward",
                        definition="Non-GAAP Operating Margin Guidance",
                        source_provenance=f"Earnings Call ({guidance_context.source_note_path})",
                    )
                    break
        historical_gaap_item = None
        if autopsy and autopsy.periods and len(autopsy.periods) >= 2:
            fy_p = next((p for p in autopsy.periods if p.period_type == "annual"), autopsy.periods[-1])
            if fy_p and fy_p.operating_income and fy_p.total_revenue and fy_p.total_revenue > 0:
                historical_gaap_item = MarginMetricItem(
                    value_pct=round((fy_p.operating_income / fy_p.total_revenue) * 100.0, 2),
                    period_end=fy_p.fiscal_period_end,
                    period_type="annual",
                    definition="Audited Historical GAAP Operating Margin (FY2025 Base)",
                    source_provenance="SEC 10-K Audited Filing",
                )
        provider_ebit_item = None
        if info.get("operatingMargins") is not None:
            provider_ebit_item = MarginMetricItem(
                value_pct=round(info["operatingMargins"] * 100.0, 2),
                period_end=as_of_date,
                period_type="ttm",
                definition="Vendor Reported EBIT/Operating Margin Proxy",
                source_provenance="Vendor Market Data",
            )

        signals.gaap_operating_margin = gaap_margin_item
        signals.non_gaap_operating_margin = non_gaap_margin_item
        signals.historical_gaap_operating_margin = historical_gaap_item
        signals.provider_ebit_margin = provider_ebit_item
        signals.valuation_margin_source_used = "Standardized TTM GAAP Operating Margin" if gaap_margin_item else ("Non-GAAP Guidance Margin" if non_gaap_margin_item else "Vendor EBIT Proxy")

        # Assemble Evidence Snapshot Manifest (Modularized Builder)
        from tools.market.evidence_manifest_builder import build_analysis_evidence_snapshot
        raw_payloads_dict = {
            "financials": {
                **(autopsy.model_dump() if autopsy else {}),
                "raw_inputs": {
                    "freeCashflow": fcf_raw,
                    "operatingCashflow": ocf_raw,
                    "marketCap": analysis_mcap,
                    "sharesOutstanding": shares_out,
                    "forwardEps": info.get("forwardEps"),
                    "trailingEps": info.get("trailingEps"),
                    "totalDebt": total_debt_raw,
                    "totalCash": cash_raw,
                    "currentPrice": analysis_price,
                    "fiscal_period_end": latest_p.fiscal_period_end if latest_p else None,
                },
            } if (autopsy or fcf_raw is not None) else None,
            "market_data": atomic_snapshot.model_dump(),
            "technical_ohlcv": {
                "ticker": ticker,
                "market": market,
                "provider_symbol": provider_symbol,
                "price_basis": "unadjusted_close",
                "interval": "1d",
                "timezone": "Asia/Bangkok" if market == "TH" else "America/New_York",
                "count": len(df_1y) if df_1y is not None else 0,
                "source_as_of": as_of_date,
                "retrieved_at": now_iso,
                "bars": df_1y.reset_index().to_dict(orient="records") if (df_1y is not None and not df_1y.empty) else [],
            } if (df_1y is not None and not df_1y.empty) else None,
            "technical_ohlcv_1y": {
                "provider_symbol": provider_symbol,
                "price_basis": "unadjusted_close",
                "interval": "1d",
                "timezone": "America/New_York",
                "bars_count": len(df_1y) if df_1y is not None else 0,
                "source_as_of": as_of_date,
                "retrieved_at": now_iso,
                "bars": df_1y.reset_index().to_dict(orient="records") if (df_1y is not None and not df_1y.empty) else [],
            } if (df_1y is not None and not df_1y.empty) else None,
            "price_context_5y": {
                "price_percentile_5y": price_percentile_5y,
                "price_zscore_5y": price_zscore_5y,
                "is_valid": price_pctile_q.is_valid if price_pctile_q else False,
            } if price_percentile_5y is not None else None,
            "mdd_3y": {
                "mdd_pct": mdd_pct,
                "is_valid": mdd_q.is_valid if mdd_q else False,
            } if mdd_pct is not None else None,
            "beta_2y_stock": {
                "beta": beta,
                "volatility_pct": volatility_pct,
                "is_valid": beta_q.is_valid if beta_q else False,
            } if beta is not None else None,
            "beta_2y_benchmark": {
                "benchmark": benchmark,
                "is_valid": beta_q.is_valid if beta_q else False,
            } if beta is not None else None,
            "analyst_targets": {
                "targetMeanPrice": info.get("targetMeanPrice"),
                "targetHighPrice": info.get("targetHighPrice"),
                "targetLowPrice": info.get("targetLowPrice"),
                "upside_pct": upside_pct,
                "downside_pct": downside_pct,
            } if (upside_pct is not None or info.get("targetMeanPrice") is not None) else None,
            "peer_group": peer_metrics if peer_metrics and len(peer_metrics) >= 2 else None,
            "consensus": revision_data if revision_data else None,
            "ownership": {
                "heldPercentInstitutions": info.get("heldPercentInstitutions"),
                "heldPercentInsiders": info.get("heldPercentInsiders"),
                "shortPercentOfFloat": info.get("shortPercentOfFloat"),
                "sharesShort": info.get("sharesShort"),
                "shortRatio": info.get("shortRatio"),
                "insider_transactions": insider_conviction.model_dump() if insider_conviction else None,
            },
            "macro_valuation": {
                "risk_free_rate_pct": (dcf_result.risk_free_rate_pct if dcf_result else (2.75 if market == "TH" else 4.25)),
                "erp_pct": (dcf_result.erp_pct if dcf_result else 2.10),
                "observable_refs": (dcf_result.observable_refs if dcf_result else []),
                "source_uri": "macro:///registry",
            },
            "valuation_parameters": {
                "methodology_version": "2.1.0",
                "base_revenue": rev_base,
                "base_ebit_margin_pct": reported_ebit_margin_pct or 0.0,
                "shares_outstanding": shares_out,
                "total_cash": cash_raw or 0.0,
                "total_debt": total_debt_raw or 0.0,
                "beta": beta or 1.0,
                "reinvestment_rate_pct": 10.0,
                "terminal_growth_cap_pct": 2.5,
                "terminal_growth_effective_pct": round(min(0.025, (dcf_result.risk_free_rate_pct if dcf_result else 4.25) / 100.0) * 100.0, 2),
                "valuation_thresholds": {
                    "undervalued_upside_min": 15.0,
                    "overvalued_upside_max": -15.0,
                },
            },
            "guidance": guidance_context.model_dump() if guidance_context else None,
        }

        evidence_snapshot = build_analysis_evidence_snapshot(
            ticker=resolved.raw_symbol.strip().upper(),
            market=market,
            as_of_date=as_of_date,
            raw_payloads=raw_payloads_dict,
            corporate_actions_info={"ticker": ticker, "market": market},
            derived_features={
                "atr_14": tactical_setup.atr_14 if tactical_setup else None,
                "price_stage": tactical_setup.price_stage if tactical_setup else None,
                "piotroski_f_score": piotroski_breakdown.f_score if piotroski_breakdown else None,
                "reverse_dcf_implied_growth": reverse_dcf_result.market_implied_growth_pct if reverse_dcf_result else None,
                "signal_maturity_passed": signal_maturity_passed,
                "consensus_expectation": {
                    "expectation_gap_pct": expectation_gap_pct,
                    "eps_momentum_score": consensus_eps_momentum,
                    "status": consensus_status,
                    "flags": consensus_flags,
                },
                "pit_multiple_bands": pit_bands_dict,
            },
            data_quality_flags=flags,
        )
        signals.evidence_snapshot = evidence_snapshot

        # 6. Deterministic Scorecard & Falsifiers
        macro_context = load_latest_macro_regime_context()
        scorecard, falsifiers, sc_flags = compute_deterministic_scorecard(
            ticker=resolved.raw_symbol.strip().upper(),
            market=market,
            quant_signals=signals,
            piotroski=piotroski_breakdown,
            dcf=reverse_dcf_result,
            guidance=guidance_context,
            tactical=tactical_setup,
            insider=insider_conviction,
            macro_context=macro_context,
        )
        signals.deterministic_scorecard = scorecard
        signals.thesis_falsifiers = falsifiers
        for sc_flag in sc_flags:
            if sc_flag not in signals.data_quality_flags:
                signals.data_quality_flags.append(sc_flag)

        return signals.model_dump_json()
    except Exception as e:
        log.warning("compute_equity_quant_signals failed for %s (%s): %s", ticker, market, e)
        from schemas.micro_quant_schemas import QuantSignalsFailureResult
        failure_dto = QuantSignalsFailureResult(
            run_status="error",
            ticker=ticker,
            market=market,
            error_code="QUANT_EXECUTION_ERROR",
            error_message=str(e),
            data_quality_flags=["execution_exception"],
            evaluated_at=datetime.now(timezone.utc).isoformat(),
        )
        return failure_dto.model_dump_json()
