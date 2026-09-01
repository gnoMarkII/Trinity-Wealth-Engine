"""Typed Financial Autopsy สำหรับดึงและคำนวณข้อมูลการเงินย้อนหลัง 3-5 ปีจริง (FCF, Debt, Payout Ratio)"""
from typing import Optional, Literal, Dict, Tuple, List, Any
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
import time
import requests
import pandas as pd
import yfinance as yf
from pydantic import BaseModel, model_validator
from langchain_core.tools import tool
from core.logger import get_logger
from schemas.micro_quant_schemas import PiotroskiFScoreBreakdown, DataStatus
from tools.market.asset_resolver import ResolvedAsset, resolve_asset, AssetClass

log = get_logger(__name__)

from threading import BoundedSemaphore

_SHARED_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="yf_autopsy_pool")
_POOL_SEMAPHORE = BoundedSemaphore(4)


class ProviderBusyError(Exception):
    """Raised when bounded pool semaphore cannot be acquired within admission deadline."""
    pass


def _run_with_deadline(func, args, timeout_sec: float, admission_timeout: float = 0.5):
    if not _POOL_SEMAPHORE.acquire(timeout=admission_timeout):
        raise ProviderBusyError("PROVIDER_BUSY: Autopsy thread pool saturated")
    def _worker():
        try:
            return func(*args)
        finally:
            _POOL_SEMAPHORE.release()
    future = _SHARED_POOL.submit(_worker)
    try:
        return future.result(timeout=timeout_sec)
    except (FuturesTimeoutError, TimeoutError) as e:
        future.cancel()
        raise FuturesTimeoutError(f"Operation timed out after {timeout_sec}s") from e


_run_with_hard_timeout = _run_with_deadline


class FinancialAutopsyPeriod(BaseModel):
    fiscal_period_end: str
    free_cash_flow: Optional[float] = None
    operating_cash_flow: Optional[float] = None
    capital_expenditure: Optional[float] = None
    total_debt: Optional[float] = None
    total_revenue: Optional[float] = None
    net_income: Optional[float] = None
    dividends_paid: Optional[float] = None
    payout_ratio_pct: Optional[float] = None
    ebit: Optional[float] = None
    operating_income: Optional[float] = None
    interest_expense: Optional[float] = None
    tax_expense: Optional[float] = None
    income_before_tax: Optional[float] = None
    # Additive fields for Piotroski 9 Criteria & Quality of Earnings
    gross_profit: Optional[float] = None
    total_assets: Optional[float] = None
    current_assets: Optional[float] = None
    current_liabilities: Optional[float] = None
    long_term_debt: Optional[float] = None
    stockholders_equity: Optional[float] = None
    shares_outstanding: Optional[float] = None
    source_tier: Optional[Literal["filing_authoritative", "primary_best_effort", "fallback", "unknown"]] = None
    period_type: Optional[Literal["annual", "quarterly", "ttm", "unknown"]] = None


class FinancialAutopsySnapshot(BaseModel):
    ticker: str
    provider_symbol: str
    market: Literal["TH", "US"]
    currency: str
    unit: str = "raw"
    source: str = "Yahoo Finance (yfinance)"
    source_tier: Optional[Literal["filing_authoritative", "primary_best_effort", "fallback", "unknown"]] = None
    retrieval_timestamp: str
    periods: List[FinancialAutopsyPeriod]
    current_pe: Optional[float] = None
    forward_pe: Optional[float] = None
    market_cap: Optional[float] = None
    health_summary: Optional[str] = None

    @property
    def market_cap_formatted(self) -> str:
        if self.market_cap is None:
            return "N/A"
        return f"{self.market_cap:,.2f} {self.currency}"

    @property
    def fcf_formatted(self) -> str:
        if not self.periods or self.periods[0].free_cash_flow is None:
            return "N/A"
        return f"{self.periods[0].free_cash_flow:,.2f} {self.currency}"

    @property
    def total_debt_formatted(self) -> str:
        if not self.periods or self.periods[0].total_debt is None:
            return "N/A"
        return f"{self.periods[0].total_debt:,.2f} {self.currency}"

    @property
    def revenue_formatted(self) -> str:
        if not self.periods or self.periods[0].total_revenue is None:
            return "N/A"
        return f"{self.periods[0].total_revenue:,.2f} {self.currency}"

    @property
    def net_income_formatted(self) -> str:
        if not self.periods or self.periods[0].net_income is None:
            return "N/A"
        return f"{self.periods[0].net_income:,.2f} {self.currency}"


class FinancialAutopsyFetchResult(BaseModel):
    asset: ResolvedAsset
    status: Literal["success", "unavailable", "error"]
    snapshot: Optional[FinancialAutopsySnapshot] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None

    @property
    def availability_block(self) -> bool:
        return self.status != "success"

    @model_validator(mode="after")
    def validate_state(self):
        if self.status == "success" and self.snapshot is None:
            raise ValueError("Successful result requires snapshot")
        if self.status != "success" and self.snapshot is not None:
            raise ValueError("Failed result must not contain snapshot")
        if self.status == "error" and not self.error_message:
            raise ValueError("Error result requires error_message")
        return self


# --- Caches ---
_AUTOPSY_SUCCESS_CACHE: Dict[str, Tuple[FinancialAutopsyFetchResult, float]] = {}
_AUTOPSY_ERROR_CACHE: Dict[str, Tuple[FinancialAutopsyFetchResult, float]] = {}
_SUCCESS_TTL_SECONDS = 12 * 3600
_ERROR_TTL_SECONDS = 60.0


def _to_float(val: Any) -> Optional[float]:
    if val is None:
        return None
    try:
        f = float(val)
        if pd.isna(f):
            return None
        return f
    except (TypeError, ValueError):
        return None


def _extract_df_val(df: Any, col: Any, candidate_rows: List[str]) -> Optional[float]:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty or col not in df.columns:
        return None
    for row_name in candidate_rows:
        if row_name in df.index:
            v = _to_float(df.loc[row_name, col])
            if v is not None:
                return v
    return None


def _extract_df_val_aligned(df: Any, target_dt: Any, candidate_rows: List[str], tolerance_days: int = 45) -> Optional[float]:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None
    best_col = None
    min_diff = tolerance_days + 1
    try:
        if hasattr(pd, "to_datetime"):
            t_dt = pd.to_datetime(target_dt)
            for col in df.columns:
                try:
                    c_dt = pd.to_datetime(col)
                    diff = abs((c_dt - t_dt).days)
                    if diff <= tolerance_days and diff < min_diff:
                        min_diff = diff
                        best_col = col
                except Exception:
                    continue
    except Exception:
        pass

    if best_col is None and target_dt in df.columns:
        best_col = target_dt

    if best_col is not None and best_col in df.columns:
        return _extract_df_val(df, best_col, candidate_rows)
    return None


def _fetch_autopsy_raw(provider_symbol: str, timeout: float = 10.0) -> Tuple[dict, Any, Any, Any]:
    session = None
    try:
        from curl_cffi import requests as c_requests
        # impersonate="chrome" จำเป็น — Session เปล่าไม่มี browser TLS fingerprint ทำให้ Yahoo
        # ตรวจจับเป็น bot และคืน YFRateLimitError ทันทีตั้งแต่ request แรก (ไม่เกี่ยวกับจำนวนครั้งที่ยิงจริง)
        session = c_requests.Session(timeout=timeout, impersonate="chrome")
    except Exception:
        pass
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            if session is not None:
                tk = yf.Ticker(provider_symbol, session=session)
            else:
                tk = yf.Ticker(provider_symbol)
            info = tk.info or {}
            financials = tk.financials
            cashflow = tk.cashflow
            balance_sheet = tk.balance_sheet
            return info, financials, cashflow, balance_sheet
        except Exception as e:
            if "Too Many Requests" in str(e) or "Rate limited" in str(e):
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
            raise


def get_financial_autopsy(resolved_asset: ResolvedAsset) -> FinancialAutopsyFetchResult:
    """ดึงข้อมูล Snapshot การเงินเชิงลึกย้อนหลัง 3-5 ปีจริง โดย Align วันที่งบให้ตรงกัน และจัดการ timeout 10s พร้อม cache"""
    prov_sym = resolved_asset.provider_symbol or resolved_asset.raw_symbol
    now = time.time()

    # Check eligibility
    if not resolved_asset.eligible_for_financial_autopsy:
        return FinancialAutopsyFetchResult(
            asset=resolved_asset,
            status="unavailable",
            error_code="INELIGIBLE_ASSET",
            error_message=f"Symbol {resolved_asset.raw_symbol} ({resolved_asset.asset_class}) is not eligible for financial autopsy",
        )

    # Check success cache
    if prov_sym in _AUTOPSY_SUCCESS_CACHE:
        res, ts = _AUTOPSY_SUCCESS_CACHE[prov_sym]
        if now - ts < _SUCCESS_TTL_SECONDS:
            return res
        else:
            del _AUTOPSY_SUCCESS_CACHE[prov_sym]

    # Check error cache
    if prov_sym in _AUTOPSY_ERROR_CACHE:
        res, ts = _AUTOPSY_ERROR_CACHE[prov_sym]
        if now - ts < _ERROR_TTL_SECONDS:
            return res
        else:
            del _AUTOPSY_ERROR_CACHE[prov_sym]

    try:
        info, financials, cashflow, balance_sheet = _run_with_deadline(_fetch_autopsy_raw, (prov_sym, 10.0), timeout_sec=10.0, admission_timeout=0.5)
    except ProviderBusyError as e:
        log.warning("get_financial_autopsy busy for %s: %s", prov_sym, e)
        err_res = FinancialAutopsyFetchResult(
            asset=resolved_asset,
            status="error",
            error_code="PROVIDER_BUSY",
            error_message="Autopsy thread pool saturated (max_workers=4)",
        )
        _AUTOPSY_ERROR_CACHE[prov_sym] = (err_res, now)
        return err_res
    except (FuturesTimeoutError, TimeoutError):
        log.warning("get_financial_autopsy timed out after 10s for %s", prov_sym)
        err_res = FinancialAutopsyFetchResult(
            asset=resolved_asset,
            status="error",
            error_code="PROVIDER_TIMEOUT",
            error_message="Yahoo Finance request timed out after 10s",
        )
        _AUTOPSY_ERROR_CACHE[prov_sym] = (err_res, now)
        return err_res
    except Exception as e:
        log.warning("get_financial_autopsy failed for %s: %s", prov_sym, e)
        err_res = FinancialAutopsyFetchResult(
            asset=resolved_asset,
            status="error",
            error_code="PROVIDER_ERROR",
            error_message=str(e),
        )
        _AUTOPSY_ERROR_CACHE[prov_sym] = (err_res, now)
        return err_res

    market: Literal["TH", "US"] = resolved_asset.market or ("TH" if prov_sym.endswith(".BK") else "US")
    currency = info.get("currency") or ("THB" if market == "TH" else "USD")
    current_pe = _to_float(info.get("trailingPE"))
    forward_pe = _to_float(info.get("forwardPE"))
    market_cap = _to_float(info.get("marketCap"))

    # Gather all available date columns across statements
    all_cols = set()
    for df in (financials, cashflow, balance_sheet):
        if df is not None and isinstance(df, pd.DataFrame) and not df.empty:
            all_cols.update(df.columns)

    if not all_cols:
        unavail_res = FinancialAutopsyFetchResult(
            asset=resolved_asset,
            status="unavailable",
            error_code="NO_FINANCIAL_DATA",
            error_message=f"No statement data available for {prov_sym}",
        )
        _AUTOPSY_ERROR_CACHE[prov_sym] = (unavail_res, now)
        return unavail_res

    # Sort dates descending (most recent first) and cluster within 45 days for canonical fiscal periods
    sorted_raw = sorted(list(all_cols), key=lambda c: pd.to_datetime(c) if hasattr(pd, "to_datetime") else str(c), reverse=True)
    canonical_cols = []
    for col in sorted_raw:
        try:
            if hasattr(pd, "to_datetime"):
                col_dt = pd.to_datetime(col)
                if not any(abs((pd.to_datetime(c) - col_dt).days) <= 45 for c in canonical_cols):
                    canonical_cols.append(col)
            else:
                if col not in canonical_cols:
                    canonical_cols.append(col)
        except Exception:
            if col not in canonical_cols:
                canonical_cols.append(col)
    sorted_cols = canonical_cols[:5]

    periods: List[FinancialAutopsyPeriod] = []
    for col in sorted_cols:
        if hasattr(col, "strftime"):
            dt_str = col.strftime("%Y-%m-%d")
        else:
            try:
                dt_str = pd.to_datetime(col).strftime("%Y-%m-%d")
            except Exception:
                dt_str = str(col)[:10]

        # Cashflow items
        fcf = _extract_df_val_aligned(cashflow, col, ["Free Cash Flow"])
        ocf = _extract_df_val_aligned(cashflow, col, ["Operating Cash Flow", "Total Cash From Operating Activities"])
        capex = _extract_df_val_aligned(cashflow, col, ["Capital Expenditure", "Capital Expenditures"])
        div_paid = _extract_df_val_aligned(cashflow, col, ["Common Stock Dividend Paid", "Cash Dividends Paid", "Dividends Paid"])

        if fcf is None and ocf is not None and capex is not None:
            fcf = ocf - abs(capex)

        # Balance sheet items
        total_debt = _extract_df_val_aligned(
            balance_sheet, col,
            ["Total Debt", "Total Debt And Capital Lease Obligation"]
        )
        if total_debt is None:
            lt_debt = _extract_df_val_aligned(balance_sheet, col, ["Long Term Debt And Capital Lease Obligation", "Long Term Debt", "Long Term Debt Noncurrent"])
            st_debt = _extract_df_val_aligned(balance_sheet, col, ["Short Term Debt And Capital Lease Obligation", "Short Term Debt", "Current Debt", "Short Term Borrowings"])
            if lt_debt is not None or st_debt is not None:
                total_debt = (lt_debt or 0.0) + (st_debt or 0.0)

        # Expanded Balance Sheet Items for Piotroski F-Score
        tot_assets = _extract_df_val_aligned(balance_sheet, col, ["Total Assets", "Assets"])
        curr_assets = _extract_df_val_aligned(balance_sheet, col, ["Current Assets", "Total Current Assets", "Assets Current"])
        curr_liab = _extract_df_val_aligned(balance_sheet, col, ["Current Liabilities", "Total Current Liabilities", "Liabilities Current"])
        lt_debt_val = _extract_df_val_aligned(balance_sheet, col, ["Long Term Debt And Capital Lease Obligation", "Long Term Debt", "Long Term Debt Noncurrent"])
        stock_equity = _extract_df_val_aligned(balance_sheet, col, ["Stockholders Equity", "Total Stockholder Equity", "Common Stock Equity", "Total Equity Gross Minority Interest"])
        shares_out = _extract_df_val_aligned(balance_sheet, col, ["Share Issued", "Ordinary Shares Number", "Common Stock"])

        # Financials / Income statement items
        rev = _extract_df_val_aligned(financials, col, ["Total Revenue", "Operating Revenue", "Revenue"])
        gross_prof = _extract_df_val_aligned(financials, col, ["Gross Profit"])
        net_inc = _extract_df_val_aligned(financials, col, ["Net Income", "Net Income Common Stockholders", "Net Income Loss"])
        ebit_val = _extract_df_val_aligned(financials, col, ["EBIT", "Total Operating Income As Reported", "Operating Income"])
        op_income_val = _extract_df_val_aligned(financials, col, ["Operating Income", "Total Operating Income As Reported", "EBIT"])
        interest_exp = _extract_df_val_aligned(financials, col, ["Interest Expense", "Interest Expense Non Operating"])
        tax_exp = _extract_df_val_aligned(financials, col, ["Tax Provision", "Tax Effect Of Unusual Items"])
        pretax_inc = _extract_df_val_aligned(financials, col, ["Pretax Income", "Net Income From Continuing Operation Net Minority Interest"])

        # Payout ratio
        payout_pct = None
        if div_paid is not None and net_inc is not None and net_inc > 0:
            payout_pct = round((abs(div_paid) / net_inc) * 100.0, 2)

        # Only add period if at least one quantitative metric exists
        if any(v is not None for v in (fcf, ocf, capex, total_debt, rev, net_inc, div_paid, ebit_val, op_income_val, tot_assets)):
            periods.append(
                FinancialAutopsyPeriod(
                    fiscal_period_end=dt_str,
                    free_cash_flow=fcf,
                    operating_cash_flow=ocf,
                    capital_expenditure=capex,
                    total_debt=total_debt,
                    total_revenue=rev,
                    net_income=net_inc,
                    dividends_paid=div_paid,
                    payout_ratio_pct=payout_pct,
                    ebit=ebit_val,
                    operating_income=op_income_val,
                    interest_expense=interest_exp,
                    tax_expense=tax_exp,
                    income_before_tax=pretax_inc,
                    gross_profit=gross_prof,
                    total_assets=tot_assets,
                    current_assets=curr_assets,
                    current_liabilities=curr_liab,
                    long_term_debt=lt_debt_val,
                    stockholders_equity=stock_equity,
                    shares_outstanding=shares_out,
                    source_tier="primary_best_effort",
                    period_type="annual",
                )
            )

    if not periods:
        unavail_res = FinancialAutopsyFetchResult(
            asset=resolved_asset,
            status="unavailable",
            error_code="NO_FINANCIAL_DATA",
            error_message=f"No periods could be extracted for {prov_sym}",
        )
        _AUTOPSY_ERROR_CACHE[prov_sym] = (unavail_res, now)
        return unavail_res

    health_notes = []
    latest = periods[0] if periods else None
    if latest and latest.free_cash_flow is not None and latest.total_debt is not None:
        if latest.total_debt > 0:
            ratio = latest.free_cash_flow / latest.total_debt
            if ratio < 0:
                health_notes.append("Negative FCF relative to Debt")
            elif ratio < 0.1:
                health_notes.append("Low FCF/Debt coverage (<10%)")
            else:
                health_notes.append(f"FCF/Debt: {ratio:.2f}x")
        elif latest.free_cash_flow > 0:
            health_notes.append("Positive FCF with zero reported debt")
    if current_pe is not None:
        health_notes.append(f"PE: {current_pe:.1f}")
    if forward_pe is not None:
        health_notes.append(f"Fwd PE: {forward_pe:.1f}")
    health_summary = "; ".join(health_notes) if health_notes else "Basic financial statements verified"

    snapshot = FinancialAutopsySnapshot(
        ticker=resolved_asset.raw_symbol,
        provider_symbol=prov_sym,
        market=market,
        currency=str(currency),
        unit="raw",
        source="Yahoo Finance (yfinance)",
        source_tier="primary_best_effort",
        retrieval_timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        periods=periods,
        current_pe=current_pe,
        forward_pe=forward_pe,
        market_cap=market_cap,
        health_summary=health_summary,
    )

    succ_res = FinancialAutopsyFetchResult(
        asset=resolved_asset,
        status="success",
        snapshot=snapshot,
    )
    _AUTOPSY_SUCCESS_CACHE[prov_sym] = (succ_res, now)
    return succ_res


def calculate_piotroski_f_score(
    periods: List[FinancialAutopsyPeriod],
    sector: Optional[str] = None,
) -> PiotroskiFScoreBreakdown:
    """คำนวณ Piotroski F-Score 9 ข้อเต็มรูปแบบ (0-9) พร้อมกฎ Sector Exclusion และ Data Status ชัดเจน"""
    excluded_sectors = ["financials", "financial services", "real estate", "banks", "insurance", "reit"]
    if sector and sector.strip().lower() in excluded_sectors:
        return PiotroskiFScoreBreakdown(
            f_score=None,
            status="not_applicable",
            is_eligible=False,
            exclusion_reason=f"Sector '{sector}' is excluded from standard Piotroski F-Score (bank/financial/REIT structure)",
        )

    if not periods or len(periods) < 2:
        return PiotroskiFScoreBreakdown(
            f_score=None,
            status="unavailable" if not periods else "partial",
            is_eligible=True,
            exclusion_reason="At least 2 historical periods are required to compute Piotroski F-Score",
        )

    t0 = periods[0]
    t1 = periods[1]

    roa_pos = (t0.net_income > 0) if (t0.net_income is not None) else None

    cfo_pos = (t0.operating_cash_flow > 0) if (t0.operating_cash_flow is not None) else None

    delta_roa = None
    if (t0.net_income is not None and t0.total_assets and t0.total_assets > 0 and
        t1.net_income is not None and t1.total_assets and t1.total_assets > 0):
        roa0 = t0.net_income / t0.total_assets
        roa1 = t1.net_income / t1.total_assets
        delta_roa = roa0 > roa1
    elif t0.net_income is not None and t1.net_income is not None:
        delta_roa = t0.net_income > t1.net_income

    accrual = None
    if t0.operating_cash_flow is not None and t0.net_income is not None:
        accrual = t0.operating_cash_flow > t0.net_income

    delta_lev = None
    lt_debt0 = t0.long_term_debt if t0.long_term_debt is not None else t0.total_debt
    lt_debt1 = t1.long_term_debt if t1.long_term_debt is not None else t1.total_debt
    if lt_debt0 is not None and lt_debt1 is not None:
        if t0.total_assets and t0.total_assets > 0 and t1.total_assets and t1.total_assets > 0:
            lev0 = lt_debt0 / t0.total_assets
            lev1 = lt_debt1 / t1.total_assets
            delta_lev = lev0 <= lev1
        else:
            delta_lev = lt_debt0 <= lt_debt1

    delta_liq = None
    if (t0.current_assets is not None and t0.current_liabilities and t0.current_liabilities > 0 and
        t1.current_assets is not None and t1.current_liabilities and t1.current_liabilities > 0):
        cr0 = t0.current_assets / t0.current_liabilities
        cr1 = t1.current_assets / t1.current_liabilities
        delta_liq = cr0 > cr1

    no_dilution = None
    if t0.shares_outstanding is not None and t1.shares_outstanding is not None and t1.shares_outstanding > 0:
        no_dilution = t0.shares_outstanding <= (t1.shares_outstanding * 1.01)

    delta_gm = None
    if (t0.gross_profit is not None and t0.total_revenue and t0.total_revenue > 0 and
        t1.gross_profit is not None and t1.total_revenue and t1.total_revenue > 0):
        gm0 = t0.gross_profit / t0.total_revenue
        gm1 = t1.gross_profit / t1.total_revenue
        delta_gm = gm0 > gm1
    elif (t0.ebit is not None and t0.total_revenue and t0.total_revenue > 0 and
          t1.ebit is not None and t1.total_revenue and t1.total_revenue > 0):
        delta_gm = (t0.ebit / t0.total_revenue) > (t1.ebit / t1.total_revenue)

    delta_at = None
    if (t0.total_revenue is not None and t0.total_assets and t0.total_assets > 0 and
        t1.total_revenue is not None and t1.total_assets and t1.total_assets > 0):
        at0 = t0.total_revenue / t0.total_assets
        at1 = t1.total_revenue / t1.total_assets
        delta_at = at0 > at1
    elif t0.total_revenue is not None and t1.total_revenue is not None:
        delta_at = t0.total_revenue > t1.total_revenue

    criteria_list = [
        roa_pos, cfo_pos, delta_roa, accrual,
        delta_lev, delta_liq, no_dilution,
        delta_gm, delta_at
    ]

    valid_count = sum(1 for c in criteria_list if c is not None)
    if valid_count < 5:
        return PiotroskiFScoreBreakdown(
            roa_positive=roa_pos,
            cfo_positive=cfo_pos,
            delta_roa_positive=delta_roa,
            accrual_quality=accrual,
            delta_leverage_improved=delta_lev,
            delta_liquidity_improved=delta_liq,
            no_share_dilution=no_dilution,
            delta_gross_margin_improved=delta_gm,
            delta_asset_turnover_improved=delta_at,
            f_score=None,
            status="partial",
            is_eligible=True,
            exclusion_reason=f"Insufficient data to score F-Score (only {valid_count}/9 criteria available)",
        )

    score = sum(1 for c in criteria_list if c is True)
    return PiotroskiFScoreBreakdown(
        roa_positive=roa_pos,
        cfo_positive=cfo_pos,
        delta_roa_positive=delta_roa,
        accrual_quality=accrual,
        delta_leverage_improved=delta_lev,
        delta_liquidity_improved=delta_liq,
        no_share_dilution=no_dilution,
        delta_gross_margin_improved=delta_gm,
        delta_asset_turnover_improved=delta_at,
        f_score=score,
        status="available" if valid_count == 9 else "partial",
        is_eligible=True,
    )


@tool
def ingest_financial_autopsy(symbol: str, market: str = "US") -> str:
    """ดึงข้อมูล Financial Autopsy เชิงลึกของหุ้น (FCF, Debt, Payout Ratio ย้อนหลัง 3-5 ปี) พร้อมตรวจสอบสิทธิของสินทรัพย์

    Args:
        symbol (str): Ticker symbol เช่น 'NVDA', 'PTT'
        market (str): 'US' หรือ 'TH' (default: 'US')
    """
    mkt: Optional[Literal["TH", "US"]] = "TH" if market.upper() == "TH" else "US"
    asset = resolve_asset(symbol, market_hint=mkt)
    if not asset.eligible_for_financial_autopsy:
        res = FinancialAutopsyFetchResult(
            asset=asset,
            status="unavailable",
            error_code="INELIGIBLE_ASSET",
            error_message=f"Symbol {symbol} ({asset.asset_class}) is not eligible for financial autopsy",
        )
        return res.model_dump_json()

    result = get_financial_autopsy(asset)
    return result.model_dump_json()
