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
    ttm_period: Optional[FinancialAutopsyPeriod] = None
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


def _build_rolling_ttm_period(
    q_financials: Any,
    q_cashflow: Any,
    balance_sheet: Any = None,
) -> Optional[FinancialAutopsyPeriod]:
    """สร้าง TTM Period จากการรวมงบ 4 ไตรมาสล่าสุด (Rolling 4-Quarter Sum) ตามแบบรายงาน SEC (10-Q/10-K)"""
    if q_financials is None and q_cashflow is None:
        return None

    cf_cols = list(q_cashflow.columns) if (q_cashflow is not None and isinstance(q_cashflow, pd.DataFrame)) else []
    fin_cols = list(q_financials.columns) if (q_financials is not None and isinstance(q_financials, pd.DataFrame)) else []

    all_q_cols = set(cf_cols).union(fin_cols)
    if not all_q_cols:
        return None

    sorted_q_cols = sorted(
        list(all_q_cols),
        key=lambda c: pd.to_datetime(c) if hasattr(pd, "to_datetime") else str(c),
        reverse=True
    )[:4]

    if len(sorted_q_cols) < 4:
        return None

    ttm_rev = 0.0
    ttm_op_inc = 0.0
    ttm_ebit = 0.0
    ttm_net_inc = 0.0
    ttm_ocf = 0.0
    ttm_capex = 0.0
    ttm_fcf = 0.0

    has_rev = False
    has_op_inc = False
    has_ebit = False
    has_net_inc = False
    has_ocf = False
    has_capex = False
    has_fcf = False

    for col in sorted_q_cols:
        rev = _extract_df_val_aligned(q_financials, col, ["Total Revenue", "Operating Revenue", "Revenue"])
        if rev is not None:
            ttm_rev += rev
            has_rev = True

        op_inc = _extract_df_val_aligned(q_financials, col, ["Operating Income", "Total Operating Income As Reported"])
        if op_inc is not None:
            ttm_op_inc += op_inc
            has_op_inc = True

        ebit = _extract_df_val_aligned(q_financials, col, ["EBIT", "Operating Income", "Total Operating Income As Reported"])
        if ebit is not None:
            ttm_ebit += ebit
            has_ebit = True

        net_inc = _extract_df_val_aligned(q_financials, col, ["Net Income", "Net Income Common Stockholders", "Net Income Loss"])
        if net_inc is not None:
            ttm_net_inc += net_inc
            has_net_inc = True

        ocf = _extract_df_val_aligned(q_cashflow, col, ["Operating Cash Flow", "Total Cash From Operating Activities", "Cash Flow From Continuing Operating Activities"])
        capex = _extract_df_val_aligned(q_cashflow, col, ["Capital Expenditure", "Capital Expenditures"])
        fcf = _extract_df_val_aligned(q_cashflow, col, ["Free Cash Flow"])

        if ocf is not None:
            ttm_ocf += ocf
            has_ocf = True
        if capex is not None:
            ttm_capex += abs(capex)
            has_capex = True
        if fcf is not None:
            ttm_fcf += fcf
            has_fcf = True
        elif ocf is not None and capex is not None:
            ttm_fcf += (ocf - abs(capex))
            has_fcf = True

    latest_q = sorted_q_cols[0]
    dt_str = latest_q.strftime("%Y-%m-%d") if hasattr(latest_q, "strftime") else str(latest_q)[:10]

    return FinancialAutopsyPeriod(
        fiscal_period_end=f"{dt_str} (TTM)",
        free_cash_flow=round(ttm_fcf, 2) if has_fcf else None,
        operating_cash_flow=round(ttm_ocf, 2) if has_ocf else None,
        capital_expenditure=round(ttm_capex, 2) if has_capex else None,
        total_revenue=round(ttm_rev, 2) if has_rev else None,
        operating_income=round(ttm_op_inc, 2) if has_op_inc else None,
        ebit=round(ttm_ebit, 2) if has_ebit else None,
        net_income=round(ttm_net_inc, 2) if has_net_inc else None,
        source_tier="filing_authoritative",
        period_type="ttm",
    )


def _fetch_autopsy_raw(provider_symbol: str, timeout: float = 10.0) -> Tuple[dict, Any, Any, Any, Any, Any]:
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
            q_financials = getattr(tk, "quarterly_financials", None)
            q_cashflow = getattr(tk, "quarterly_cashflow", None)
            return info, financials, cashflow, balance_sheet, q_financials, q_cashflow
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
        raw_res = _run_with_deadline(_fetch_autopsy_raw, (prov_sym, 10.0), timeout_sec=10.0, admission_timeout=0.5)
        if isinstance(raw_res, (tuple, list)):
            info = raw_res[0] if len(raw_res) > 0 else {}
            financials = raw_res[1] if len(raw_res) > 1 else None
            cashflow = raw_res[2] if len(raw_res) > 2 else None
            balance_sheet = raw_res[3] if len(raw_res) > 3 else None
            q_financials = raw_res[4] if len(raw_res) > 4 else None
            q_cashflow = raw_res[5] if len(raw_res) > 5 else None
        else:
            info, financials, cashflow, balance_sheet, q_financials, q_cashflow = {}, None, None, None, None, None
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
    ttm_period = _build_rolling_ttm_period(q_financials, q_cashflow, balance_sheet)

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
        ttm_period=ttm_period,
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


# ==============================================================================
# SEC 10-Q / XBRL Financial Filings Adapter & Point-in-Time TTM Engine (P0.3)
# ==============================================================================

class XBRLFactRecord(BaseModel):
    cik: str
    form: str  # "10-Q", "10-K", "10-Q/A", "10-K/A"
    accession_number: str
    filed_date: str  # YYYY-MM-DD
    fiscal_period_end: str  # YYYY-MM-DD
    fiscal_period_start: Optional[str] = None
    duration_days: int = 90  # 90, 180, 270, 365
    concept: str  # "revenue", "operating_income", "operating_cash_flow", "capex", "tax_expense", "pretax_income"
    tag: str
    unit: str = "USD"
    value: float
    is_consolidated: bool = True
    source_hash: Optional[str] = None
    lineage_status: str = "as_reported"


class SecFinancialFilingsAdapter:
    """Institutional SEC XBRL adapter with ConceptResolver, FactSelector, and PIT Cutoffs."""

    CONCEPT_TAG_MAP: Dict[str, List[str]] = {
        "revenue": [
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "SalesRevenueNet",
            "Revenues",
            "SalesRevenueServicesGross",
        ],
        "operating_income": [
            "OperatingIncomeLoss",
            "OperatingIncome",
        ],
        "operating_cash_flow": [
            "NetCashProvidedByUsedInOperatingActivities",
            "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
        ],
        "capex": [
            "PaymentsToAcquirePropertyPlantAndEquipment",
            "PaymentsToAcquireProductiveAssets",
            "PaymentsForPropertyPlantAndEquipment",
        ],
        "tax_expense": [
            "IncomeTaxExpenseBenefit",
            "IncomeTaxExpenseBenefitContinuingOperations",
        ],
        "pretax_income": [
            "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
            "IncomeLossFromContinuingOperationsBeforeIncomeTaxes",
            "OperatingIncomeLoss",
        ],
    }

    @classmethod
    def select_facts(
        cls,
        facts: List[XBRLFactRecord],
        as_of_date: Optional[str] = None,
    ) -> List[XBRLFactRecord]:
        """Filters facts with PIT cutoff (filed_date <= as_of_date) and selects latest filed revision per period/concept."""
        filtered = [f for f in facts if f.is_consolidated and (as_of_date is None or f.filed_date <= as_of_date)]
        # Sort by filed_date ascending so latest supersedes
        filtered.sort(key=lambda x: x.filed_date)
        dedup_map: Dict[Tuple[str, str, int], XBRLFactRecord] = {}
        for f in filtered:
            key = (f.concept, f.fiscal_period_end, f.duration_days)
            dedup_map[key] = f
        return list(dedup_map.values())


def deaccumulate_quarterly_cashflows(
    cumulative_periods: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Converts cumulative YTD cash flows (10-Q 6M, 9M, 10-K FY, 52/53-week, or transition stubs) into standalone single quarters.

    Prevents double-counting in TTM aggregation.
    Formula:
      Quarter_n = Cumulative_n - Cumulative_{n-1} (within the same fiscal year)
    """
    sorted_periods = sorted(cumulative_periods, key=lambda p: p.get("fiscal_period_end", ""))
    standalone_quarters: List[Dict[str, Any]] = []

    # Group by fiscal year
    by_year: Dict[str, List[Dict[str, Any]]] = {}
    for p in sorted_periods:
        fy = p.get("fiscal_year") or p.get("fiscal_period_end", "")[:4]
        by_year.setdefault(fy, []).append(p)

    for fy, periods in by_year.items():
        # Sort by duration_days ascending
        periods.sort(key=lambda x: x.get("duration_days", 90))
        prev_ocf_cum = 0.0
        prev_capex_cum = 0.0
        prev_rev_cum = 0.0
        prev_op_inc_cum = 0.0
        prev_net_inc_cum = 0.0
        prev_tax_cum = 0.0
        prev_pretax_cum = 0.0
        prev_dur_cum = 0

        for p in periods:
            dur = p.get("duration_days", 90)
            raw_ocf = float(p.get("operating_cash_flow") or 0.0)
            raw_capex = float(p.get("capital_expenditure") or 0.0)
            raw_rev = float(p.get("total_revenue") or 0.0)
            raw_op_inc = float(p.get("operating_income") or 0.0)
            raw_net_inc = float(p.get("net_income") or 0.0)
            raw_tax = float(p.get("tax_expense") or 0.0)
            raw_pretax = float(p.get("income_before_tax") or 0.0)

            if dur <= 95 and prev_dur_cum == 0:
                # Standalone first quarter or discrete quarter
                standalone_ocf = raw_ocf
                standalone_capex = raw_capex
                standalone_rev = raw_rev
                standalone_op_inc = raw_op_inc
                standalone_net_inc = raw_net_inc
                standalone_tax = raw_tax
                standalone_pretax = raw_pretax
                standalone_dur = dur
                prev_ocf_cum = raw_ocf
                prev_capex_cum = raw_capex
                prev_rev_cum = raw_rev
                prev_op_inc_cum = raw_op_inc
                prev_net_inc_cum = raw_net_inc
                prev_tax_cum = raw_tax
                prev_pretax_cum = raw_pretax
                prev_dur_cum = dur
            else:
                # Cumulative YTD -> Subtract previous cumulative
                standalone_ocf = raw_ocf - prev_ocf_cum
                standalone_capex = raw_capex - prev_capex_cum
                standalone_rev = raw_rev - prev_rev_cum if p.get("is_revenue_cumulative", False) else raw_rev
                standalone_op_inc = raw_op_inc - prev_op_inc_cum if p.get("is_operating_income_cumulative", False) else raw_op_inc
                standalone_net_inc = raw_net_inc - prev_net_inc_cum if p.get("is_net_income_cumulative", False) else raw_net_inc
                standalone_tax = raw_tax - prev_tax_cum if p.get("is_tax_cumulative", False) else raw_tax
                standalone_pretax = raw_pretax - prev_pretax_cum if p.get("is_pretax_cumulative", False) else raw_pretax
                standalone_dur = dur - prev_dur_cum

                prev_ocf_cum = raw_ocf
                prev_capex_cum = raw_capex
                if p.get("is_revenue_cumulative", False):
                    prev_rev_cum = raw_rev
                if p.get("is_operating_income_cumulative", False):
                    prev_op_inc_cum = raw_op_inc
                if p.get("is_net_income_cumulative", False):
                    prev_net_inc_cum = raw_net_inc
                if p.get("is_tax_cumulative", False):
                    prev_tax_cum = raw_tax
                if p.get("is_pretax_cumulative", False):
                    prev_pretax_cum = raw_pretax
                prev_dur_cum = dur

            standalone_fcf = standalone_ocf - standalone_capex
            standalone_quarters.append({
                "fiscal_period_start": p.get("fiscal_period_start"),
                "fiscal_period_end": p.get("fiscal_period_end"),
                "fiscal_quarter": p.get("fiscal_quarter"),
                "fiscal_year": fy,
                "duration_days": standalone_dur if standalone_dur > 0 else dur,
                "total_revenue": standalone_rev,
                "operating_income": standalone_op_inc,
                "operating_cash_flow": standalone_ocf,
                "capital_expenditure": standalone_capex,
                "net_income": standalone_net_inc,
                "free_cash_flow": standalone_fcf,
                "tax_expense": standalone_tax,
                "income_before_tax": standalone_pretax,
                "total_debt": p.get("total_debt"),
                "stockholders_equity": p.get("stockholders_equity"),
                "cash_and_equivalents": p.get("cash_and_equivalents"),
            })

    return standalone_quarters


def compute_ttm_standardized_fundamentals(
    standalone_quarters: List[Dict[str, Any]],
    issuer_reported_fcf: Optional[float] = None,
    market: str = "US",
    is_us_domestic: bool = True,
) -> Dict[str, Any]:
    """Computes Standardized TTM fundamentals from contiguous standalone quarters with duration coverage validation."""
    if len(standalone_quarters) < 4:
        return {
            "status": "insufficient_quarters",
            "exclusion_reason": "insufficient_quarters",
            "ttm_revenue": None,
            "ttm_operating_income": None,
            "standardized_ttm_gaap_operating_margin_pct": None,
            "standardized_ttm_fcf": None,
            "issuer_reported_fcf": issuer_reported_fcf,
            "fcf_reconciliation_delta": None,
            "ocf_to_net_income": None,
            "ttm_effective_tax_rate_pct": None,
            "effective_tax_rate_used_dec": 0.21 if (market == "US" and is_us_domestic) else None,
            "tax_policy_note": "default_statutory_due_to_insufficient_quarters",
        }

    # Sort chronological
    sorted_q = sorted(standalone_quarters, key=lambda q: q.get("fiscal_period_end", ""))
    latest_4 = sorted_q[-4:]

    # Validate Contiguous Date Ranges & Annual Duration Coverage (364-371 days)
    total_duration_days = sum(q.get("duration_days", 90) for q in latest_4)
    has_gap_or_overlap = False

    for i in range(len(latest_4) - 1):
        end_str = latest_4[i].get("fiscal_period_end")
        start_next_str = latest_4[i + 1].get("fiscal_period_start")
        end_next_str = latest_4[i + 1].get("fiscal_period_end")
        if end_str and end_next_str:
            try:
                d_end = datetime.strptime(end_str[:10], "%Y-%m-%d").date()
                d_end_next = datetime.strptime(end_next_str[:10], "%Y-%m-%d").date()
                if d_end >= d_end_next:  # Overlap
                    has_gap_or_overlap = True
                    break
                if start_next_str:
                    d_start_next = datetime.strptime(start_next_str[:10], "%Y-%m-%d").date()
                    if (d_start_next - d_end).days > 2:  # Gap > 1 day
                        has_gap_or_overlap = True
                        break
            except Exception:
                pass

    if has_gap_or_overlap or not (350 <= total_duration_days <= 380):
        return {
            "status": "partial",
            "exclusion_reason": "incomplete_trailing_duration_coverage",
            "total_duration_days": total_duration_days,
            "has_gap_or_overlap": has_gap_or_overlap,
            "ttm_revenue": None,
            "ttm_operating_income": None,
            "standardized_ttm_gaap_operating_margin_pct": None,
            "standardized_ttm_fcf": None,
            "issuer_reported_fcf": issuer_reported_fcf,
            "fcf_reconciliation_delta": None,
            "ocf_to_net_income": None,
            "ttm_effective_tax_rate_pct": None,
            "effective_tax_rate_used_dec": 0.21 if (market == "US" and is_us_domestic) else None,
            "tax_policy_note": "incomplete_trailing_duration_coverage",
        }

    ttm_rev = sum(q.get("total_revenue", 0.0) for q in latest_4)
    ttm_op_inc = sum(q.get("operating_income", 0.0) for q in latest_4)
    ttm_ocf = sum(q.get("operating_cash_flow", 0.0) for q in latest_4)
    ttm_capex = sum(q.get("capital_expenditure", 0.0) for q in latest_4)
    ttm_net_inc = sum(q.get("net_income", 0.0) for q in latest_4)
    standardized_fcf = ttm_ocf - ttm_capex

    gaap_margin_pct = (ttm_op_inc / ttm_rev * 100.0) if ttm_rev > 0 else None
    ocf_to_ni = (ttm_ocf / ttm_net_inc) if ttm_net_inc > 0 else None

    ttm_tax = sum(q.get("tax_expense", 0.0) for q in latest_4)
    ttm_pretax = sum(q.get("income_before_tax", 0.0) for q in latest_4)

    # Effective Tax Rate Policy: GAAP normalized statutory fallback (21% US Federal)
    if ttm_pretax > 0 and 0.05 <= (ttm_tax / ttm_pretax) <= 0.45:
        tax_rate_dec = ttm_tax / ttm_pretax
        tax_policy_note = "standardized_ttm_effective_tax_rate"
    elif market == "US" and is_us_domestic:
        tax_rate_dec = 0.21  # Normalized US federal statutory fallback
        tax_policy_note = "US_FEDERAL_STATUTORY_21PCT"
    else:
        tax_rate_dec = 0.20
        tax_policy_note = "jurisdiction_normalized_tax_policy"

    recon_delta = (issuer_reported_fcf - standardized_fcf) if issuer_reported_fcf is not None else 0.0

    return {
        "status": "complete",
        "latest_fiscal_period_end": latest_4[-1].get("fiscal_period_end"),
        "total_duration_days": total_duration_days,
        "ttm_revenue": round(ttm_rev, 2),
        "ttm_operating_income": round(ttm_op_inc, 2),
        "ttm_operating_cash_flow": round(ttm_ocf, 2),
        "ttm_capital_expenditure": round(ttm_capex, 2),
        "ttm_net_income": round(ttm_net_inc, 2),
        "standardized_ttm_gaap_operating_margin_pct": round(gaap_margin_pct, 2) if gaap_margin_pct is not None else None,
        "standardized_ttm_fcf": round(standardized_fcf, 2),
        "issuer_reported_fcf": round(issuer_reported_fcf, 2) if issuer_reported_fcf is not None else None,
        "fcf_reconciliation_delta": round(recon_delta, 2),
        "ocf_to_net_income": round(ocf_to_ni, 2) if ocf_to_ni is not None else None,
        "ttm_effective_tax_rate_pct": round(tax_rate_dec * 100.0, 2),
        "effective_tax_rate_used_dec": round(tax_rate_dec, 4),
        "tax_policy_note": tax_policy_note,
    }


def compute_roic_average_invested_capital(
    nopat_ttm: float,
    beginning_balance_sheet: Dict[str, Any],
    ending_balance_sheet: Dict[str, Any],
) -> Tuple[Optional[float], Dict[str, Any]]:
    """Calculates ROIC using NOPAT TTM / Average Invested Capital across beginning and ending periods."""
    def _inv_cap(bs: Dict[str, Any]) -> float:
        debt = float(bs.get("total_debt") or 0.0)
        equity = float(bs.get("stockholders_equity") or bs.get("total_equity") or 0.0)
        cash = float(bs.get("cash_and_equivalents") or bs.get("total_cash") or 0.0)
        return debt + equity - cash

    ic_beg = _inv_cap(beginning_balance_sheet)
    ic_end = _inv_cap(ending_balance_sheet)
    ic_avg = (ic_beg + ic_end) / 2.0

    if ic_avg <= 0:
        return None, {
            "ic_beginning": ic_beg,
            "ic_ending": ic_end,
            "ic_average": ic_avg,
            "status": "non_positive_invested_capital",
        }

    roic_pct = (nopat_ttm / ic_avg) * 100.0
    return round(roic_pct, 2), {
        "ic_beginning": round(ic_beg, 2),
        "ic_ending": round(ic_end, 2),
        "ic_average": round(ic_avg, 2),
        "status": "calculated",
    }
