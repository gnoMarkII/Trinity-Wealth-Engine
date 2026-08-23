import concurrent.futures
import csv
import io
from datetime import datetime, timedelta, date
from typing import Any, Literal

import pandas as pd
import yfinance as yf

from core.logger import get_logger
from core.retry import with_retry
from .models import _MONEY_DP, _COST_DP, _FLOAT_EPS, DividendRound
from .core import (
    _load_or_init,
    _save,
    _find_holding,
    _get_portfolio_lock,
    _holding_currency,
    _normalize_portfolio_id,
)
from .prices import _yf_symbol, _USDTHB_TICKER
from .trading import _get_trades_log_filepath, _migrate_trades_log_if_needed, _now_iso

log = get_logger(__name__)


def _normalize_tz_series(s: pd.Series | None) -> pd.Series | None:
    """Normalize Series DatetimeIndex to timezone-naive dates."""
    if s is None or s.empty:
        return s
    s = s.copy()
    if hasattr(s.index, "tz") and s.index.tz is not None:
        s.index = s.index.tz_localize(None)
    if hasattr(s.index, "normalize"):
        s.index = s.index.normalize()
    return s


def _fetch_batch_fx(earliest_date_str: str, fallback_rate: float) -> pd.Series | None:
    """Download historical USDTHB=X series in batch from earliest_date to today."""
    def _fetch():
        try:
            target_dt = datetime.strptime(earliest_date_str, "%Y-%m-%d")
        except Exception:
            target_dt = datetime.now() - timedelta(days=365)

        start_dt = target_dt - timedelta(days=7)
        end_dt = datetime.now() + timedelta(days=1)
        df = yf.download(
            _USDTHB_TICKER,
            start=start_dt.strftime("%Y-%m-%d"),
            end=end_dt.strftime("%Y-%m-%d"),
            progress=False,
        )
        if df is not None and not df.empty:
            close = df["Close"]
            if hasattr(close, "columns"):
                close = close.iloc[:, 0]
            return _normalize_tz_series(close)
        return None

    try:
        return with_retry(_fetch)
    except Exception as e:
        log.warning("batch fx download failed: %s", e)
        return None


def _get_fx_at_date(fx_series: pd.Series | None, target_date: date, fallback_rate: float) -> float:
    """Lookup FX rate on or prior to target_date from in-memory fx_series."""
    if fx_series is not None and not fx_series.empty:
        try:
            ts = pd.Timestamp(target_date)
            val = fx_series.asof(ts)
            if val is not None:
                val_float = float(val)
                if val_float > 0 and val_float == val_float:
                    return round(val_float, 4)
        except Exception as e:
            log.debug("fx asof failed for date %s: %s", target_date, e)
    return fallback_rate


def _get_withholding_tax_rate(currency: str) -> float:
    """Tax rate determined by currency context: THB -> 10%, USD -> 15%, other -> 0.0%."""
    if currency == "THB":
        return 0.10
    if currency == "USD":
        return 0.15
    return 0.0


def _parse_trades_for_symbol(trades_rows: list[list[str]], symbol: str) -> list[tuple[date, str, float]]:
    """Parse trades rows for symbol into a chronological list of (trade_date, action, units)."""
    sym_norm = symbol.strip().upper()
    timeline: list[tuple[date, str, float]] = []

    for r in trades_rows:
        if len(r) < 5:
            continue
        r_sym = r[2].strip().upper()
        if r_sym != sym_norm:
            continue
        try:
            raw_ts = r[1].strip()
            trade_dt = datetime.fromisoformat(raw_ts).date() if "T" in raw_ts else datetime.strptime(raw_ts[:10], "%Y-%m-%d").date()
            action = r[3].strip().upper()
            units = float(r[4])
            if units > 0 and action in ("BUY", "SELL"):
                timeline.append((trade_dt, action, units))
        except Exception as e:
            log.debug("error parsing trade row %s: %s", r, e)

    timeline.sort(key=lambda x: x[0])
    return timeline


def _calculate_units_held_before(timeline: list[tuple[date, str, float]], xd_date: date) -> float:
    """Calculate units held prior to xd_date (i.e. trades on or before xd_date - 1 day)."""
    units_held = 0.0
    for trade_date, action, units in timeline:
        if trade_date < xd_date:
            if action == "BUY":
                units_held += units
            elif action == "SELL":
                units_held = max(0.0, units_held - units)
        else:
            break
    return round(units_held, _COST_DP)


def _calculate_symbol_dividends(
    symbol: str,
    currency: str,
    trades_rows: list[list[str]],
    fx_series: pd.Series | None,
    fallback_fx: float,
) -> dict:
    """Fetch dividend history from yfinance for symbol and compute net dividends (received vs upcoming)."""
    empty_res = {
        "rounds": [],
        "received_net_thb": 0.0,
        "received_net_native": 0.0,
        "upcoming_net_thb": 0.0,
        "upcoming_net_native": 0.0,
        "received_count": 0,
        "upcoming_count": 0,
    }
    yf_sym = _yf_symbol(symbol, currency)
    timeline = _parse_trades_for_symbol(trades_rows, symbol)
    if not timeline:
        return empty_res

    earliest_buy = min(t[0] for t in timeline if t[1] == "BUY")

    calendar_pay_date: date | None = None
    calendar_ex_date: date | None = None
    try:
        tk = yf.Ticker(yf_sym)
        divs = with_retry(lambda: tk.dividends)
        try:
            cal = with_retry(lambda: tk.calendar)
            if cal and isinstance(cal, dict):
                p_date = cal.get("Dividend Date")
                if p_date:
                    if hasattr(p_date, "date"):
                        calendar_pay_date = p_date.date()
                    elif isinstance(p_date, date):
                        calendar_pay_date = p_date
                    elif isinstance(p_date, str):
                        calendar_pay_date = datetime.strptime(p_date[:10], "%Y-%m-%d").date()
                e_date = cal.get("Ex-Dividend Date")
                if e_date:
                    if hasattr(e_date, "date"):
                        calendar_ex_date = e_date.date()
                    elif isinstance(e_date, date):
                        calendar_ex_date = e_date
                    elif isinstance(e_date, str):
                        calendar_ex_date = datetime.strptime(e_date[:10], "%Y-%m-%d").date()
        except Exception as e:
            log.debug("failed to fetch calendar for %s: %s", yf_sym, e)
    except Exception as e:
        log.warning("failed to fetch dividends for %s: %s", yf_sym, e)
        return empty_res

    if divs is None or divs.empty:
        return empty_res

    divs = _normalize_tz_series(divs)
    tax_rate = _get_withholding_tax_rate(currency)
    rounds: list[dict] = []
    received_net_thb = 0.0
    received_net_native = 0.0
    upcoming_net_thb = 0.0
    upcoming_net_native = 0.0
    received_count = 0
    upcoming_count = 0

    today = date.today()

    latest_xd_date: date | None = None
    for ts in divs.index:
        d = ts.date() if hasattr(ts, "date") else ts
        if latest_xd_date is None or d > latest_xd_date:
            latest_xd_date = d

    for ts, dps_raw in divs.items():
        try:
            xd_date = ts.date() if hasattr(ts, "date") else ts
            dps = float(dps_raw)
        except Exception:
            continue

        if dps <= 0 or xd_date < earliest_buy:
            continue

        units_held = _calculate_units_held_before(timeline, xd_date)
        if units_held <= _FLOAT_EPS:
            continue

        pay_date_val: date | None = None
        if calendar_pay_date and calendar_ex_date and xd_date == calendar_ex_date:
            pay_date_val = calendar_pay_date
        elif (
            calendar_pay_date
            and not calendar_ex_date
            and xd_date == latest_xd_date
            and calendar_pay_date >= xd_date
            and (calendar_pay_date - xd_date).days <= 45
        ):
            pay_date_val = calendar_pay_date
        else:
            pay_date_val = xd_date + timedelta(days=21)

        status = "received" if (pay_date_val <= today and xd_date <= today) else "upcoming"

        fx_rate = _get_fx_at_date(fx_series, xd_date, fallback_fx) if currency == "USD" else 1.0
        gross_native = round(units_held * dps, 4)
        net_native = round(gross_native * (1.0 - tax_rate), 4)
        gross_thb = round(gross_native * fx_rate, _MONEY_DP)
        net_thb = round(net_native * fx_rate, _MONEY_DP)

        round_dict = {
            "symbol": symbol,
            "ex_date": xd_date.isoformat(),
            "pay_date": pay_date_val.isoformat() if pay_date_val else None,
            "dps": round(dps, 4),
            "currency": currency,
            "units_held": units_held,
            "status": status,
            "gross_native": gross_native,
            "net_native": net_native,
            "gross_thb": gross_thb,
            "tax_rate": tax_rate,
            "net_thb": net_thb,
            "fx_rate": round(fx_rate, 4),
        }
        rounds.append(round_dict)

        if status == "received":
            received_net_thb += net_thb
            received_net_native += net_native
            received_count += 1
        else:
            upcoming_net_thb += net_thb
            upcoming_net_native += net_native
            upcoming_count += 1

    rounds.sort(key=lambda r: r["ex_date"], reverse=True)
    return {
        "rounds": rounds,
        "received_net_thb": round(received_net_thb, _MONEY_DP),
        "received_net_native": round(received_net_native, 4),
        "upcoming_net_thb": round(upcoming_net_thb, _MONEY_DP),
        "upcoming_net_native": round(upcoming_net_native, 4),
        "received_count": received_count,
        "upcoming_count": upcoming_count,
    }


def sync_dividends_from_history(portfolio_id: str = "default") -> dict:
    """Calculate and synchronize accumulated dividends from Trades_Log and yfinance history."""
    pid = _normalize_portfolio_id(portfolio_id)
    lock = _get_portfolio_lock(pid)

    with lock:
        _migrate_trades_log_if_needed(pid)
        post, state = _load_or_init(portfolio_id=pid)
        fallback_fx = state.fx_rates.get("USDTHB", 36.5)

        targets: list[tuple[str, str]] = []
        skipped_manual: list[str] = []

        for h in state.holdings:
            if h.asset_type == "Cash":
                continue
            if h.dividend_source == "manual":
                skipped_manual.append(h.symbol)
            else:
                curr = _holding_currency(h)
                targets.append((h.symbol, curr))

        trades_path = _get_trades_log_filepath(pid)
        trades_rows: list[list[str]] = []
        if trades_path.exists():
            try:
                content = trades_path.read_text(encoding="utf-8")
                reader = csv.reader(io.StringIO(content))
                rows = list(reader)
                if rows:
                    trades_rows = rows[1:]
            except Exception as e:
                log.warning("failed to read trades log for %s: %s", pid, e)

    if not targets:
        return {
            "synced_symbols": 0,
            "total_rounds": 0,
            "total_received_rounds": 0,
            "total_upcoming_rounds": 0,
            "total_dividend_thb": 0.0,
            "total_upcoming_thb": 0.0,
            "skipped_manual": skipped_manual,
            "details": {},
        }

    earliest_date_str = "2020-01-01"
    for r in trades_rows:
        if len(r) > 1 and r[1].strip():
            raw_d = r[1].strip()[:10]
            if raw_d < earliest_date_str or earliest_date_str == "2020-01-01":
                earliest_date_str = raw_d

    fx_series = _fetch_batch_fx(earliest_date_str, fallback_rate=fallback_fx)
    calc_results: dict[str, dict] = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(targets))) as ex:
        future_map = {
            ex.submit(_calculate_symbol_dividends, sym, curr, trades_rows, fx_series, fallback_fx): sym
            for sym, curr in targets
        }
        for future in concurrent.futures.as_completed(future_map):
            sym = future_map[future]
            try:
                res = future.result()
                calc_results[sym] = res
            except Exception as e:
                log.warning("error calculating dividends for %s: %s", sym, e)
                calc_results[sym] = {
                    "rounds": [],
                    "received_net_thb": 0.0,
                    "received_net_native": 0.0,
                    "upcoming_net_thb": 0.0,
                    "upcoming_net_native": 0.0,
                    "received_count": 0,
                    "upcoming_count": 0,
                }

    with lock:
        post, state = _load_or_init(portfolio_id=pid)
        total_rounds = 0
        total_received_rounds = 0
        total_upcoming_rounds = 0
        total_dividend_thb = 0.0
        total_upcoming_thb = 0.0
        details: dict[str, list[dict]] = {}
        final_skipped_manual = list(skipped_manual)

        for sym, res in calc_results.items():
            rounds = res["rounds"]
            target_h = _find_holding(state, sym)
            if target_h is None:
                continue

            if target_h.dividend_source == "manual":
                if sym not in final_skipped_manual:
                    final_skipped_manual.append(sym)
                continue

            target_h.accumulated_dividend_thb = res["received_net_thb"]
            target_h.accumulated_dividend_native = res["received_net_native"]
            target_h.upcoming_dividend_thb = res["upcoming_net_thb"]
            target_h.upcoming_dividend_native = res["upcoming_net_native"]
            target_h.dividend_rounds = [DividendRound(**r) for r in rounds]
            target_h.dividend_source = "synced"

            total_rounds += len(rounds)
            total_received_rounds += res["received_count"]
            total_upcoming_rounds += res["upcoming_count"]
            total_dividend_thb += res["received_net_thb"]
            total_upcoming_thb += res["upcoming_net_thb"]
            details[sym] = rounds

        state.summary.total_accumulated_dividend = round(
            sum(h.accumulated_dividend_thb or 0.0 for h in state.holdings), _MONEY_DP
        )
        _save(post, state, portfolio_id=pid)

    return {
        "synced_symbols": len(details),
        "total_rounds": total_rounds,
        "total_received_rounds": total_received_rounds,
        "total_upcoming_rounds": total_upcoming_rounds,
        "total_dividend_thb": round(total_dividend_thb, _MONEY_DP),
        "total_upcoming_thb": round(total_upcoming_thb, _MONEY_DP),
        "skipped_manual": final_skipped_manual,
        "details": details,
    }
