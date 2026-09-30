"""Cboe Keyless Adapter for US Listed Equity Options Chains.

Fetches full delayed option chains from Cboe public quote API.
Parses OCC-21 symbol syntax from the right to handle variable-length roots safely.
Converts percent IV30 to decimal and treats zero-IV placeholders as None.
"""
from datetime import date
import logging
import time
from typing import Any, Optional, Tuple
import requests

from tools.market.terminal_v2.application.cache import BROWSER_HEADERS, ThreadSafeTTLCache
from tools.market.terminal_v2.domain.errors import DataUnavailableError, ProviderError
from tools.market.terminal_v2.domain.models import (
    OptionContract,
    OptionsChainSnapshot,
)
from tools.market.terminal_v2.ports.driven_ports import OptionsChainPort

logger = logging.getLogger(__name__)

CBOE_BASE_URL = "https://cdn.cboe.com/api/global/delayed_quotes/options/{ticker}.json"
CBOE_TTL_SECONDS = 300.0          # 5 minutes
CBOE_MAX_STALE_SECONDS = 900.0    # 15 minutes ceiling
FETCH_TIMEOUT_SECONDS = 25.0
DELAY_MINUTES = 15
TAIL_LENGTH = 15


def parse_occ_symbol(occ_symbol: str, expected_root: str) -> Optional[Tuple[str, str, str, float, bool]]:
    """Parse OCC-21 contract symbol from the right.

    Format: <ROOT><YY><MM><DD><C|P><strike * 1000, 8 digits>
    Returns: (root, expiry_iso, side, strike, is_standard) or None if invalid.
    """
    if not isinstance(occ_symbol, str):
        return None
    sym = occ_symbol.strip().upper()
    if len(sym) < TAIL_LENGTH + 1:
        return None

    root = sym[:-TAIL_LENGTH]
    date_part = sym[-TAIL_LENGTH:-9]
    side_char = sym[-9:-8]
    strike_part = sym[-8:]

    if not date_part.isdigit() or not strike_part.isdigit() or side_char not in ("C", "P"):
        return None

    yy = int(date_part[:2])
    mm = int(date_part[2:4])
    dd = int(date_part[4:6])

    try:
        expiry_date = date(2000 + yy, mm, dd)
    except ValueError:
        return None

    strike = float(strike_part) / 1000.0
    if strike <= 0:
        return None

    side = "call" if side_char == "C" else "put"
    is_standard = (root == expected_root.upper()) and root.isalpha()

    return (root, expiry_date.isoformat(), side, strike, is_standard)


class CboeOptionsAdapter(OptionsChainPort):
    """Adapter reading delayed equity option chains from Cboe."""

    def __init__(self, cache: Optional[ThreadSafeTTLCache] = None):
        self._cache = cache or ThreadSafeTTLCache(default_ttl_seconds=CBOE_TTL_SECONDS, max_entries=50)

    def _strip_symbol(self, raw_symbol: str) -> str:
        idx = raw_symbol.find(":")
        clean = raw_symbol[idx + 1 :] if idx != -1 else raw_symbol
        return clean.strip().upper()

    def _fetch_chain(self, ticker: str) -> OptionsChainSnapshot:
        url = CBOE_BASE_URL.format(ticker=ticker)
        try:
            resp = requests.get(url, headers=BROWSER_HEADERS, timeout=FETCH_TIMEOUT_SECONDS)
            if resp.status_code == 404:
                raise ProviderError(f"Cboe options chain not found for symbol '{ticker}'", source="Cboe", status_code=404)
            resp.raise_for_status()
            data = resp.json()
        except ProviderError:
            raise
        except Exception as exc:
            raise ProviderError(f"Failed to fetch Cboe options chain for '{ticker}': {exc}", source="Cboe") from exc

        root_data = data.get("data", {})
        current_price = root_data.get("current_price")
        if current_price is not None:
            try:
                current_price = float(current_price)
            except (ValueError, TypeError):
                current_price = None

        # iv30 is published in percent (e.g. 22.215) -> convert to decimal (0.22215)
        raw_iv30 = root_data.get("iv30")
        iv30_decimal = None
        if raw_iv30 is not None:
            try:
                iv30_val = float(raw_iv30)
                if iv30_val > 0:
                    iv30_decimal = iv30_val / 100.0
            except (ValueError, TypeError):
                iv30_decimal = None

        options_raw = root_data.get("options", [])
        contracts = []
        now_epoch = time.time()

        for row in options_raw:
            occ_id = row.get("option")
            if not occ_id:
                continue

            parsed = parse_occ_symbol(occ_id, expected_root=ticker)
            if parsed is None:
                continue

            root, expiry, side, strike, is_standard = parsed

            # IV: Cboe uses 0 as null placeholder. Convert to float decimal if > 0
            raw_iv = row.get("iv")
            iv_val = None
            if raw_iv is not None:
                try:
                    fiv = float(raw_iv)
                    if fiv > 0:
                        iv_val = fiv
                except (ValueError, TypeError):
                    iv_val = None

            def _flt(v: Any) -> Optional[float]:
                if v is None:
                    return None
                try:
                    val = float(v)
                    return val
                except (ValueError, TypeError):
                    return None

            def _int(v: Any) -> int:
                if v is None:
                    return 0
                try:
                    return int(float(v))
                except (ValueError, TypeError):
                    return 0

            contracts.append(
                OptionContract(
                    occ_symbol=occ_id,
                    underlying=ticker,
                    expiry=expiry,
                    strike=strike,
                    side=side,
                    open_interest=_int(row.get("open_interest")),
                    volume=_int(row.get("volume")),
                    bid=_flt(row.get("bid")),
                    ask=_flt(row.get("ask")),
                    last_price=_flt(row.get("last_trade_price")),
                    implied_volatility=iv_val,
                    delta=_flt(row.get("delta")),
                    gamma=_flt(row.get("gamma")),
                    vega=_flt(row.get("vega")),
                    theta=_flt(row.get("theta")),
                    rho=_flt(row.get("rho")),
                    multiplier=100,
                    is_standard=is_standard,
                )
            )

        return OptionsChainSnapshot(
            underlying=ticker,
            underlying_price=current_price,
            iv30_decimal=iv30_decimal,
            delay_minutes=DELAY_MINUTES,
            contracts=tuple(contracts),
            fetched_at=now_epoch,
            source="Cboe",
        )

    def get_options_chain(self, symbol: str) -> OptionsChainSnapshot:
        """Fetch delayed listed options chain for underlying symbol."""
        ticker = self._strip_symbol(symbol)
        if not ticker:
            raise ProviderError("Empty symbol provided for options chain", source="Cboe", status_code=400)

        cache_key = f"cboe:options:{ticker}"
        return self._cache.get_or_set(
            key=cache_key,
            loader=lambda: self._fetch_chain(ticker),
            ttl_seconds=CBOE_TTL_SECONDS,
            max_stale_seconds=CBOE_MAX_STALE_SECONDS,
        )
