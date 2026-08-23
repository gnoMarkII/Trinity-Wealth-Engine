"""Pure domain constants for portfolio management (Zero IO / Zero filesystem paths)."""

CASH_THB_SYMBOL = "CASH_THB"
CASH_USD_SYMBOL = "CASH_USD"
_CASH_SYMBOLS = (CASH_THB_SYMBOL, CASH_USD_SYMBOL)
# Back-compat alias
CASH_SYMBOL = CASH_THB_SYMBOL

_FLOAT_EPS = 1e-6
_MONEY_DP = 2
_COST_DP = 6
_PCT_DP = 2

FUNDAMENTALS_TTL_SECONDS = 86400  # 24 hours
MARKET_CAP_MEGA_USD = 200_000_000_000
MARKET_CAP_LARGE_USD = 10_000_000_000
MARKET_CAP_MID_USD = 2_000_000_000

_EDITABLE_HOLDING_FIELDS = ("units", "avg_cost", "accumulated_dividend_thb", "asset_type")
_TOP_LEVEL_KEY_ORDER = (
    "schema_version",
    "doc_type",
    "last_updated",
    "base_currency",
    "summary",
    "fx_rates",
    "allocation_targets",
    "holdings",
)
