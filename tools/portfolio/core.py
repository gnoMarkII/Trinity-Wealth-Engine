from typing import Optional, List, Tuple, Literal
import frontmatter
from tools.portfolio import get_default_service
from tools.portfolio.domain.models import (
    Holding,
    Summary,
    PortfolioState,
    PortfolioMeta,
    AllocationTarget,
    WatchlistItem,
    WatchlistState,
    GoalItem,
    GoalsState,
    _now_iso,
    _coerce_iso_string,
    default_allocation_targets,
)
from tools.portfolio.domain.constants import (
    CASH_THB_SYMBOL,
    CASH_USD_SYMBOL,
    CASH_SYMBOL,
    _CASH_SYMBOLS,
    _FLOAT_EPS,
    _MONEY_DP,
    _COST_DP,
    _PCT_DP,
    FUNDAMENTALS_TTL_SECONDS,
    MARKET_CAP_MEGA_USD,
    MARKET_CAP_LARGE_USD,
    MARKET_CAP_MID_USD,
    _EDITABLE_HOLDING_FIELDS,
    _TOP_LEVEL_KEY_ORDER,
)
from tools.portfolio.domain.calculations import (
    calc_weighted_avg_cost,
    calc_realized_pnl,
    calc_holding_currency as _holding_currency,
    recalc_holding as _recalc_holding,
    compute_total_cost as _compute_total_cost,
    recalc_summary as _recalc_summary,
    recalc_fundamentals_derived as _recalc_fundamentals_derived,
    recalc_all as _recalc_all,
)
from tools.portfolio.domain.validator import validate_portfolio_id as _normalize_portfolio_id
from tools.portfolio.adapters.markdown.paths import (
    get_portfolio_filepath as _get_portfolio_filepath,
    get_portfolio_dir as _get_portfolio_dir,
    get_holdings_dir as _get_holdings_dir,
    PORTFOLIOS_DIR,
    VAULT_PATH,
)
from tools.portfolio.adapters.markdown.repository_adapter import (
    _get_portfolio_lock,
    _initial_state,
    _holding_to_md,
)
from tools.portfolio.agent_tools import (
    get_portfolio_state,
    compute_allocation_breakdown,
    tool_list_portfolios,
    tool_create_portfolio,
    tool_delete_portfolio,
    tool_rename_portfolio,
)

# --- Structured API Functions Delegation ---

def create_portfolio(name: str, portfolio_id: Optional[str] = None) -> PortfolioMeta:
    return get_default_service().create_portfolio(name=name, portfolio_id=portfolio_id)

def delete_portfolio(portfolio_id: str) -> None:
    get_default_service().delete_portfolio(portfolio_id=portfolio_id)

def update_portfolio_name(portfolio_id: str, name: str) -> PortfolioMeta:
    return get_default_service().update_portfolio_name(portfolio_id=portfolio_id, name=name)

def list_portfolios() -> List[PortfolioMeta]:
    return get_default_service().list_portfolios()

def get_structured_portfolio_state(
    refresh_prices: bool = False, fetch_fundamentals: bool = False, portfolio_id: str = "default"
) -> PortfolioState:
    return get_default_service().get_structured_portfolio_state(
        refresh_prices=refresh_prices, fetch_fundamentals=fetch_fundamentals, portfolio_id=portfolio_id
    )

def get_structured_bucket_allocation(
    state: Optional[PortfolioState] = None, portfolio_id: str = "default"
) -> Tuple[List[dict], Optional[str]]:
    return get_default_service().get_structured_bucket_allocation(state=state, portfolio_id=portfolio_id)

def structured_assign_holding_bucket(
    symbol: str, bucket_id: Optional[str], portfolio_id: str = "default"
) -> PortfolioState:
    return get_default_service().structured_assign_holding_bucket(
        symbol=symbol, bucket_id=bucket_id, portfolio_id=portfolio_id
    )

def structured_batch_assign_holding_buckets(
    symbols: List[str], bucket_id: Optional[str], portfolio_id: str = "default"
) -> PortfolioState:
    return get_default_service().structured_batch_assign_holding_buckets(
        symbols=symbols, bucket_id=bucket_id, portfolio_id=portfolio_id
    )

def structured_batch_remove_holdings(
    symbols: List[str], portfolio_id: str = "default"
) -> PortfolioState:
    return get_default_service().structured_batch_remove_holdings(
        symbols=symbols, portfolio_id=portfolio_id
    )

def structured_reset_clean_slate(portfolio_id: str = "default") -> PortfolioState:
    return get_default_service().structured_reset_clean_slate(portfolio_id=portfolio_id)

def structured_upsert_allocation_targets(
    targets: List[AllocationTarget], portfolio_id: str = "default"
) -> PortfolioState:
    return get_default_service().structured_upsert_allocation_targets(
        targets=targets, portfolio_id=portfolio_id
    )

# --- Compatibility Hooks for tests/conftest.py ---

def _find_holding(state: PortfolioState, symbol: str) -> Optional[Holding]:
    return next((h for h in state.holdings if h.symbol == symbol), None)

def _require_cash(state: PortfolioState, currency: Literal["THB", "USD"] = "THB") -> Holding:
    sym = CASH_THB_SYMBOL if currency == "THB" else CASH_USD_SYMBOL
    cash = _find_holding(state, sym)
    if cash is None:
        cash = Holding(symbol=sym, asset_type="Cash", units=0.0, market_value_thb=0.0)
        state.holdings.append(cash)
    return cash

def _require_fx(state: PortfolioState) -> float:
    fx = state.fx_rates.get("USDTHB")
    if fx is None or fx <= 0:
        raise ValueError("ไม่พบ fx_rates.USDTHB ที่ valid ใน portfolio")
    return fx

def _get_portfolios_dir():
    return PORTFOLIOS_DIR

def _portfolio_exists(portfolio_id: str = "default") -> bool:
    return _get_portfolio_filepath(portfolio_id).exists()

def _load_or_init(portfolio_id: str = "default") -> Tuple[frontmatter.Post, PortfolioState]:
    from tools.portfolio.adapters.markdown.repository_adapter import MarkdownVaultRepositoryAdapter
    md_repo = MarkdownVaultRepositoryAdapter()
    return md_repo._load_or_init_locked(portfolio_id)

def _save(post: frontmatter.Post, state: PortfolioState, portfolio_id: str = "default") -> None:
    from tools.portfolio.adapters.markdown.repository_adapter import MarkdownVaultRepositoryAdapter
    from tools.portfolio.domain.ledger_change import LedgerChange
    md_repo = MarkdownVaultRepositoryAdapter()
    md_repo._commit_locked(portfolio_id, state, LedgerChange(kind="unchanged"))
