import os
from pathlib import Path
from tools.portfolio.domain.validator import validate_portfolio_id
from tools.archivist.maintenance_guard import assert_write_allowed

VAULT_PATH = Path(os.getenv("OBSIDIAN_VAULT_PATH", "./memories"))
PORTFOLIOS_DIR = VAULT_PATH / "20_Portfolio_Management/Current_Holdings/Portfolios"

GOALS_REL = os.getenv("GOALS_FILE", "20_Portfolio_Management/Goals/Goals.md")
GOALS_PATH = VAULT_PATH / GOALS_REL
GOALS_ITEMS_DIR = VAULT_PATH / "20_Portfolio_Management/Goals/Items"

_PERFORMANCE_LOG_HEADER = [
    "Date",
    "Total_NAV",
    "Total_Cost",
    "Unrealized_PnL",
    "Cash_Balance",
    "Realized_PnL_YTD",
    "Passive_Income_YTD",
]

_TRADES_LOG_HEADER = [
    "Transaction_ID",
    "Timestamp",
    "Symbol",
    "Action",
    "Units",
    "Price",
    "Currency",
    "FX_Rate",
    "Cost_THB",
    "Realized_PnL_THB",
    "Notes",
    "Gross_Amount",
    "Commission",
    "VAT",
    "Other_Fees",
    "Net_Amount",
    "Fee_Currency",
    "Confirmation_No",
    "Order_ID",
    "Settlement_Date",
    "Source",
    "Fingerprint",
    "Cash_Adjusted",
    "Related_Transaction_ID",
]

_LOCK_TIMEOUT = 15.0  # seconds


def get_vault_path() -> Path:
    return Path(os.getenv("OBSIDIAN_VAULT_PATH", str(VAULT_PATH)))


def get_portfolios_dir() -> Path:
    return get_vault_path() / "20_Portfolio_Management/Current_Holdings/Portfolios"


def get_portfolio_dir(portfolio_id: str = "default") -> Path:
    pid = validate_portfolio_id(portfolio_id)
    pdir = get_portfolios_dir() / pid
    if not pdir.exists():
        assert_write_allowed(pdir)
        pdir.mkdir(parents=True, exist_ok=True)
    return pdir


def get_portfolio_filepath(portfolio_id: str = "default") -> Path:
    return get_portfolio_dir(portfolio_id) / "Portfolio_Holdings.md"


def get_portfolio_lock_path(portfolio_id: str = "default") -> str:
    return str(get_portfolio_dir(portfolio_id) / "Portfolio_Holdings.md.lock")


def get_trades_log_filepath(portfolio_id: str = "default") -> Path:
    return get_portfolio_dir(portfolio_id) / "Trades_Log.csv"


def get_pending_manifest_path(portfolio_id: str = "default") -> Path:
    return get_portfolio_dir(portfolio_id) / ".pending_commit.json"


def get_holdings_dir(portfolio_id: str = "default") -> Path:
    hdir = get_portfolio_dir(portfolio_id) / "Holdings"
    if not hdir.exists():
        assert_write_allowed(hdir)
        hdir.mkdir(parents=True, exist_ok=True)
    return hdir


def get_watchlist_filepath(portfolio_id: str = "default") -> Path:
    return get_portfolio_dir(portfolio_id) / "Watchlist.md"


def get_watchlist_items_dir(portfolio_id: str = "default") -> Path:
    wdir = get_portfolio_dir(portfolio_id) / "Watchlist_Items"
    if not wdir.exists():
        assert_write_allowed(wdir)
        wdir.mkdir(parents=True, exist_ok=True)
    return wdir


def get_performance_filepath(portfolio_id: str = "default") -> Path:
    return get_portfolio_dir(portfolio_id) / "Performance_Log.csv"


def get_journal_filepath(portfolio_id: str = "default") -> Path:
    return get_portfolio_dir(portfolio_id) / "Trading_Journal.md"
