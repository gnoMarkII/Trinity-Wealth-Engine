import os
from pathlib import Path

VAULT_PATH = Path(os.getenv("OBSIDIAN_VAULT_PATH", "./memories"))

PORTFOLIO_REL = os.getenv("PORTFOLIO_FILE", "20_Portfolio_Management/Current_Holdings/Portfolio_Holdings.md")
PORTFOLIO_PATH = VAULT_PATH / PORTFOLIO_REL

TRADING_JOURNAL_REL = os.getenv("TRADING_JOURNAL_FILE", "20_Portfolio_Management/Journals_and_Reports/Trading_Journal.md")
TRADING_JOURNAL_PATH = VAULT_PATH / TRADING_JOURNAL_REL

WATCHLIST_REL = os.getenv("WATCHLIST_FILE", "20_Portfolio_Management/Current_Holdings/Watchlist.md")
WATCHLIST_PATH = VAULT_PATH / WATCHLIST_REL

PERFORMANCE_LOG_REL = os.getenv("PERFORMANCE_LOG_FILE", "20_Portfolio_Management/Journals_and_Reports/Performance_Log.csv")
PERFORMANCE_LOG_PATH = VAULT_PATH / PERFORMANCE_LOG_REL

HOLDINGS_DIR = VAULT_PATH / "20_Portfolio_Management/Current_Holdings/Holdings"
WATCHLIST_ITEMS_DIR = VAULT_PATH / "20_Portfolio_Management/Current_Holdings/WatchlistItems"

GOALS_REL = os.getenv("GOALS_FILE", "20_Portfolio_Management/Goals/Goals.md")
GOALS_PATH = VAULT_PATH / GOALS_REL
GOALS_ITEMS_DIR = VAULT_PATH / "20_Portfolio_Management/Goals/Items"

_PERFORMANCE_LOG_HEADER = ["Date", "Total_NAV", "Total_Cost", "Unrealized_PnL", "Cash_Balance"]

FUNDAMENTALS_TTL_SECONDS = 86400  # 24 hours
MARKET_CAP_MEGA_USD = 200_000_000_000
MARKET_CAP_LARGE_USD = 10_000_000_000
MARKET_CAP_MID_USD = 2_000_000_000

