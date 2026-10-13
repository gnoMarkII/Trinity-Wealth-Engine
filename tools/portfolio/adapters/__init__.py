"""Portfolio Adapters (Hexagonal Architecture).

Uses lazy imports (PEP 562) to prevent eager loading of heavy dependencies
(e.g., PyTorch, sentence-transformers, ChromaDB via repository/archivist)
when lightweight adapters like WealthX or Dime parsers are imported in child processes.
"""
from typing import Any

__all__ = [
    "MarkdownVaultRepositoryAdapter",
    "MarkdownWatchlistAdapter",
    "MarkdownGoalsAdapter",
    "MarkdownPerformanceAdapter",
    "JournalVaultAdapter",
    "SqliteMirroredPortfolioRepository",
    "PriceYFinanceAdapter",
    "CompositeMarketPriceAdapter",
]


def __getattr__(name: str) -> Any:
    if name == "MarkdownVaultRepositoryAdapter":
        from .markdown.repository_adapter import MarkdownVaultRepositoryAdapter
        return MarkdownVaultRepositoryAdapter
    if name == "MarkdownWatchlistAdapter":
        from .markdown.watchlist_adapter import MarkdownWatchlistAdapter
        return MarkdownWatchlistAdapter
    if name == "MarkdownGoalsAdapter":
        from .markdown.goals_adapter import MarkdownGoalsAdapter
        return MarkdownGoalsAdapter
    if name == "MarkdownPerformanceAdapter":
        from .markdown.performance_adapter import MarkdownPerformanceAdapter
        return MarkdownPerformanceAdapter
    if name == "JournalVaultAdapter":
        from .markdown.journal_vault_adapter import JournalVaultAdapter
        return JournalVaultAdapter
    if name == "SqliteMirroredPortfolioRepository":
        from .sqlite_mirror_decorator import SqliteMirroredPortfolioRepository
        return SqliteMirroredPortfolioRepository
    if name == "PriceYFinanceAdapter":
        from .price_yfinance_adapter import PriceYFinanceAdapter
        return PriceYFinanceAdapter
    if name == "CompositeMarketPriceAdapter":
        from .composite_price_adapter import CompositeMarketPriceAdapter
        return CompositeMarketPriceAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
