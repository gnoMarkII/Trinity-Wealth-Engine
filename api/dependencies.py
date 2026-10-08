"""FastAPI Application Dependencies & Composition Provider."""
from pathlib import Path
from typing import Any, Optional
from fastapi import Request
import yfinance as yf

from tools.portfolio.bootstrap import (
    PortfolioDependencies,
    PortfolioApplication,
    build_default_portfolio_dependencies,
    build_portfolio_application,
    build_legacy_portfolio_service,
)
from tools.portfolio.service import PortfolioService
from application.jobs.service import JobApplicationService
from application.kanban.service import KanbanApplicationService
from application.notebooklm.service import (
    NotebookLMApplicationService,
    NotebookLMPreflightError,
    NOTEBOOKLM_SOURCES_DIR,
)
from application.notebooklm.bootstrap import build_notebooklm_service
from application.earnings_call.service import EarningsCallApplicationService
from application.earnings_call.bootstrap import build_earnings_call_service
from tools.content.earnings_call.adapters.llm_adapter import LlmEarningsCallSummarizerAdapter
from tools.content.earnings_call.adapters.obsidian_adapter import ObsidianEarningsCallAdapter
from tools.content.earnings_call.adapters.kanban_adapter import KanbanEarningsCallAdapter
from tools.market.ohlcv.bootstrap import build_ohlcv_service
from tools.market.ohlcv.application.query_service import OHLCVQueryService
from tools.market.financials.service import FinancialsService
from tools.market.financials.bootstrap import build_default_financials_service
from api.db.adapters import (
    SqliteJobRepositoryAdapter,
    SqliteKanbanRepositoryAdapter,
    SqliteNotebookLMJobRepositoryAdapter,
    SqliteNotebookLMCardRepositoryAdapter,
    SqliteAnalystCacheAdapter,
    SqliteValuationLedgerAdapter,
    SqliteInsiderLedgerAdapter,
    SqliteInsiderSyncAdapter,
    SqliteEarningsCallWorkflowAdapter,
)
from application.equity.service import (
    EquityAnalystApplicationService,
    EquityFinancialsApplicationService,
    EquityInsiderApplicationService,
    EquityValuationApplicationService,
)
from application.equity.query_service import EquityResearchQueryService
from application.macro.service import MacroApplicationService, PortfolioCalendarApplicationService
from application.macro.sector_rotation_service import SectorRotationApplicationService
from application.macro.card_service import NewsFunnelCardApplicationService
from api.db.legacy_adapter import LegacyNewsFunnelCardAdapter, NewsFunnelPromptAdapter
from application.notebooklm.ports import NotebookLMDispatchPort, NotebookLMBinaryPort
from tools.content.notebooklm.adapters.filesystem import (
    FilesystemManifestAdapter,
    FilesystemSourceCatalogAdapter,
)
from tools.content.notebooklm.adapter import check_binary_available
from tools.archivist import core as archivist_core
from application.knowledge.write_service import KnowledgeWriteService
from tools.archivist.vault_paths import VaultPaths
from tools.archivist.composition import build_knowledge_note_writer, build_knowledge_write_port
from tools.market.calendar import get_asset_calendar
from tools.market.earnings import fetch_earnings_dates
from tools.market.adapters.equity_research import (
    MarketAssetResolverAdapter,
    YFinanceAnalystProviderAdapter,
    EquitySidecarValuationAdapter,
)
from tools.market.adapters.insider_provider import YFinanceInsiderHistoryAdapter
from tools.market.adapters.equity_vault_query_adapter import EquityVaultQueryAdapter
from tools.macro.adapters.market_calendar_adapter import (
    MarketAssetResolverAdapter as MacroMarketAssetResolverAdapter,
    MarketCalendarAdapter,
)
from tools.macro.sector_rotation.bootstrap import get_sector_rotation_service
from tools.macro.adapters.news_funnel_store_adapter import NewsFunnelStoreAdapter
from tools.macro.adapters.strategy_vault_adapter import IndicatorSeriesAdapter, StrategyVaultAdapter

# Compatibility seam for legacy tests/importers that patch the composition
# root's vault path.  Production code resolves adapters here; routers and
# application services never import this symbol.
VAULT_PATH = archivist_core.VAULT_PATH
_INITIAL_VAULT_PATH = VAULT_PATH


def _get_vault_path() -> Path:
    """Resolve the injectable vault path while honoring legacy patch seams.

    New callers can patch the archivist core path; older tests/importers patch
    ``api.dependencies.VAULT_PATH``.  Both remain composition-root concerns
    and neither leaks into the application layer.
    """
    if VAULT_PATH != _INITIAL_VAULT_PATH:
        return Path(VAULT_PATH)
    return Path(archivist_core.VAULT_PATH)


def get_knowledge_write_service() -> KnowledgeWriteService:
    """Compose the local durable write broker for multi-app callers."""
    vault_paths = VaultPaths(_get_vault_path())
    return KnowledgeWriteService(build_knowledge_write_port(vault_paths=vault_paths))

# Process-scoped query service: the cache is application state, not a new
# object per request.  Tests and compatibility callers can clear the exposed
# cache through ``api.routes_ohlcv``.
_OHLCV_SERVICE = build_ohlcv_service()


def get_portfolio_service() -> PortfolioService:
    """Dependency provider for unified PortfolioService facade."""
    # Construct concrete adapters in the composition root and inject them into
    # the compatibility facade.  This keeps the production path explicit;
    # ``PortfolioService()`` remains available only for legacy callers.
    deps = build_default_portfolio_dependencies()
    return build_legacy_portfolio_service(deps)


def get_portfolio_application() -> PortfolioApplication:
    """Dependency provider for granular PortfolioApplication aggregate."""
    return build_portfolio_application()


def get_job_service() -> JobApplicationService:
    """Dependency provider for JobApplicationService."""
    return JobApplicationService(repo=SqliteJobRepositoryAdapter())


def get_kanban_service() -> KanbanApplicationService:
    """Dependency provider for KanbanApplicationService."""
    return KanbanApplicationService(repo=SqliteKanbanRepositoryAdapter())


class _JobQueueDispatchAdapter(NotebookLMDispatchPort):
    def __init__(self, queue) -> None:
        self._queue = queue

    def dispatch(
        self,
        instruction: str,
        card_id: Optional[str] = None,
        flow: str = "notebooklm",
        scope: str = "both",
    ) -> str:
        if self._queue is None:
            raise RuntimeError("NotebookLM job queue is not running")
        return self._queue.dispatch(instruction, card_id, flow=flow, scope=scope)


class _NotebookLMBinaryAdapter(NotebookLMBinaryPort):
    def __init__(self, checker) -> None:
        self._checker = checker

    def check_available(self) -> None:
        try:
            self._checker()
        except Exception as exc:
            # Translate the concrete CLI adapter's exception into an
            # application-owned error before it crosses the port boundary.
            raise NotebookLMPreflightError(str(exc)) from exc


def get_notebooklm_service(request: Request) -> NotebookLMApplicationService:
    """Dependency provider for NotebookLMApplicationService."""
    return build_notebooklm_service(
        repo=SqliteNotebookLMJobRepositoryAdapter(),
        source_catalog=FilesystemSourceCatalogAdapter(NOTEBOOKLM_SOURCES_DIR),
        manifest_port=FilesystemManifestAdapter(),
        card_repo=SqliteNotebookLMCardRepositoryAdapter(),
        dispatcher=_JobQueueDispatchAdapter(getattr(request.app.state, "notebooklm_job_queue", None)),
        binary=_NotebookLMBinaryAdapter(check_binary_available),
    )


def get_macro_notebooklm_export_service(
    request: Request,
) -> "MacroNotebookLMExportService":
    """Dependency provider for MacroNotebookLMExportService."""
    queue = getattr(request.app.state, "notebooklm_job_queue", None)
    return build_macro_notebooklm_export_service(queue=queue)


def build_macro_notebooklm_export_service(
    queue: Any = None,
) -> "MacroNotebookLMExportService":
    from application.macro.notebooklm_export_service import MacroNotebookLMExportService
    from tools.macro.adapters.macro_corpus_adapter import MacroCorpusAdapter
    from tools.macro.notebooklm_bundle_builder import MacroExportBundleBuilder
    from api.db.adapters import SqliteMacroExportRepositoryAdapter
    from tools.content.notebooklm.research_export_pipeline import (
        NotebookLMResearchExportPipelineAdapter,
    )

    return MacroNotebookLMExportService(
        corpus_reader=MacroCorpusAdapter(),
        bundle_builder=MacroExportBundleBuilder(),
        repo=SqliteMacroExportRepositoryAdapter(),
        dispatcher=_JobQueueDispatchAdapter(queue),
        pipeline=NotebookLMResearchExportPipelineAdapter(),
    )


def get_ohlcv_service() -> OHLCVQueryService:
    """Dependency provider for the canonical OHLCV query service."""
    return _OHLCV_SERVICE


def get_financials_service() -> FinancialsService:
    """Dependency provider for FinancialsService."""
    return build_default_financials_service()


def get_equity_financials_service() -> EquityFinancialsApplicationService:
    """Compose the equity financials use case and outbound dependencies."""
    return EquityFinancialsApplicationService(
        resolver=MarketAssetResolverAdapter(),
        financials=build_default_financials_service(),
    )


def get_equity_analyst_service() -> EquityAnalystApplicationService:
    """Compose the analyst application service and its outbound adapters."""
    return EquityAnalystApplicationService(
        resolver=MarketAssetResolverAdapter(),
        cache=SqliteAnalystCacheAdapter(),
        provider=YFinanceAnalystProviderAdapter(
            ticker_factory=yf.Ticker,
            calendar_fetcher=get_asset_calendar,
            earnings_fetcher=fetch_earnings_dates,
        ),
    )


def get_equity_query_service() -> EquityResearchQueryService:
    """Compose the read-only Equity Vault query service."""
    return EquityResearchQueryService(query_port=EquityVaultQueryAdapter())


def get_asset_resolver() -> MarketAssetResolverAdapter:
    """Provide the market asset resolver through an outbound port."""
    return MarketAssetResolverAdapter()


def get_macro_service() -> MacroApplicationService:
    """Compose Macro read-model ports at the application boundary."""
    strategy = StrategyVaultAdapter(vault_path=_get_vault_path())
    card_service = NewsFunnelCardApplicationService(
        storage=LegacyNewsFunnelCardAdapter(),
        prompt=NewsFunnelPromptAdapter(),
    )
    return MacroApplicationService(
        strategy=strategy,
        indicators=IndicatorSeriesAdapter(strategy),
        news_funnel=NewsFunnelStoreAdapter(card_sync=card_service.upsert),
    )


class _PortfolioReadAdapter:
    """Narrow read port over the legacy Portfolio facade."""

    def __init__(self, service: PortfolioService) -> None:
        self._service = service

    def get_state(self, portfolio_id: str):
        return self._service.get_structured_portfolio_state(portfolio_id=portfolio_id)

    def get_watchlist(self, portfolio_id: str):
        return self._service.get_structured_watchlist(portfolio_id=portfolio_id)


def get_portfolio_calendar_service() -> PortfolioCalendarApplicationService:
    """Compose the Portfolio Calendar application service."""
    return PortfolioCalendarApplicationService(
        portfolio=_PortfolioReadAdapter(get_portfolio_service()),
        resolver=MacroMarketAssetResolverAdapter(),
        calendar=MarketCalendarAdapter(),
    )


def get_equity_valuation_service() -> EquityValuationApplicationService:
    """Compose the valuation application service and its ledger/sidecar ports."""
    return EquityValuationApplicationService(
        ledger=SqliteValuationLedgerAdapter(),
        sidecar=EquitySidecarValuationAdapter(),
    )


def get_equity_insider_service() -> EquityInsiderApplicationService:
    """Compose the insider application service and its ledger/sync ports."""
    return EquityInsiderApplicationService(
        ledger=SqliteInsiderLedgerAdapter(),
        sync=SqliteInsiderSyncAdapter(provider=YFinanceInsiderHistoryAdapter()),
    )


def get_earnings_call_service() -> EarningsCallApplicationService:
    """Compose the earnings call application service with driven adapters."""
    kanban_service = get_kanban_service()
    return build_earnings_call_service(
        llm_port=LlmEarningsCallSummarizerAdapter(),
        writer_port=ObsidianEarningsCallAdapter(
            vault_path=_get_vault_path(),
            note_writer=build_knowledge_note_writer(vault_paths=VaultPaths(_get_vault_path())),
        ),
        workflow_port=SqliteEarningsCallWorkflowAdapter(),
        kanban_port=KanbanEarningsCallAdapter(kanban_service=kanban_service),
    )
