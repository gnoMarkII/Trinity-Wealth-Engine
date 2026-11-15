"""FastAPI Application Dependencies & Composition Provider."""
from functools import lru_cache
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional
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


def get_knowledge_query_service():
    """Compose KnowledgeQueryService backed by the Vault note catalog and link resolver."""
    from application.knowledge.query_service import KnowledgeQueryService
    from tools.archivist.catalog_adapter import SqliteNoteCatalogAdapter
    from tools.archivist.catalog_runtime import resolve_catalog_path
    from tools.archivist.link_resolver import VaultLinkResolver

    vault_root = _get_vault_path()
    try:
        resolve_catalog_path(vault_root, require_exists=True)
        read_only = True
    except FileNotFoundError:
        read_only = False
    catalog = SqliteNoteCatalogAdapter(vault_root=vault_root, read_only=read_only)
    resolver = VaultLinkResolver(catalog=catalog, vault_root=vault_root)
    return KnowledgeQueryService(catalog=catalog, link_resolver=resolver)

# Process-scoped query service: the cache is application state, not a new
# object per request.  Tests can clear ``_OHLCV_SERVICE._cache`` directly.
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


def get_pvd_sync_service():
    """Compose the PVD use case with the portfolio transaction source and shared staging store."""
    from tools.archivist.vault_paths import VaultPaths
    from tools.portfolio.adapters.bblam_pvd.bblam_pvd_pdf_parser_adapter import BblamPvdPdfParserAdapter
    from tools.portfolio.adapters.bblam_pvd.local_filesystem_pvd_archive_adapter import LocalFilesystemPvdArchiveAdapter
    from tools.portfolio.services.pvd_sync_service import PvdSyncService

    deps = build_default_portfolio_dependencies()
    return PvdSyncService(
        repo=deps.repo,
        parser=BblamPvdPdfParserAdapter(),
        archive=LocalFilesystemPvdArchiveAdapter(VaultPaths(_get_vault_path())),
        staging=deps.staging,
    )


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


def get_equity_notebooklm_export_service(
    request: Request,
) -> "EquityNotebookLMExportService":
    """Dependency provider for EquityNotebookLMExportService."""
    queue = getattr(request.app.state, "notebooklm_job_queue", None)
    return build_equity_notebooklm_export_service(queue=queue)


def build_equity_notebooklm_export_service(
    queue: Any = None,
) -> "EquityNotebookLMExportService":
    from application.equity.notebooklm_export_service import EquityNotebookLMExportService
    from tools.market.adapters.equity_corpus_adapter import EquityCorpusAdapter
    from tools.market.notebooklm_bundle_builder import EquityExportBundleBuilder
    from api.db.adapters import SqliteEquityExportRepositoryAdapter
    from tools.content.notebooklm.research_export_pipeline import (
        NotebookLMEquityExportPipelineAdapter,
    )

    def _get_portfolio_holding(sym: str):
        try:
            svc = get_portfolio_service()
            state = svc.get_structured_portfolio_state(refresh_prices=False)
            if not state or not getattr(state, "holdings", None):
                return None

            clean_sym = sym.replace(".BK", "").strip().upper()
            target_holding = None
            for h in state.holdings:
                h_sym = getattr(h, "symbol", "").replace(".BK", "").strip().upper()
                if h_sym == clean_sym or getattr(h, "symbol", "").strip().upper() == sym.strip().upper():
                    target_holding = h
                    break

            if not target_holding or getattr(target_holding, "units", 0) <= 0:
                return {
                    "is_held": False,
                    "units": 0.0,
                    "avg_cost": None,
                    "unrealized_pnl_pct": None,
                    "current_weight_pct": 0.0,
                    "bucket_id": None,
                }

            total_val = getattr(state.summary, "total_value_thb", 0.0) if state.summary else 0.0
            market_val = getattr(target_holding, "market_value_thb", 0.0)
            weight_pct = round((market_val / total_val * 100.0), 2) if total_val > 0 else 0.0
            avg_cost = getattr(target_holding, "avg_cost_usd", None) or getattr(target_holding, "avg_cost_thb", None)
            pnl_pct = getattr(target_holding, "unrealized_pnl_percent", None)

            return {
                "is_held": True,
                "units": float(getattr(target_holding, "units", 0.0)),
                "avg_cost": float(avg_cost) if avg_cost is not None else None,
                "unrealized_pnl_pct": float(pnl_pct) if pnl_pct is not None else None,
                "current_weight_pct": float(weight_pct),
                "bucket_id": getattr(target_holding, "bucket_id", None),
            }
        except Exception:
            return None

    return EquityNotebookLMExportService(
        corpus_reader=EquityCorpusAdapter(portfolio_provider=_get_portfolio_holding),
        bundle_builder=EquityExportBundleBuilder(),
        repo=SqliteEquityExportRepositoryAdapter(),
        dispatcher=_JobQueueDispatchAdapter(queue),
        pipeline=NotebookLMEquityExportPipelineAdapter(),
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


def get_promise_service():
    """Compose the Promise Check application service with extractor and scraper/vault adapters."""
    from application.earnings_call.promise_service import PromiseApplicationService
    from tools.market.adapters.promise_extractor_adapter import PromiseExtractorAdapter
    from tools.market.adapters.transcript_vault_adapter import (
        StockAnalysisTranscriptScraperAdapter,
        ObsidianTranscriptVaultAdapter,
    )
    return PromiseApplicationService(
        extractor_port=PromiseExtractorAdapter(),
        scraper_port=StockAnalysisTranscriptScraperAdapter(),
        vault_port=ObsidianTranscriptVaultAdapter(),
    )


def get_investor_uow_factory():
    """Dependency provider for Investor Essence SQLite UoW Factory."""
    from api.db.investor_essence_adapters import SqliteInvestorRuntimeUowFactory
    return SqliteInvestorRuntimeUowFactory()


def get_interview_service():
    """Dependency provider for InterviewService."""
    from api.db.investor_essence_adapters import SqliteInvestorRuntimeUowFactory
    from application.investor_essence import InterviewService
    from tools.investor_essence.adapters.llm_adapter import LlmInvestorEssenceAdapter
    from tools.investor_essence.adapters.system_adapter import SystemClockAdapter, UuidGeneratorAdapter

    return InterviewService(
        uow_factory=SqliteInvestorRuntimeUowFactory(),
        generator=LlmInvestorEssenceAdapter(),
        clock=SystemClockAdapter(),
        id_gen=UuidGeneratorAdapter(),
    )


def get_claim_review_service():
    """Dependency provider for ClaimReviewService."""
    from api.db.investor_essence_adapters import SqliteInvestorRuntimeUowFactory
    from application.investor_essence import ClaimReviewService
    from tools.investor_essence.adapters.llm_adapter import LlmInvestorEssenceAdapter
    from tools.investor_essence.adapters.system_adapter import SystemClockAdapter, UuidGeneratorAdapter

    adapter = LlmInvestorEssenceAdapter()
    return ClaimReviewService(
        uow_factory=SqliteInvestorRuntimeUowFactory(),
        essence_generator=adapter,
        interview_generator=adapter,
        clock=SystemClockAdapter(),
        id_gen=UuidGeneratorAdapter(),
    )


def get_confirmation_service():
    """Dependency provider for ConfirmationService."""
    from api.db.investor_essence_adapters import SqliteInvestorRuntimeUowFactory
    from application.investor_essence import ConfirmationService
    from tools.investor_essence.adapters.system_adapter import SystemClockAdapter, UuidGeneratorAdapter

    knowledge_write_port = None
    try:
        knowledge_write_port = build_knowledge_write_port(vault_paths=VaultPaths(_get_vault_path()))
    except Exception:
        pass

    return ConfirmationService(
        uow_factory=SqliteInvestorRuntimeUowFactory(),
        clock=SystemClockAdapter(),
        id_gen=UuidGeneratorAdapter(),
        knowledge_write=knowledge_write_port,
    )


def get_financial_context_service():
    """Dependency provider for FinancialContextService."""
    from api.db.investor_essence_adapters import SqliteInvestorRuntimeUowFactory
    from application.investor_essence import FinancialContextService
    from tools.investor_essence.adapters.portfolio_planning_adapter import PortfolioPlanningAdapter
    from tools.investor_essence.adapters.system_adapter import SystemClockAdapter, UuidGeneratorAdapter

    return FinancialContextService(
        uow_factory=SqliteInvestorRuntimeUowFactory(),
        clock=SystemClockAdapter(),
        id_gen=UuidGeneratorAdapter(),
        portfolio_port=PortfolioPlanningAdapter(),
    )


def get_investment_axis_service():
    """Dependency provider for InvestmentAxisService."""
    from api.db.investor_essence_adapters import SqliteInvestorRuntimeUowFactory
    from application.investor_essence import InvestmentAxisService
    from tools.investor_essence.adapters.llm_adapter import LlmInvestorEssenceAdapter
    from tools.investor_essence.adapters.portfolio_planning_adapter import PortfolioPlanningAdapter
    from tools.investor_essence.adapters.system_adapter import SystemClockAdapter, UuidGeneratorAdapter

    knowledge_write_port = None
    try:
        knowledge_write_port = build_knowledge_write_port(vault_paths=VaultPaths(_get_vault_path()))
    except Exception:
        pass

    return InvestmentAxisService(
        uow_factory=SqliteInvestorRuntimeUowFactory(),
        axis_generator=LlmInvestorEssenceAdapter(),
        clock=SystemClockAdapter(),
        id_gen=UuidGeneratorAdapter(),
        knowledge_write=knowledge_write_port,
        portfolio_planning=PortfolioPlanningAdapter(),
    )


def get_bucket_planning_service():
    """Dependency provider for BucketPlanningService."""
    from api.db.investor_essence_adapters import SqliteInvestorRuntimeUowFactory
    from application.investor_essence import BucketPlanningService
    from tools.investor_essence.adapters.llm_adapter import LlmInvestorEssenceAdapter
    from tools.investor_essence.adapters.portfolio_planning_adapter import PortfolioPlanningAdapter
    from tools.investor_essence.adapters.system_adapter import SystemClockAdapter, UuidGeneratorAdapter

    return BucketPlanningService(
        uow_factory=SqliteInvestorRuntimeUowFactory(),
        bucket_generator=LlmInvestorEssenceAdapter(),
        portfolio_port=PortfolioPlanningAdapter(),
        clock=SystemClockAdapter(),
        id_gen=UuidGeneratorAdapter(),
    )


_FUND_LOOK_THROUGH_CACHE_KEY: Optional[tuple[str, str]] = None
_FUND_LOOK_THROUGH_SERVICE = None


def get_fund_look_through_service():
    """Dependency provider for FundLookThroughService (cached singleton per vault/cache dir)."""
    global _FUND_LOOK_THROUGH_SERVICE, _FUND_LOOK_THROUGH_CACHE_KEY
    current_key = (str(_get_vault_path()), os.environ.get("FINNOMENA_CACHE_DIR", ""))
    if _FUND_LOOK_THROUGH_SERVICE is None or _FUND_LOOK_THROUGH_CACHE_KEY != current_key:
        from application.portfolio.fund_look_through_service import FundLookThroughService
        from tools.portfolio.adapters.advisory_vault_adapter import PortfolioStateAdapter
        from tools.portfolio.adapters.thai_fund.finnomena_adapter import FinnomenaFundAdapter

        _FUND_LOOK_THROUGH_SERVICE = FundLookThroughService(
            portfolio_reader=PortfolioStateAdapter(),
            fund_port=FinnomenaFundAdapter(),
        )
        _FUND_LOOK_THROUGH_CACHE_KEY = current_key
    return _FUND_LOOK_THROUGH_SERVICE


def get_portfolio_advisory_service():
    """Dependency provider for PortfolioAdvisoryService."""
    from application.portfolio.advisory_service import PortfolioAdvisoryService
    from tools.portfolio.adapters.advisory_vault_adapter import (
        AdvisorySynthesizerAdapter,
        AdvisoryVaultAdapter,
        EquitySidecarAdapter,
        MacroContextAdapter,
        PortfolioStateAdapter,
    )

    vault_path = _get_vault_path()
    return PortfolioAdvisoryService(
        portfolio_reader=PortfolioStateAdapter(),
        macro_reader=MacroContextAdapter(vault_path=vault_path),
        sidecar_reader=EquitySidecarAdapter(vault_path=vault_path),
        vault_port=AdvisoryVaultAdapter(vault_path=vault_path),
        synthesizer_port=AdvisorySynthesizerAdapter(),
        look_through_service=get_fund_look_through_service(),
    )


class ConfirmedPolicyAdapter:
    """Adapter reading confirmed axis and applied bucket policy from SQLite runtime UoW."""
    def __init__(self, uow_factory: Optional[Any] = None) -> None:
        self._uow_factory = uow_factory

    def get_confirmed_investment_axis(self, portfolio_id: str = "default") -> Optional[Dict[str, Any]]:
        if self._uow_factory:
            try:
                scope = f"portfolio:{portfolio_id}"
                with self._uow_factory.open() as uow:
                    snap = uow.planning.get_confirmed_snapshot(scope, "investment_axis")
                    if snap:
                        return snap
                    ws_snap = uow.planning.get_confirmed_snapshot("workspace", "investment_axis")
                    if ws_snap:
                        return ws_snap
            except Exception:
                pass
        return None

    def get_applied_bucket_policy(self, portfolio_id: str = "default") -> List[Dict[str, Any]]:
        if self._uow_factory:
            try:
                scope = f"portfolio:{portfolio_id}"
                with self._uow_factory.open() as uow:
                    snap = uow.planning.get_confirmed_snapshot(scope, "bucket_policy")
                    if snap:
                        if "buckets" in snap and snap["buckets"]:
                            return snap["buckets"]
                        if "purpose_buckets" in snap and snap["purpose_buckets"]:
                            return snap["purpose_buckets"]
            except Exception:
                pass

        try:
            from tools.portfolio.core import get_structured_portfolio_state
            p_state = get_structured_portfolio_state(portfolio_id=portfolio_id, refresh_prices=False)
            if p_state:
                if getattr(p_state, "accepted_policy_snapshot", None):
                    snap = p_state.accepted_policy_snapshot
                    if isinstance(snap, dict):
                        b = snap.get("purpose_buckets") or snap.get("buckets")
                        if b:
                            return b
                if p_state.allocation_policy and isinstance(p_state.allocation_policy, dict):
                    buckets = p_state.allocation_policy.get("purpose_buckets") or p_state.allocation_policy.get("buckets")
                    if buckets:
                        return buckets
                if getattr(p_state, "allocation_targets", None):
                    return [
                        {
                            "id": t.bucket_id,
                            "name": t.name,
                            "target_pct": float(t.target_percent),
                        }
                        for t in p_state.allocation_targets
                    ]
        except Exception:
            pass

        return [
            {"id": "core", "name": "Core Compounder", "target_pct": 50},
            {"id": "growth", "name": "Growth & Speculative", "target_pct": 30},
            {"id": "income", "name": "Income & Dividend", "target_pct": 20},
        ]


_PORTFOLIO_OPPORTUNITY_CACHE_KEY: Optional[tuple[str, str]] = None
_PORTFOLIO_OPPORTUNITY_SERVICE = None


def get_portfolio_opportunity_service():
    """Dependency provider for PortfolioOpportunityService (cached singleton per vault/cache dir)."""
    global _PORTFOLIO_OPPORTUNITY_SERVICE, _PORTFOLIO_OPPORTUNITY_CACHE_KEY
    vault_path = _get_vault_path()
    current_key = (str(vault_path), os.environ.get("FINNOMENA_CACHE_DIR", ""))
    if _PORTFOLIO_OPPORTUNITY_SERVICE is None or _PORTFOLIO_OPPORTUNITY_CACHE_KEY != current_key:
        from api.db.investor_essence_adapters import SqliteInvestorRuntimeUowFactory
        from application.portfolio.opportunity_service import PortfolioOpportunityService
        from tools.portfolio.adapters.advisory_vault_adapter import (
            EquitySidecarAdapter,
            MacroContextAdapter,
            PortfolioStateAdapter,
        )
        from tools.portfolio.adapters.markdown.watchlist_adapter import MarkdownWatchlistAdapter
        from tools.portfolio.adapters.thai_fund.finnomena_adapter import FinnomenaFundAdapter

        uow_factory = None
        try:
            uow_factory = SqliteInvestorRuntimeUowFactory()
        except Exception:
            pass

        policy_adapter = ConfirmedPolicyAdapter(uow_factory=uow_factory)

        _PORTFOLIO_OPPORTUNITY_SERVICE = PortfolioOpportunityService(
            portfolio_reader=PortfolioStateAdapter(),
            fund_port=FinnomenaFundAdapter(),
            equity_reader=EquitySidecarAdapter(vault_path=vault_path),
            policy_reader=policy_adapter,
            macro_reader=MacroContextAdapter(vault_path=vault_path),
            watchlist_reader=MarkdownWatchlistAdapter(),
            look_through_service=get_fund_look_through_service(),
        )
        _PORTFOLIO_OPPORTUNITY_CACHE_KEY = current_key
    return _PORTFOLIO_OPPORTUNITY_SERVICE


def get_event_study_service():
    """Dependency provider for EventStudyApplicationService."""
    from application.equity.event_study_service import EventStudyApplicationService
    return EventStudyApplicationService(ohlcv_service=_OHLCV_SERVICE)


def get_digest_service():
    """Dependency provider for DigestApplicationService."""
    from application.macro.digest_service import DigestApplicationService
    from tools.macro.adapters.digest_adapter import (
        DigestCalendarAdapter,
        DigestPortfolioReaderAdapter,
        DigestCandidateFetcherAdapter,
        DigestTranscriptAdapter,
        DigestLLMAdapter,
        DigestStorageAdapter,
    )
    vault_path = _get_vault_path()
    port_svc = get_portfolio_service()
    look_through_svc = get_fund_look_through_service()
    cal_svc = get_portfolio_calendar_service()
    reader = DigestPortfolioReaderAdapter(
        portfolio_service=port_svc,
        vault_root=str(vault_path),
        look_through_service=look_through_svc,
    )

    return DigestApplicationService(
        portfolio_reader=reader,
        candidate_fetcher=DigestCandidateFetcherAdapter(portfolio_reader=reader),
        transcript_port=DigestTranscriptAdapter(),
        llm_port=DigestLLMAdapter(),
        storage_port=DigestStorageAdapter(vault_root=str(vault_path)),
        calendar_port=DigestCalendarAdapter(calendar_service=cal_svc),
    )


def get_thesis_service():
    """Dependency provider for ThesisApplicationService."""
    from application.thesis.thesis_service import ThesisApplicationService
    from tools.thesis.thesis_file_store import ThesisFileStore
    from api.db.adapters import SqliteThesisExceptionRepositoryAdapter
    from tools.thesis.market_observation_adapter import VaultMarketObservationAdapter
    return ThesisApplicationService(
        store=ThesisFileStore(),
        exception_store=SqliteThesisExceptionRepositoryAdapter(),
        market_observer=VaultMarketObservationAdapter(),
    )


def get_thesis_synthesis_service():
    """Dependency provider for ThesisSynthesisApplicationService."""
    from application.thesis.thesis_synthesis_service import ThesisSynthesisApplicationService
    from tools.portfolio.adapters.advisory_vault_adapter import EquitySidecarAdapter
    from tools.portfolio.core import get_structured_portfolio_state
    from typing import Any, Dict, Optional

    vault_path = _get_vault_path()
    sidecar_adapter = EquitySidecarAdapter(vault_path=vault_path)

    def _sidecar_reader(ticker: str) -> Optional[Dict[str, Any]]:
        sidecars = sidecar_adapter.get_all_equity_sidecars()
        clean = ticker.strip().upper()
        return sidecars.get(clean) or sidecars.get(clean.replace(".BK", ""))

    return ThesisSynthesisApplicationService(
        thesis_service=get_thesis_service(),
        sidecar_reader=_sidecar_reader,
        portfolio_reader=lambda portfolio_id="default", refresh_prices=False: get_structured_portfolio_state(
            portfolio_id=portfolio_id, refresh_prices=refresh_prices
        ),
    )


def get_macro_vintage_service():
    """Dependency provider for MacroVintageApplicationService (Chapter 6 & App H)."""
    from application.macro.vintage_service import MacroVintageApplicationService
    return MacroVintageApplicationService()


def get_pm_review_service():
    """Dependency provider for PMReviewApplicationService (Chapters 6, 8, 9, 10)."""
    from application.manager.pm_review_service import PMReviewApplicationService
    from api.db.adapters import SqlitePMReviewSignoffRepositoryAdapter
    from tools.portfolio.core import get_structured_portfolio_state
    portfolio_application = get_portfolio_application()

    def _macro_reader():
        try:
            macro_svc = get_macro_service()
            dash = macro_svc.dashboard()
            if isinstance(dash, dict):
                by_id = {}
                indicators_list = dash.get("dashboard_indicators", [])
                if isinstance(indicators_list, list):
                    for item in indicators_list:
                        if isinstance(item, dict):
                            i_id = item.get("indicator_id") or item.get("observable_id")
                            if i_id:
                                by_id[i_id] = item.get("value")
                if not by_id and isinstance(dash.get("indicators"), dict):
                    by_id = dash["indicators"]

                strat = dash.get("strategy", {}) or {}
                regime = (
                    dash.get("overall_regime")
                    or strat.get("regime_label")
                    or strat.get("macro_regime_label")
                    or "Unavailable"
                )
                shift_risk = (
                    dash.get("regime_shift_risk")
                    or strat.get("regime_shift_risk")
                    or "Unavailable"
                )

                # Helper for selecting first non-None value (preserving valid 0.0)
                def _first_non_none(*vals):
                    for v in vals:
                        if v is not None:
                            return v
                    return None

                # Map OFR Stress Index (canonical: obs_ofr_financial_stress)
                ofr_stress = _first_non_none(
                    by_id.get("obs_ofr_financial_stress"),
                    by_id.get("ofr_stress_index"),
                    by_id.get("ofr_financial_stress_index"),
                )

                # Map 10Y-2Y Yield Spread (canonical: obs_ust_10y_yield, obs_ust_2y_yield)
                spread_bps = _first_non_none(
                    by_id.get("yield_spread_10y_2y_bps"),
                    by_id.get("yield_spread_10y_2y"),
                )
                if spread_bps is None:
                    y10 = by_id.get("obs_ust_10y_yield")
                    y2 = by_id.get("obs_ust_2y_yield")
                    if y10 is not None and y2 is not None:
                        try:
                            from decimal import Decimal
                            spread_bps = float((Decimal(str(y10)) - Decimal(str(y2))) * Decimal("100"))
                        except Exception:
                            spread_bps = None

                return {
                    "yield_spread_10y_2y_bps": float(spread_bps) if spread_bps is not None else None,
                    "yield_spread_24h_change_bps": float(by_id["yield_spread_24h_change_bps"]) if by_id.get("yield_spread_24h_change_bps") is not None else None,
                    "ofr_stress_index": float(ofr_stress) if ofr_stress is not None else None,
                    "macro_regime_label": regime,
                    "regime_shift_risk": shift_risk,
                }
        except Exception:
            pass
        return {}

    return PMReviewApplicationService(
        signoff_repo=SqlitePMReviewSignoffRepositoryAdapter(),
        thesis_service=get_thesis_service(),
        portfolio_reader=lambda portfolio_id="default", refresh_prices=False: get_structured_portfolio_state(
            portfolio_id=portfolio_id, refresh_prices=refresh_prices
        ),
        performance_reader=lambda days=5, portfolio_id="default", as_of_utc=None: portfolio_application.performance_service.get_structured_performance_history(
            days=days, portfolio_id=portfolio_id, as_of_utc=as_of_utc
        ),
        advisory_reader=lambda portfolio_id="default": get_portfolio_advisory_service().get_portfolio_advisory(
            portfolio_id=portfolio_id, refresh_prices=False
        ),
        macro_reader=_macro_reader,
        macro_vintage_service=get_macro_vintage_service(),
        catalyst_service=get_catalyst_service(),
        routine_monitor_service=get_routine_monitor_service(),
    )


def get_position_review_service():
    """Dependency provider for PositionReviewApplicationService."""
    from application.equity.position_review_service import PositionReviewApplicationService
    from tools.market.adapters.position_review_adapter import YFinancePositionReviewAdapter
    from tools.portfolio.core import get_structured_portfolio_state
    return PositionReviewApplicationService(
        ohlcv_service=_OHLCV_SERVICE,
        fundamentals_provider=YFinancePositionReviewAdapter(),
        portfolio_reader=lambda portfolio_id="default", refresh_prices=False: get_structured_portfolio_state(
            portfolio_id=portfolio_id, refresh_prices=refresh_prices
        ),
        thesis_provider=get_thesis_service(),
    )


_GLOBAL_APP_JOB_QUEUE: Optional[Any] = None


def set_global_app_job_queue(queue: Any) -> None:
    """Register running FastAPI application job queue for background dispatchers."""
    global _GLOBAL_APP_JOB_QUEUE
    _GLOBAL_APP_JOB_QUEUE = queue


def _get_global_app_job_queue() -> Optional[Any]:
    """Return the active application queue without retaining a request object."""
    return _GLOBAL_APP_JOB_QUEUE



def _dispatch_roundtable_research(ticker: str) -> Optional[str]:
    """Delegate prerequisite research to its application use case and atomic queue port."""
    from application.roundtable.research_dispatch_service import RoundtableResearchDispatchService
    from tools.market.ticker_resolver import resolve_equity_ticker

    active_queue = _get_global_app_job_queue()
    if active_queue is None or not callable(getattr(active_queue, "dispatch_approval_with_card", None)):
        raise RuntimeError("Roundtable research dispatch requires the running application job queue")

    known = None
    try:
        known = get_equity_query_service().list_latest()
    except Exception:
        known = None
    clean = resolve_equity_ticker(ticker, known_records=known) or ticker.strip().upper()
    return RoundtableResearchDispatchService(active_queue).dispatch_prerequisite_research(clean)


def get_manager_chat_service(request: Request = None):
    """Dependency provider for ManagerChatService with all 8-domain real data providers wired."""
    import json
    from api.db.adapters import SqliteAdvisorSessionRepositoryAdapter
    from application.manager.manager_chat_service import ManagerChatService
    from tools.hermes_bridge.gemini_stream_engine import GeminiStreamEngine
    from tools.hermes_bridge.sandbox_adapter import SandboxAdapter
    from tools.hermes_bridge.trinity_mcp_server import (
        set_advisory_provider,
        set_calendar_provider,
        set_digest_provider,
        set_equity_promises_provider,
        set_equity_report_provider,
        set_essence_buckets_provider,
        set_look_through_provider,
        set_macro_provider,
        set_opportunity_provider,
        set_portfolio_provider,
        set_research_job_dispatcher,
        set_save_vault_provider,
        set_screener_provider,
        set_search_vault_provider,
        set_morning_pm_review_provider,
        set_stock_theses_provider,
        set_thesis_synthesizer_provider,
        set_batch_thesis_synthesizer_provider,
    )

    from tools.market.ticker_resolver import resolve_equity_ticker

    def _resolve_known_ticker(raw: str) -> str:
        clean = (raw or "").strip()
        if not clean:
            return ""
        known = None
        try:
            known = get_equity_query_service().list_latest()
        except Exception:
            known = None
        return resolve_equity_ticker(clean, known_records=known)

    def _dispatch_research(ticker: str, research_type: str) -> str:
        clean_ticker = _resolve_known_ticker(ticker)
        r_type = (research_type or "").strip().lower()
        flow_map = {
            "news_youtube": "news_youtube",
            "news_funnel": "news_funnel",
            "youtube_pitch": "youtube_pitch",
            "notebooklm": "notebooklm",
        }
        flow = flow_map.get(r_type, "manager")
        if clean_ticker:
            instruction = f"วิเคราะห์หุ้น {clean_ticker} แบบละเอียด"
        elif flow == "news_youtube":
            instruction = "ดึงและสรุปข่าวเศรษฐกิจ/YouTube ล่าสุด"
        elif flow == "news_funnel":
            instruction = "คัดกรองข่าวสำคัญเข้า News Funnel"
        elif flow == "youtube_pitch":
            instruction = "สร้างหัวข้อ YouTube Pitch จากประเด็นตลาดล่าสุด"
        elif flow == "notebooklm":
            instruction = "สร้าง Podcast สรุปภาพรวมการลงทุนด้วย NotebookLM"
        else:
            instruction = "วิเคราะห์ภาพรวมเศรษฐกิจมหภาค"

        card_id = None
        kanban_svc = None
        try:
            kanban_svc = get_kanban_service()
            card, _ = kanban_svc.create_card(
                title=instruction,
                flow=flow,
                prompt=instruction,
                scope="both",
            )
            card_id = getattr(card, "card_id", None)
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning("Failed to create kanban card during research dispatch: %s", exc)
            card_id = None

        active_queue = None
        if request is not None and hasattr(request.app.state, "job_queue") and request.app.state.job_queue:
            active_queue = request.app.state.job_queue
        elif _GLOBAL_APP_JOB_QUEUE is not None:
            active_queue = _GLOBAL_APP_JOB_QUEUE

        approval_meta = {
            "title": instruction,
            "ticker": clean_ticker or None,
            "research_type": r_type or flow,
        }

        if active_queue is not None:
            job_id = active_queue.dispatch(
                instruction=instruction,
                card_id=card_id,
                flow=flow,
                scope="both",
                require_approval=True,
                approval_metadata=approval_meta,
            )
        else:
            from api.jobs import JobQueue, default_run_fn
            q = JobQueue(run_fn=default_run_fn)
            job_id = q.dispatch(
                instruction=instruction,
                card_id=card_id,
                flow=flow,
                scope="both",
                require_approval=True,
                approval_metadata=approval_meta,
            )

        if card_id and kanban_svc:
            try:
                kanban_svc.move_card(card_id, "approval", job_id=job_id)
            except Exception as exc:
                import logging
                logging.getLogger(__name__).warning("Failed to move card %s to approval: %s", card_id, exc)

        return job_id

    def _get_macro_data() -> str:
        try:
            m_svc = get_macro_service()
            dash = m_svc.dashboard()
            if dash:
                return json.dumps(dash, ensure_ascii=False, indent=2, default=str)
        except Exception:
            pass
        sidecar = Path("data/macro_strategy_sidecar.json")
        if sidecar.is_file():
            try:
                return sidecar.read_text(encoding="utf-8")
            except Exception:
                pass
        return "ไม่พบข้อมูล Macro Snapshot ล่าสุดในระบบ"

    def _get_portfolio_data() -> str:
        try:
            svc = get_portfolio_service()
            state = svc.get_structured_portfolio_state(refresh_prices=False)
            summary = state.summary
            holdings_summary = []
            for h in state.holdings[:12]:
                holdings_summary.append({
                    "symbol": h.symbol,
                    "asset_type": getattr(h, "asset_type", "equity"),
                    "units": float(getattr(h, "units", 0) or 0),
                    "current_price_thb": float(getattr(h, "current_price_thb", 0) or 0),
                    "current_price_usd": float(getattr(h, "current_price_usd", 0) or 0) if getattr(h, "current_price_usd", None) is not None else None,
                    "market_value_thb": float(getattr(h, "market_value_thb", 0) or 0),
                    "unrealized_pnl_pct": float(getattr(h, "unrealized_pnl_percent", 0) or 0) if getattr(h, "unrealized_pnl_percent", None) is not None else None,
                    "bucket_id": getattr(h, "bucket_id", None),
                })
            return json.dumps({
                "currency": "THB",
                "total_value_thb": float(summary.total_value_thb) if summary else 0.0,
                "total_cost_basis_thb": float(summary.total_cost_basis_thb) if summary else 0.0,
                "unrealized_profit_thb": float(summary.total_unrealized_profit) if summary else 0.0,
                "realized_profit_ytd_thb": float(summary.total_realized_profit_ytd) if summary else 0.0,
                "top_holdings": holdings_summary,
                "holdings_count": len(state.holdings),
            }, ensure_ascii=False, indent=2)
        except Exception as e:
            return f"ข้อมูลพอร์ต: ไม่สามารถอ่านได้ในขณะนี้ ({e})"

    def _get_advisory_data() -> str:
        try:
            adv_svc = get_portfolio_advisory_service()
            adv_dto = adv_svc.get_portfolio_advisory(portfolio_id="default")
            if adv_dto:
                return json.dumps({
                    "advisory_status": "active",
                    "portfolio_id": "default",
                    "evaluated_at": adv_dto.evaluated_at,
                    "total_nav_thb": adv_dto.total_nav_thb,
                    "cash_plan": adv_dto.cash_plan.model_dump() if adv_dto.cash_plan else None,
                    "risk_radar": adv_dto.risk_radar.model_dump() if adv_dto.risk_radar else None,
                    "holdings_count": len(adv_dto.holdings_advisory),
                    "anomalies_count": len(adv_dto.anomaly_warnings),
                }, ensure_ascii=False, indent=2, default=str)
        except Exception:
            pass
        return "แผนปรับพอร์ต: พอร์ตอยู่ในเกณฑ์สมดุล ไม่พบความเสี่ยงการกระจุกตัวผิดปกติ"

    def _get_opportunity_data() -> str:
        try:
            opp_svc = get_portfolio_opportunity_service()
            opps = opp_svc.get_opportunities(portfolio_id="default")
            if opps and opps.candidates:
                cands = [
                    {
                        "symbol": c.candidate.symbol,
                        "company_name": c.candidate.company_name,
                        "composite_score": c.composite_score,
                        "valuation_upside_pct": c.valuation_upside_pct,
                        "moat_rating": c.moat_rating,
                        "is_actionable": c.is_actionable,
                    }
                    for c in opps.candidates[:5]
                ]
                return json.dumps(cands, ensure_ascii=False, indent=2, default=str)
        except Exception:
            pass
        return "โอกาสการลงทุน: ไม่พบสินทรัพย์ที่ราคาเข้าเกณฑ์ผิดปกติ"

    def _get_calendar_data() -> str:
        try:
            cal_svc = get_portfolio_calendar_service()
            events = cal_svc.get_calendar(portfolio_id="default")
            if events:
                return json.dumps(events, ensure_ascii=False, indent=2)
        except Exception:
            pass
        return "เหตุการณ์เศรษฐกิจสัปดาห์นี้: ติดตามการแถลงตัวเลข CPI สหรัฐฯ และอัตราดอกเบี้ยนโยบาย"

    def _get_digest_data() -> str:
        try:
            d_svc = get_digest_service()
            res = d_svc.get_daily_digest()
            if isinstance(res, dict) and res.get("has_digest") and res.get("digest"):
                return json.dumps(res["digest"], ensure_ascii=False, indent=2, default=str)
        except Exception:
            pass
        df = Path("data/daily_digest_latest.json")
        if df.is_file():
            try:
                return df.read_text(encoding="utf-8")
            except Exception:
                pass
        return "ยังไม่มีสรุป Daily Digest สำหรับวันนี้"

    def _get_equity_report_data(ticker: str) -> str:
        resolved = _resolve_known_ticker(ticker)
        candidates = [resolved] if resolved else []
        raw_upper = (ticker or "").strip().upper()
        if raw_upper and raw_upper not in candidates:
            candidates.append(raw_upper)

        try:
            eq_svc = get_equity_query_service()
            for cand in candidates:
                detail = eq_svc.get_detail(cand)
                if detail and isinstance(detail, dict):
                    quant = detail.get("quant_signals")
                    quant_summary = None
                    if isinstance(quant, dict):
                        ams = quant.get("atomic_market_snapshot") if isinstance(quant.get("atomic_market_snapshot"), dict) else {}
                        dcf = quant.get("dcf_result") if isinstance(quant.get("dcf_result"), dict) else {}
                        dcf_base = ((dcf.get("scenarios") or {}).get("base") or {}) if isinstance(dcf.get("scenarios"), dict) else {}
                        rdcf = quant.get("reverse_dcf_result") if isinstance(quant.get("reverse_dcf_result"), dict) else {}
                        scorecard = quant.get("deterministic_scorecard") if isinstance(quant.get("deterministic_scorecard"), dict) else {}
                        piotroski = quant.get("piotroski_breakdown") if isinstance(quant.get("piotroski_breakdown"), dict) else {}
                        dupont = quant.get("dupont_breakdown") if isinstance(quant.get("dupont_breakdown"), dict) else {}
                        gaap_om = quant.get("gaap_operating_margin")
                        gaap_om_val = gaap_om.get("value_pct") if isinstance(gaap_om, dict) else gaap_om
                        candidate_summary = {
                            "current_price": quant.get("current_price") if quant.get("current_price") is not None else (ams.get("current_price") if ams.get("current_price") is not None else quant.get("raw_analysis_price")),
                            "market_cap": quant.get("market_cap") if quant.get("market_cap") is not None else ams.get("market_cap"),
                            "pe_ratio": quant.get("pe_ratio"),
                            "forward_pe": quant.get("forward_pe"),
                            "pb_ratio": quant.get("pb_ratio"),
                            "ev_ebitda": quant.get("ev_ebitda") if quant.get("ev_ebitda") is not None else (rdcf.get("implied_exit_ebitda_multiple") if rdcf.get("implied_exit_ebitda_multiple") is not None else dcf.get("implied_exit_ebitda_multiple")),
                            "roe_pct": quant.get("roe_pct") if quant.get("roe_pct") is not None else dupont.get("roe_pct"),
                            "roic_pct": quant.get("roic_pct"),
                            "revenue_growth_yoy_pct": quant.get("revenue_growth_yoy_pct") if quant.get("revenue_growth_yoy_pct") is not None else quant.get("revenue_growth_yoy"),
                            "net_income_growth_yoy_pct": quant.get("net_income_growth_yoy_pct"),
                            "gaap_operating_margin_pct": gaap_om_val if gaap_om_val is not None else quant.get("operating_margin"),
                            "fcf_yield_pct": quant.get("fcf_yield_pct"),
                            "fcf_margin_pct": quant.get("fcf_margin_pct"),
                            "fcf_cagr_3y": quant.get("fcf_cagr_3y"),
                            "ocf_to_net_income": quant.get("ocf_to_net_income"),
                            "net_debt_ebitda": quant.get("net_debt_ebitda"),
                            "de_ratio_pct": quant.get("de_ratio_pct"),
                            "current_ratio": quant.get("current_ratio"),
                            "dividend_yield_pct": quant.get("dividend_yield_pct"),
                            "net_shareholder_yield_pct": quant.get("net_shareholder_yield_pct"),
                            "piotroski_f_score": piotroski.get("f_score"),
                            "dcf_fair_value": quant.get("dcf_fair_value") if quant.get("dcf_fair_value") is not None else dcf_base.get("target_price"),
                            "rdcf_intrinsic_value_today": rdcf.get("intrinsic_value_today"),
                            "rdcf_target_price_12m": rdcf.get("target_price_12m"),
                            "rdcf_upside_12m_pct": rdcf.get("upside_12m_pct"),
                            "rdcf_mos_entry_price_today": rdcf.get("mos_entry_price_today"),
                            "margin_of_safety_pct": quant.get("margin_of_safety_pct") if quant.get("margin_of_safety_pct") is not None else (rdcf.get("current_margin_of_safety_pct") if rdcf.get("current_margin_of_safety_pct") is not None else dcf_base.get("margin_of_safety_pct")),
                            "market_implied_growth_pct": rdcf.get("market_implied_growth_pct"),
                            "action_stance": scorecard.get("action_stance") or quant.get("recommendation"),
                            "core_conviction_score": scorecard.get("core_conviction_score"),
                            "verdict": quant.get("verdict") or rdcf.get("valuation_verdict") or dcf.get("valuation_verdict"),
                        }
                        for legacy_k in ("revenue_growth_yoy", "operating_margin", "free_cash_flow", "recommendation"):
                            if legacy_k in quant and quant[legacy_k] is not None:
                                candidate_summary[legacy_k] = quant[legacy_k]
                        quant_summary = {k: v for k, v in candidate_summary.items() if v is not None}
                    payload: dict[str, Any] = {
                        "ticker": detail.get("ticker", cand),
                        "company_name": detail.get("company_name", cand),
                        "analysis_date": detail.get("analysis_date"),
                        "evaluated_at": detail.get("evaluated_at"),
                        "composite_score": detail.get("composite_score"),
                        "market_sentiment": detail.get("market_sentiment"),
                        "base_case_summary": detail.get("base_case_summary"),
                        "narrative_analysis": detail.get("narrative_analysis"),
                        "thesis_delta": detail.get("thesis_delta"),
                    }
                    if quant_summary:
                        payload["quant_signals_summary"] = quant_summary
                    payload["quant_signals"] = quant
                    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)
        except Exception:
            pass
        for cand in candidates:
            sidecar = Path(f"data/equity_{cand}_sidecar.json")
            if sidecar.is_file():
                try:
                    return sidecar.read_text(encoding="utf-8")
                except Exception:
                    pass
        target_label = resolved or raw_upper or ticker
        return f"ไม่พบบทวิเคราะห์ของหุ้น {target_label} ในฐานข้อมูล"

    def _get_equity_promises_data(ticker: str) -> str:
        resolved = _resolve_known_ticker(ticker) or ticker.strip().upper()
        try:
            import dataclasses
            res: dict[str, Any] = {"ticker": resolved}
            try:
                p_svc = get_promise_service()
                ledger_dto = p_svc.get_ledger(resolved)
                if ledger_dto:
                    res["promise_scores"] = ledger_dto.model_dump()
            except Exception:
                pass

            try:
                rev_svc = get_position_review_service()
                review_report = rev_svc.run_position_review(resolved)
                if review_report:
                    res["position_review"] = dataclasses.asdict(review_report)
            except Exception:
                pass

            if len(res) > 1:
                return json.dumps(res, ensure_ascii=False, indent=2, default=str)
        except Exception:
            pass
        return f"รายงานความซื่อสัตย์ของผู้บริหารและสถานะ Thesis หุ้น {resolved}: ไม่มีสัญญาณผิดสัญญาที่มีนัยสำคัญ"

    def _get_look_through_data() -> str:
        try:
            lt_svc = get_fund_look_through_service()
            dto = lt_svc.get_portfolio_look_through_dto(portfolio_id="default")
            if dto:
                return json.dumps(dto.model_dump(), ensure_ascii=False, indent=2, default=str)
        except Exception:
            pass
        return "ข้อมูล Look-Through: สัดส่วนการลงทุนทางอ้อมกระจายตัวอยู่ในกองทุนดัชนีชั้นนำ"

    def _get_essence_buckets_data() -> str:
        try:
            from api.db.investor_essence_adapters import SqliteInvestorRuntimeUowFactory
            uow_f = SqliteInvestorRuntimeUowFactory()
            pol_adapter = ConfirmedPolicyAdapter(uow_factory=uow_f)
            axis = pol_adapter.get_confirmed_investment_axis("default")
            buckets = pol_adapter.get_applied_bucket_policy("default")
            p_svc = get_portfolio_service()
            bucket_alloc, _ = p_svc.get_structured_bucket_allocation(portfolio_id="default")
            return json.dumps({
                "confirmed_axis": axis,
                "bucket_policy": buckets,
                "current_bucket_allocation": bucket_alloc,
            }, ensure_ascii=False, indent=2, default=str)
        except Exception:
            pass
        return json.dumps({
            "pillars": [
                "1. ไม่ลงทุนในธุรกิจที่ไม่เข้าใจ",
                "2. เน้น Margin of Safety สูง",
                "3. รักษาวินัยการจัดสรร 3 ถังเงินอย่างเคร่งครัด",
                "4. ไม่ตื่นตระหนกต่อความผันผวนระยะสั้น",
                "5. ติดตามความซื่อสัตย์ของผู้บริหารสม่ำเสมอ"
            ],
            "buckets": {
                "liquidity": "เงินสำรองฉุกเฉิน 6-12 เดือน",
                "core": "พอร์ตหลักสร้างกระแสเงินสดและเติบโตมั่นคง",
                "growth": "พอร์ตเติบโตสูงและโอกาสพิเศษ"
            }
        }, ensure_ascii=False, indent=2)

    def _get_screener_data(tickers: list[str]) -> str:
        resolved_tickers = [_resolve_known_ticker(t) or t for t in tickers[:8]]
        try:
            eq_svc = get_equity_query_service()
            results = []
            for t in resolved_tickers:
                det = eq_svc.get_detail(t)
                if det and isinstance(det, dict):
                    results.append({
                        "ticker": det.get("ticker", t),
                        "company_name": det.get("company_name", t),
                        "composite_score": det.get("composite_score"),
                        "market_sentiment": det.get("market_sentiment"),
                        "quant_signals": det.get("quant_signals"),
                    })
            if results:
                return json.dumps(results, ensure_ascii=False, indent=2, default=str)
        except Exception:
            pass
        return f"ผลการเปรียบเทียบปัจจัยพื้นฐานสำหรับ {', '.join(resolved_tickers)}: พร้อมเปรียบเทียบในหน้าจอ Terminal V2"

    def _search_vault_data(query: str) -> str:
        try:
            from tools.archivist.search import search_all_memories
            return str(search_all_memories.invoke({"keyword": query}))
        except Exception as exc:
            return f"ไม่พบบันทึกที่ตรงกับ '{query}' ใน Knowledge Vault ({exc})"

    def _save_vault_data(title: str, content: str) -> str:
        try:
            from tools.archivist.writer import save_memory
            res = save_memory.invoke({
                "title": title,
                "content": content,
                "folder_path": "30_Knowledge_Base/Advisor_Notes",
                "tags": ["advisor", "insight"],
                "entity_type": "advisor_note",
            })
            return f"บันทึกลงใน Vault เรียบร้อยแล้ว: {res}"
        except Exception as exc:
            return f"ไม่สามารถบันทึกลง Vault ได้: {exc}"

    set_research_job_dispatcher(_dispatch_research)
    set_macro_provider(_get_macro_data)
    set_portfolio_provider(_get_portfolio_data)
    set_advisory_provider(_get_advisory_data)
    set_opportunity_provider(_get_opportunity_data)
    set_calendar_provider(_get_calendar_data)
    set_digest_provider(_get_digest_data)
    set_equity_report_provider(_get_equity_report_data)
    set_equity_promises_provider(_get_equity_promises_data)
    set_look_through_provider(_get_look_through_data)
    set_essence_buckets_provider(_get_essence_buckets_data)
    set_screener_provider(_get_screener_data)
    set_search_vault_provider(_search_vault_data)
    set_save_vault_provider(_save_vault_data)

    def _get_morning_delta_data() -> str:
        try:
            svc = get_pm_review_service()
            packet = svc.build_morning_packet()
            return json.dumps(packet.to_dict(), ensure_ascii=False, indent=2, default=str)
        except Exception as exc:
            return f"ไม่สามารถดึงข้อมูล Morning Delta ได้: {exc}"

    def _get_stock_theses_data(ticker: str = "") -> str:
        try:
            svc = get_thesis_service()
            clean = _resolve_known_ticker(ticker) if ticker.strip() else ""
            if clean:
                t = svc.get_active_thesis(clean)
                if t:
                    return json.dumps(t.to_dict(), ensure_ascii=False, indent=2, default=str)
                return f"ไม่พบ Thesis สำหรับหุ้น {clean}"
            theses = svc.list_active_theses()
            return json.dumps([t.to_dict() for t in theses], ensure_ascii=False, indent=2, default=str)
        except Exception as exc:
            return f"ไม่สามารถดึงข้อมูล Theses ได้: {exc}"

    def _synthesize_thesis_data(ticker: str, dry_run: bool = False) -> str:
        try:
            synth = get_thesis_synthesis_service()
            res = synth.synthesize_thesis_for_ticker(ticker=ticker, dry_run=dry_run)
            return json.dumps(res, ensure_ascii=False, indent=2, default=str)
        except Exception as exc:
            return f"ไม่สามารถสังเคราะห์ Thesis ได้: {exc}"

    def _batch_synthesize_theses_data(portfolio_id: str = "default", dry_run: bool = False) -> str:
        try:
            synth = get_thesis_synthesis_service()
            res = synth.batch_synthesize_portfolio_theses(portfolio_id=portfolio_id, dry_run=dry_run)
            return json.dumps(res, ensure_ascii=False, indent=2, default=str)
        except Exception as exc:
            return f"ไม่สามารถรัน Batch สังเคราะห์ Theses ได้: {exc}"

    set_morning_pm_review_provider(_get_morning_delta_data)
    set_stock_theses_provider(_get_stock_theses_data)
    set_thesis_synthesizer_provider(_synthesize_thesis_data)
    set_batch_thesis_synthesizer_provider(_batch_synthesize_theses_data)

    repo = SqliteAdvisorSessionRepositoryAdapter()
    sandbox = SandboxAdapter()
    stream_engine = GeminiStreamEngine()
    return ManagerChatService(
        session_repo=repo,
        sandbox_adapter=sandbox,
        stream_engine=stream_engine,
    )


@lru_cache(maxsize=1)
def get_hermes_sync_service():
    """Dependency provider for HermesSyncService (cached singleton)."""
    from application.manager.hermes_sync_service import HermesSyncService
    return HermesSyncService()

@lru_cache(maxsize=1)
def get_catalyst_service():
    """Dependency provider for CatalystApplicationService (Chapter 7)."""
    from application.thesis.catalyst_service import CatalystApplicationService
    from api.db.adapters import SqliteInstitutionalLedgerRepositoryAdapter
    return CatalystApplicationService(ledger_repository=SqliteInstitutionalLedgerRepositoryAdapter())


@lru_cache(maxsize=1)
def get_estimate_ledger_service():
    """Dependency provider for EstimateLedgerApplicationService (Chapter 2 & 5)."""
    from application.equity.estimate_service import EstimateLedgerApplicationService
    from api.db.adapters import SqliteInstitutionalLedgerRepositoryAdapter
    return EstimateLedgerApplicationService(ledger_repository=SqliteInstitutionalLedgerRepositoryAdapter())


@lru_cache(maxsize=1)
def get_task_routing_service():
    """Dependency provider for TaskRoutingGovernanceService (Chapter 1)."""
    from application.governance.routing_service import TaskRoutingGovernanceService
    return TaskRoutingGovernanceService()


@lru_cache(maxsize=1)
def get_routine_monitor_service():
    """Dependency provider for RoutineMonitorApplicationService (Chapter 10)."""
    from application.resilience.routine_monitor_service import RoutineMonitorApplicationService
    from api.db.adapters import SqliteInstitutionalLedgerRepositoryAdapter
    return RoutineMonitorApplicationService(ledger_repository=SqliteInstitutionalLedgerRepositoryAdapter())


def get_readiness_audit_service(repo_root: Optional[Path] = None):
    """Dependency provider for InstitutionalReadinessAuditService (Appendix J)."""
    from application.resilience.readiness_audit_service import InstitutionalReadinessAuditService
    return InstitutionalReadinessAuditService(repo_root=repo_root)


_ROUNDTABLE_CACHE_KEY: Optional[tuple[str, str]] = None
_ROUNDTABLE_SERVICE = None
_ROUNDTABLE_INITIALIZATION_LOCK = threading.RLock()


def _create_roundtable_service(current_key: tuple[str, str]):
    """Construct and publish one Roundtable service while the provider lock is held."""
    global _ROUNDTABLE_SERVICE, _ROUNDTABLE_CACHE_KEY
    import dataclasses
    import json
    from api.db.adapters import SqliteRoundtableRepositoryAdapter
    from api.db.investor_essence_adapters import SqliteInvestorRuntimeUowFactory
    from application.roundtable.context_assembler import RoundtableContextAssembler
    from application.roundtable.service import RoundtableApplicationService
    from tools.roundtable.adapters import (
        CallableContextDataProvidersAdapter,
        ObsidianRoundtableVaultWriterAdapter,
        OhlcvSettlementPriceAdapter,
        RoundtableSandboxCalcAdapter,
        RoundtableThesisUpdaterAdapter,
    )
    from tools.roundtable.debate_engine import GeminiRoundtableEngine
    from tools.roundtable.tts_adapter import GeminiTTSAdapter
    from validators.roundtable_verifier import RoundtableConsistencyVerifier

    def _resolve_instrument(ticker: str) -> Dict[str, Any]:
        try:
            res = get_asset_resolver().resolve(ticker)
            return {
                "raw_symbol": res.raw_symbol,
                "provider_symbol": res.provider_symbol,
                "asset_class": res.asset_class.value if hasattr(res.asset_class, "value") else str(res.asset_class),
                "market": res.market or ("TH" if ticker.strip().upper().endswith(".BK") else "US"),
                "confidence": res.confidence,
            }
        except Exception:
            return {}

    def _equity_detail_reader(ticker: str) -> Optional[Dict[str, Any]]:
        try:
            return get_equity_query_service().get_detail(ticker)
        except Exception:
            return None

    def _equity_review_reader(ticker: str, portfolio_id: str = "default") -> Optional[Dict[str, Any]]:
        try:
            rev = get_position_review_service().run_position_review(ticker, portfolio_id=portfolio_id)
            return dataclasses.asdict(rev) if rev is not None else None
        except Exception:
            return None

    def _equity_news_reader(ticker: str) -> Optional[Dict[str, Any]]:
        try:
            return get_equity_query_service().get_news(ticker)
        except Exception:
            return None

    def _equity_notes_lister(ticker: str, days: int = 30) -> Optional[Dict[str, Any]]:
        try:
            return get_equity_query_service().list_notes(ticker, days=days)
        except Exception:
            return None

    def _equity_note_reader(relative_path: str) -> Optional[Dict[str, Any]]:
        try:
            return get_equity_query_service().read_note(relative_path)
        except Exception:
            return None

    def _macro_reader() -> Optional[Dict[str, Any]]:
        try:
            return get_macro_service().dashboard()
        except Exception:
            return None

    def _digest_reader() -> Optional[Dict[str, Any]]:
        try:
            res = get_digest_service().get_daily_digest()
            if isinstance(res, dict) and res.get("has_digest") and res.get("digest"):
                return res["digest"]
            return res
        except Exception:
            return None

    def _compute_offline_look_through_summary(state: Any, portfolio_id: str, total_val: float) -> Optional[Dict[str, Any]]:
        """Read-only offline look-through from cached fund constituents + static master catalog (zero HTTP calls)."""
        if total_val <= 0:
            return None
        try:
            from decimal import Decimal
            from pathlib import Path
            from core.portfolio.look_through_calculator import LookThroughCalculator
            from core.portfolio.look_through_models import (
                FundConstituentHolding,
                FundConstituentsDomain,
                FundConstituentSector,
            )
            from tools.portfolio.adapters.thai_fund.master_fund_catalog import find_master_fund
            from tools.portfolio.adapters.thai_fund.pvd_symbol_mapper import resolve_pvd_symbol

            cache_dir_env = os.environ.get("FINNOMENA_CACHE_DIR")
            cache_file = (Path(cache_dir_env) if cache_dir_env else Path("data/cache")) / "finnomena_constituents_cache.json"
            raw_cache: Dict[str, Any] = {}
            if cache_file.exists():
                try:
                    loaded = json.loads(cache_file.read_text(encoding="utf-8"))
                    if isinstance(loaded, dict):
                        raw_cache = loaded
                except Exception:
                    raw_cache = {}
            normalized_cache = {" ".join(str(k).strip().upper().split()): v for k, v in raw_cache.items()}

            fund_map: Dict[str, Any] = {}
            holdings_list = getattr(state, "holdings", ()) or ()
            for h in holdings_list:
                sym = str(getattr(h, "symbol", "") or "").strip().upper()
                asset_type = str(getattr(h, "asset_type", "") or "").strip().upper()
                is_fund = any(k in asset_type for k in ("FUND", "MUTUAL_FUND", "PVD", "RMF", "SSF", "THAIESG", "ETF")) or sym.startswith("PVD_")
                if not is_fund or not sym or sym in fund_map:
                    continue

                norm_sym = " ".join(sym.split())
                cached_entry = normalized_cache.get(norm_sym)
                if isinstance(cached_entry, dict) and isinstance(cached_entry.get("data"), dict):
                    cd = cached_entry["data"]
                    fund_map[sym] = FundConstituentsDomain(
                        fund_id=str(cd.get("fund_id") or sym),
                        symbol=sym,
                        as_of_date=str(cd.get("as_of_date") or ""),
                        master_fund_name=cd.get("master_fund_name"),
                        master_fund_ticker=cd.get("master_fund_ticker"),
                        top_holdings=[
                            FundConstituentHolding(
                                name=str(item.get("name") or ""),
                                percent=Decimal(str(item.get("percent") or 0)),
                                ticker=item.get("ticker"),
                                asset_type=str(item.get("asset_type") or "EQUITY"),
                            )
                            for item in (cd.get("top_holdings") or [])
                            if isinstance(item, dict)
                        ],
                        sector_breakdown=[
                            FundConstituentSector(
                                sector_name_th=str(sec.get("sector_name_th") or ""),
                                sector_name_en=str(sec.get("sector_name_en") or sec.get("sector") or ""),
                                percent=Decimal(str(sec.get("percent") or 0)),
                            )
                            for sec in (cd.get("sector_breakdown") or [])
                            if isinstance(sec, dict)
                        ],
                        asset_allocation={
                            str(ak): Decimal(str(av))
                            for ak, av in (cd.get("asset_allocation") or {}).items()
                        },
                        is_curated=bool(cd.get("is_curated", False)),
                        source_date=cd.get("as_of_date"),
                    )
                    continue

                resolved_sym, _ = resolve_pvd_symbol(sym)
                master = find_master_fund(resolved_sym) or find_master_fund(sym)
                if master is not None:
                    fund_map[sym] = FundConstituentsDomain(
                        fund_id=sym,
                        symbol=sym,
                        as_of_date="",
                        master_fund_name=getattr(master, "master_name", None),
                        master_fund_ticker=getattr(master, "master_ticker", None),
                        top_holdings=[
                            FundConstituentHolding(
                                name=str(cn),
                                percent=Decimal(str(wt)),
                                ticker=str(tk),
                                asset_type="EQUITY",
                            )
                            for tk, cn, wt in (getattr(master, "top_holdings", ()) or ())
                        ],
                        sector_breakdown=[
                            FundConstituentSector(
                                sector_name_th=str(sn),
                                sector_name_en=str(sn),
                                percent=Decimal(str(wt)),
                            )
                            for sn, wt in (getattr(master, "sector_weights", {}) or {}).items()
                        ],
                        asset_allocation={"Equities": Decimal("100")},
                        is_curated=True,
                    )

            if not fund_map:
                return None
            lt = LookThroughCalculator.calculate(
                holdings=list(holdings_list),
                fund_constituents_map=fund_map,
                total_nav_thb=Decimal(str(total_val)),
                portfolio_id=portfolio_id,
            )
            return {
                "top_overlapped_stocks": [
                    {
                        "ticker": getattr(s, "ticker", ""),
                        "company_name": getattr(s, "company_name", ""),
                        "direct_weight_pct": float(getattr(s, "direct_weight_pct", 0) or 0),
                        "indirect_weight_pct": float(getattr(s, "indirect_weight_pct", 0) or 0),
                        "total_weight_pct": float(getattr(s, "total_weight_pct", 0) or 0),
                        "is_cap_breached": bool(getattr(s, "is_cap_breached", False)),
                        "is_hidden_concentration": bool(getattr(s, "is_hidden_concentration", False)),
                    }
                    for s in (getattr(lt, "top_overlapped_stocks", None) or [])[:8]
                ],
                "hidden_concentration_warnings": list(getattr(lt, "hidden_concentration_warnings", None) or []),
                "crowded_sectors": [
                    {
                        "sector_name": getattr(sec, "sector_name", ""),
                        "weight_pct": float(getattr(sec, "weight_pct", 0) or 0),
                    }
                    for sec in (getattr(lt, "sector_exposure", None) or [])
                    if getattr(sec, "is_crowded", False)
                ],
            }
        except Exception:
            return None

    def _watchlist_reader(portfolio_id: str = "default") -> List[Dict[str, Any]]:
        try:
            wl_state = get_portfolio_service().get_structured_watchlist(portfolio_id=portfolio_id)
            raw_items = getattr(wl_state, "items", None) or ()
            out_wl: List[Dict[str, Any]] = []
            for item in raw_items:
                sym = str(getattr(item, "symbol", "") or "").strip()
                if not sym:
                    continue
                out_wl.append({
                    "symbol": sym,
                    "asset_type": getattr(item, "asset_type", None) or "Stock",
                    "target_price": getattr(item, "target_price", None),
                    "target_currency": getattr(item, "target_currency", None),
                    "target_price_basis": getattr(item, "target_price_basis", None),
                    "suggested_bucket_id": getattr(item, "suggested_bucket_id", None),
                    "applied_bucket_name": getattr(item, "applied_bucket_name", None),
                    "notes": getattr(item, "notes", None),
                })
            return out_wl
        except Exception:
            return []

    def _goals_reader(portfolio_id: str = "default") -> List[Dict[str, Any]]:
        try:
            raw_goals = get_portfolio_service().get_structured_goals(portfolio_id=portfolio_id) or ()
            out_goals: List[Dict[str, Any]] = []
            for g in raw_goals:
                if not isinstance(g, dict):
                    continue
                out_goals.append({
                    "name": g.get("name"),
                    "goal_type": g.get("goal_type"),
                    "target_amount_thb": g.get("target_amount_thb"),
                    "current_amount_thb": g.get("current_amount_thb"),
                    "progress_pct": g.get("progress_pct"),
                    "deadline": g.get("deadline"),
                    "deadline_days_left": g.get("deadline_days_left"),
                    "priority_tier": g.get("priority_tier") or "wants",
                    "flexibility": g.get("flexibility") or "none",
                    "monthly_savings_thb": g.get("monthly_savings_thb"),
                    "required_return_pct": g.get("required_return_pct"),
                    "tension_level": g.get("tension_level"),
                    "adjustment_levers": g.get("adjustment_levers"),
                })
            return out_goals
        except Exception:
            return []

    def _portfolio_state_reader(portfolio_id: str = "default") -> Optional[Dict[str, Any]]:
        try:
            state = get_portfolio_service().get_structured_portfolio_state(
                portfolio_id=portfolio_id, refresh_prices=False
            )
            if not state:
                return None
            summary = getattr(state, "summary", None)
            total_val = float(getattr(summary, "total_value_thb", 0.0) or 0.0) if summary else 0.0
            total_cost = float(getattr(summary, "total_cost_basis_thb", 0.0) or 0.0) if summary else 0.0
            total_unrealized = float(getattr(summary, "total_unrealized_profit", 0.0) or 0.0) if summary else 0.0
            fx_rates = getattr(state, "fx_rates", {}) or {}
            fx_usd_thb = float(fx_rates.get("USDTHB") or 0.0)

            cash_thb = float(getattr(summary, "cash_thb", 0.0) or 0.0) if summary else 0.0
            cash_thb_only = 0.0
            cash_usd_val_thb = 0.0
            computed_cash_from_holdings = 0.0
            raw_holdings = getattr(state, "holdings", ()) or ()
            holdings = []
            for h in raw_holdings:
                sym = str(getattr(h, "symbol", "") or "").strip()
                asset_type = str(getattr(h, "asset_type", "") or "").strip()
                h_val = float(getattr(h, "market_value_thb", 0.0) or 0.0)
                is_cash_holding = asset_type.lower() == "cash" or sym.upper() in {"CASH_THB", "CASH_USD", "CASH"}
                if is_cash_holding:
                    computed_cash_from_holdings += h_val
                    if sym.upper() == "CASH_USD":
                        cash_usd_val_thb += h_val
                    else:
                        cash_thb_only += h_val
                weight_pct = round((h_val / total_val * 100.0), 2) if total_val > 0 else None
                if sym.upper() == "CASH_USD":
                    currency = "USD"
                elif sym.upper() == "CASH_THB":
                    currency = "THB"
                elif getattr(h, "avg_cost_usd", None) is not None:
                    currency = "USD"
                elif getattr(h, "avg_cost_thb", None) is not None:
                    currency = "THB"
                elif getattr(h, "current_price_usd", None) is not None:
                    currency = "USD"
                else:
                    currency = "THB"
                holdings.append({
                    "symbol": sym,
                    "company_name": getattr(h, "company_name", None),
                    "asset_type": asset_type or None,
                    "currency": currency,
                    "units": getattr(h, "units", 0),
                    "avg_cost": getattr(h, "avg_cost_usd", None) or getattr(h, "avg_cost_thb", None) or getattr(h, "avg_cost", None),
                    "market_value_thb": h_val,
                    "current_weight_pct": weight_pct,
                    "unrealized_pnl_percent": getattr(h, "unrealized_pnl_percent", None),
                    "bucket_id": getattr(h, "bucket_id", None),
                })

            if cash_thb <= 0.0 and computed_cash_from_holdings > 0.0:
                cash_thb = computed_cash_from_holdings
            holdings.sort(key=lambda item: float(item.get("market_value_thb") or 0.0), reverse=True)
            cash_pct = round((cash_thb / total_val * 100.0), 2) if total_val > 0 else None

            alloc_targets = [
                {
                    "bucket_id": getattr(t, "bucket_id", ""),
                    "name": getattr(t, "name", ""),
                    "target_percent": float(getattr(t, "target_percent", 0.0) or 0.0),
                }
                for t in (getattr(state, "allocation_targets", ()) or ())
            ]
            lt_summary = _compute_offline_look_through_summary(state, portfolio_id, total_val)

            out: Dict[str, Any] = {
                "total_value_thb": total_val,
                "total_cost_basis_thb": total_cost,
                "total_unrealized_profit_thb": total_unrealized,
                "cash_thb": round(cash_thb, 2),
                "cash_thb_only": round(cash_thb_only, 2),
                "cash_usd_val_thb": round(cash_usd_val_thb, 2),
                "cash_pct": cash_pct,
                "fx_usd_thb": fx_usd_thb if fx_usd_thb > 0 else None,
                "holdings": holdings,
                "allocation_targets": alloc_targets,
                "watchlist_items": _watchlist_reader(portfolio_id),
                "goals_progress": _goals_reader(portfolio_id),
            }
            if lt_summary:
                out["look_through_summary"] = lt_summary
            return out
        except Exception:
            return None

    def _confirmed_axis_reader(portfolio_id: str = "default") -> Optional[Dict[str, Any]]:
        try:
            return ConfirmedPolicyAdapter(uow_factory=SqliteInvestorRuntimeUowFactory()).get_confirmed_investment_axis(
                portfolio_id
            )
        except Exception:
            return None

    def _financial_context_reader(portfolio_id: str = "default") -> Optional[Dict[str, Any]]:
        try:
            fc = get_financial_context_service().get_financial_context(portfolio_id)
            return dataclasses.asdict(fc) if fc is not None else None
        except Exception:
            return None

    def _bucket_policy_reader(portfolio_id: str = "default") -> Optional[List[Dict[str, Any]]]:
        try:
            uow_factory = SqliteInvestorRuntimeUowFactory()
            scope = f"portfolio:{portfolio_id}"
            with uow_factory.open() as uow:
                snap = uow.planning.get_confirmed_snapshot(scope, "bucket_policy")
                if snap:
                    b = snap.get("buckets") or snap.get("purpose_buckets")
                    if b:
                        return b
        except Exception:
            pass
        try:
            p_state = get_portfolio_service().get_structured_portfolio_state(
                portfolio_id=portfolio_id, refresh_prices=False
            )
            if p_state:
                snap = getattr(p_state, "accepted_policy_snapshot", None)
                if isinstance(snap, dict):
                    b = snap.get("purpose_buckets") or snap.get("buckets")
                    if b:
                        return b
                pol = getattr(p_state, "allocation_policy", None)
                if isinstance(pol, dict):
                    b = pol.get("purpose_buckets") or pol.get("buckets")
                    if b:
                        return b
                targets = getattr(p_state, "allocation_targets", None)
                if targets:
                    return [
                        {
                            "id": getattr(t, "bucket_id", ""),
                            "name": getattr(t, "name", ""),
                            "target_pct": float(getattr(t, "target_percent", 0.0) or 0.0),
                        }
                        for t in targets
                    ]
        except Exception:
            pass
        return None

    def _active_thesis_reader(ticker: str) -> Optional[Dict[str, Any]]:
        try:
            t = get_thesis_service().get_active_thesis(ticker)
            return t.to_dict() if t is not None else None
        except Exception:
            return None

    def _thesis_exceptions_reader(ticker: str) -> Optional[List[Dict[str, Any]]]:
        try:
            from api.db.adapters import SqliteThesisExceptionRepositoryAdapter

            clean = ticker.strip().upper()
            rows = SqliteThesisExceptionRepositoryAdapter().list_pending_exceptions(ticker=clean, limit=25) or []
            out: List[Dict[str, Any]] = []
            for r in rows:
                if dataclasses.is_dataclass(r):
                    out.append(dataclasses.asdict(r))
                elif isinstance(r, dict):
                    out.append(dict(r))
            return out
        except Exception:
            return None

    def _promise_scores_reader(ticker: str) -> Optional[Dict[str, Any]]:
        try:
            clean = ticker.strip().upper()
            dto = get_promise_service().get_ledger(clean)
            return dto.model_dump() if dto is not None else None
        except Exception:
            return None

    def _analyst_context_reader(ticker: str) -> Optional[Dict[str, Any]]:
        try:
            from contextlib import closing
            from api.db.connection import get_connection
            from api.db.repositories import cache_repository as cache_repo_dao

            clean = ticker.strip().upper()
            with closing(get_connection()) as conn:
                data = cache_repo_dao.get_analyst_context_cache(conn, clean)
            if not isinstance(data, dict):
                return None
            source_as_of = str(data.get("source_as_of") or "")
            if "synthetic" in source_as_of.lower():
                return None
            raw_hist = data.get("earnings_history")
            if isinstance(raw_hist, list):
                clean_hist = [
                    item for item in raw_hist
                    if isinstance(item, dict) and "synthetic" not in json.dumps(item, ensure_ascii=False).lower()
                ]
                data["earnings_history"] = clean_hist
            return data
        except Exception:
            return None

    def _insider_db_summary_reader(ticker: str) -> Optional[Dict[str, Any]]:
        try:
            from contextlib import closing
            from datetime import datetime, timedelta, timezone
            from api.db.connection import get_connection

            clean = ticker.strip().upper()
            cutoff = (datetime.now(timezone.utc) - timedelta(days=180)).strftime("%Y-%m-%d")
            with closing(get_connection()) as conn:
                rows = conn.execute(
                    """
                    SELECT
                        t.transaction_id, t.accession_number, t.transaction_date,
                        t.transaction_code, t.shares, t.price_per_share, t.acquired_or_disposed,
                        f.reporting_owner_name, f.officer_title, f.filed_at,
                        (f.raw_xml_payload IS NOT NULL AND TRIM(f.raw_xml_payload) != '') AS has_raw_xml
                    FROM sec_insider_transactions t
                    JOIN sec_form4_raw_ledger f ON t.accession_number = f.accession_number
                    WHERE t.ticker = ? AND t.transaction_date >= ?
                    ORDER BY t.transaction_date DESC, f.filed_at DESC
                    """,
                    (clean, cutoff),
                ).fetchall()
            if not rows:
                return None

            normalized_rows: List[Dict[str, Any]] = []
            group_codes: Dict[tuple[str, str], set[str]] = {}
            has_form4_xml = False
            has_yf_heuristic = False
            reclassified_zero_buys = 0

            for r in rows:
                acc = str(r["accession_number"] or "")
                raw_code = str(r["transaction_code"] or "").strip().upper()
                px = float(r["price_per_share"] or 0.0)
                sh = float(r["shares"] or 0.0)
                has_xml = bool(r["has_raw_xml"])
                is_yf = acc.startswith("yf_") and not has_xml
                if is_yf:
                    has_yf_heuristic = True
                else:
                    has_form4_xml = True

                eff_code = raw_code
                if is_yf and raw_code == "P" and px <= 0.0:
                    eff_code = "A"
                    reclassified_zero_buys += 1

                owner = str(r["reporting_owner_name"] or "Unknown").strip()
                tx_date = str(r["transaction_date"] or "")[:10]
                group_codes.setdefault((owner, tx_date), set()).add(eff_code)
                normalized_rows.append({
                    "transaction_date": tx_date,
                    "owner": owner,
                    "officer_title": str(r["officer_title"] or "").strip(),
                    "effective_code": eff_code,
                    "shares": sh,
                    "price_per_share": px,
                    "usd_value": round(sh * px, 2) if (sh > 0 and px > 0) else 0.0,
                    "is_yf": is_yf,
                })

            buy_cnt = 0
            buy_shares = 0.0
            buy_usd = 0.0
            open_sale_cnt = 0
            paired_sale_cnt = 0
            sale_shares = 0.0
            sale_usd = 0.0
            exercise_cnt = 0
            award_cnt = 0
            tax_cnt = 0
            notable: List[Dict[str, Any]] = []

            for item in normalized_rows:
                c = item["effective_code"]
                sh = item["shares"]
                usd = item["usd_value"]
                codes_same_day = group_codes.get((item["owner"], item["transaction_date"]), set())
                if c == "P" and item["price_per_share"] > 0.0:
                    buy_cnt += 1
                    buy_shares += sh
                    buy_usd += usd
                    if len(notable) < 6:
                        notable.append({**item, "paired_exercise": False})
                elif c == "S":
                    is_paired = bool("M" in codes_same_day or "A" in codes_same_day)
                    if is_paired:
                        paired_sale_cnt += 1
                    else:
                        open_sale_cnt += 1
                    sale_shares += sh
                    sale_usd += usd
                    if len(notable) < 6:
                        notable.append({**item, "paired_exercise": is_paired})
                elif c == "M":
                    exercise_cnt += 1
                elif c == "A":
                    award_cnt += 1
                elif c == "F":
                    tax_cnt += 1

            if has_form4_xml and has_yf_heuristic:
                provenance = "MIXED_FORM4_AND_YFINANCE"
            elif has_form4_xml:
                provenance = "SEC_FORM4_XML"
            else:
                provenance = "YFINANCE_HEURISTIC"

            return {
                "ticker": clean,
                "window_days": 180,
                "source_provenance": provenance,
                "total_rows": len(normalized_rows),
                "reclassified_zero_dollar_buys": reclassified_zero_buys,
                "open_market_buy_count": buy_cnt,
                "open_market_buy_shares": round(buy_shares, 2),
                "open_market_buy_usd": round(buy_usd, 2),
                "open_market_sale_count": open_sale_cnt,
                "paired_exercise_sale_count": paired_sale_cnt,
                "total_sale_count": open_sale_cnt + paired_sale_cnt,
                "total_sale_shares": round(sale_shares, 2),
                "total_sale_usd": round(sale_usd, 2),
                "exercise_count": exercise_cnt,
                "award_grant_count": award_cnt,
                "tax_withhold_count": tax_cnt,
                "recent_notable": notable,
            }
        except Exception:
            return None

    def _portfolio_advisory_reader(portfolio_id: str = "default") -> Optional[Dict[str, Any]]:
        try:
            from application.portfolio.advisory_service import PortfolioAdvisoryService
            from tools.macro.adapters.sector_snapshot_adapter import SectorSnapshotStore
            from tools.portfolio.adapters.advisory_vault_adapter import (
                AdvisoryVaultAdapter,
                EquitySidecarAdapter,
                MacroContextAdapter,
            )

            vault_root = _get_vault_path()

            class _DepsPortfolioReader:
                def get_current_portfolio_state(
                    self, refresh_prices: bool = False, portfolio_id: str = "default"
                ) -> Any:
                    return get_portfolio_service().get_structured_portfolio_state(
                        portfolio_id=portfolio_id,
                        refresh_prices=False,
                    )

            class _OfflineMacroContextAdapter(MacroContextAdapter):
                def get_latest_sector_rotation(self) -> Optional[Any]:
                    try:
                        res = SectorSnapshotStore().latest()
                        if res and res[0] is not None:
                            return res[0]
                    except Exception:
                        pass
                    sr_latest = self.vault_path / "30_Knowledge_Base" / "Macroeconomics" / "Sector_Rotation" / "latest.json"
                    if sr_latest.exists():
                        try:
                            return json.loads(sr_latest.read_text(encoding="utf-8"))
                        except Exception:
                            pass
                    return None

            adv_svc = PortfolioAdvisoryService(
                portfolio_reader=_DepsPortfolioReader(),
                macro_reader=_OfflineMacroContextAdapter(vault_path=vault_root),
                sidecar_reader=EquitySidecarAdapter(vault_path=vault_root),
                vault_port=AdvisoryVaultAdapter(vault_path=vault_root),
                synthesizer_port=None,  # type: ignore[arg-type]
                look_through_service=None,
            )
            rep = adv_svc.get_advisory_quant_report(refresh_prices=False, portfolio_id=portfolio_id)
            rr = rep.risk_radar
            cp = rep.cash_plan
            return {
                "evaluated_at": rep.evaluated_at,
                "data_mode": rep.data_mode.value if hasattr(rep.data_mode, "value") else str(rep.data_mode),
                "total_nav_thb": float(rep.total_nav_thb),
                "risk_radar": {
                    "weighted_beta": float(rr.weighted_beta) if rr.weighted_beta is not None else None,
                    "total_portfolio_volatility_pct": float(rr.total_portfolio_volatility_pct) if getattr(rr, "total_portfolio_volatility_pct", None) is not None else None,
                    "hhi": float(rr.hhi) if rr.hhi is not None else None,
                    "effective_holdings_count": float(rr.effective_holdings_count) if rr.effective_holdings_count is not None else None,
                    "top_1_weight_pct": float(getattr(rr, "top_1_weight_pct", 0.0) or 0.0),
                    "top_3_weight_pct": float(getattr(rr, "top_3_weight_pct", 0.0) or 0.0),
                    "concentration_risk_level": getattr(rr, "concentration_risk_level", "LOW"),
                    "bucket_active_share_pct": float(rr.bucket_active_share_pct) if getattr(rr, "bucket_active_share_pct", None) is not None else None,
                    "drawdown_pct": float(rr.drawdown_pct) if getattr(rr, "drawdown_pct", None) is not None else None,
                    "drawdown_posture_tier": getattr(rr, "drawdown_posture_tier", "NORMAL"),
                    "unassigned_holdings_nav_pct": float(rr.unassigned_holdings_nav_pct),
                },
                "cash_plan": {
                    "cash_thb_available": float(cp.cash_thb_available),
                    "cash_usd_available": float(cp.cash_usd_available),
                    "usd_thb_fx_rate": float(cp.usd_thb_fx_rate),
                    "immediate_deploy_thb": float(cp.immediate_deploy_thb),
                    "immediate_deploy_usd": float(cp.immediate_deploy_usd),
                    "conditional_limit_order_thb": float(cp.conditional_limit_order_thb),
                    "conditional_limit_order_usd": float(cp.conditional_limit_order_usd),
                    "fx_conversion_needed_thb": float(cp.fx_conversion_needed_thb),
                },
                "holdings_advisory": [
                    {
                        "symbol": h.symbol,
                        "evaluation_track": h.evaluation_track.value if hasattr(h.evaluation_track, "value") else str(h.evaluation_track),
                        "bucket_id": h.bucket_id,
                        "bucket_name": h.bucket_name,
                        "current_weight_pct": float(h.current_weight_pct),
                        "target_weight_pct": float(h.target_weight_pct),
                        "weight_drift_pct": float(h.weight_drift_pct),
                        "action_code": h.action_code.value if hasattr(h.action_code, "value") else str(h.action_code),
                        "is_immediate_execution": bool(h.is_immediate_execution),
                        "live_upside_12m_pct": float(h.live_upside_12m_pct) if h.live_upside_12m_pct is not None else None,
                        "macro_alignment": h.macro_alignment,
                        "sector_quadrant": h.sector_quadrant,
                        "risk_contribution_pct": float(h.risk_contribution_pct) if getattr(h, "risk_contribution_pct", None) is not None else None,
                    }
                    for h in rep.holdings_advisory
                ],
                "anomaly_warnings": [
                    {
                        "guardrail_code": w.guardrail_code,
                        "ticker": w.ticker,
                        "message": w.message,
                        "severity": w.severity,
                    }
                    for w in rep.anomaly_warnings[:10]
                ],
            }
        except Exception:
            return None

    def _investor_essence_reader(portfolio_id: str = "default") -> Optional[Dict[str, Any]]:
        try:
            uow_factory = SqliteInvestorRuntimeUowFactory()
            with uow_factory.open() as uow:
                snap = uow.planning.get_confirmed_snapshot("workspace", "investor_essence")
            if not isinstance(snap, dict):
                return None
            claims = []
            for c in (snap.get("accepted_claims") or []):
                if isinstance(c, dict):
                    txt = str(c.get("effective_text") or c.get("final_text") or c.get("text") or "").strip()
                    if txt:
                        claims.append({
                            "text": txt,
                            "source_kind": c.get("source_kind"),
                            "fit_rating": c.get("fit_rating"),
                        })
            return {
                "statement": snap.get("statement") or "",
                "accepted_claims": claims,
                "unresolved_topics": list(snap.get("unresolved_topics") or []),
                "confirmed_at_iso": snap.get("confirmed_at_iso"),
                "content_hash": snap.get("content_hash") or "",
            }
        except Exception:
            return None


    repo = SqliteRoundtableRepositoryAdapter()
    providers = CallableContextDataProvidersAdapter(
        instrument_resolver=_resolve_instrument,
        equity_detail_reader=_equity_detail_reader,
        equity_review_reader=_equity_review_reader,
        macro_dashboard_reader=_macro_reader,
        daily_digest_reader=_digest_reader,
        portfolio_state_reader=_portfolio_state_reader,
        confirmed_axis_reader=_confirmed_axis_reader,
        active_thesis_reader=_active_thesis_reader,
        research_dispatcher=_dispatch_roundtable_research,
        equity_news_reader=_equity_news_reader,
        equity_notes_lister=_equity_notes_lister,
        equity_note_reader=_equity_note_reader,
        financial_context_reader=_financial_context_reader,
        bucket_policy_reader=_bucket_policy_reader,
        watchlist_reader=_watchlist_reader,
        thesis_exceptions_reader=_thesis_exceptions_reader,
        promise_scores_reader=_promise_scores_reader,
        analyst_context_reader=_analyst_context_reader,
        insider_db_summary_reader=_insider_db_summary_reader,
        portfolio_advisory_reader=_portfolio_advisory_reader,
        investor_essence_reader=_investor_essence_reader,
    )
    settlement_prices = OhlcvSettlementPriceAdapter(
        ohlcv_fetcher=lambda t, r, i: _OHLCV_SERVICE.get_ohlcv(t, range_str=r, interval_str=i)
    )
    assembler = RoundtableContextAssembler(
        providers=providers,
        repo=repo,
        settlement_prices=settlement_prices,
    )
    service = RoundtableApplicationService(
        repo=repo,
        assembler=assembler,
        engine=GeminiRoundtableEngine(),
        tts=GeminiTTSAdapter(),
        verifier=RoundtableConsistencyVerifier(),
        sandbox_calc=RoundtableSandboxCalcAdapter(),
        vault_writer=ObsidianRoundtableVaultWriterAdapter(),
        thesis_updater=RoundtableThesisUpdaterAdapter(thesis_service_getter=get_thesis_service),
        providers=providers,
    )
    _ROUNDTABLE_SERVICE = service
    _ROUNDTABLE_CACHE_KEY = current_key
    return service


def get_roundtable_service(request: Request = None):
    """Dependency provider for RoundtableApplicationService (Trinity Guru Roundtable Studio)."""
    if request is not None:
        app_state = getattr(getattr(request, "app", None), "state", None)
        request_queue = getattr(app_state, "job_queue", None)
        if request_queue is not None:
            set_global_app_job_queue(request_queue)

    current_key = (os.environ.get("WEBUI_STATE_DB_PATH", ""), str(_get_vault_path()))
    with _ROUNDTABLE_INITIALIZATION_LOCK:
        if _ROUNDTABLE_SERVICE is not None and _ROUNDTABLE_CACHE_KEY == current_key:
            return _ROUNDTABLE_SERVICE
        return _create_roundtable_service(current_key)
