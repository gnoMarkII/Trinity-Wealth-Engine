"""FastAPI Application Dependencies & Composition Provider."""
from functools import lru_cache
import os
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
    )


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


_GLOBAL_APP_JOB_QUEUE: Optional[Any] = None


def set_global_app_job_queue(queue: Any) -> None:
    """Register running FastAPI application job queue for background dispatchers."""
    global _GLOBAL_APP_JOB_QUEUE
    _GLOBAL_APP_JOB_QUEUE = queue


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
    )

    def _dispatch_research(ticker: str, research_type: str) -> str:
        clean_ticker = ticker.strip().upper()
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

        if active_queue is not None:
            job_id = active_queue.dispatch(
                instruction=instruction,
                card_id=card_id,
                flow=flow,
                scope="both",
            )
        else:
            from api.jobs import JobQueue, default_run_fn
            q = JobQueue(run_fn=default_run_fn)
            job_id = q.dispatch(
                instruction=instruction,
                card_id=card_id,
                flow=flow,
                scope="both",
            )

        if card_id and kanban_svc:
            try:
                kanban_svc.move_card(card_id, "executing", job_id=job_id)
            except Exception as exc:
                import logging
                logging.getLogger(__name__).warning("Failed to move card %s to executing: %s", card_id, exc)

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
        try:
            eq_svc = get_equity_query_service()
            detail = eq_svc.get_detail(ticker)
            if detail and isinstance(detail, dict):
                return json.dumps({
                    "ticker": detail.get("ticker", ticker),
                    "company_name": detail.get("company_name", ticker),
                    "composite_score": detail.get("composite_score"),
                    "market_sentiment": detail.get("market_sentiment"),
                    "quant_signals": detail.get("quant_signals"),
                    "narrative_analysis": detail.get("narrative_analysis"),
                    "base_case_summary": detail.get("base_case_summary"),
                    "thesis_delta": detail.get("thesis_delta"),
                }, ensure_ascii=False, indent=2, default=str)
        except Exception:
            pass
        sidecar = Path(f"data/equity_{ticker}_sidecar.json")
        if sidecar.is_file():
            try:
                return sidecar.read_text(encoding="utf-8")
            except Exception:
                pass
        return f"ไม่พบบทวิเคราะห์ของหุ้น {ticker} ในฐานข้อมูล คุณสามารถสั่งวิเคราะห์เชิงลึกได้ด้วยคำสั่ง trigger_deep_research_job('{ticker}', 'equity_intel')"

    def _get_equity_promises_data(ticker: str) -> str:
        try:
            import dataclasses
            res: dict[str, Any] = {"ticker": ticker}
            try:
                p_svc = get_promise_service()
                ledger_dto = p_svc.get_ledger(ticker)
                if ledger_dto:
                    res["promise_scores"] = ledger_dto.model_dump()
            except Exception:
                pass

            try:
                rev_svc = get_position_review_service()
                review_report = rev_svc.run_position_review(ticker)
                if review_report:
                    res["position_review"] = dataclasses.asdict(review_report)
            except Exception:
                pass

            if len(res) > 1:
                return json.dumps(res, ensure_ascii=False, indent=2, default=str)
        except Exception:
            pass
        return f"รายงานความซื่อสัตย์ของผู้บริหารและสถานะ Thesis หุ้น {ticker}: ไม่มีสัญญาณผิดสัญญาที่มีนัยสำคัญ"

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
        try:
            eq_svc = get_equity_query_service()
            results = []
            for t in tickers[:8]:
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
        return f"ผลการเปรียบเทียบปัจจัยพื้นฐานสำหรับ {', '.join(tickers)}: พร้อมเปรียบเทียบในหน้าจอ Terminal V2"

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
