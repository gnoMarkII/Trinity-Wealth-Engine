"""PortfolioStateService — CRUD Portfolio, Bucket Allocation, Reset."""
import json
from typing import Optional, List, Dict, Literal, Tuple

from tools.portfolio.domain.constants import _FLOAT_EPS
from tools.portfolio.domain.models import (
    PortfolioState,
    PortfolioMeta,
    AllocationTarget,
    _now_iso,
    default_allocation_targets,
)
from tools.portfolio.domain.ledger_change import LedgerChange
from tools.portfolio.domain.calculations import (
    compute_target_allocation_variance,
    compute_allocation_breakdown,
    recalc_all,
)
from tools.portfolio.domain.validator import validate_portfolio_id
from tools.portfolio.ports.repository_port import PortfolioRepositoryPort
from tools.portfolio.ports.price_port import MarketPricePort


def _find_holding(state: PortfolioState, symbol: str):
    return next((h for h in state.holdings if h.symbol == symbol), None)


class PortfolioStateService:
    """Handles Portfolio lifecycle, bucket allocation, and state queries."""

    def __init__(
        self,
        repo: PortfolioRepositoryPort,
        price_provider: Optional[MarketPricePort] = None,
    ) -> None:
        self.repo = repo
        self.price_provider = price_provider

    # ------------------------------------------------------------------
    # Portfolio CRUD
    # ------------------------------------------------------------------

    def list_portfolios(self) -> List[PortfolioMeta]:
        return self.repo.list_portfolios()

    def create_portfolio(self, name: str, portfolio_id: Optional[str] = None) -> PortfolioMeta:
        return self.repo.create_portfolio(name=name, portfolio_id=portfolio_id)

    def delete_portfolio(self, portfolio_id: str) -> None:
        self.repo.delete_portfolio(portfolio_id=portfolio_id)

    def update_portfolio_name(self, portfolio_id: str, name: str) -> PortfolioMeta:
        return self.repo.rename_portfolio(portfolio_id=portfolio_id, new_name=name)

    # ------------------------------------------------------------------
    # State Queries
    # ------------------------------------------------------------------

    def get_structured_portfolio_state(
        self, refresh_prices: bool = False, fetch_fundamentals: bool = False, portfolio_id: str = "default"
    ) -> PortfolioState:
        pid = validate_portfolio_id(portfolio_id)
        if refresh_prices or fetch_fundamentals:
            with self.repo.unit_of_work(pid) as uow:
                state = uow.load_state()
                if refresh_prices and self.price_provider:
                    self.price_provider.refresh_portfolio_prices(state)
                if fetch_fundamentals and self.price_provider:
                    self.price_provider.fetch_fundamentals(state, force=False)
                recalc_all(state)
                uow.commit(state, LedgerChange(kind="unchanged"))
                return state
        return self.repo.load_state(pid)

    def get_portfolio_state(self, refresh_prices: bool = True, portfolio_id: str = "default") -> str:
        from filelock import Timeout
        pid = validate_portfolio_id(portfolio_id)
        refresh_info: Dict[str, str] = {}
        try:
            with self.repo.unit_of_work(pid) as uow:
                state = uow.load_state()
                if refresh_prices and self.price_provider:
                    refresh_info = self.price_provider.refresh_portfolio_prices(state)
                    uow.commit(state, LedgerChange(kind="unchanged"))
                else:
                    recalc_all(state)
        except Timeout:
            return json.dumps({"error": f"portfolio lock timeout for '{portfolio_id}'"}, ensure_ascii=False)

        dump = state.model_dump(exclude_none=True)
        if refresh_info or getattr(state, "price_refresh_info", None):
            dump["_price_refresh"] = refresh_info or getattr(state, "price_refresh_info", {})
        return json.dumps(dump, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # Bucket Allocation
    # ------------------------------------------------------------------

    def get_structured_bucket_allocation(
        self, state: Optional[PortfolioState] = None, portfolio_id: str = "default"
    ) -> Tuple[List[Dict], Optional[str]]:
        pid = validate_portfolio_id(portfolio_id)
        st = state or self.repo.load_state(pid)
        return compute_target_allocation_variance(st)

    def structured_assign_holding_bucket(
        self, symbol: str, bucket_id: Optional[str], portfolio_id: str = "default"
    ) -> PortfolioState:
        pid = validate_portfolio_id(portfolio_id)
        with self.repo.unit_of_work(pid) as uow:
            state = uow.load_state()
            if bucket_id is not None:
                valid_bucket_ids = {t.bucket_id for t in state.allocation_targets}
                if bucket_id not in valid_bucket_ids:
                    raise ValueError(f"ไม่พบ bucket_id '{bucket_id}' ใน allocation targets")
            clean_sym = symbol.strip().upper()
            h = _find_holding(state, clean_sym)
            if not h:
                raise ValueError(f"ไม่พบสินทรัพย์ '{clean_sym}' ในพอร์ต")
            h.bucket_id = bucket_id
            uow.commit(state, LedgerChange(kind="unchanged"))
            return state

    def structured_batch_assign_holding_buckets(
        self, symbols: List[str], bucket_id: Optional[str], portfolio_id: str = "default"
    ) -> PortfolioState:
        pid = validate_portfolio_id(portfolio_id)
        with self.repo.unit_of_work(pid) as uow:
            state = uow.load_state()
            if bucket_id is not None:
                valid_bucket_ids = {t.bucket_id for t in state.allocation_targets}
                if bucket_id not in valid_bucket_ids:
                    raise ValueError(f"ไม่พบ bucket_id '{bucket_id}' ใน allocation targets")
            sym_set = {s.strip().upper() for s in symbols}
            found = 0
            for h in state.holdings:
                if h.symbol in sym_set:
                    h.bucket_id = bucket_id
                    found += 1
            if found == 0 and symbols:
                raise ValueError("ไม่พบสินทรัพย์ตามที่ระบุในพอร์ต")
            uow.commit(state, LedgerChange(kind="unchanged"))
            return state

    def structured_batch_remove_holdings(
        self, symbols: List[str], portfolio_id: str = "default"
    ) -> PortfolioState:
        pid = validate_portfolio_id(portfolio_id)
        with self.repo.unit_of_work(pid) as uow:
            state = uow.load_state()
            sym_set = {s.strip().upper() for s in symbols}
            before_len = len(state.holdings)
            state.holdings = [h for h in state.holdings if h.symbol not in sym_set]
            if len(state.holdings) == before_len and symbols:
                raise ValueError("ไม่พบสินทรัพย์ตามที่ระบุเพื่อลบ")
            uow.commit(state, LedgerChange(kind="unchanged"))
            return state

    def structured_reset_clean_slate(self, portfolio_id: str = "default") -> PortfolioState:
        pid = validate_portfolio_id(portfolio_id)
        return self.repo.backup_and_reset_clean_slate(portfolio_id=pid)

    def structured_upsert_allocation_targets(
        self, targets: List[AllocationTarget], portfolio_id: str = "default"
    ) -> PortfolioState:
        pid = validate_portfolio_id(portfolio_id)
        with self.repo.unit_of_work(pid) as uow:
            state = uow.load_state()
            total_pct = sum(t.target_percent for t in targets)
            if abs(total_pct - 100.0) > 0.01:
                raise ValueError(f"ผลรวม target_percent ({total_pct:.1f}%) ต้องเท่ากับ 100%")
            state.allocation_targets = targets
            uow.commit(state, LedgerChange(kind="unchanged"))
            return state

    def compute_allocation_breakdown(
        self, group_by: Literal["asset_type", "currency"] = "asset_type", portfolio_id: str = "default"
    ) -> str:
        from filelock import Timeout
        from tools.tool_errors import LOCK_TIMEOUT
        pid = validate_portfolio_id(portfolio_id)
        try:
            state = self.repo.load_state(pid)
            breakdown = compute_allocation_breakdown(state, group_by=group_by)
            total_nav = state.summary.total_value_thb
            return json.dumps(
                {
                    "group_by": group_by,
                    "total_nav_thb": total_nav,
                    "breakdown": breakdown,
                    "generated_at": _now_iso(),
                },
                ensure_ascii=False,
                indent=2,
            )
        except Timeout:
            return LOCK_TIMEOUT.format(detail=f"portfolio lock '{pid}'")
