"""PortfolioWatchlistService — Watchlist CRUD operations."""
import json
from typing import Optional

from tools.portfolio.domain.models import WatchlistState, WatchlistItem, _now_iso
from tools.portfolio.domain.validator import validate_portfolio_id
from tools.portfolio.ports.watchlist_port import WatchlistRepositoryPort


class PortfolioWatchlistService:
    """Handles all Watchlist add/update/remove/read operations."""

    def __init__(self, watchlist_repo: WatchlistRepositoryPort) -> None:
        self.watchlist_repo = watchlist_repo

    def add_to_watchlist(
        self, symbol: str, asset_type: str, target_price: Optional[float] = None, notes: str = "", portfolio_id: str = "default"
    ) -> str:
        from filelock import Timeout
        from tools.tool_errors import LOCK_TIMEOUT
        clean_sym = symbol.strip().upper()
        if not clean_sym:
            return "Error: symbol ต้องไม่ว่าง"
        if target_price is not None and target_price <= 0:
            return "Error: target_price ต้องมากกว่า 0"

        pid = validate_portfolio_id(portfolio_id)
        try:
            state = self.watchlist_repo.load_watchlist(pid)
            item = next((it for it in state.items if it.symbol == clean_sym), None)
            is_update = item is not None
            if item:
                item.asset_type = asset_type
                if target_price is not None:
                    item.target_price = target_price
                if notes:
                    item.notes = notes
            else:
                state.items.append(
                    WatchlistItem(
                        symbol=clean_sym,
                        asset_type=asset_type,
                        target_price=target_price,
                        notes=notes or None,
                        added_date=_now_iso()[:10],
                    )
                )
            self.watchlist_repo.save_watchlist(state, pid)
            if is_update:
                return f"[WATCH UPD] อัปเดต {clean_sym} ใน Watchlist (target_price={target_price})"
            return f"[WATCH ADD] เพิ่ม {clean_sym} ({asset_type}) เข้า Watchlist (target_price={target_price})"
        except Timeout:
            return LOCK_TIMEOUT.format(detail=f"watchlist lock '{pid}'")
        except OSError:
            raise
        except Exception as e:
            return f"Error: {e}"

    def remove_from_watchlist(self, symbol: str, portfolio_id: str = "default") -> str:
        from filelock import Timeout
        from tools.tool_errors import LOCK_TIMEOUT
        clean_sym = symbol.strip().upper()
        if not clean_sym:
            return "Error: symbol ต้องไม่ว่าง"

        pid = validate_portfolio_id(portfolio_id)
        try:
            state = self.watchlist_repo.load_watchlist(pid)
            orig_len = len(state.items)
            state.items = [it for it in state.items if it.symbol != clean_sym]
            if len(state.items) == orig_len:
                return f"Error: ไม่พบ {clean_sym} ใน Watchlist"
            self.watchlist_repo.save_watchlist(state, pid)
            return f"[WATCH DEL] ลบ {clean_sym} ออกจาก Watchlist สำเร็จ (remaining: {len(state.items)})"
        except Timeout:
            return LOCK_TIMEOUT.format(detail=f"watchlist lock '{pid}'")
        except Exception as e:
            return f"Error: {e}"

    def read_watchlist(self, portfolio_id: str = "default") -> str:
        from filelock import Timeout
        pid = validate_portfolio_id(portfolio_id)
        try:
            state = self.get_structured_watchlist(portfolio_id=pid)
            items_list = [it.model_dump(exclude_none=True) for it in state.items]
            return json.dumps({"n_items": len(items_list), "items": items_list}, ensure_ascii=False, indent=2)
        except Timeout:
            return json.dumps({"error": f"watchlist lock timeout for '{pid}'"})
        except Exception as e:
            return json.dumps({"error": str(e)})

    def get_structured_watchlist(self, portfolio_id: str = "default") -> WatchlistState:
        return self.watchlist_repo.load_watchlist(portfolio_id)

    def structured_upsert_watchlist_item(
        self, symbol: str, asset_type: str, target_price: Optional[float] = None, notes: str = "", portfolio_id: str = "default"
    ) -> WatchlistState:
        clean_sym = symbol.strip().upper()
        if not clean_sym:
            raise ValueError("symbol ต้องไม่ว่าง")
        if target_price is not None and target_price <= 0:
            raise ValueError("target_price ต้องมากกว่า 0")
        state = self.watchlist_repo.load_watchlist(portfolio_id)
        item = next((it for it in state.items if it.symbol == clean_sym), None)
        if item:
            item.asset_type = asset_type
            if target_price is not None:
                item.target_price = target_price
            if notes:
                item.notes = notes
        else:
            state.items.append(
                WatchlistItem(
                    symbol=clean_sym,
                    asset_type=asset_type,
                    target_price=target_price,
                    notes=notes or None,
                    added_date=_now_iso()[:10],
                )
            )
        self.watchlist_repo.save_watchlist(state, portfolio_id)
        return state

    def structured_remove_watchlist_item(self, symbol: str, portfolio_id: str = "default") -> WatchlistState:
        clean_sym = symbol.strip().upper()
        if not clean_sym:
            raise ValueError("symbol ต้องไม่ว่าง")
        state = self.watchlist_repo.load_watchlist(portfolio_id)
        orig_len = len(state.items)
        state.items = [it for it in state.items if it.symbol != clean_sym]
        if len(state.items) == orig_len:
            raise ValueError(f"ไม่พบ {clean_sym} ใน Watchlist")
        self.watchlist_repo.save_watchlist(state, portfolio_id)
        return state
