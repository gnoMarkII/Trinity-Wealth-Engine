"""Application use cases for Macro dashboards, news funnel, and calendars."""
from __future__ import annotations

import concurrent.futures
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional

from application.macro.ports import (
    AssetResolverPort,
    IndicatorSeriesPort,
    MarketCalendarPort,
    NewsFunnelPort,
    PortfolioReadPort,
    StrategySnapshotPort,
)


class MacroApplicationService:
    """Coordinates Macro read models without knowing their storage format."""

    def __init__(
        self,
        strategy: StrategySnapshotPort,
        indicators: IndicatorSeriesPort,
        news_funnel: NewsFunnelPort,
    ) -> None:
        self._strategy = strategy
        self._indicators = indicators
        self._news_funnel = news_funnel

    def latest_portfolio(self) -> Dict[str, Any]:
        return self._strategy.latest()

    def dashboard(self) -> Dict[str, Any]:
        return self._strategy.latest()

    def report_by_id(self, strategy_report_id: str) -> Dict[str, Any]:
        reader = getattr(self._strategy, "report_by_id", None)
        if not callable(reader):
            raise LookupError("Archived Macro reports are unavailable")
        try:
            return reader(strategy_report_id)
        except FileNotFoundError as exc:
            raise LookupError("Macro report not found") from exc

    def indicator_series(self, indicator_id: str, range_name: str) -> Dict[str, Any]:
        if range_name not in {"1m", "3m", "1y"}:
            raise ValueError("range must be one of: 1m, 3m, 1y")
        raw = self._strategy.latest()
        indicator = next(
            (
                item
                for item in raw.get("dashboard_indicators", [])
                if isinstance(item, dict) and item.get("indicator_id") == indicator_id
            ),
            None,
        )
        if indicator is None:
            raise LookupError("Macro indicator not found in the latest report")
        try:
            points = self._indicators.load(str(indicator.get("series_key", "")), range_name)
        except ValueError as exc:
            raise LookupError("Macro indicator series is unavailable") from exc
        return {
            "indicator_id": indicator_id,
            "series_key": str(indicator.get("series_key", "")),
            "label": str(indicator.get("label", "")),
            "unit": str(indicator.get("unit", "")),
            "range": range_name,
            "points": points,
        }

    def pending_news(self) -> List[Dict[str, Any]]:
        return self._news_funnel.pending()

    def filtered_news(self) -> List[Dict[str, Any]]:
        return self._news_funnel.filtered()

    def reject_news(self, event_id: str) -> Dict[str, Any]:
        return {"ok": True, "remaining_count": self._news_funnel.reject(event_id)}


class PortfolioCalendarApplicationService:
    """Builds a calendar read model from portfolio and market ports."""

    _CASH_SYMBOLS = {"CASH_THB", "CASH_USD", "CASH"}

    def __init__(
        self,
        portfolio: PortfolioReadPort,
        resolver: AssetResolverPort,
        calendar: MarketCalendarPort,
    ) -> None:
        self._portfolio = portfolio
        self._resolver = resolver
        self._calendar = calendar

    def get_calendar(self, portfolio_id: str = "default") -> Dict[str, Any]:
        state = self._portfolio.get_state(portfolio_id)
        watchlist = self._portfolio.get_watchlist(portfolio_id)
        holding_names = {
            item.symbol.strip().upper(): item.company_name
            for item in getattr(state, "holdings", [])
            if item.symbol and item.symbol not in self._CASH_SYMBOLS and item.asset_type != "Cash"
        }
        tracked: list[dict[str, Any]] = []
        seen: set[str] = set()

        def add(symbol: str, bucket: str, company_name: Optional[str]) -> None:
            clean = symbol.strip().upper()
            if not clean or clean in seen or clean in self._CASH_SYMBOLS:
                return
            seen.add(clean)
            resolved = self._resolver.resolve(clean)
            if resolved is not None and getattr(resolved, "confidence", None) == "low":
                return
            tracked.append({
                "symbol": clean,
                "provider_symbol": getattr(resolved, "provider_symbol", None) or clean,
                "company_name": company_name,
                "bucket": bucket,
            })

        for holding in getattr(state, "holdings", []):
            if holding.symbol and holding.asset_type != "Cash":
                add(holding.symbol, "holding", holding_names.get(holding.symbol.strip().upper()))
        for item in getattr(watchlist, "items", []):
            if item.symbol and item.asset_type != "Cash":
                add(item.symbol, "watchlist", None)


        now = datetime.now(timezone.utc).isoformat()
        if not tracked:
            return {"generated_at": now, "events": [], "tickers_fetched": 0, "tickers_failed": []}

        events: list[dict[str, Any]] = []
        failed: list[str] = []
        today = date.today()
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = {executor.submit(self._calendar.fetch, item["provider_symbol"]): item for item in tracked}
            for future in concurrent.futures.as_completed(futures):
                item = futures[future]
                try:
                    self._append_events(events, item, future.result(), today)
                except Exception:
                    failed.append(item["symbol"])
        events.sort(key=lambda value: value["event_date"])
        return {
            "generated_at": now,
            "events": events,
            "tickers_fetched": len(tracked) - len(failed),
            "tickers_failed": failed,
        }

    @staticmethod
    def _coerce_date(value: Any) -> Optional[date]:
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, date):
            return value
        if isinstance(value, str):
            try:
                return datetime.strptime(value[:10], "%Y-%m-%d").date()
            except ValueError:
                return None
        return None

    @classmethod
    def _append_events(cls, events: list[dict[str, Any]], item: dict[str, Any], calendar: Dict[str, Any], today: date) -> None:
        earnings = calendar.get("Earnings Date")
        if isinstance(earnings, list) and earnings:
            event_date = cls._coerce_date(earnings[0])
            if event_date:
                events.append({
                    "ticker": item["symbol"],
                    "company_name": item["company_name"],
                    "event_type": "earnings",
                    "event_date": event_date.isoformat(),
                    "days_until": (event_date - today).days,
                    "bucket": item["bucket"],
                    "eps_estimate": _as_float(calendar.get("Earnings Average")),
                    "eps_low": _as_float(calendar.get("Earnings Low")),
                    "eps_high": _as_float(calendar.get("Earnings High")),
                })
        ex_date = cls._coerce_date(calendar.get("Ex-Dividend Date"))
        if ex_date:
            events.append({
                "ticker": item["symbol"],
                "company_name": item["company_name"],
                "event_type": "ex_dividend",
                "event_date": ex_date.isoformat(),
                "days_until": (ex_date - today).days,
                "bucket": item["bucket"],
            })


def _as_float(value: Any) -> Optional[float]:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
