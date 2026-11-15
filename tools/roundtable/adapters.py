"""Concrete outbound adapters for Trinity Guru Roundtable (tools layer).

Architecture Rule: Lives in ``tools/roundtable/``; NEVER imports ``api.*``.
"""
from __future__ import annotations

from datetime import datetime
from decimal import Decimal, InvalidOperation
import logging
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

from application.roundtable.ports import (
    ContextDataProvidersPort,
    SandboxCalcPort,
    SettlementPricePort,
    ThesisUpdaterPort,
    VaultNoteWriterPort,
)
from core.roundtable.models import FrozenContextPacket
from tools.hermes_bridge.sandbox_adapter import SandboxAdapter

log = logging.getLogger(__name__)


_NUMERIC_TOKEN_RE = re.compile(r"(?<![\d.])[-+]?\d+(?:\.\d+)?(?![\d.])")


def _numeric_token_spans(text: str) -> List[Tuple[int, int]]:
    """Return signed decimal spans so clause splitting never treats a decimal point as punctuation."""
    return [match.span() for match in _NUMERIC_TOKEN_RE.finditer(text or "")]


class CallableContextDataProvidersAdapter(ContextDataProvidersPort):
    """Bridges narrow domain query callables into ContextDataProvidersPort without api.* imports."""

    def __init__(
        self,
        *,
        instrument_resolver: Callable[[str], Dict[str, Any]],
        equity_detail_reader: Callable[[str], Optional[Dict[str, Any]]],
        equity_review_reader: Callable[[str, str], Optional[Dict[str, Any]]],
        macro_dashboard_reader: Callable[[], Optional[Dict[str, Any]]],
        daily_digest_reader: Callable[[], Optional[Dict[str, Any]]],
        portfolio_state_reader: Callable[[str], Optional[Dict[str, Any]]],
        confirmed_axis_reader: Callable[[str], Optional[Dict[str, Any]]],
        active_thesis_reader: Callable[[str], Optional[Dict[str, Any]]],
        research_dispatcher: Optional[Callable[[str], Optional[str]]] = None,
        equity_news_reader: Optional[Callable[[str], Optional[Dict[str, Any]]]] = None,
        equity_notes_lister: Optional[Callable[[str, int], Optional[Dict[str, Any]]]] = None,
        equity_note_reader: Optional[Callable[[str], Optional[Dict[str, Any]]]] = None,
        financial_context_reader: Optional[Callable[[str], Optional[Dict[str, Any]]]] = None,
        bucket_policy_reader: Optional[Callable[[str], Optional[List[Dict[str, Any]]]]] = None,
        watchlist_reader: Optional[Callable[[str], Optional[List[Dict[str, Any]]]]] = None,
        thesis_exceptions_reader: Optional[Callable[[str], Optional[List[Dict[str, Any]]]]] = None,
        promise_scores_reader: Optional[Callable[[str], Optional[Dict[str, Any]]]] = None,
        analyst_context_reader: Optional[Callable[[str], Optional[Dict[str, Any]]]] = None,
        insider_db_summary_reader: Optional[Callable[[str], Optional[Dict[str, Any]]]] = None,
        portfolio_advisory_reader: Optional[Callable[[str], Optional[Dict[str, Any]]]] = None,
        investor_essence_reader: Optional[Callable[[str], Optional[Dict[str, Any]]]] = None,
    ) -> None:
        self._instrument_resolver = instrument_resolver
        self._equity_detail_reader = equity_detail_reader
        self._equity_review_reader = equity_review_reader
        self._macro_dashboard_reader = macro_dashboard_reader
        self._daily_digest_reader = daily_digest_reader
        self._portfolio_state_reader = portfolio_state_reader
        self._confirmed_axis_reader = confirmed_axis_reader
        self._active_thesis_reader = active_thesis_reader
        self._research_dispatcher = research_dispatcher
        self._equity_news_reader = equity_news_reader
        self._equity_notes_lister = equity_notes_lister
        self._equity_note_reader = equity_note_reader
        self._financial_context_reader = financial_context_reader
        self._bucket_policy_reader = bucket_policy_reader
        self._watchlist_reader = watchlist_reader
        self._thesis_exceptions_reader = thesis_exceptions_reader
        self._promise_scores_reader = promise_scores_reader
        self._analyst_context_reader = analyst_context_reader
        self._insider_db_summary_reader = insider_db_summary_reader
        self._portfolio_advisory_reader = portfolio_advisory_reader
        self._investor_essence_reader = investor_essence_reader

    def resolve_instrument(self, ticker: str) -> Dict[str, Any]:
        try:
            return self._instrument_resolver(ticker) or {}
        except Exception as exc:
            log.debug("resolve_instrument(%s) error: %s", ticker, exc)
            return {}

    def get_equity_detail(self, ticker: str) -> Optional[Dict[str, Any]]:
        try:
            return self._equity_detail_reader(ticker)
        except Exception as exc:
            log.debug("get_equity_detail(%s) error: %s", ticker, exc)
            return None

    def get_equity_position_review(self, ticker: str, portfolio_id: str = "default") -> Optional[Dict[str, Any]]:
        try:
            return self._equity_review_reader(ticker, portfolio_id)
        except Exception as exc:
            log.debug("get_equity_position_review(%s) error: %s", ticker, exc)
            return None

    def get_equity_news(self, ticker: str) -> Optional[Dict[str, Any]]:
        if self._equity_news_reader is None:
            return None
        try:
            return self._equity_news_reader(ticker)
        except Exception as exc:
            log.debug("get_equity_news(%s) error: %s", ticker, exc)
            return None

    def list_equity_notes(self, ticker: str, days: int = 30) -> Optional[Dict[str, Any]]:
        if self._equity_notes_lister is None:
            return None
        try:
            return self._equity_notes_lister(ticker, days)
        except Exception as exc:
            log.debug("list_equity_notes(%s) error: %s", ticker, exc)
            return None

    def read_equity_note(self, relative_path: str) -> Optional[Dict[str, Any]]:
        if self._equity_note_reader is None:
            return None
        try:
            return self._equity_note_reader(relative_path)
        except Exception as exc:
            log.debug("read_equity_note(%s) error: %s", relative_path, exc)
            return None

    def get_macro_dashboard(self) -> Optional[Dict[str, Any]]:
        try:
            return self._macro_dashboard_reader()
        except Exception as exc:
            log.debug("get_macro_dashboard error: %s", exc)
            return None

    def get_daily_digest(self) -> Optional[Dict[str, Any]]:
        try:
            return self._daily_digest_reader()
        except Exception as exc:
            log.debug("get_daily_digest error: %s", exc)
            return None

    def get_portfolio_state(self, portfolio_id: str = "default") -> Optional[Dict[str, Any]]:
        try:
            return self._portfolio_state_reader(portfolio_id)
        except Exception as exc:
            log.debug("get_portfolio_state error: %s", exc)
            return None

    def get_confirmed_investment_axis(self, portfolio_id: str = "default") -> Optional[Dict[str, Any]]:
        try:
            return self._confirmed_axis_reader(portfolio_id)
        except Exception as exc:
            log.debug("get_confirmed_investment_axis error: %s", exc)
            return None

    def get_financial_context(self, portfolio_id: str = "default") -> Optional[Dict[str, Any]]:
        if self._financial_context_reader is None:
            return None
        try:
            return self._financial_context_reader(portfolio_id)
        except Exception as exc:
            log.debug("get_financial_context error: %s", exc)
            return None

    def get_bucket_policy(self, portfolio_id: str = "default") -> Optional[List[Dict[str, Any]]]:
        if self._bucket_policy_reader is None:
            return None
        try:
            return self._bucket_policy_reader(portfolio_id)
        except Exception as exc:
            log.debug("get_bucket_policy error: %s", exc)
            return None

    def get_watchlist(self, portfolio_id: str = "default") -> Optional[List[Dict[str, Any]]]:
        if self._watchlist_reader is None:
            return None
        try:
            return self._watchlist_reader(portfolio_id)
        except Exception as exc:
            log.debug("get_watchlist error: %s", exc)
            return None

    def get_active_thesis(self, ticker: str) -> Optional[Dict[str, Any]]:
        try:
            return self._active_thesis_reader(ticker)
        except Exception as exc:
            log.debug("get_active_thesis(%s) error: %s", ticker, exc)
            return None

    def get_thesis_pending_exceptions(self, ticker: str) -> Optional[List[Dict[str, Any]]]:
        if self._thesis_exceptions_reader is None:
            return None
        try:
            return self._thesis_exceptions_reader(ticker)
        except Exception as exc:
            log.debug("get_thesis_pending_exceptions(%s) error: %s", ticker, exc)
            return None

    def get_promise_ledger_scores(self, ticker: str) -> Optional[Dict[str, Any]]:
        if self._promise_scores_reader is None:
            return None
        try:
            return self._promise_scores_reader(ticker)
        except Exception as exc:
            log.debug("get_promise_ledger_scores(%s) error: %s", ticker, exc)
            return None

    def get_analyst_context(self, ticker: str) -> Optional[Dict[str, Any]]:
        if self._analyst_context_reader is None:
            return None
        try:
            return self._analyst_context_reader(ticker)
        except Exception as exc:
            log.debug("get_analyst_context(%s) error: %s", ticker, exc)
            return None

    def get_insider_db_summary(self, ticker: str) -> Optional[Dict[str, Any]]:
        if self._insider_db_summary_reader is None:
            return None
        try:
            return self._insider_db_summary_reader(ticker)
        except Exception as exc:
            log.debug("get_insider_db_summary(%s) error: %s", ticker, exc)
            return None

    def get_portfolio_advisory_summary(self, portfolio_id: str = "default") -> Optional[Dict[str, Any]]:
        if self._portfolio_advisory_reader is None:
            return None
        try:
            return self._portfolio_advisory_reader(portfolio_id)
        except Exception as exc:
            log.debug("get_portfolio_advisory_summary error: %s", exc)
            return None

    def get_investor_essence(self, portfolio_id: str = "default") -> Optional[Dict[str, Any]]:
        if self._investor_essence_reader is None:
            return None
        try:
            return self._investor_essence_reader(portfolio_id)
        except Exception as exc:
            log.debug("get_investor_essence error: %s", exc)
            return None

    def dispatch_equity_research_if_needed(self, ticker: str) -> Optional[str]:
        if self._research_dispatcher is None:
            return None
        try:
            return self._research_dispatcher(ticker)
        except Exception as exc:
            log.warning("dispatch_equity_research_if_needed(%s) error: %s", ticker, exc)
            return None


class OhlcvSettlementPriceAdapter(SettlementPricePort):
    """Fetches daily close prices via injected OHLCV query callable using Decimal."""

    def __init__(self, ohlcv_fetcher: Callable[[str, str, str], Any]) -> None:
        self._fetcher = ohlcv_fetcher

    def _candles_with_dates(self, ticker: str) -> list[tuple[str, Decimal]]:
        try:
            res = self._fetcher(ticker, "1y", "1d")
            candles = getattr(res, "candles", None) or []
            tz = ZoneInfo("Asia/Bangkok") if ticker.strip().upper().endswith(".BK") else ZoneInfo("UTC")
            out: list[tuple[str, Decimal]] = []
            for c in candles:
                ts_ms = getattr(c, "timestamp", None)
                close_val = getattr(c, "close", None)
                if ts_ms is None or close_val is None:
                    continue
                date_str = datetime.fromtimestamp(float(ts_ms) / 1000.0, tz=tz).date().isoformat()
                try:
                    out.append((date_str, Decimal(str(close_val))))
                except InvalidOperation:
                    continue
            return out
        except Exception as exc:
            log.debug("Settlement OHLCV fetch error for %s: %s", ticker, exc)
            return []

    def get_close_on_or_after(self, ticker: str, target_date_iso: str) -> Optional[Tuple[str, Decimal]]:
        target = target_date_iso[:10]
        rows = self._candles_with_dates(ticker)
        for d, px in rows:
            if d >= target:
                return d, px
        return rows[-1] if rows else None

    def get_latest_close(self, ticker: str) -> Optional[Tuple[str, Decimal]]:
        rows = self._candles_with_dates(ticker)
        return rows[-1] if rows else None


class RoundtableSandboxCalcAdapter(SandboxCalcPort):
    """Runs safe AST-whitelisted Python Decimal calculations inside manager_sandbox/."""

    def __init__(self, sandbox: Optional[SandboxAdapter] = None) -> None:
        self._sandbox = sandbox or SandboxAdapter()

    def _extract_usd_thb_fx_rate(self, packet: FrozenContextPacket) -> Optional[str]:
        eq_data = packet.equity_data if isinstance(packet.equity_data, dict) else {}
        for key in ("fx_usd_thb", "usd_thb_fx_rate", "usd_thb", "fx_rate"):
            val = eq_data.get(key)
            if val is not None:
                try:
                    d = Decimal(str(val).replace(",", ""))
                    if d > 0:
                        return str(d)
                except InvalidOperation:
                    pass
        fx_rates = eq_data.get("fx_rates") if isinstance(eq_data.get("fx_rates"), dict) else {}
        for k in ("THB", "USD_THB", "USD/THB"):
            if fx_rates.get(k) is not None:
                try:
                    d = Decimal(str(fx_rates[k]).replace(",", ""))
                    if d > 0:
                        return str(d)
                except InvalidOperation:
                    pass

        for txt in (packet.portfolio_guard.summary_text or "", packet.macro_report_text or ""):
            m = re.search(r"(?:FX\s+)?USD/THB\s*[:=]\s*(\d+(?:\.\d+)?)", txt, re.IGNORECASE)
            if m:
                try:
                    d = Decimal(m.group(1))
                    if d > 0:
                        return str(d)
                except InvalidOperation:
                    pass
        return None

    def run_what_if_calculation(self, question: str, packet: FrozenContextPacket) -> Dict[str, Any]:
        # Build deterministic Decimal calculation script anchored on packet numbers + user question parameters
        clean_q = (question or "").replace("\u2212", "-").replace(",", "")
        cur_price = (
            packet.instrument_anchor.current_price
            if packet.instrument_anchor and packet.instrument_anchor.current_price
            else None
        )
        stock_currency = (
            (packet.instrument_anchor.currency or "USD").strip().upper()
            if packet.instrument_anchor
            else "USD"
        )
        usd_thb_fx = self._extract_usd_thb_fx_rate(packet)

        pct_matches = list(re.finditer(r"(?<![\d.])([-+]?\d+(?:\.\d+)?)\s*%", clean_q))
        allocation_cue_re = re.compile(
            r"(?i)(?:\btarget\s+allocation\b|\ballocation\b|\ballocate\b|"
            r"\bportfolio\s+(?:weight|allocation)\b|\btarget\s+weight\b|\bweight\s+target\b|"
            r"\u0e2a\u0e31\u0e14\u0e2a\u0e48\u0e27\u0e19|\u0e08\u0e31\u0e14\u0e2a\u0e23\u0e23|"
            r"\u0e19\u0e49\u0e33\u0e2b\u0e19\u0e31\u0e01|\u0e40\u0e1b\u0e49\u0e32\u0e2b\u0e21\u0e32\u0e22)"
        )
        price_drop_cue_re = re.compile(
            r"(?i)(?:\bprice\s*(?:drops?|declines?|falls?|down)\b|"
            r"\b(?:drop(?:s|ped)?|declin(?:e|es|ed)|fall(?:s|en)?|down)\s*(?:by\s*)?\d*\s*%?\s*(?:in\s+)?price\b|"
            r"\u0e23\u0e32\u0e04\u0e32(?:\u0e2b\u0e38\u0e49\u0e19)?\s*(?:\u0e25\u0e07|\u0e25\u0e14(?:\u0e25\u0e07)?|\u0e1b\u0e23\u0e31\u0e1a\u0e25\u0e07)|"
            r"\u0e1b\u0e23\u0e31\u0e1a\u0e23\u0e32\u0e04\u0e32\s*(?:\u0e25\u0e07|\u0e25\u0e14(?:\u0e25\u0e07)?)|"
            r"(?:\u0e25\u0e14\u0e25\u0e07|\u0e1b\u0e23\u0e31\u0e1a\u0e25\u0e07))"
        )

        allocation_pct_candidates: list[str] = []
        price_drop_pct_candidates: list[str] = []
        recognized_numeric_spans: set[Tuple[int, int]] = set()
        for pct_match in pct_matches:
            window_start = max(0, pct_match.start() - 64)
            window_end = min(len(clean_q), pct_match.end() + 32)
            window = clean_q[window_start:window_end]
            allocation_cues = list(allocation_cue_re.finditer(window))
            price_drop_cues = list(price_drop_cue_re.finditer(window))

            def _cue_distance(cue: re.Match[str]) -> int:
                if cue.end() <= pct_match.start() - window_start:
                    return pct_match.start() - window_start - cue.end()
                if cue.start() >= pct_match.end() - window_start:
                    return cue.start() - (pct_match.end() - window_start)
                return 0

            allocation_distance = min((_cue_distance(c) for c in allocation_cues), default=None)
            price_distance = min((_cue_distance(c) for c in price_drop_cues), default=None)
            if allocation_distance is not None and (
                price_distance is None or allocation_distance < price_distance
            ):
                allocation_pct_candidates.append(pct_match.group(1))
                recognized_numeric_spans.add(pct_match.span(1))
            elif price_distance is not None and (
                allocation_distance is None or price_distance < allocation_distance
            ):
                price_drop_pct_candidates.append(pct_match.group(1))
                recognized_numeric_spans.add(pct_match.span(1))

        allocation_pct_values = list(dict.fromkeys(allocation_pct_candidates))
        price_drop_pct_values = list(dict.fromkeys(price_drop_pct_candidates))
        allocation_pct = allocation_pct_values[0] if len(allocation_pct_values) == 1 else None
        price_drop_pct = price_drop_pct_values[0] if len(price_drop_pct_values) == 1 else None
        percentage_ambiguities = []
        if len(allocation_pct_values) > 1:
            percentage_ambiguities.append("multiple_allocation_percentages")
        if len(price_drop_pct_values) > 1:
            percentage_ambiguities.append("multiple_price_drop_percentages")

        shares_matches = list(re.finditer(
            r"(?<![\d.])(?P<sign>[-+]?)(?P<amount>\d+(?:\.\d+)?)\s*(?:\u0e2b\u0e38\u0e49\u0e19|shares?|units?)",
            clean_q,
            re.IGNORECASE,
        ))
        share_action_re = re.compile(
            r"(?i)(?:\b(?:buy|purchase|add|invest|accumulate)\b|"
            r"\u0e0b\u0e37\u0e49\u0e2d|\u0e25\u0e07\u0e17\u0e38\u0e19|\u0e40\u0e1e\u0e34\u0e48\u0e21)"
        )
        negative_action_re = re.compile(
            r"(?i)(?:\b(?:not|never|don't|do not|wouldn't|won't|shouldn't|can't|cannot)\s+"
            r"(?:(?:want|plan|intend)\s+to|to)?\s*(?:buy|purchase|add|invest|accumulate)\b|"
            r"(?:\u0e44\u0e21\u0e48\s*(?:\u0e04\u0e27\u0e23\s*)?|"
            r"\u0e2b\u0e49\u0e32\u0e21\s*|\u0e2d\u0e22\u0e48\u0e32\s*)"
            r"(?:buy|purchase|add|invest|accumulate|\u0e0b\u0e37\u0e49\u0e2d|\u0e25\u0e07\u0e17\u0e38\u0e19|\u0e40\u0e1e\u0e34\u0e48\u0e21))"
        )
        sell_action_re = re.compile(
            r"(?i)(?:\b(?:sell|sale|reduce|trim|exit)\b|"
            r"\u0e02\u0e32\u0e22|\u0e25\u0e14\u0e2b\u0e38\u0e49\u0e19|\u0e25\u0e14\u0e1e\u0e2d\u0e23\u0e4c|\u0e15\u0e31\u0e14\u0e02\u0e32\u0e22)"
        )
        held_action_re = re.compile(
            r"(?i)(?:\b(?:hold|holds|holding|own|owns|have)\b|\u0e16\u0e37\u0e2d|\u0e04\u0e23\u0e2d\u0e1a\u0e04\u0e23\u0e2d\u0e07)"
        )
        transaction_clause_boundary_re = re.compile(
            r"(?i)[.!?;\n]+|\b(?:but|however|although|or)\b|\u0e41\u0e25\u0e49\u0e27|\u0e41\u0e15\u0e48|\u0e2b\u0e23\u0e37\u0e2d"
        )
        clauses = []
        numeric_token_spans = _numeric_token_spans(clean_q)
        clause_start = 0
        for boundary in transaction_clause_boundary_re.finditer(clean_q):
            if boundary.group(0).strip() == "." and any(
                span_start < boundary.start() < span_end
                for span_start, span_end in numeric_token_spans
            ):
                continue
            if boundary.start() > clause_start:
                clauses.append((clause_start, boundary.start(), clean_q[clause_start:boundary.start()]))
            clause_start = boundary.end()
        if clause_start < len(clean_q):
            clauses.append((clause_start, len(clean_q), clean_q[clause_start:]))

        def _clause_for_span(span: Tuple[int, int]) -> str:
            for clause_start, clause_end, clause_text in clauses:
                if clause_start <= span[0] < clause_end:
                    return clause_text
            return clean_q

        def _transaction_action(clause: str) -> str:
            has_negative_buy = bool(negative_action_re.search(clause))
            positive_clause = negative_action_re.sub(" ", clause)
            has_buy = bool(share_action_re.search(positive_clause))
            has_sell = bool(sell_action_re.search(clause))
            has_held = bool(held_action_re.search(clause))
            if has_negative_buy and has_buy:
                return "AMBIGUOUS"
            if has_negative_buy:
                return "NEGATED"
            if has_buy and has_sell:
                return "AMBIGUOUS"
            if has_sell:
                return "SELL"
            if has_buy:
                return "BUY"
            if has_held:
                return "HELD"
            return "NONE"

        explicit_shares = None
        explicit_share_action = "NONE"
        explicit_share_ambiguous = len(shares_matches) > 1
        if len(shares_matches) == 1:
            shares_m = shares_matches[0]
            explicit_share_action = _transaction_action(_clause_for_span(shares_m.span()))
            if explicit_share_action == "BUY":
                explicit_shares = f"{shares_m.group('sign')}{shares_m.group('amount')}"
                recognized_numeric_spans.add(shares_m.span("amount"))

        price_m = re.search(
            r"(?i)(?:\u0e23\u0e32\u0e04\u0e32(?:\u0e25\u0e07(?:\u0e21\u0e32|\u0e40\u0e2b\u0e25\u0e37\u0e2d)?|"
            r"\u0e1b\u0e23\u0e31\u0e1a(?:\u0e15\u0e31\u0e27)?\u0e25\u0e07(?:\u0e40\u0e2b\u0e25\u0e37\u0e2d)?|"
            r"\u0e2d\u0e22\u0e39\u0e48\u0e17\u0e35\u0e48|\u0e40\u0e2b\u0e25\u0e37\u0e2d|"
            r"\u0e17\u0e35\u0e48|\u0e40\u0e1b\u0e49\u0e32\u0e2b\u0e21\u0e32\u0e22)?|\bat\b|\bprice\s*(?:at|of|=)?)"
            r"\s*(?:\u0e40\u0e1b\u0e47\u0e19|\u0e16\u0e36\u0e07|\u0e23\u0e30\u0e14\u0e31\u0e1a)?"
            r"\s*(?P<leading_sign>[-+]?)(?P<prefix_currency>USD\b|THB\b|\$|\u0e3f)?\s*(?P<inner_sign>[-+]?)\s*(?<![\d.])(?P<amount>\d+(?:\.\d+)?)(?![\d.%])"
            r"\s*(?P<suffix_currency>USD\b|THB\b|dollars?\b|baht\b|\u0e1a\u0e32\u0e17|\u0e14\u0e2d\u0e25\u0e25\u0e32\u0e23\u0e4c|\u0e3f)?",
            clean_q,
        )
        scenario_price = (f"{price_m.group('leading_sign') or price_m.group('inner_sign')}{price_m.group('amount')}" if price_m else None)
        scenario_price_span = price_m.span("amount") if price_m else None
        scenario_price_currency = stock_currency
        if price_m is not None:
            currency_token = price_m.group("prefix_currency") or price_m.group("suffix_currency") or ""
            currency_upper = currency_token.upper()
            if currency_upper in ("$", "USD", "DOLLAR", "DOLLARS", "\u0e14\u0e2d\u0e25\u0e25\u0e32\u0e23\u0e4c"):
                scenario_price_currency = "USD"
            elif currency_upper in ("THB", "BAHT", "\u0e1a\u0e32\u0e17", "\u0e3f"):
                scenario_price_currency = "THB"
            recognized_numeric_spans.add(scenario_price_span)

        budget_context_re = re.compile(
            r"(?i)(?:\bbuy\b|\bpurchase\b|\binvest(?:ment)?\b|\badd\b|\bspend\b|"
            r"\bsell\b|\bbudget\b|\bworth\b|\bshares?\b|\bunits?\b|"
            r"\u0e0b\u0e37\u0e49\u0e2d|\u0e25\u0e07\u0e17\u0e38\u0e19|\u0e40\u0e1e\u0e34\u0e48\u0e21|"
            r"\u0e02\u0e32\u0e22|\u0e07\u0e1a\u0e25\u0e07\u0e17\u0e38\u0e19|\u0e2b\u0e38\u0e49\u0e19)"
        )
        money_token_re = re.compile(
            r"(?<![A-Za-z0-9.])(?P<leading_sign>[-+]?)\s*(?:"
            r"(?P<prefix_currency>\$|\u0e3f)\s*(?P<inner_sign>[-+]?)\s*"
            r"(?P<prefix_amount>\d+(?:\.\d+)?)(?![\d.%])|"
            r"(?P<suffix_amount>\d+(?:\.\d+)?)(?![\d.%])\s*"
            r"(?P<suffix_currency>USD|THB|dollars?|baht|\u0e1a\u0e32\u0e17|\u0e14\u0e2d\u0e25\u0e25\u0e32\u0e23\u0e4c|\u0e3f|\$)(?![A-Za-z]))",
            re.IGNORECASE,
        )
        transaction_money_candidates: list[Tuple[str, str, Tuple[int, int], str]] = []
        budget_candidates: list[Tuple[str, str, Tuple[int, int], str]] = []
        for money_match in money_token_re.finditer(clean_q):
            amount_group = "prefix_amount" if money_match.group("prefix_amount") else "suffix_amount"
            currency_token = money_match.group("prefix_currency") or money_match.group("suffix_currency") or ""
            amount_span = money_match.span(amount_group)
            sign = money_match.group("leading_sign") or money_match.group("inner_sign") or ""
            amount_value = f"{sign}{money_match.group(amount_group)}"
            recognized_numeric_spans.add(amount_span)
            if scenario_price_span is not None and max(amount_span[0], scenario_price_span[0]) < min(amount_span[1], scenario_price_span[1]):
                continue
            currency_upper = currency_token.upper()
            currency = "USD" if currency_upper in (
                "$", "USD", "DOLLAR", "DOLLARS", "\u0e14\u0e2d\u0e25\u0e25\u0e32\u0e23\u0e4c"
            ) else "THB"
            context_start = max(0, money_match.start() - 48)
            context_end = min(len(clean_q), money_match.end() + 48)
            if not budget_context_re.search(clean_q[context_start:context_end]):
                continue
            action = _transaction_action(_clause_for_span(amount_span))
            candidate = (amount_value, currency, amount_span, action)
            transaction_money_candidates.append(candidate)
            if action == "BUY":
                budget_candidates.append(candidate)

        budget_amount: Optional[str] = None
        budget_currency: Optional[str] = None
        budget_span: Optional[Tuple[int, int]] = None
        budget_ambiguous = len(transaction_money_candidates) > 1
        if len(transaction_money_candidates) == 1 and len(budget_candidates) == 1:
            budget_amount, budget_currency, budget_span, _action = budget_candidates[0]
        if budget_span is not None:
            recognized_numeric_spans.add(budget_span)

        observed_actions = {candidate[3] for candidate in transaction_money_candidates}
        if explicit_share_action != "NONE":
            observed_actions.add(explicit_share_action)
        if budget_ambiguous or explicit_share_ambiguous or "AMBIGUOUS" in observed_actions:
            transaction_action = "AMBIGUOUS"
        elif "SELL" in observed_actions:
            transaction_action = "SELL"
        elif "NEGATED" in observed_actions:
            transaction_action = "NEGATED"
        elif "HELD" in observed_actions:
            transaction_action = "HELD"
        elif "BUY" in observed_actions:
            transaction_action = "BUY"
        elif transaction_money_candidates:
            transaction_action = "UNSPECIFIED"
        else:
            transaction_action = "NONE"
        unclassified_numeric_values = [
            number_match.group(0)
            for number_match in re.finditer(r"(?<![\d.])\d+(?:\.\d+)?(?![\d.])", clean_q)
            if not any(
                number_match.start() >= span[0] and number_match.end() <= span[1]
                for span in recognized_numeric_spans
            )
        ]
        clarification_reasons = []
        if unclassified_numeric_values:
            clarification_reasons.append("unclassified_numeric_inputs")
        clarification_reasons.extend(percentage_ambiguities)
        if budget_ambiguous:
            clarification_reasons.append("multiple_budget_amounts")
        if explicit_share_ambiguous:
            clarification_reasons.append("multiple_explicit_share_amounts")
        if transaction_action in ("SELL", "NEGATED"):
            clarification_reasons.append("unsupported_transaction_direction")
        elif transaction_action in ("HELD", "UNSPECIFIED", "AMBIGUOUS"):
            clarification_reasons.append("purchase_intent_missing_or_ambiguous")
        currency_conversion_missing = bool(
            usd_thb_fx is None
            and (
                (scenario_price is not None and scenario_price_currency != stock_currency)
                or (budget_amount is not None and budget_currency != stock_currency)
            )
        )
        if currency_conversion_missing:
            clarification_reasons.append("fx_conversion_missing")
        invalid_numeric_parameters = []
        for parameter_name, parameter_value in (
            ("budget_amount", budget_amount),
            ("explicit_shares", explicit_shares),
            ("scenario_price", scenario_price),
        ):
            if parameter_value is None:
                continue
            try:
                parameter_decimal = Decimal(parameter_value)
                if not parameter_decimal.is_finite() or parameter_decimal <= 0:
                    invalid_numeric_parameters.append(f"{parameter_name}_must_be_positive")
            except (InvalidOperation, ValueError):
                invalid_numeric_parameters.append(f"{parameter_name}_invalid")
        for parameter_name, parameter_value in (
            ("allocation_pct", allocation_pct),
            ("price_drop_pct", price_drop_pct),
        ):
            if parameter_value is None:
                continue
            try:
                parameter_decimal = Decimal(parameter_value)
                if not parameter_decimal.is_finite() or not Decimal("0") <= parameter_decimal <= Decimal("100"):
                    invalid_numeric_parameters.append(f"{parameter_name}_outside_0_to_100")
            except (InvalidOperation, ValueError):
                invalid_numeric_parameters.append(f"{parameter_name}_invalid")
        clarification_reasons.extend(invalid_numeric_parameters)
        needs_clarification = bool(clarification_reasons)
        has_scenario_parameters = any(
            value is not None
            for value in (allocation_pct, price_drop_pct, explicit_shares, scenario_price, budget_amount)
        )
        calculation_status = (
            "needs_clarification"
            if needs_clarification
            else ("calculated" if has_scenario_parameters else "no_transaction_parameters")
        )

        pg = packet.portfolio_guard
        code_lines = [
            "from decimal import Decimal, ROUND_HALF_UP",
            "import json",
            f"question = {question!r}",
            f"stock_currency = {stock_currency!r}",
            f"current_price = Decimal({cur_price!r}) if {cur_price is not None!r} else None",
            f"units_held = Decimal({pg.units!r}) if {pg.units is not None!r} else Decimal('0')",
            f"avg_cost = Decimal({pg.avg_cost!r}) if {pg.avg_cost is not None!r} else Decimal('0')",
            f"nav_thb = Decimal({pg.portfolio_total_nav_thb!r}) if {pg.portfolio_total_nav_thb is not None!r} else Decimal('0')",
            f"usd_thb_fx = Decimal({usd_thb_fx!r}) if {usd_thb_fx is not None!r} else None",
            f"allocation_pct = Decimal({allocation_pct!r}) if {allocation_pct is not None!r} else None",
            f"price_drop_pct = Decimal({price_drop_pct!r}) if {price_drop_pct is not None!r} else None",
            f"explicit_shares = Decimal({explicit_shares!r}) if {explicit_shares is not None!r} else None",
            f"scenario_price = Decimal({scenario_price!r}) if {scenario_price is not None!r} else None",
            f"scenario_price_currency = {scenario_price_currency!r}",
            f"transaction_action = {transaction_action!r}",
            f"currency_conversion_missing = {currency_conversion_missing!r}",
            f"budget_amount = Decimal({budget_amount!r}) if {budget_amount is not None!r} else None",
            f"budget_currency = {budget_currency!r}",
            f"transaction_calculation_blocked = {needs_clarification!r}",
            f"calculation_status = {calculation_status!r}",
            f"clarification_reasons = {clarification_reasons!r}",
            f"unclassified_numeric_values = {unclassified_numeric_values!r}",
            "result = {",
            "    'question': question,",
            "    'stock_currency': stock_currency,",
            "    'current_price': str(current_price) if current_price is not None else None,",
            "    'units_held': str(units_held),",
            "    'avg_cost': str(avg_cost),",
            "    'nav_thb': str(nav_thb),",
            "    'calculation_status': calculation_status,",
            "    'transaction_action': transaction_action,",
            "    'scenario_supported': calculation_status != 'needs_clarification',",
            "    'clarification_reasons': clarification_reasons,",
            "    'unclassified_numeric_values': unclassified_numeric_values,",
            "}",
            "if currency_conversion_missing:",
            "    result['fx_conversion_missing'] = 'USD/THB FX rate not available in packet; cannot convert amounts safely'",
            "if usd_thb_fx is not None:",
            "    result['usd_thb_fx_rate'] = str(usd_thb_fx)",
            "if current_price and units_held > 0 and avg_cost > 0:",
            "    cost_total = units_held * avg_cost",
            "    mkt_val = units_held * current_price",
            "    pnl_pct = ((current_price / avg_cost) - Decimal('1')) * Decimal('100')",
            "    result['position_pnl_pct'] = str(pnl_pct.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))",
            "    result['position_cost_basis'] = str(cost_total.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))",
            "    result['position_market_value'] = str(mkt_val.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))",
            "effective_price = current_price",
            "if scenario_price is not None and scenario_price <= 0:",
            "    effective_price = None",
            "if scenario_price is not None and scenario_price > 0:",
            "    result['scenario_price'] = str(scenario_price.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))",
            "    result['scenario_price_currency'] = scenario_price_currency",
            "    scenario_price_in_stock_ccy = scenario_price",
            "    if scenario_price_currency != stock_currency:",
            "        if usd_thb_fx is not None and usd_thb_fx > 0:",
            "            if scenario_price_currency == 'THB' and stock_currency == 'USD':",
            "                scenario_price_in_stock_ccy = scenario_price / usd_thb_fx",
            "            elif scenario_price_currency == 'USD' and stock_currency == 'THB':",
            "                scenario_price_in_stock_ccy = scenario_price * usd_thb_fx",
            "            else:",
            "                scenario_price_in_stock_ccy = None",
            "        else:",
            "            scenario_price_in_stock_ccy = None",
            "        if scenario_price_in_stock_ccy is None:",
            "            result['fx_conversion_missing'] = 'USD/THB FX rate not available in packet; cannot convert scenario price safely'",
            "    if scenario_price_in_stock_ccy is not None:",
            "        effective_price = scenario_price_in_stock_ccy",
            "        result['scenario_price_in_stock_currency'] = str(scenario_price_in_stock_ccy.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))",
            "if allocation_pct is not None and Decimal('0') <= allocation_pct <= Decimal('100') and nav_thb > 0:",
            "    alloc_amt = (nav_thb * allocation_pct / Decimal('100')).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)",
            "    result['target_allocation_pct'] = str(allocation_pct)",
            "    result['target_allocation_thb'] = str(alloc_amt)",
            "if price_drop_pct is not None and Decimal('0') <= price_drop_pct <= Decimal('100') and current_price and current_price > 0:",
            "    discounted = (current_price * (Decimal('1') - price_drop_pct / Decimal('100'))).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)",
            "    result['price_drop_percent'] = str(price_drop_pct)",
            "    result['price_after_drop_pct'] = str(discounted)",
            "    if scenario_price is None:",
            "        effective_price = discounted",
            "added_shares = None",
            "if not transaction_calculation_blocked and budget_amount is not None and budget_amount > 0 and effective_price is not None and effective_price > 0:",
            "    result['budget_amount'] = str(budget_amount)",
            "    result['budget_currency'] = budget_currency",
            "    budget_in_stock_ccy = budget_amount",
            "    if budget_currency == 'THB' and stock_currency == 'USD':",
            "        if usd_thb_fx is not None and usd_thb_fx > 0:",
            "            budget_in_stock_ccy = budget_amount / usd_thb_fx",
            "            result['budget_converted_usd'] = str(budget_in_stock_ccy.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))",
            "        else:",
            "            budget_in_stock_ccy = None",
            "            result['fx_conversion_missing'] = 'USD/THB FX rate not available in packet; cannot convert THB to USD safely'",
            "    elif budget_currency == 'USD' and stock_currency == 'THB':",
            "        if usd_thb_fx is not None and usd_thb_fx > 0:",
            "            budget_in_stock_ccy = budget_amount * usd_thb_fx",
            "            result['budget_converted_thb'] = str(budget_in_stock_ccy.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))",
            "        else:",
            "            budget_in_stock_ccy = None",
            "            result['fx_conversion_missing'] = 'USD/THB FX rate not available in packet; cannot convert USD to THB safely'",
            "    if budget_in_stock_ccy is not None and budget_in_stock_ccy > 0 and explicit_shares is None:",
            "        added_shares = (budget_in_stock_ccy / effective_price).quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP)",
            "        result['estimated_shares_at_effective_price'] = str(added_shares)",
            "if not transaction_calculation_blocked and explicit_shares is not None and explicit_shares > 0 and effective_price is not None and effective_price > 0:",
            "    added_shares = explicit_shares",
            "if added_shares is not None and added_shares > 0 and effective_price is not None and effective_price > 0:",
            "    result['if_add_units'] = str(added_shares)",
            "    result['additional_cost_in_stock_ccy'] = str((added_shares * effective_price).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))",
            "    if units_held > 0 and avg_cost > 0:",
            "        new_units = units_held + added_shares",
            "        blended = ((units_held * avg_cost) + (added_shares * effective_price)) / new_units",
            "        result['blended_avg_cost'] = str(blended.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))",
            "        result['new_total_units'] = str(new_units.quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP))",
            "print(json.dumps(result, ensure_ascii=True))",
        ]
        script = "\n".join(code_lines)
        exec_res = self._sandbox.execute_scratch_script("roundtable_calc.py", script)
        return {
            "is_success": exec_res.is_success,
            "output": exec_res.output,
            "duration_ms": int(exec_res.duration_seconds * 1000),
            "script": script,
        }


class ObsidianRoundtableVaultWriterAdapter(VaultNoteWriterPort):
    """Writes distilled Roundtable notes to Obsidian Vault using save_memory / KnowledgeWriteService."""

    def __init__(self, writer_fn: Optional[Callable[[str, str, Optional[str], Sequence[str]], Dict[str, Any]]] = None) -> None:
        self._writer_fn = writer_fn

    def save_roundtable_note(
        self,
        *,
        title: str,
        markdown_content: str,
        ticker: Optional[str],
        tags: Sequence[str],
    ) -> Dict[str, Any]:
        if self._writer_fn is not None:
            return self._writer_fn(title, markdown_content, ticker, tags)
        from tools.archivist.writer import save_memory

        folder = f"20_Stocks/{ticker.strip().upper()}" if ticker else "30_Knowledge_Base/Advisor_Notes"
        res = save_memory.invoke(
            {
                "title": title,
                "content": markdown_content,
                "folder_path": folder,
                "tags": list(tags),
                "entity_type": "advisor_note",
            }
        )
        return {"ok": True, "vault_path": str(res)}


class RoundtableThesisUpdaterAdapter(ThesisUpdaterPort):
    """Revises active stock thesis using an injected ThesisApplicationService."""

    def __init__(self, thesis_service_getter: Callable[[], Any]) -> None:
        self._getter = thesis_service_getter

    def revise_thesis_from_roundtable(
        self,
        *,
        ticker: str,
        claim_statement: str,
        kill_switches: Sequence[str],
    ) -> Dict[str, Any]:
        svc = self._getter()
        existing = svc.get_active_thesis(ticker)
        falsifiers = list(getattr(existing, "falsifiers", ()) or ()) if existing else []
        catalysts = list(getattr(existing, "linked_catalysts", ()) or ()) if existing else []

        ks_note = ""
        if kill_switches:
            ks_note = "\n\n[Kill-Switches from Guru Roundtable]\n" + "\n".join(f"- {k}" for k in kill_switches)
        full_claim = f"{claim_statement.strip()}{ks_note}"

        if existing is None:
            rec = svc.record_new_thesis(
                ticker=ticker,
                claim_statement=full_claim,
                falsifiers=falsifiers,
                catalysts=catalysts,
                author_role="HERMES_ROUNDTABLE",
            )
        else:
            rec = svc.revise_thesis(
                ticker=ticker,
                claim_statement=full_claim,
                falsifiers=falsifiers,
                catalysts=catalysts,
                author_role="HERMES_ROUNDTABLE",
                prior_claim_id=getattr(existing, "claim_id", None),
            )
        return {"claim_id": getattr(rec, "claim_id", ""), "ticker": ticker}
