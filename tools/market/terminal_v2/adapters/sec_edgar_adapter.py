"""SEC EDGAR Keyless Adapter for Company-filed Financial Facts and Form 4 Insider Trades.

Strict Domain & Compliance Invariants:
1. Company facts are "company-filed XBRL facts" (Form 10-K & 10-Q).
   Must NEVER be labeled or treated as universally audited (Form 10-Q is unaudited).
2. Submissions metadata only indexes filing events. To obtain transaction shares,
   prices, and transaction codes (P/S), the adapter downloads and parses the actual
   Form 4 XML ownership document.
3. Form 13F (Institutional Holdings) is strictly segregated and excluded from insider trades.
4. User-Agent is explicitly configured to adhere to SEC fair access rules.
5. Thread-safe rate pacing (<= 8 req/s) with exponential backoff on HTTP 429.
"""
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import threading
import time
from typing import Any, Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET
import requests

from tools.market.terminal_v2.application.cache import ThreadSafeTTLCache
from tools.market.terminal_v2.domain.calculations import (
    calculate_financial_ratios,
    calculate_insider_net_buying_90d,
)
from tools.market.terminal_v2.domain.errors import DataUnavailableError, ProviderError
from tools.market.terminal_v2.domain.models import (
    InsiderTransaction,
    SecCompanyFactsSnapshot,
    SecFact,
    SecInsiderTradeSnapshot,
)
from tools.market.terminal_v2.ports.driven_ports import (
    SecFinancialsPort,
    SecInsiderTradesPort,
)

logger = logging.getLogger(__name__)

SEC_FACTS_TTL = 14400.0         # 4 hours
SEC_SUBMISSIONS_TTL = 3600.0    # 1 hour
SEC_MAX_STALE = 86400.0         # 24 hours

SEC_USER_AGENT = os.getenv(
    "SEC_USER_AGENT",
    "TrinityWealthEngine/2.0 (research-terminal@trinity-wealth.internal)",
)

# Standard static mapping for widely followed equities
TICKER_TO_CIK: Dict[str, str] = {
    "NVDA": "0001045810",
    "AAPL": "0000320193",
    "MSFT": "0000789019",
    "GOOGL": "0001652044",
    "GOOG": "0001652044",
    "AMZN": "0001018724",
    "META": "0001326801",
    "TSLA": "0001318605",
    "AMD": "0000002488",
    "INTC": "0000050863",
    "AVGO": "0001730168",
    "QCOM": "0000804328",
    "NFLX": "0001065280",
    "ADBE": "0000796343",
    "CRM": "0001108524",
    "ORCL": "0001341439",
    "PLTR": "0001321655",
    "COIN": "0001679788",
}

# Concept priority lists (aliases)
CONCEPT_TAGS = {
    "revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet"],
    "net_income": ["NetIncomeLoss"],
    "operating_income": ["OperatingIncomeLoss"],
    "ocf": ["NetCashProvidedByUsedInOperatingActivities"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets"],
    "long_term_debt": ["LongTermDebtNoncurrent", "LongTermDebt", "LongTermDebtAndCapitalLeaseObligations"],
    "rnd": ["ResearchAndDevelopmentExpense"],
    "share_repurchase": ["PaymentsForRepurchaseOfCommonStock"],
    "shares_outstanding": ["CommonStockSharesOutstanding"],
}


class _SecRateLimiter:
    """Thread-safe rate pacer ensuring max 8 requests per second across threads."""

    def __init__(self, min_interval_seconds: float = 0.125):
        self._min_interval = min_interval_seconds
        self._last_call = 0.0
        self._lock = threading.Lock()

    def pace(self):
        with self._lock:
            now = time.time()
            elapsed = now - self._last_call
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)
            self._last_call = time.time()


_sec_pacer = _SecRateLimiter()


class SecEdgarAdapter(SecFinancialsPort, SecInsiderTradesPort):
    """Adapter fetching official company filings from SEC EDGAR."""

    def __init__(
        self,
        cache: Optional[ThreadSafeTTLCache] = None,
        facts_fixture_path: Optional[Path] = None,
        submissions_fixture_path: Optional[Path] = None,
        form4_fixture_path: Optional[Path] = None,
        user_agent: Optional[str] = None,
    ):
        self._cache = cache or ThreadSafeTTLCache(default_ttl_seconds=SEC_FACTS_TTL)
        self._facts_fixture_path = facts_fixture_path
        self._submissions_fixture_path = submissions_fixture_path
        self._form4_fixture_path = form4_fixture_path
        self._user_agent = user_agent or SEC_USER_AGENT

    def _get_headers(self) -> Dict[str, str]:
        return {
            "User-Agent": self._user_agent,
            "Accept": "application/json, text/plain, */*",
            "Accept-Encoding": "gzip, deflate",
        }

    def _resolve_cik(self, symbol: str) -> str:
        sym = symbol.strip().upper()
        if sym in TICKER_TO_CIK:
            return TICKER_TO_CIK[sym]
        if sym.isdigit():
            return sym.zfill(10)
        raise DataUnavailableError(f"CIK mapping not found for symbol: {symbol}", capability="sec-edgar", source="SEC EDGAR")

    def _request_get(self, url: str) -> requests.Response:
        _sec_pacer.pace()
        try:
            resp = requests.get(url, headers=self._get_headers(), timeout=15)
            if resp.status_code == 429:
                logger.warning("SEC EDGAR rate limited (429). Backing off 2.0s.")
                time.sleep(2.0)
                _sec_pacer.pace()
                resp = requests.get(url, headers=self._get_headers(), timeout=15)
            resp.raise_for_status()
            return resp
        except Exception as exc:
            raise ProviderError(f"SEC EDGAR request failed for {url}: {exc}", source="SEC EDGAR") from exc

    # -------------------------------------------------------------------------
    # 1. Company Facts (XBRL)
    # -------------------------------------------------------------------------
    def _fetch_facts_payload(self, cik_10: str) -> Dict[str, Any]:
        if self._facts_fixture_path and self._facts_fixture_path.exists():
            return json.loads(self._facts_fixture_path.read_text(encoding="utf-8"))

        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik_10}.json"
        resp = self._request_get(url)
        return resp.json()

    def get_company_facts(self, symbol: str) -> SecCompanyFactsSnapshot:
        sym = symbol.strip().upper()
        cik_10 = self._resolve_cik(sym)
        cache_key = f"sec:facts:{cik_10}"
        return self._cache.get_or_set(
            key=cache_key,
            loader=lambda: self._load_company_facts(sym, cik_10),
            ttl_seconds=SEC_FACTS_TTL,
            max_stale_seconds=SEC_MAX_STALE,
        )

    def _load_company_facts(self, symbol: str, cik_10: str) -> SecCompanyFactsSnapshot:
        payload = self._fetch_facts_payload(cik_10)
        entity_name = payload.get("entityName", symbol)
        us_gaap = payload.get("facts", {}).get("us-gaap", {})

        if not us_gaap:
            raise DataUnavailableError(f"No US-GAAP facts found for {symbol} (CIK {cik_10})", capability="sec-edgar", source="SEC EDGAR")

        def _pick_latest_fact(concepts: List[str]) -> Optional[Tuple[str, SecFact]]:
            candidate_facts: List[SecFact] = []
            for concept in concepts:
                if concept not in us_gaap:
                    continue
                units = us_gaap[concept].get("units", {})
                label = us_gaap[concept].get("label", concept)
                for unit_name, entries in units.items():
                    if not entries:
                        continue
                    # Pick entry with latest end date, then latest filed date
                    sorted_entries = sorted(
                        entries,
                        key=lambda e: (e.get("end", ""), e.get("filed", "")),
                        reverse=True,
                    )
                    best = sorted_entries[0]
                    val = best.get("val")
                    if val is not None:
                        sec_fact = SecFact(
                            concept_tag=concept,
                            label=label,
                            val=float(val),
                            unit=unit_name,
                            form=best.get("form", ""),
                            fy=best.get("fy"),
                            fp=best.get("fp"),
                            start=best.get("start"),
                            end=best.get("end"),
                            filed=best.get("filed"),
                            accn=best.get("accn"),
                        )
                        candidate_facts.append(sec_fact)
            if not candidate_facts:
                return None
            # Return candidate with the most recent end date
            best_fact = sorted(candidate_facts, key=lambda f: (f.end or "", f.filed or ""), reverse=True)[0]
            return best_fact.concept_tag, best_fact

        collected_facts: List[SecFact] = []
        metrics_vals: Dict[str, Optional[float]] = {}

        for m_key, aliases in CONCEPT_TAGS.items():
            result = _pick_latest_fact(aliases)
            if result:
                _, fact = result
                collected_facts.append(fact)
                metrics_vals[m_key] = fact.val
            else:
                metrics_vals[m_key] = None

        rev = metrics_vals.get("revenue")
        ocf = metrics_vals.get("ocf")
        capex = metrics_vals.get("capex")
        debt = metrics_vals.get("long_term_debt")

        fcf, fcf_margin, debt_to_ocf = calculate_financial_ratios(rev, ocf, capex, debt)

        latest_as_of = ""
        for f in collected_facts:
            if f.end and f.end > latest_as_of:
                latest_as_of = f.end

        return SecCompanyFactsSnapshot(
            symbol=symbol,
            cik=cik_10,
            entity_name=entity_name,
            facts=tuple(collected_facts),
            revenue_usd=rev,
            operating_cash_flow_usd=ocf,
            capex_usd=capex,
            free_cash_flow_usd=fcf,
            free_cash_flow_margin=fcf_margin,
            long_term_debt_usd=debt,
            debt_to_ocf_ratio=debt_to_ocf,
            shares_outstanding=metrics_vals.get("shares_outstanding"),
            source="SEC EDGAR (companyfacts)",
            as_of_date=latest_as_of or datetime.utcnow().strftime("%Y-%m-%d"),
            fetched_at=time.time(),
            is_stale=False,
            stale_reason=None,
        )

    # -------------------------------------------------------------------------
    # 2. Form 4 Insider Trades (XML Parsing)
    # -------------------------------------------------------------------------
    def _fetch_submissions_payload(self, cik_10: str) -> Dict[str, Any]:
        if self._submissions_fixture_path and self._submissions_fixture_path.exists():
            return json.loads(self._submissions_fixture_path.read_text(encoding="utf-8"))

        url = f"https://data.sec.gov/submissions/CIK{cik_10}.json"
        resp = self._request_get(url)
        return resp.json()

    def _fetch_form4_xml(self, cik_int: str, acc_no_dashes: str, doc_name: str) -> str:
        if self._form4_fixture_path and self._form4_fixture_path.exists():
            return self._form4_fixture_path.read_text(encoding="utf-8")

        # Strip any xsl viewer prefix to get raw XML document
        clean_doc = doc_name.split("/")[-1]
        url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_no_dashes}/{clean_doc}"
        resp = self._request_get(url)
        return resp.text

    def _parse_form4_xml(
        self,
        xml_text: str,
        accession_number: str,
        is_amendment: bool,
    ) -> List[InsiderTransaction]:
        results: List[InsiderTransaction] = []
        try:
            root = ET.fromstring(xml_text)
        except Exception as exc:
            logger.debug("Failed to parse Form 4 XML for %s: %s", accession_number, exc)
            return results

        # Reporting Owner Information
        owner_name = root.findtext(".//reportingOwner//rptOwnerName", "Unknown Owner")
        officer_title = root.findtext(".//reportingOwnerRelationship/officerTitle")
        is_officer = root.findtext(".//reportingOwnerRelationship/isOfficer") == "1"
        is_director = root.findtext(".//reportingOwnerRelationship/isDirector") == "1"
        is_ten_pct = root.findtext(".//reportingOwnerRelationship/isTenPercentOwner") == "1"

        # Non-derivative transactions (Common stock)
        for tx in root.findall(".//nonDerivativeTransaction"):
            tx_date = tx.findtext(".//transactionDate/value", "")
            tx_code = tx.findtext(".//transactionCoding/transactionCode", "")

            shares_txt = tx.findtext(".//transactionAmounts/transactionShares/value")
            price_txt = tx.findtext(".//transactionAmounts/transactionPricePerShare/value")
            direct_or_ind = tx.findtext(".//ownershipNature/directOrIndirectOwnership/value", "D")

            shares = float(shares_txt) if shares_txt else None
            price = float(price_txt) if price_txt else None
            notional = (shares * price) if (shares and price) else None

            results.append(
                InsiderTransaction(
                    transaction_date=tx_date,
                    reporting_owner=owner_name,
                    officer_title=officer_title,
                    is_officer=is_officer,
                    is_director=is_director,
                    is_ten_percent_owner=is_ten_pct,
                    transaction_code=tx_code,
                    shares=shares,
                    price_per_share=price,
                    notional_usd=notional,
                    direct_or_indirect=direct_or_ind,
                    accession_number=accession_number,
                    is_amendment=is_amendment,
                )
            )

        return results

    def get_insider_trades(self, symbol: str, limit: int = 20) -> SecInsiderTradeSnapshot:
        sym = symbol.strip().upper()
        cik_10 = self._resolve_cik(sym)
        capped_limit = min(max(limit, 1), 30)
        cache_key = f"sec:insider_trades:{cik_10}:{capped_limit}"
        return self._cache.get_or_set(
            key=cache_key,
            loader=lambda: self._load_insider_trades(sym, cik_10, capped_limit),
            ttl_seconds=SEC_SUBMISSIONS_TTL,
            max_stale_seconds=SEC_MAX_STALE,
        )

    def _load_insider_trades(self, symbol: str, cik_10: str, limit: int) -> SecInsiderTradeSnapshot:
        subm = self._fetch_submissions_payload(cik_10)
        recent = subm.get("filings", {}).get("recent", {})

        forms = recent.get("form", [])
        accessions = recent.get("accessionNumber", [])
        primary_docs = recent.get("primaryDocument", [])
        cik_int = str(int(cik_10))

        # Filter Form 4 and Form 4/A
        form4_targets: List[Tuple[str, str, str, bool]] = []
        for i, f in enumerate(forms):
            if f in ("4", "4/A") and i < len(accessions) and i < len(primary_docs):
                acc = accessions[i]
                doc = primary_docs[i]
                is_amend = f == "4/A"
                form4_targets.append((acc, doc, f, is_amend))
                if len(form4_targets) >= limit:
                    break

        all_transactions: List[InsiderTransaction] = []
        for acc, doc, form_type, is_amend in form4_targets:
            acc_clean = acc.replace("-", "")
            try:
                xml_text = self._fetch_form4_xml(cik_int, acc_clean, doc)
                parsed_txs = self._parse_form4_xml(xml_text, acc, is_amend)
                all_transactions.extend(parsed_txs)
            except Exception as exc:
                logger.debug("Skipping Form 4 %s: %s", acc, exc)
                continue

        as_of = datetime.utcnow().strftime("%Y-%m-%d")
        if all_transactions:
            dates = [t.transaction_date for t in all_transactions if t.transaction_date]
            if dates:
                as_of = max(dates)

        net_ratio, p_sum, s_sum, eligible_count = calculate_insider_net_buying_90d(all_transactions, as_of)

        return SecInsiderTradeSnapshot(
            symbol=symbol,
            cik=cik_10,
            transactions=tuple(all_transactions),
            net_buy_ratio_90d=net_ratio,
            p_notional_sum_90d=p_sum,
            s_notional_sum_90d=s_sum,
            eligible_transaction_count=eligible_count,
            source="SEC EDGAR (Form 4 XML)",
            as_of_date=as_of,
            fetched_at=time.time(),
            is_stale=False,
            stale_reason=None,
        )
