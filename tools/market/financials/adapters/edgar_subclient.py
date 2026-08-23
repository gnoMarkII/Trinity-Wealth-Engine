"""Sub-client for SEC EDGAR Filings Query and Retrieval."""
import logging
import os
from typing import Any, Callable, Optional

log = logging.getLogger(__name__)


class EdgarSubclient:
    """Sub-client สำหรับการดึงเอกสาร SEC EDGAR (10-K, 10-Q, 8-K) โดยรองรับ Dependency Injection สำหรับ Test"""

    def __init__(self, company_factory: Optional[Callable[[str], Any]] = None):
        self._company_factory = company_factory

    def init_identity(self) -> bool:
        """ตั้งค่าตัวตน SEC User-Agent Identity ตามมาตรฐาน SEC Fair Access"""
        if self._company_factory is not None:
            return True
        ua = os.environ.get("SEC_EDGAR_USER_AGENT", "").strip()
        if not ua or "@" not in ua:
            log.warning("SEC_EDGAR_USER_AGENT is not configured or invalid (requires Name and Email). Skipping EDGAR.")
            return False
        try:
            from edgar import set_identity
            set_identity(ua)
            return True
        except Exception as e:
            log.warning("Failed to initialize SEC EDGAR identity: %s", e)
            return False

    def get_company_filings(self, ticker: str) -> Optional[tuple[list[Any], list[Any], list[Any]]]:
        """ดึงรายการ Filings (10-K, 10-Q, 8-K) ของบริษัท

        Returns:
            Optional[tuple[list[10-K], list[10-Q], list[8-K]]]: None หากดึงไม่สำเร็จ
        """
        if not self.init_identity():
            return None

        try:
            if self._company_factory:
                company = self._company_factory(ticker)
            else:
                from edgar import Company
                company = Company(ticker)

            filings_10k_raw = company.get_filings(form=["10-K", "10-K/A"])
            filings_10q_raw = company.get_filings(form=["10-Q", "10-Q/A"])
            filings_8k_raw = company.get_filings(form=["8-K", "8-K/A"])

            filings_10k = list(filings_10k_raw) if filings_10k_raw else []
            filings_10q = list(filings_10q_raw) if filings_10q_raw else []
            filings_8k = list(filings_8k_raw) if filings_8k_raw else []

            return filings_10k, filings_10q, filings_8k
        except Exception as e:
            log.warning("Failed to fetch SEC EDGAR filings for %s: %s", ticker, e)
            return None
