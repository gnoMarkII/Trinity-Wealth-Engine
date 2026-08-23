"""Financial Provider Port Interface."""
from abc import ABC, abstractmethod
from tools.market.financials.domain.models import FinancialStatementsDTO


class FinancialStatementProviderPort(ABC):
    """Outbound Port: Ingestion Contract for Financial Statements Data Sources"""

    @abstractmethod
    def fetch_statements(self, ticker: str, provider_symbol: str) -> FinancialStatementsDTO:
        """ดึงและประกอบร่างงบการเงินทั้งหมด (Income, BS, CF, Non-GAAP, Ratios, Warnings) ในรอบเดียว"""
        ...
