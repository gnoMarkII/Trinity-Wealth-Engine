"""Financial Cache Port Interface and Domain Data Object."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
from tools.market.financials.domain.models import FinancialStatementsDTO


@dataclass
class CacheEntry:
    statements: FinancialStatementsDTO
    provider: str
    synced_at: float


class FinancialCachePort(ABC):
    """Outbound Port: Persistence and Cache Access for Financial Statements"""

    @abstractmethod
    def get(self, market: str, provider_symbol: str) -> Optional[CacheEntry]:
        """ดึง CacheEntry โดย Adapter จะเป็นผู้ Deserialize JSON กลับมาเป็น FinancialStatementsDTO"""
        ...

    @abstractmethod
    def save(self, market: str, provider_symbol: str, entry: CacheEntry) -> None:
        """บันทึก CacheEntry ลง Persistence โดย Adapter จะเป็นผู้ Serialize DTO สู่ JSON"""
        ...

    @abstractmethod
    def delete(self, market: str, provider_symbol: str) -> None:
        """ลบแคชของสัญลักษณ์ที่ระบุ"""
        ...
