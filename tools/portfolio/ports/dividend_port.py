from abc import ABC, abstractmethod
from typing import Dict, List, Optional


class DividendHistoryPort(ABC):
    """Port interface for fetching external dividend historical records."""

    @abstractmethod
    def fetch_dividend_history(self, symbols: List[str]) -> Dict[str, List[Dict]]:
        """Fetch dividend payout history for a list of ticker symbols.

        Args:
            symbols: List of equity symbols (e.g. ['AAPL', 'MSFT', 'BDMS.BK'])

        Returns:
            Dict mapping symbol -> list of dividend records, each containing:
            {'date': 'YYYY-MM-DD', 'amount': float, 'currency': 'THB' | 'USD'}
        """
        ...
