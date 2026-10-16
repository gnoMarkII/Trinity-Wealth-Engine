"""Driving and driven ports for PVD statement synchronization."""
from dataclasses import dataclass
from typing import Optional, Protocol

from tools.portfolio.domain.pvd_models import PvdStatement


class PvdParserPort(Protocol):
    def parse_statement(self, pdf_bytes: bytes, password: Optional[str] = None) -> PvdStatement: ...


class PvdDocumentArchivePort(Protocol):
    def archive(self, pdf_bytes: bytes, *, portfolio_id: str, statement: PvdStatement) -> str: ...

    def read(self, *, portfolio_id: str, metadata: dict) -> bytes: ...


@dataclass(frozen=True)
class PvdScanResult:
    scan_id: Optional[str]
    can_commit: bool
    status: str
    message: str
    statement: Optional[dict] = None
    holdings: tuple[dict, ...] = ()
    current_nav: Optional[str] = None
    new_nav: Optional[str] = None
    delta_nav: Optional[str] = None
    vesting_percent: str = "100"
    vested_amount: Optional[str] = None
    archive_id: Optional[str] = None


@dataclass(frozen=True)
class PvdCommitResult:
    state: dict
    status: str
    message: str
    archive_id: Optional[str] = None
