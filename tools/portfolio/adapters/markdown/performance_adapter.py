import csv
import io
from datetime import datetime, timedelta
from typing import Optional, List, Dict

from core.logger import get_logger
from tools._atomic_io import _atomic_write_to
from tools.portfolio.domain.validator import validate_portfolio_id
from tools.portfolio.ports.performance_port import PerformanceRepositoryPort
from .paths import get_performance_filepath, _PERFORMANCE_LOG_HEADER
from .repository_adapter import _get_portfolio_lock

log = get_logger(__name__)


class MarkdownPerformanceAdapter(PerformanceRepositoryPort):
    """Markdown Vault storage adapter for Performance Log CSV."""

    def upsert_snapshot(self, portfolio_id: str, row: Dict) -> None:
        pid = validate_portfolio_id(portfolio_id)
        lock = _get_portfolio_lock(pid)
        with lock:
            perf_path = get_performance_filepath(pid)
            perf_path.parent.mkdir(parents=True, exist_ok=True)

            date_val = str(row.get("Date") or datetime.now().strftime("%Y-%m-%d"))
            row_list = [
                date_val,
                f"{float(row.get('Total_NAV', 0.0)):.2f}",
                f"{float(row.get('Total_Cost', 0.0)):.2f}",
                f"{float(row.get('Unrealized_PnL', 0.0)):.2f}",
                f"{float(row.get('Cash_Balance', 0.0)):.2f}",
                f"{float(row.get('Realized_PnL_YTD', 0.0)):.2f}",
                f"{float(row.get('Passive_Income_YTD', 0.0)):.2f}",
            ]

            existing_rows: List[List[str]] = []
            if perf_path.exists() and perf_path.stat().st_size > 0:
                with perf_path.open("r", encoding="utf-8", newline="") as f:
                    reader = csv.reader(f)
                    header_read = False
                    for r in reader:
                        if not header_read:
                            header_read = True
                            if not r or (r and r[0] == "Date"):
                                continue
                        if r and len(r) >= 5:
                            if len(r) < len(_PERFORMANCE_LOG_HEADER):
                                r.extend([""] * (len(_PERFORMANCE_LOG_HEADER) - len(r)))
                            existing_rows.append(r[:len(_PERFORMANCE_LOG_HEADER)])

            replaced = False
            for idx, r in enumerate(existing_rows):
                if r[0] == date_val:
                    existing_rows[idx] = row_list
                    replaced = True
                    break
            if not replaced:
                existing_rows.append(row_list)

            output = io.StringIO()
            writer = csv.writer(output, lineterminator="\n")
            writer.writerow(_PERFORMANCE_LOG_HEADER)
            writer.writerows(existing_rows)
            _atomic_write_to(perf_path, output.getvalue())

    def read_history(self, portfolio_id: str = "default", days: Optional[int] = None) -> List[Dict]:
        pid = validate_portfolio_id(portfolio_id)
        lock = _get_portfolio_lock(pid)
        with lock:
            perf_path = get_performance_filepath(pid)
            if not perf_path.exists() or perf_path.stat().st_size == 0:
                return []

            cutoff_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d") if days else None
            rows: List[Dict] = []
            with perf_path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    d = r.get("Date", "")
                    if cutoff_date and d < cutoff_date:
                        continue
                    rows.append(dict(r))
            return rows
