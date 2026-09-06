import csv
import hashlib
import json
import os
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Union

import frontmatter
from filelock import FileLock, Timeout

from core.logger import get_logger
from tools._atomic_io import _atomic_write_to
from decimal import Decimal
from tools.portfolio.domain.constants import (
    CASH_THB_SYMBOL,
    CASH_USD_SYMBOL,
    _TOP_LEVEL_KEY_ORDER,
    _MONEY_DP,
)
from tools.portfolio.domain.models import (
    PortfolioState,
    Holding,
    Summary,
    PortfolioMeta,
    _now_iso,
    default_allocation_targets,
    MONEY_QUANTUM,
    quantize_decimal,
)
from tools.portfolio.domain.ledger_change import LedgerChange
from tools.portfolio.domain.mutation import PortfolioMutation
from tools.portfolio.domain.events import SystemJournalEvent
from tools.portfolio.domain.errors import RecoveryConflictError, PortfolioNotFoundError
from tools.portfolio.domain.calculations import recalc_all
from tools.portfolio.domain.validator import validate_portfolio_id
from tools.portfolio.ports.repository_port import PortfolioRepositoryPort, PortfolioUnitOfWork
from .paths import (
    get_portfolio_dir,
    get_portfolio_filepath,
    get_portfolio_lock_path,
    get_trades_log_filepath,
    get_pending_manifest_path,
    get_holdings_dir,
    get_journal_filepath,
    _TRADES_LOG_HEADER,
    _LOCK_TIMEOUT,
)
from .journal_format import inject_journal_wikilinks

log = get_logger(__name__)

_locks_registry_lock = threading.Lock()
_locks: Dict[str, FileLock] = {}


def _get_portfolio_lock(portfolio_id: str = "default") -> FileLock:
    pid = validate_portfolio_id(portfolio_id)
    lock_path = get_portfolio_lock_path(pid)
    with _locks_registry_lock:
        if pid not in _locks:
            _locks[pid] = FileLock(lock_path, timeout=_LOCK_TIMEOUT)
        return _locks[pid]


def _portfolio_exists(portfolio_id: str) -> bool:
    pid = validate_portfolio_id(portfolio_id)
    if pid == "default":
        return True
    return get_portfolio_filepath(pid).exists()


def _compute_sha256(filepath: Path) -> str:
    if not filepath.exists():
        return ""
    h = hashlib.sha256()
    with filepath.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def _sanitize_csv_field(value: str) -> str:
    """Sanitize CSV/Formula injection, preserving valid numbers."""
    if not value:
        return value
    try:
        float(value)
        return value
    except ValueError:
        pass
    if value[0] in ("=", "+", "-", "@", "\t", "\r"):
        return "'" + value
    return value


def _desanitize_csv_field(value: str) -> str:
    """Reverse CSV/Formula injection escaping on read."""
    if value and value.startswith("'") and len(value) > 1 and value[1] in ("=", "+", "-", "@", "\t", "\r"):
        return value[1:]
    return value


def _initial_state() -> PortfolioState:
    return PortfolioState(
        last_updated=_now_iso(),
        summary=Summary(),
        fx_rates={"USDTHB": 36.5},
        holdings=[
            Holding(symbol=CASH_THB_SYMBOL, asset_type="Cash", units=0.0, market_value_thb=0.0),
            Holding(symbol=CASH_USD_SYMBOL, asset_type="Cash", units=0.0, market_value_thb=0.0),
        ],
    )


def _holding_to_md(h: Holding) -> str:
    """Generate YAML frontmatter markdown for holding sidecar note."""
    if h.avg_cost_usd is not None:
        currency = "USD"
        avg_cost = h.avg_cost_usd
        current_price = h.current_price_usd
    else:
        currency = "THB"
        avg_cost = h.avg_cost_thb
        current_price = h.current_price_thb

    lines = [
        "---",
        f"schema_version: {h.schema_version}",
        "entity_type: holding",
        "derived: true",
        f"symbol: {h.symbol}",
        f"asset_type: {h.asset_type}",
        f"status: {h.status}",
    ]
    if h.archived_at is not None:
        lines.append(f'archived_at: "{h.archived_at}"')
    lines.append(f"currency: {currency}")
    lines.append(f"units: {h.units}")
    if avg_cost is not None:
        lines.append(f"avg_cost: {avg_cost}")
    if current_price is not None:
        lines.append(f"current_price: {current_price}")
    lines.append(f"market_value_thb: {h.market_value_thb}")
    if h.unrealized_pnl_percent is not None:
        lines.append(f"unrealized_pnl_pct: {h.unrealized_pnl_percent}")
    if h.accumulated_dividend_thb is not None:
        lines.append(f"dividend_thb: {h.accumulated_dividend_thb}")
    lines.append("---")
    lines.append("")
    lines.append(f"# {h.symbol}")
    lines.append("")
    lines.append("> [!CAUTION]")
    lines.append("> **ไฟล์นี้ถูกสร้างและอัปเดตอัตโนมัติโดยระบบ**")
    lines.append("> กรุณาอย่าบันทึกโน้ตส่วนตัวที่นี่เพราะจะถูกเขียนทับเมื่อระบบทำการ Sync")
    lines.append("")
    return "\n".join(lines)


class MarkdownPortfolioUnitOfWork(PortfolioUnitOfWork):
    """Concrete Unit of Work for Markdown Repository under FileLock with Decision Matrix Recovery."""

    supports_staged_mutations = True

    def __init__(self, repo: "MarkdownVaultRepositoryAdapter", portfolio_id: str = "default"):
        self.repo = repo
        self.portfolio_id = validate_portfolio_id(portfolio_id)
        self.lock = _get_portfolio_lock(self.portfolio_id)
        self.acquired = False
        self._cached_post: Optional[frontmatter.Post] = None
        self._cached_state: Optional[PortfolioState] = None

    def __enter__(self) -> "MarkdownPortfolioUnitOfWork":
        self.lock.acquire()
        self.acquired = True
        # Perform crash recovery reconciliation before allowing operations
        self.repo._reconcile_pending_commit(self.portfolio_id)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> Optional[bool]:
        if self.acquired:
            self.lock.release()
            self.acquired = False
        return None

    def load_state(self) -> PortfolioState:
        post, state = self.repo._load_or_init_locked(self.portfolio_id)
        self._cached_post = post
        self._cached_state = state
        return state

    def read_trade_log_locked(self) -> List[Dict]:
        fpath = get_trades_log_filepath(self.portfolio_id)
        raw_rows = _read_and_migrate_trade_log_locked(fpath)
        rows: List[Dict] = []
        for r in raw_rows:
            item_dict = {k.lower(): v for k, v in r.items()}
            item_dict.update({k: v for k, v in r.items()})
            rows.append(item_dict)
        return rows

    def commit(
        self,
        state: PortfolioState,
        ledger_change: Optional[Union[LedgerChange, PortfolioMutation]] = None,
    ) -> None:
        self.repo._commit_locked(self.portfolio_id, state, ledger_change)

    def rollback(self) -> None:
        manifest_path = get_pending_manifest_path(self.portfolio_id)
        if manifest_path.exists():
            try:
                with manifest_path.open("r", encoding="utf-8") as f:
                    manifest = json.load(f)
                self.repo._cleanup_staged_files(manifest)
                manifest_path.unlink(missing_ok=True)
            except Exception as e:
                log.warning("Failed to rollback manifest for %s: %s", self.portfolio_id, e)


def _read_and_migrate_trade_log_locked(fpath: Path) -> List[Dict[str, str]]:
    if not fpath.exists():
        return []
    with fpath.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        all_rows = list(reader)
    if not all_rows:
        return []

    header = [h.strip() for h in all_rows[0]]
    needs_rewrite = False
    migrated_rows: List[Dict[str, str]] = []

    if header != _TRADES_LOG_HEADER:
        needs_rewrite = True

    # Read rows into dicts
    with fpath.open("r", encoding="utf-8", newline="") as f:
        dreader = csv.DictReader(f)
        for r in dreader:
            if not r or not any(r.values()):
                continue
            row_dict = dict(r)

            # Check Transaction_ID
            if not row_dict.get("Transaction_ID"):
                row_dict["Transaction_ID"] = f"tx_{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}"
                needs_rewrite = True

            # Calculate Gross_Amount and Net_Amount in Trade Currency if missing
            units_raw = row_dict.get("Units") or "0"
            price_raw = row_dict.get("Price") or "0"
            try:
                u_dec = Decimal(str(units_raw))
                p_dec = Decimal(str(price_raw))
                computed_amt = str(quantize_decimal(u_dec * p_dec, MONEY_QUANTUM))
            except Exception:
                computed_amt = "0.00"

            if not row_dict.get("Gross_Amount"):
                row_dict["Gross_Amount"] = computed_amt
                needs_rewrite = True

            if not row_dict.get("Net_Amount"):
                row_dict["Net_Amount"] = computed_amt
                needs_rewrite = True

            if not row_dict.get("Commission"):
                row_dict["Commission"] = "0.00"
                needs_rewrite = True
            if not row_dict.get("VAT"):
                row_dict["VAT"] = "0.00"
                needs_rewrite = True
            if not row_dict.get("Other_Fees"):
                row_dict["Other_Fees"] = "0.00"
                needs_rewrite = True

            curr = row_dict.get("Currency") or "THB"
            if not row_dict.get("Fee_Currency"):
                row_dict["Fee_Currency"] = curr
                needs_rewrite = True

            if not row_dict.get("Confirmation_No"):
                row_dict["Confirmation_No"] = ""
            if not row_dict.get("Settlement_Date"):
                row_dict["Settlement_Date"] = ""
            if not row_dict.get("Fingerprint"):
                row_dict["Fingerprint"] = ""

            # Critical: Legacy rows were cash adjusted and manual
            if not row_dict.get("Cash_Adjusted"):
                row_dict["Cash_Adjusted"] = "YES"
                needs_rewrite = True
            if not row_dict.get("Source"):
                row_dict["Source"] = "MANUAL"
                needs_rewrite = True
            if not row_dict.get("Related_Transaction_ID"):
                row_dict["Related_Transaction_ID"] = ""

            canonical_row = {col: _desanitize_csv_field(str(row_dict.get(col, "") or "")) for col in _TRADES_LOG_HEADER}
            migrated_rows.append(canonical_row)

    if needs_rewrite:
        with fpath.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_TRADES_LOG_HEADER, lineterminator="\n")
            writer.writeheader()
            for r in migrated_rows:
                writer.writerow(r)
            f.flush()
            os.fsync(f.fileno())

    return migrated_rows


class MarkdownVaultRepositoryAdapter(PortfolioRepositoryPort):
    """Authoritative Markdown Vault Storage Adapter."""

    def unit_of_work(self, portfolio_id: str = "default") -> PortfolioUnitOfWork:
        return MarkdownPortfolioUnitOfWork(self, portfolio_id=portfolio_id)

    def load_state(self, portfolio_id: str = "default") -> PortfolioState:
        pid = validate_portfolio_id(portfolio_id)
        lock = _get_portfolio_lock(pid)
        with lock:
            self._reconcile_pending_commit(pid)
            _, state = self._load_or_init_locked(pid)
            return state

    def read_trade_log(self, portfolio_id: str = "default", symbol: Optional[str] = None) -> List[Dict]:
        pid = validate_portfolio_id(portfolio_id)
        lock = _get_portfolio_lock(pid)
        with lock:
            self._reconcile_pending_commit(pid)
            fpath = get_trades_log_filepath(pid)
            raw_rows = _read_and_migrate_trade_log_locked(fpath)
            rows: List[Dict] = []
            for r in raw_rows:
                sym_val = r.get("Symbol") or r.get("symbol") or ""
                if symbol and sym_val.upper() != symbol.strip().upper():
                    continue
                item_dict = {k.lower(): v for k, v in r.items()}
                item_dict.update({k: v for k, v in r.items()})
                rows.append(item_dict)
            return rows

    def backup_and_reset_clean_slate(self, portfolio_id: str = "default") -> PortfolioState:
        pid = validate_portfolio_id(portfolio_id)
        with self.unit_of_work(pid) as uow:
            state = uow.load_state()

            # Create backup before wiping
            portfolio_file = get_portfolio_filepath(pid)
            holdings_dir = get_holdings_dir(pid)
            backups_dir = portfolio_file.parent / ".backups"
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            backup_dest = backups_dir / timestamp

            try:
                backup_dest.mkdir(parents=True, exist_ok=True)
                if portfolio_file.exists():
                    shutil.copy2(portfolio_file, backup_dest / portfolio_file.name)
                if holdings_dir.exists():
                    dest_holdings = backup_dest / "Holdings"
                    dest_holdings.mkdir(parents=True, exist_ok=True)
                    for f in holdings_dir.glob("*.md"):
                        shutil.copy2(f, dest_holdings / f.name)
            except Exception as e:
                raise ValueError(f"สำรองข้อมูลก่อนล้างพอร์ตไม่สำเร็จ: {e}")

            # Clean sidecars
            if holdings_dir.exists():
                for f in holdings_dir.glob("*.md"):
                    try:
                        f.unlink(missing_ok=True)
                    except Exception:
                        pass

            new_state = PortfolioState(
                last_updated=_now_iso(),
                allocation_targets=default_allocation_targets(),
                fx_rates={"USDTHB": 36.5},
                holdings=[],
            )
            uow.commit(new_state, LedgerChange(kind="replace_all", rows=[]))
            return new_state

    def list_portfolios(self) -> List[PortfolioMeta]:
        portfolios_dir = get_portfolio_dir("default").parent
        default_name = "พอร์ตลงทุนหลัก"
        default_fpath = get_portfolio_filepath("default")
        if default_fpath.exists():
            try:
                with default_fpath.open("r", encoding="utf-8") as f:
                    post = frontmatter.load(f)
                    if post.metadata and post.metadata.get("name"):
                        default_name = post.metadata.get("name")
            except Exception as e:
                log.warning("Failed to load default portfolio meta: %s", e)

        results: List[PortfolioMeta] = [
            PortfolioMeta(id="default", name=default_name, is_default=True)
        ]

        if portfolios_dir.exists():
            for pdir in portfolios_dir.iterdir():
                if not pdir.is_dir():
                    continue
                pid = pdir.name
                if pid == "default":
                    continue
                fpath = pdir / "Portfolio_Holdings.md"
                if not fpath.exists():
                    continue
                name = pid
                try:
                    with fpath.open("r", encoding="utf-8") as f:
                        post = frontmatter.load(f)
                        if post.metadata and post.metadata.get("name"):
                            name = post.metadata.get("name")
                except Exception as e:
                    log.warning("Failed to load portfolio meta for %s: %s", pid, e)
                results.append(PortfolioMeta(id=pid, name=name, is_default=False))

        return results

    def create_portfolio(self, name: str, portfolio_id: Optional[str] = None) -> PortfolioMeta:
        clean_name = (name or "").strip()
        if not clean_name:
            raise ValueError("ชื่อพอร์ตต้องไม่ว่างเปล่า")
        if portfolio_id:
            pid = validate_portfolio_id(portfolio_id)
            if pid == "default":
                raise ValueError("ไม่สามารถใช้ id 'default' ในการสร้างพอร์ตใหม่ได้")
        else:
            base_id = clean_name.lower().replace(" ", "_")
            ascii_chars = "".join(c for c in base_id if c.isascii() and (c.isalnum() or c in ("_", "-"))).strip("_")
            pid = ascii_chars or f"port_{uuid.uuid4().hex[:6]}"
            if pid == "default":
                pid = f"port_{pid}"

        fpath = get_portfolio_filepath(pid)
        if fpath.exists():
            raise ValueError(f"พอร์ตไอดี '{pid}' มีอยู่แล้วในระบบ")

        pdir = get_portfolio_dir(pid)
        pdir.mkdir(parents=True, exist_ok=True)
        post = frontmatter.Post(content="", metadata={"name": clean_name})
        state = _initial_state()
        state.name = clean_name
        self._commit_locked(pid, state, LedgerChange(kind="unchanged"))
        return PortfolioMeta(id=pid, name=clean_name, is_default=False)

    def delete_portfolio(self, portfolio_id: str) -> None:
        pid = validate_portfolio_id(portfolio_id)
        if pid == "default":
            raise ValueError("ไม่สามารถลบพอร์ตหลัก (default) ได้")
        pdir = get_portfolio_dir(pid)
        if not pdir.exists():
            raise PortfolioNotFoundError(f"ไม่พบพอร์ตไอดี '{pid}' ในระบบ")
        shutil.rmtree(pdir, ignore_errors=True)

    def rename_portfolio(self, portfolio_id: str, new_name: str) -> PortfolioMeta:
        pid = validate_portfolio_id(portfolio_id)
        clean_name = (new_name or "").strip()
        if not clean_name:
            raise ValueError("ชื่อพอร์ตต้องไม่ว่างเปล่า")
        with self.unit_of_work(pid) as uow:
            state = uow.load_state()
            state.name = clean_name
            uow.commit(state, LedgerChange(kind="unchanged"))
        return PortfolioMeta(id=pid, name=clean_name, is_default=(pid == "default"))

    def portfolio_exists(self, portfolio_id: str) -> bool:
        return _portfolio_exists(portfolio_id)

    # --- Internal Locked Persistence & Recovery Helpers ---

    def _load_or_init_locked(self, portfolio_id: str = "default") -> Tuple[frontmatter.Post, PortfolioState]:
        fpath = get_portfolio_filepath(portfolio_id)
        if not fpath.exists():
            pid = validate_portfolio_id(portfolio_id)
            if pid != "default":
                raise PortfolioNotFoundError(f"ไม่พบพอร์ตไอดี '{pid}' ในระบบ")
            fpath.parent.mkdir(parents=True, exist_ok=True)
            post = frontmatter.Post(content="", metadata={})
            state = _initial_state()
            self._commit_locked(portfolio_id, state, LedgerChange(kind="unchanged"))
            return post, state

        with fpath.open("r", encoding="utf-8") as f:
            post = frontmatter.load(f)

        if not post.metadata:
            log.warning("Portfolio file %s has no YAML frontmatter -> Re-initialising", fpath)
            state = _initial_state()
            self._commit_locked(portfolio_id, state, LedgerChange(kind="unchanged"))
            return post, state

        state = PortfolioState.model_validate(post.metadata)
        if not state.name and post.metadata.get("name"):
            state.name = post.metadata.get("name")
        return post, state

    def _serialize_state_to_md(self, state: PortfolioState) -> str:
        recalc_all(state)
        state.last_updated = _now_iso()
        dump = state.model_dump(exclude_none=True)

        ordered: dict = {}
        if state.name:
            ordered["name"] = state.name
        for key in _TOP_LEVEL_KEY_ORDER:
            if key in dump:
                ordered[key] = dump.pop(key)
        ordered.update(dump)

        post = frontmatter.Post(content="", **ordered)
        return frontmatter.dumps(post, sort_keys=False)

    def _sync_sidecars(self, state: PortfolioState, portfolio_id: str = "default") -> None:
        """Sync derived sidecars Holdings/*.md atomically."""
        holdings_dir = get_holdings_dir(portfolio_id)
        holdings_dir.mkdir(parents=True, exist_ok=True)
        live: set[str] = set()

        for h in state.holdings:
            if h.asset_type == "Cash":
                continue
            safe = h.symbol.replace("/", "_")
            _atomic_write_to(holdings_dir / f"{safe}.md", _holding_to_md(h))
            live.add(safe)

        for old in holdings_dir.glob("*.md"):
            if old.stem not in live:
                try:
                    with old.open("r", encoding="utf-8") as f:
                        post = frontmatter.load(f)
                    if post.metadata.get("status") != "archived":
                        post.metadata["status"] = "archived"
                        post.metadata["archived_at"] = _now_iso()
                        _atomic_write_to(old, frontmatter.dumps(post, sort_keys=False))
                except Exception as e:
                    log.warning("Failed to archive sidecar %s: %s", old.name, e)

    def _cleanup_staged_files(self, manifest: dict) -> None:
        for key in ("staged_master_file", "staged_ledger_file", "staged_journal_file"):
            p = manifest.get(key)
            if p:
                Path(p).unlink(missing_ok=True)

    def _reconcile_pending_commit(self, portfolio_id: str) -> None:
        """Deterministic Multi-File Crash Recovery Decision Matrix."""
        manifest_path = get_pending_manifest_path(portfolio_id)
        if not manifest_path.exists():
            return

        try:
            with manifest_path.open("r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception as e:
            log.critical("[RECOVERY CONFLICT] Unreadable manifest on %s: %s", portfolio_id, e)
            return

        master_file = get_portfolio_filepath(portfolio_id)
        ledger_file = get_trades_log_filepath(portfolio_id)
        journal_file = get_journal_filepath(portfolio_id)

        disk_master_sha = _compute_sha256(master_file)
        disk_ledger_sha = _compute_sha256(ledger_file)
        disk_journal_sha = _compute_sha256(journal_file)

        pre_master_sha = manifest.get("pre_master_sha256") or ""
        staged_master_sha = manifest.get("staged_master_sha256") or ""
        pre_ledger_sha = manifest.get("pre_ledger_sha256") or ""
        target_ledger_sha = manifest.get("staged_ledger_sha256") or pre_ledger_sha or ""
        pre_journal_sha = manifest.get("pre_journal_sha256") or ""
        target_journal_sha = manifest.get("staged_journal_sha256") or pre_journal_sha or ""
        ledger_kind = manifest.get("ledger_kind", "unchanged")

        # Step 1: Clean Rollback (Master & Ledger & Journal all at pre-state)
        if disk_master_sha == pre_master_sha and disk_ledger_sha == pre_ledger_sha and disk_journal_sha == pre_journal_sha:
            self._cleanup_staged_files(manifest)
            manifest_path.unlink(missing_ok=True)
            log.info("[RECOVERY] Rollback uncommitted transaction %s for %s", manifest.get("tx_id"), portfolio_id)
            return

        # Step 2: Commit Already Complete (Master & Ledger & Journal at target state)
        if disk_master_sha == staged_master_sha and disk_ledger_sha == target_ledger_sha and disk_journal_sha == target_journal_sha:
            _, state = self._load_or_init_locked(portfolio_id)
            self._sync_sidecars(state, portfolio_id=portfolio_id)
            self._cleanup_staged_files(manifest)
            manifest_path.unlink(missing_ok=True)
            log.info("[RECOVERY] Cleaned completed manifest for %s (ledger_kind=%s)", portfolio_id, ledger_kind)
            return

        # Step 3: Roll-Forward Ledger / Journal (Master committed, remaining files at pre-state)
        if disk_master_sha == staged_master_sha:
            # Roll forward ledger if staged
            if disk_ledger_sha == pre_ledger_sha and ledger_kind != "unchanged":
                staged_ledger = Path(manifest["staged_ledger_file"]) if manifest.get("staged_ledger_file") else None
                if staged_ledger and staged_ledger.exists() and _compute_sha256(staged_ledger) == target_ledger_sha:
                    os.replace(staged_ledger, ledger_file)
                    log.info("[RECOVERY] Roll-forward committed ledger for %s", portfolio_id)
                elif staged_ledger:
                    log.critical("[RECOVERY CONFLICT] Staged ledger missing or corrupted for %s", portfolio_id)
                    raise RecoveryConflictError(
                        f"ไม่สามารถ Roll-forward Ledger ของพอร์ต '{portfolio_id}' ได้เนื่องจากไฟล์ staged เสียหาย — ระงับการทำงานเพื่อความปลอดภัย"
                    )

            # Roll forward journal if staged
            if disk_journal_sha == pre_journal_sha and manifest.get("staged_journal_file"):
                staged_journal = Path(manifest["staged_journal_file"])
                if staged_journal and staged_journal.exists() and _compute_sha256(staged_journal) == target_journal_sha:
                    os.replace(staged_journal, journal_file)
                    log.info("[RECOVERY] Roll-forward committed journal for %s", portfolio_id)
                else:
                    log.critical("[RECOVERY CONFLICT] Staged journal missing or corrupted for %s", portfolio_id)
                    raise RecoveryConflictError(
                        f"ไม่สามารถ Roll-forward Journal ของพอร์ต '{portfolio_id}' ได้เนื่องจากไฟล์ staged เสียหาย — ระงับการทำงานเพื่อความปลอดภัย"
                    )

            _, state = self._load_or_init_locked(portfolio_id)
            self._sync_sidecars(state, portfolio_id=portfolio_id)
            self._cleanup_staged_files(manifest)
            manifest_path.unlink(missing_ok=True)
            return

        # Step 4: Recovery Conflict (Any unknown hash or unexpected state combination)
        log.critical(
            "[RECOVERY CONFLICT] Unresolvable hash divergence on %s (Master: %s, Ledger: %s, Journal: %s). Manifest preserved.",
            portfolio_id, disk_master_sha, disk_ledger_sha, disk_journal_sha,
        )
        raise RecoveryConflictError(
            f"ตรวจพบความขัดแย้งของไฟล์ Master/Ledger/Journal ระหว่างกู้คืนพอร์ต '{portfolio_id}' — ไฟล์ถูกแก้ไขภายนอกหรือเสียหาย ระงับการเขียนทับโดยเด็ดขาด"
        )

    def _commit_locked(
        self,
        portfolio_id: str,
        state: PortfolioState,
        ledger_change: Optional[Union[LedgerChange, PortfolioMutation]] = None,
    ) -> None:
        """Durable staged commit with crash-consistent recovery sequencing."""
        pdir = get_portfolio_dir(portfolio_id)
        master_file = get_portfolio_filepath(portfolio_id)
        ledger_file = get_trades_log_filepath(portfolio_id)
        journal_file = get_journal_filepath(portfolio_id)
        manifest_path = get_pending_manifest_path(portfolio_id)

        pre_master_sha = _compute_sha256(master_file)
        pre_ledger_sha = _compute_sha256(ledger_file)
        pre_journal_sha = _compute_sha256(journal_file)

        mutation = PortfolioMutation.from_change(ledger_change)
        change = mutation.ledger_change or LedgerChange(kind="unchanged")
        tx_id = change.tx_id or str(uuid.uuid4())

        # 1. Stage Master File
        staged_master = pdir / f".master_{tx_id}.staged"
        serialized_master = self._serialize_state_to_md(state)
        with staged_master.open("w", encoding="utf-8") as f:
            f.write(serialized_master)
            f.flush()
            os.fsync(f.fileno())
        staged_master_sha = _compute_sha256(staged_master)

        # 2. Stage Ledger File if mutated
        staged_ledger: Optional[Path] = None
        staged_ledger_sha: Optional[str] = None

        if change.kind in ("append", "replace_all"):
            staged_ledger = pdir / f".ledger_{tx_id}.staged"
            rows_to_write: List[Dict] = []

            if change.kind == "replace_all":
                rows_to_write = change.rows or []
            elif change.kind == "append":
                if ledger_file.exists():
                    rows_to_write = _read_and_migrate_trade_log_locked(ledger_file)
                if change.row:
                    rows_to_write.append(change.row)

            with staged_ledger.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=_TRADES_LOG_HEADER, lineterminator="\n")
                writer.writeheader()
                for r in rows_to_write:
                    sanitized = {
                        col: _sanitize_csv_field(str(r.get(col) if r.get(col) is not None else (r.get(col.lower()) if r.get(col.lower()) is not None else "")))
                        for col in _TRADES_LOG_HEADER
                    }
                    writer.writerow(sanitized)
                f.flush()
                os.fsync(f.fileno())
            staged_ledger_sha = _compute_sha256(staged_ledger)

        # 3. Stage Journal File if system journal events present
        staged_journal: Optional[Path] = None
        staged_journal_sha: Optional[str] = None

        if mutation.system_journal_events:
            staged_journal = pdir / f".journal_{tx_id}.staged"
            existing_journal = journal_file.read_text(encoding="utf-8") if journal_file.exists() else ""
            journal_blocks = []
            for event in mutation.system_journal_events:
                rendered_message = inject_journal_wikilinks(event.message)
                journal_blocks.append(f"\n## [{event.timestamp}]\n\n{rendered_message}\n")
            with staged_journal.open("w", encoding="utf-8") as f:
                f.write(existing_journal + "".join(journal_blocks))
                f.flush()
                os.fsync(f.fileno())
            staged_journal_sha = _compute_sha256(staged_journal)

        # 4. Write Durable Pending Manifest
        manifest_data = {
            "schema_version": 2,
            "tx_id": tx_id,
            "portfolio_id": portfolio_id,
            "timestamp": time.time(),
            "ledger_kind": change.kind,
            "has_journal_events": bool(mutation.system_journal_events),
            "pre_master_sha256": pre_master_sha,
            "staged_master_file": str(staged_master),
            "staged_master_sha256": staged_master_sha,
            "pre_ledger_sha256": pre_ledger_sha,
            "staged_ledger_file": str(staged_ledger) if staged_ledger else None,
            "staged_ledger_sha256": staged_ledger_sha,
            "pre_journal_sha256": pre_journal_sha,
            "staged_journal_file": str(staged_journal) if staged_journal else None,
            "staged_journal_sha256": staged_journal_sha,
            "sidecars_dirty": True,
        }
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())

        # 5. Atomic Master Replace (Master Commit Point)
        os.replace(staged_master, master_file)

        # 6. Atomic Ledger Replace (Ledger Commit Point)
        if staged_ledger and staged_ledger.exists():
            os.replace(staged_ledger, ledger_file)

        # 7. Atomic Journal Replace (Journal Commit Point)
        if staged_journal and staged_journal.exists():
            os.replace(staged_journal, journal_file)

        # 8. Sync Derived Sidecars (Deferred Sync)
        self._sync_sidecars(state, portfolio_id=portfolio_id)

        # 9. Unlink Manifest
        manifest_path.unlink(missing_ok=True)
