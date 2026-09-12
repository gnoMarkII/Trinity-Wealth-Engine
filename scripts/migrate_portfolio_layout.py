"""One-off migration: รวมทุกพอร์ต (รวมถึง 'default') เข้า layout เดียวกัน
Portfolios/{id}/Portfolio_Holdings.md, Watchlist.md, Trading_Journal.md,
Trades_Log.csv, Performance_Log.csv, Holdings/, WatchlistItems/

- migrate(): ย้ายพอร์ตรอง (portfolio_id != 'default') จาก flat layout เดิม
  (Portfolios/{id}.md, Portfolios/{id}_watchlist.md, ...) เข้าโฟลเดอร์ต่อพอร์ต
- migrate_default(): ย้ายพอร์ตหลักจากตำแหน่งเดิม (Current_Holdings/Portfolio_Holdings.md,
  Journals_and_Reports/*, ...) เข้า Portfolios/default/ ตำแหน่งเดียวกับพอร์ตอื่น

รันด้วยมือครั้งเดียว: uv run python scripts/migrate_portfolio_layout.py
Idempotent — รันซ้ำได้ปลอดภัย (ไฟล์ที่ย้ายไปแล้วจะไม่มีให้ย้ายซ้ำ)
"""
import os
import shutil
import logging
from datetime import datetime
from pathlib import Path

import frontmatter

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("PortfolioMigration")


def get_vault_path() -> Path:
    return Path(os.getenv("OBSIDIAN_VAULT_PATH", "./memories")).resolve()


def get_portfolios_dir(vault_path: Path) -> Path:
    return vault_path / "20_Portfolio_Management" / "Current_Holdings" / "Portfolios"


def _move_if_exists(src: Path, dest: Path) -> None:
    if not src.exists():
        return
    if dest.exists():
        log.warning("ปลายทางมีอยู่แล้ว ข้าม: %s -> %s", src, dest)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dest))
    log.info("ย้าย %s -> %s", src, dest)


def _move_dir_if_exists(src: Path, dest: Path) -> None:
    """ย้ายทั้งโฟลเดอร์ (รวมไฟล์ข้างใน) — ข้ามถ้า src ไม่มีหรือ dest มีอยู่แล้ว"""
    if not src.exists():
        return
    if dest.exists():
        log.warning("ปลายทางมีอยู่แล้ว ข้าม: %s -> %s", src, dest)
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dest))
    log.info("ย้ายโฟลเดอร์ %s -> %s", src, dest)


def migrate(vault_path: Path) -> None:
    pdir = get_portfolios_dir(vault_path)
    if not pdir.exists():
        log.info("ไม่พบโฟลเดอร์ Portfolios/ — ไม่มีอะไรต้อง migrate")
        return

    flat_masters = [
        f for f in sorted(pdir.glob("*.md"))
        if not f.name.endswith("_watchlist.md") and not f.name.endswith("_journal.md")
    ]

    migrated_ids: set[str] = set()

    for master in flat_masters:
        pid = master.stem
        try:
            with master.open("r", encoding="utf-8") as fh:
                post = frontmatter.load(fh)
            if post.metadata.get("doc_type") != "portfolio_master":
                log.warning("ข้าม %s — ไม่ใช่ portfolio_master (doc_type=%r)", master, post.metadata.get("doc_type"))
                continue
        except Exception as e:
            log.warning("อ่าน %s ไม่สำเร็จ ข้าม: %s", master, e)
            continue

        target_dir = pdir / pid
        _move_if_exists(master, target_dir / "Portfolio_Holdings.md")
        _move_if_exists(pdir / f"{pid}_watchlist.md", target_dir / "Watchlist.md")
        _move_if_exists(pdir / f"{pid}_journal.md", target_dir / "Trading_Journal.md")
        _move_if_exists(pdir / f"{pid}_performance.csv", target_dir / "Performance_Log.csv")
        migrated_ids.add(pid)

    # ไฟล์กำพร้า (sidecar ที่ไม่มี master คู่กัน) — ย้ายเข้า .backups/ แทนลบทิ้ง เผื่อยังมีประโยชน์
    orphan_suffixes = ("_watchlist.md", "_journal.md", "_performance.csv")
    remaining = [
        f for f in pdir.glob("*")
        if f.is_file() and any(f.name.endswith(s) for s in orphan_suffixes)
    ]
    if remaining:
        backup_dir = pdir / ".backups" / f"orphaned_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        for f in remaining:
            dest = backup_dir / f.name
            backup_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(f), str(dest))
            log.warning("ไฟล์กำพร้า (ไม่มี master คู่กัน) ย้ายไป backup: %s -> %s", f, dest)

    log.info("Migration เสร็จสิ้น — ย้ายพอร์ตทั้งหมด %d พอร์ต: %s", len(migrated_ids), sorted(migrated_ids))


def migrate_default(vault_path: Path) -> None:
    """ย้ายพอร์ต default จากตำแหน่งเดิม (Current_Holdings/ + Journals_and_Reports/) เข้า Portfolios/default/
    ให้ layout เหมือนพอร์ตอื่นทุกประการ — ไม่มี special-case อีกต่อไป
    """
    pdir = get_portfolios_dir(vault_path)
    target_dir = pdir / "default"
    current_holdings = vault_path / "20_Portfolio_Management" / "Current_Holdings"
    journals = vault_path / "20_Portfolio_Management" / "Journals_and_Reports"

    _move_if_exists(current_holdings / "Portfolio_Holdings.md", target_dir / "Portfolio_Holdings.md")
    _move_if_exists(current_holdings / "Watchlist.md", target_dir / "Watchlist.md")
    _move_dir_if_exists(current_holdings / "Holdings", target_dir / "Holdings")
    _move_dir_if_exists(current_holdings / "WatchlistItems", target_dir / "WatchlistItems")
    _move_if_exists(journals / "Trading_Journal.md", target_dir / "Trading_Journal.md")
    _move_if_exists(journals / "Trades_Log.csv", target_dir / "Trades_Log.csv")
    _move_if_exists(journals / "Performance_Log.csv", target_dir / "Performance_Log.csv")

    # ลบโฟลเดอร์เก่าที่ว่างเปล่าหลังย้าย (ไม่ลบถ้ายังมีไฟล์อื่นเหลืออยู่ — เผื่อผู้ใช้เก็บไฟล์อื่นไว้)
    for old_dir in (journals, current_holdings / "Holdings", current_holdings / "WatchlistItems"):
        try:
            if old_dir.exists() and not any(old_dir.iterdir()):
                old_dir.rmdir()
                log.info("ลบโฟลเดอร์ว่างเปล่า: %s", old_dir)
        except OSError as e:
            log.warning("ลบโฟลเดอร์ %s ไม่สำเร็จ: %s", old_dir, e)

    log.info("Migration พอร์ต default เสร็จสิ้น")


def main():
    vault_path = get_vault_path()
    log.info("เริ่ม migrate portfolio layout ใน vault: %s", vault_path)
    migrate(vault_path)
    migrate_default(vault_path)


if __name__ == "__main__":
    main()
