"""Automated Vault Backup and Disaster Recovery (DR) Snapshot Engine.

Provides atomic zip archiving, SHA256 integrity checks, rotation, and recovery verification
for the Obsidian Vault.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

from core.logger import get_logger

logger = get_logger(__name__)

# System and transient directories excluded from backup archives
_BACKUP_EXCLUDE_PARTS = {
    ".chroma_index",
    ".trash",
    ".sync_history",
    ".system/locks",
    "__pycache__",
}


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def create_vault_snapshot(
    vault_root: Optional[Union[str, Path]] = None,
    backup_dir: Optional[Union[str, Path]] = None,
) -> tuple[Path, str]:
    """Creates a compressed zip snapshot of the Obsidian vault.

    Returns:
        tuple[Path, str]: Path to the created zip file and its SHA256 checksum.
    """
    v_root = Path(vault_root).resolve() if vault_root else Path(os.getenv("OBSIDIAN_VAULT_PATH", "./memories")).resolve()
    if not v_root.exists():
        raise FileNotFoundError(f"Vault root does not exist: {v_root}")

    b_dir = Path(backup_dir).resolve() if backup_dir else v_root.parent / "data" / "backups"
    b_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    zip_name = f"vault_snapshot_{timestamp}.zip"
    tmp_zip = b_dir / f".tmp_{zip_name}"
    final_zip = b_dir / zip_name

    try:
        with zipfile.ZipFile(tmp_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            for item in v_root.rglob("*"):
                if item.is_dir():
                    continue
                try:
                    rel = item.relative_to(v_root).as_posix()
                except ValueError:
                    continue

                if any(excl in rel for excl in _BACKUP_EXCLUDE_PARTS):
                    continue

                zf.write(item, arcname=rel)

        # Atomic rename
        shutil.move(str(tmp_zip), str(final_zip))
        checksum = _file_sha256(final_zip)

        # Write sidecar checksum
        chk_file = final_zip.with_suffix(".zip.sha256")
        chk_file.write_text(f"{checksum}  {zip_name}\n", encoding="utf-8")

        logger.info("Vault snapshot created: %s (SHA256: %s)", final_zip, checksum[:12])
        return final_zip, checksum

    finally:
        if tmp_zip.exists():
            tmp_zip.unlink(missing_ok=True)


def rotate_snapshots(
    backup_dir: Optional[Union[str, Path]] = None,
    max_keep: int = 5,
) -> list[Path]:
    """Prunes older snapshots, retaining the latest `max_keep` snapshots."""
    b_dir = Path(backup_dir).resolve() if backup_dir else Path("./data/backups").resolve()
    from tools.archivist.maintenance_guard import assert_write_allowed
    assert_write_allowed(b_dir)
    if not b_dir.exists():
        return []

    snapshots = sorted(b_dir.glob("vault_snapshot_*.zip"), key=lambda p: p.stat().st_mtime)
    to_delete = snapshots[:-max_keep] if len(snapshots) > max_keep else []

    deleted: list[Path] = []
    for snap in to_delete:
        snap.unlink(missing_ok=True)
        snap.with_suffix(".zip.sha256").unlink(missing_ok=True)
        deleted.append(snap)

    if deleted:
        logger.info("Rotated %d old snapshot(s)", len(deleted))
    return deleted


def restore_vault_snapshot(
    zip_path: Union[str, Path],
    target_dir: Union[str, Path],
    verify_checksum: bool = True,
) -> int:
    """Extracts snapshot archive into target directory for disaster recovery or verification."""
    z_path = Path(zip_path).resolve()
    t_dir = Path(target_dir).resolve()
    if not z_path.exists():
        raise FileNotFoundError(f"Snapshot not found: {z_path}")

    if verify_checksum:
        chk_file = z_path.with_suffix(".zip.sha256")
        if chk_file.exists():
            expected = chk_file.read_text(encoding="utf-8").split()[0].strip()
            actual = _file_sha256(z_path)
            if expected.lower() != actual.lower():
                raise ValueError(f"Snapshot checksum mismatch! Expected: {expected}, Actual: {actual}")

    t_dir.mkdir(parents=True, exist_ok=True)
    extracted = 0
    with zipfile.ZipFile(z_path, "r") as zf:
        for member in zf.infolist():
            # ``ZipFile.extractall`` accepts ``../`` members on some Python/
            # platform combinations.  Validate every archive name before
            # creating anything so a snapshot can never escape its restore
            # root or overwrite an unrelated path.
            member_name = member.filename.replace("\\", "/")
            target = (t_dir / member_name).resolve()
            if not target.is_relative_to(t_dir):
                raise ValueError(f"Snapshot member escapes restore root: {member.filename!r}")
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member, "r") as source, target.open("wb") as dest:
                shutil.copyfileobj(source, dest)
            extracted += 1
    return extracted
