"""Immutable local archive for uploaded PVD statement PDFs."""
from __future__ import annotations

import hashlib
import os
import re
import tempfile
from pathlib import Path

from tools.archivist.vault_paths import VaultPaths
from tools.portfolio.domain.pvd_models import PvdStatement
from tools.portfolio.ports.pvd_port import PvdDocumentArchivePort


class LocalFilesystemPvdArchiveAdapter(PvdDocumentArchivePort):
    def __init__(self, vault_paths: VaultPaths | None = None, *, root: str | Path | None = None):
        if vault_paths is not None and root is not None:
            raise ValueError("provide vault_paths or root, not both")
        self.root = Path(root).resolve() if root is not None else (vault_paths or VaultPaths()).safe_resolve("attachments/pvd")

    def _path(self, portfolio_id: str, metadata: dict) -> Path:
        pid = str(portfolio_id).strip()
        fund_code = str(metadata.get("fund_code") or "")
        member_hash = str(metadata.get("member_hash") or "")
        as_of = str(metadata.get("as_of_date") or "")[:10]
        sha = str(metadata.get("source_pdf_sha256") or "")
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", pid) or not re.fullmatch(r"[A-Za-z0-9_-]{1,32}", fund_code):
            raise ValueError("Invalid PVD archive identity")
        if not re.fullmatch(r"[0-9a-f]{64}", member_hash) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", as_of) or not re.fullmatch(r"[0-9a-f]{64}", sha):
            raise ValueError("Invalid PVD archive metadata")
        path = (self.root / pid / fund_code / member_hash / as_of / f"{sha}.pdf").resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("PVD archive path escaped its configured root")
        return path

    def archive(self, pdf_bytes: bytes, *, portfolio_id: str, statement: PvdStatement) -> str:
        digest = hashlib.sha256(pdf_bytes).hexdigest()
        if statement.source_pdf_sha256 and digest != statement.source_pdf_sha256:
            raise ValueError("PDF content changed after analysis")
        metadata = {
            "fund_code": statement.fund_code,
            "member_hash": statement.member_hash,
            "as_of_date": statement.as_of_date.isoformat(),
            "source_pdf_sha256": digest,
        }
        target = self._path(portfolio_id, metadata)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                raise ValueError("Existing PVD archive has a hash mismatch")
            return digest
        fd, temp_name = tempfile.mkstemp(prefix=".pvd-", suffix=".tmp", dir=target.parent)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(pdf_bytes)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, target)
        finally:
            Path(temp_name).unlink(missing_ok=True)
        return digest

    def read(self, *, portfolio_id: str, metadata: dict) -> bytes:
        target = self._path(portfolio_id, metadata)
        data = target.read_bytes()
        expected = str(metadata.get("source_pdf_sha256") or "")
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError("Archived PVD statement hash check failed")
        return data
