import hashlib
from datetime import date
from decimal import Decimal

import pytest

from tools.portfolio.adapters.bblam_pvd.local_filesystem_pvd_archive_adapter import LocalFilesystemPvdArchiveAdapter
from tools.portfolio.domain.pvd_models import PvdContributionSummary, PvdStatement


def test_archive_is_atomic_immutable_and_hash_checked(tmp_path):
    payload = b"%PDF-pvd-statement"
    statement = PvdStatement(
        member_id="1234567890", member_hash="a" * 64, as_of_date=date(2026, 9, 30),
        source_pdf_sha256=hashlib.sha256(payload).hexdigest(),
        summary=PvdContributionSummary(
            member_principal=Decimal("1"), member_benefit=Decimal("0"), employer_principal=Decimal("0"), employer_benefit=Decimal("0"),
            total_contribution=Decimal("1"), total_nav=Decimal("1"), employer_vesting_percent=Decimal("100"), member_vested_amount=Decimal("1"), employer_vested_amount=Decimal("0"),
        ),
        holdings=[],
    )
    archive = LocalFilesystemPvdArchiveAdapter(root=tmp_path / "attachments" / "pvd")
    archive_id = archive.archive(payload, portfolio_id="default", statement=statement)
    assert archive_id == hashlib.sha256(payload).hexdigest()
    metadata = {"fund_code": "BBLAM", "member_hash": "a" * 64, "as_of_date": "2026-09-30", "source_pdf_sha256": archive_id}
    assert archive.read(portfolio_id="default", metadata=metadata) == payload
    with pytest.raises(ValueError, match="changed"):
        archive.archive(b"%PDF-changed", portfolio_id="default", statement=statement)


def test_archive_rejects_path_traversal_metadata(tmp_path):
    archive = LocalFilesystemPvdArchiveAdapter(root=tmp_path / "pvd")
    with pytest.raises(ValueError, match="identity"):
        archive.read(portfolio_id="default", metadata={"fund_code": "../escape", "member_hash": "a" * 64, "as_of_date": "2026-09-30", "source_pdf_sha256": "b" * 64})
