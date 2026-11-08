"""Unit tests for ThesisFileStore and Obsidian sync."""
from decimal import Decimal
import tempfile
from pathlib import Path
import pytest

from core.thesis.evidence_taxonomy import EvidenceTier
from core.thesis.models import (
    CatalystEvent,
    FalsifierRule,
    ThesisClaimRecord,
)
from tools.thesis.thesis_file_store import ThesisFileStore


@pytest.fixture
def temp_dirs():
    with tempfile.TemporaryDirectory() as data_d, tempfile.TemporaryDirectory() as vault_d:
        yield Path(data_d), Path(vault_d)


def test_file_store_save_and_load(temp_dirs):
    data_dir, vault_dir = temp_dirs
    store = ThesisFileStore(data_dir=data_dir, vault_dir=vault_dir)

    rule = FalsifierRule(
        falsifier_id="FAL-REV",
        metric_name="datacenter_revenue",
        operator="<",
        threshold_value=Decimal("35000000000"),
    )
    catalyst = CatalystEvent(
        catalyst_id="CAT-01",
        event_name="Earnings Call",
        expected_date_str="2026-11-20",
    )
    claim = ThesisClaimRecord.create(
        claim_id="TH-NVDA-v1",
        ticker="NVDA",
        version=1,
        as_of_utc="2026-11-02T12:00:00Z",
        claim_statement="Datacenter dominance continues.",
        falsifiers=[rule],
        linked_catalysts=[catalyst],
    )

    store.save_claim(claim)

    loaded = store.load_claims("NVDA")
    assert len(loaded) == 1
    assert loaded[0].claim_id == "TH-NVDA-v1"
    assert loaded[0].falsifiers[0].metric_name == "datacenter_revenue"
    assert loaded[0].verify_integrity() is True

    # Check Obsidian Markdown Mirror was generated
    md_file = vault_dir / "NVDA.md"
    assert md_file.exists()
    content = md_file.read_text(encoding="utf-8")
    assert "---" in content
    assert "ticker: NVDA" in content
    assert "claim_id: TH-NVDA-v1" in content
    assert "Falsifier Floor Rules" in content
    assert "datacenter_revenue" in content


def test_append_only_revision_chain(temp_dirs):
    data_dir, vault_dir = temp_dirs
    store = ThesisFileStore(data_dir=data_dir, vault_dir=vault_dir)

    claim_v1 = ThesisClaimRecord.create(
        claim_id="TH-AAPL-v1",
        ticker="AAPL",
        version=1,
        as_of_utc="2026-06-01T00:00:00Z",
        claim_statement="iPhone 16 cycle.",
        status="REVISED",
    )
    claim_v2 = ThesisClaimRecord.create(
        claim_id="TH-AAPL-v2",
        ticker="AAPL",
        version=2,
        as_of_utc="2026-10-01T00:00:00Z",
        claim_statement="Apple Intelligence adoption thesis.",
        prior_claim_id="TH-AAPL-v1",
        status="ACTIVE",
    )

    store.save_claim(claim_v1)
    store.save_claim(claim_v2)

    history = store.load_claims("AAPL")
    assert len(history) == 2
    assert history[0].version == 1
    assert history[1].version == 2
    assert history[1].prior_claim_id == "TH-AAPL-v1"

    active = store.get_active_claim("AAPL")
    assert active is not None
    assert active.claim_id == "TH-AAPL-v2"

