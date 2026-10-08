"""Unit tests for EssenceClaim review, editing, and confirmation snapshotting."""
import pytest

from core.investor_essence.models import (
    EssenceClaim,
    EvidenceRef,
    EvidenceType,
    FitRating,
    SourceKind,
)
from core.investor_essence.claims import (
    build_confirmation_snapshot,
    edit_claim_text,
    exclude_claim,
    rate_claim_fit,
)


def _make_claim(cid: str = "claim_1", text: str = "ต้องการกระแสเงินสดสม่ำเสมอ") -> EssenceClaim:
    return EssenceClaim(
        claim_id=cid,
        text=text,
        source_kind=SourceKind.USER_STATED,
        evidence_refs=[
            EvidenceRef(
                answer_id="ans_1",
                question_id="q_1",
                revision=1,
                quote="ลงทุนเพื่อรับเงินปันผลรายไตรมาส",
                evidence_type=EvidenceType.ACTUAL_EXPERIENCE,
            )
        ],
    )


class TestClaimsAndConfirmation:
    def test_rate_claim_fit(self):
        c = _make_claim()
        assert not c.is_accepted

        # Exact match -> accepted
        c_exact = rate_claim_fit(c, FitRating.EXACT)
        assert c_exact.fit_rating == FitRating.EXACT
        assert c_exact.is_accepted is True

        # Partial match -> not accepted until edited
        c_partial = rate_claim_fit(c, FitRating.PARTIAL)
        assert c_partial.fit_rating == FitRating.PARTIAL
        assert c_partial.is_accepted is False

        # Rejected -> not accepted
        c_rejected = rate_claim_fit(c, FitRating.REJECTED)
        assert c_rejected.fit_rating == FitRating.REJECTED
        assert c_rejected.is_accepted is False

    def test_edit_claim_text(self):
        c = _make_claim()
        c_edited = edit_claim_text(c, "ต้องการปันผล 5% ต่อปีเพื่อเป็นค่าใช้จ่าย")

        assert c_edited.claim_id == c.claim_id
        assert c_edited.text == c.text
        assert c_edited.edited_text == "ต้องการปันผล 5% ต่อปีเพื่อเป็นค่าใช้จ่าย"
        assert c_edited.source_kind == SourceKind.USER_EDITED
        assert c_edited.fit_rating == FitRating.EXACT
        assert c_edited.is_accepted is True
        assert c_edited.text_revision == 2

    def test_edit_claim_text_empty_fails(self):
        c = _make_claim()
        with pytest.raises(ValueError, match="Claim text cannot be empty"):
            edit_claim_text(c, "   ")

    def test_exclude_claim(self):
        c = _make_claim()
        c_ex = exclude_claim(c)
        assert c_ex.fit_rating == FitRating.REJECTED
        assert c_ex.is_accepted is False

    def test_build_confirmation_snapshot_success(self):
        c1 = rate_claim_fit(_make_claim("c1", "เป้าหมายเงินปันผล"), FitRating.EXACT)
        c2 = edit_claim_text(_make_claim("c2", "เน้นรักษาเงินต้น"), "เน้นรักษาเงินต้นเป็นหลัก")
        c3 = exclude_claim(_make_claim("c3", "ต้องการเก็งกำไรระยะสั้น"))

        snapshot = build_confirmation_snapshot(
            confirmation_id="conf_1",
            session_id="sess_100",
            claims=[c1, c2, c3],
            unresolved_topics=["กรอบเวลาเกษียณยังไม่แน่ชัด"],
            evidence_snapshot_hash="abc123hash",
            confirmed_at_iso="2026-10-08T10:00:00Z",
        )

        assert snapshot.confirmation_id == "conf_1"
        assert snapshot.session_id == "sess_100"
        assert len(snapshot.accepted_claims) == 2
        assert {c.claim_id for c in snapshot.accepted_claims} == {"c1", "c2"}
        assert len(snapshot.per_claim_snapshot) == 2
        assert len(snapshot.unresolved_topics) == 1
        assert snapshot.evidence_snapshot_hash == "abc123hash"
        assert snapshot.confirmed_at_iso == "2026-10-08T10:00:00Z"

    def test_cannot_confirm_empty_accepted_claims(self):
        c1 = exclude_claim(_make_claim("c1"))
        with pytest.raises(ValueError, match="at least 1 accepted claim is required"):
            build_confirmation_snapshot(
                confirmation_id="conf_1",
                session_id="sess_100",
                claims=[c1],
                unresolved_topics=[],
                evidence_snapshot_hash="hash",
                confirmed_at_iso="2026-10-08T10:00:00Z",
            )

    def test_cannot_confirm_rejected_claim_in_accepted_list(self):
        c1 = exclude_claim(_make_claim("c1"))
        with pytest.raises(ValueError, match="Rejected claim c1 cannot be included"):
            build_confirmation_snapshot(
                confirmation_id="conf_1",
                session_id="sess_100",
                claims=[c1],
                unresolved_topics=[],
                evidence_snapshot_hash="hash",
                confirmed_at_iso="2026-10-08T10:00:00Z",
                accepted_claim_ids=["c1"],
            )

    def test_cannot_confirm_partial_fit_without_edit(self):
        c1 = rate_claim_fit(_make_claim("c1"), FitRating.PARTIAL)
        with pytest.raises(ValueError, match="has partial fit pending edit"):
            build_confirmation_snapshot(
                confirmation_id="conf_1",
                session_id="sess_100",
                claims=[c1],
                unresolved_topics=[],
                evidence_snapshot_hash="hash",
                confirmed_at_iso="2026-10-08T10:00:00Z",
                accepted_claim_ids=["c1"],
            )
