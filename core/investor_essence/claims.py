"""Pure domain logic for EssenceClaim review, editing, and confirmation snapshotting.

Rules:
- Fit ratings (exact, partial, rejected)
- Editing increments text_revision, marks SourceKind.USER_EDITED
- Rejected and pending-edit claims are NEVER included in confirmed accepted claims
- At least 1 accepted claim is required to confirm
- Confirmation creates an auditable PerClaimConfirmationSnapshot
"""
from __future__ import annotations

import hashlib
import json
from typing import List, Optional

from core.investor_essence.models import (
    EssenceClaim,
    EssenceConfirmationSnapshot,
    FitRating,
    PerClaimConfirmationSnapshot,
    SourceKind,
)


def rate_claim_fit(claim: EssenceClaim, fit_rating: FitRating) -> EssenceClaim:
    """Updates the user fit rating on a summary claim."""
    is_accepted = (fit_rating == FitRating.EXACT)
    return EssenceClaim(
        claim_id=claim.claim_id,
        text=claim.text,
        source_kind=claim.source_kind,
        evidence_refs=claim.evidence_refs,
        counter_evidence_refs=claim.counter_evidence_refs,
        fit_rating=fit_rating,
        edited_text=claim.edited_text,
        text_revision=claim.text_revision,
        is_accepted=is_accepted,
    )


def edit_claim_text(claim: EssenceClaim, new_text: str) -> EssenceClaim:
    """Edits claim text to match user's true intent, incrementing text revision."""
    cleaned = new_text.strip()
    if not cleaned:
        raise ValueError("Claim text cannot be empty")
    return EssenceClaim(
        claim_id=claim.claim_id,
        text=claim.text,
        source_kind=SourceKind.USER_EDITED,
        evidence_refs=claim.evidence_refs,
        counter_evidence_refs=claim.counter_evidence_refs,
        fit_rating=FitRating.EXACT,
        edited_text=cleaned,
        text_revision=claim.text_revision + 1,
        is_accepted=True,
    )


def exclude_claim(claim: EssenceClaim) -> EssenceClaim:
    """Excludes/rejects a claim so it is omitted from the confirmed essence."""
    return EssenceClaim(
        claim_id=claim.claim_id,
        text=claim.text,
        source_kind=claim.source_kind,
        evidence_refs=claim.evidence_refs,
        counter_evidence_refs=claim.counter_evidence_refs,
        fit_rating=FitRating.REJECTED,
        edited_text=claim.edited_text,
        text_revision=claim.text_revision,
        is_accepted=False,
    )


def build_confirmation_snapshot(
    confirmation_id: str,
    session_id: str,
    claims: List[EssenceClaim],
    unresolved_topics: List[str],
    evidence_snapshot_hash: str,
    confirmed_at_iso: str,
    accepted_claim_ids: Optional[List[str]] = None,
) -> EssenceConfirmationSnapshot:
    """Validates and compiles confirmed essence claims into an immutable snapshot."""
    if accepted_claim_ids is not None:
        accepted_set = set(accepted_claim_ids)
        candidate_claims = [c for c in claims if c.claim_id in accepted_set]
    else:
        candidate_claims = [c for c in claims if c.is_accepted]

    accepted_claims: List[EssenceClaim] = []
    per_claim_snapshots: List[PerClaimConfirmationSnapshot] = []

    for c in candidate_claims:
        if c.fit_rating == FitRating.REJECTED:
            raise ValueError(f"Rejected claim {c.claim_id} cannot be included in confirmed essence")
        if c.fit_rating == FitRating.PARTIAL:
            raise ValueError(f"Claim {c.claim_id} has partial fit pending edit; please edit or exclude it before confirmation")

        effective_fit = c.fit_rating or FitRating.EXACT
        accepted_claim = EssenceClaim(
            claim_id=c.claim_id,
            text=c.text,
            source_kind=c.source_kind,
            evidence_refs=c.evidence_refs,
            counter_evidence_refs=c.counter_evidence_refs,
            fit_rating=effective_fit,
            edited_text=c.edited_text,
            text_revision=c.text_revision,
            is_accepted=True,
        )
        accepted_claims.append(accepted_claim)
        per_claim_snapshots.append(
            PerClaimConfirmationSnapshot(
                claim_id=accepted_claim.claim_id,
                final_text=accepted_claim.effective_text,
                text_revision=accepted_claim.text_revision,
                source_kind=accepted_claim.source_kind,
                fit_rating=effective_fit,
            )
        )

    if not accepted_claims:
        raise ValueError("Cannot confirm essence: at least 1 accepted claim is required")

    return EssenceConfirmationSnapshot(
        confirmation_id=confirmation_id,
        session_id=session_id,
        accepted_claims=accepted_claims,
        unresolved_topics=list(unresolved_topics),
        per_claim_snapshot=per_claim_snapshots,
        evidence_snapshot_hash=evidence_snapshot_hash,
        confirmed_at_iso=confirmed_at_iso,
    )
