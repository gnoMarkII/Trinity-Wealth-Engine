"""Read-only diagnostic reproductions; all investor/portfolio state is synthetic.

Run from repository root: .\.venv\Scripts\python.exe reports/principles_review_2026-10-10/backend_audit.py
The checks characterize existing defects; they are not correctness tests.
"""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from decimal import Decimal, InvalidOperation
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from application.investor_essence import (
    BucketPlanningService, ClaimReviewService, ConfirmationService,
    FinancialContextService, InterviewService, InvestmentAxisService,
)
from application.investor_essence.testing.fakes import (
    FakeAxisGenerator, FakeBucketGenerator, FakeClock, FakeEssenceGenerator,
    FakeIdGenerator, FakeInterviewGenerator, FakeInvestorRuntimeUowFactory,
    FakePortfolioPlanningAdapter,
)
from core.investor_essence.financial_context import update_readiness
from core.investor_essence.investment_axis import validate_axis_completeness, build_confirmed_axis
from core.investor_essence.models import AnswerKind, FinancialContextSnapshot
from tools.investor_essence.adapters.portfolio_planning_adapter import PortfolioPlanningAdapter
from tools.archivist.write_broker import KnowledgeWriteBroker
from application.investor_essence.errors import PortfolioConflictError

results = []


def record(case: str, reproduced: bool, detail: dict) -> None:
    results.append({"case": case, "defect_reproduced": bool(reproduced), "observed": detail})
    assert reproduced, f"Expected current defect was not reproduced: {case}"


factory = FakeInvestorRuntimeUowFactory()
clock = FakeClock()
ids = FakeIdGenerator("audit")
interview_gen = FakeInterviewGenerator()
interview = InterviewService(factory, interview_gen, clock, ids)
review = ClaimReviewService(factory, FakeEssenceGenerator(), interview_gen, clock, ids)
confirm = ConfirmationService(factory, clock, ids)
session = interview.start_session()
for sequence in range(1, 11):
    question = session.current_question
    session = interview.record_answer(
        session.session_id, question["question_id"], AnswerKind.CHOICE,
        option_id=question["options"][0]["option_id"],
    )
    if sequence < 10:
        session = interview.advance_next_question(session.session_id)

qa = interview_gen.call_history[1].qa_pairs[0]
record("adaptive_question_loses_selected_option_text",
       "quote" not in qa["answer"] and "options" not in qa,
       {"payload_for_q2": qa})

summary = review.generate_summary(session.session_id)
confirmed = confirm.confirm_essence(
    session.session_id, accepted_claim_ids=[c["claim_id"] for c in summary.claims],
)
axis_gen = FakeAxisGenerator()
portfolio = FakePortfolioPlanningAdapter()
axis = InvestmentAxisService(factory, axis_gen, clock, ids, portfolio_planning=portfolio)
draft = axis.create_axis_draft("synthetic-A")
record("batch_confirmed_claims_lost_at_axis",
       confirmed.accepted_claims_count == 2 and not axis_gen.call_history[-1]["accepted_claims"],
       {"confirmed_count": confirmed.accepted_claims_count,
        "axis_accepted_claims": axis_gen.call_history[-1]["accepted_claims"],
        "current_essence_response_keys": sorted(confirm.get_current_confirmed_essence())})

context_service = FinancialContextService(factory, clock, ids, portfolio)
context_service.update_financial_context("synthetic-A", {
    "horizon_years": "5", "target_use_amount": "250000", "target_use_timeline": "2027-12-31",
    "emergency_reserves_months": "9", "withdrawal_amount": "12000",
    "withdrawal_frequency": "monthly", "experience_description": "beginner",
    "obligations_description": "dependent parent", "unknown_fields": [],
})
draft = axis.create_axis_draft("synthetic-A")
payload = axis_gen.call_history[-1]["financial_context"]
missing = [key for key in (
    "target_use_timeline", "emergency_reserves_months", "withdrawal_amount",
    "withdrawal_frequency", "experience_description", "obligations_description",
) if key not in payload]
record("financial_context_fields_lost_at_axis", len(missing) == 6, {"missing": missing})

invalid_examples = []
for key, value in (("horizon_years", "5-10 ปี"), ("emergency_reserves_months", "6 เดือน"),
                   ("target_use_amount", "500,000 บาท")):
    try:
        context_service.update_financial_context("synthetic-input", {key: value})
    except InvalidOperation:
        invalid_examples.append({"field": key, "input": value, "error": "decimal.InvalidOperation"})
record("form_placeholder_values_fail_backend", len(invalid_examples) == 3, {"failures": invalid_examples})

bad_ready = update_readiness(FinancialContextSnapshot(
    snapshot_id="unknown", portfolio_id="synthetic-A", unknown_fields=["horizon_years"],
))
record("unknown_horizon_missing_reserves_still_ready", bad_ready.is_ready_for_numeric_policy,
       {"ready": bad_ready.is_ready_for_numeric_policy, "issues": len(bad_ready.readiness_issues)})

draft = axis.update_axis_draft(draft.draft_id, {"risk_limits": {
    "mdd_max_annual": {"value": "15", "unit": "percent", "is_confirmed": True},
    "max_loss_per_trade": {"value": "2", "unit": "percent", "is_confirmed": True},
}})
axis.confirm_axis(draft.draft_id)
domain_draft = factory.open().planning.get_axis_draft(draft.draft_id)
bad_limits = {key: replace(value, value=Decimal("-10"), unit="", calculation_basis="")
              for key, value in domain_draft.risk_limits.items()}
issues = validate_axis_completeness(replace(domain_draft, risk_limits=bad_limits))
record("invalid_risk_numbers_and_units_pass_axis", not issues, {"issues": issues})

changed_rows = [replace(row, target_percent=Decimal("50")) for row in domain_draft.allocation_rows]
hash1 = build_confirmed_axis("c1", domain_draft, clock.now_utc()).content_hash
hash2 = build_confirmed_axis("c2", replace(domain_draft, allocation_rows=changed_rows), clock.now_utc()).content_hash
record("axis_content_hash_ignores_allocation", hash1 == hash2, {"equal_content_hash": hash1 == hash2})


class SyntheticPortfolio(FakePortfolioPlanningAdapter):
    def snapshot(self, portfolio_id):
        return replace(super().snapshot(portfolio_id),
                       targets=[{"bucket_id": "core_equities", "name": "Core Equities", "target_percent": 100}],
                       holdings_summary=[{"symbol": "TEST", "asset_type": "equity",
                                          "market_value_thb": 900000, "bucket_id": "core_equities"}])


bucket_gen = FakeBucketGenerator()
synthetic_portfolio = SyntheticPortfolio()
buckets = BucketPlanningService(factory, bucket_gen, synthetic_portfolio, clock, ids)
plan = buckets.create_bucket_plan_draft("synthetic-A")
holding_payload = bucket_gen.call_history[-1]["portfolio_snapshot"]["holdings"][0]
record("holdings_value_and_weight_are_zero_in_bucket_prompt",
       holding_payload["value_thb"] == "0" and holding_payload["weight_percent"] == "0",
       {"source_value_thb": 900000, "prompt_holding": holding_payload})
record("default_remapping_unassigns_existing_holdings", plan.remapping[0]["target_bucket_id"] is None,
       {"remapping": plan.remapping})
record("bucket_provenance_uses_placeholder_hashes", plan.axis_ref["content_hash"] == "hash",
       {"axis_ref": plan.axis_ref, "essence_ref": plan.essence_ref})

changed = [dict(row) for row in plan.purpose_buckets]
changed[0]["target_percent"] = "90"
changed[1]["target_percent"] = "10"
changed[0]["cash_inflow_routing_pct"] = "90"
changed[1]["cash_inflow_routing_pct"] = "90"
plan = buckets.update_bucket_plan_draft(plan.draft_id, {"purpose_buckets": changed})
record("inconsistent_matrix_and_180_percent_dca_pass_validation", plan.is_valid,
       {"is_valid": plan.is_valid, "bucket_targets": [b["target_percent"] for b in plan.purpose_buckets],
        "mapping_weights": plan.mapping_weights, "dca_sum": 180})
buckets.apply_bucket_plan(plan.draft_id, "audit-invalid-matrix")
record("invalid_matrix_is_forwarded_to_apply", len(synthetic_portfolio.applied_commands) == 1,
       {"apply_called": True})


class SyntheticRepo:
    def __init__(self):
        self.sequence = 1
        self.state = SimpleNamespace(allocation_targets=[], holdings=[
            SimpleNamespace(symbol="TEST", bucket_id="core_equities", market_value_thb=900000),
        ])
        self.store = SimpleNamespace(checkpoint=lambda pid: SimpleNamespace(sequence=self.sequence, state_hash="synthetic"))

    def load_state(self, pid=None):
        return self.state

    def unit_of_work(self, pid):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def commit(self, state, change):
        self.state = state
        self.sequence += 1


repo = SyntheticRepo()
adapter = PortfolioPlanningAdapter(repo=repo)
command = synthetic_portfolio.applied_commands[-1]
adapter.apply(command)
target = repo.state.allocation_targets[0].model_dump()
record("practical_policy_metadata_discarded_by_portfolio_apply",
       "eligible_assets" not in target and "cash_inflow_routing_pct" not in target and "action_rules" not in target,
       {"persisted_target_fields": sorted(target), "holding_bucket_after_apply": repo.state.holdings[0].bucket_id})

repo.state.holdings[0].bucket_id = "core_equities"
adapter.apply(replace(command, expected_checkpoint_sequence=repo.sequence,
                      remapping=[{"old_bucket_id": "core_equities", "target_bucket_id": "missing-bucket"}]))
record("nonexistent_remapping_destination_is_committed",
       repo.state.holdings[0].bucket_id == "missing-bucket",
       {"holding_bucket": repo.state.holdings[0].bucket_id,
        "target_bucket_ids": [t.bucket_id for t in repo.state.allocation_targets]})


class StrictBrokerProbe:
    rejected_dict = False

    def submit(self, command):
        self.rejected_dict = isinstance(command, dict)
        # Uses the production type guard, which raises before any I/O.
        return KnowledgeWriteBroker.submit(self, command)


probe = StrictBrokerProbe()
confirmed_with_broker = ConfirmationService(factory, clock, ids, knowledge_write=probe).confirm_essence(
    session.session_id, accepted_claim_ids=[c["claim_id"] for c in summary.claims],
)
record("production_knowledge_broker_rejects_confirmation_but_success_reported",
       probe.rejected_dict and confirmed_with_broker.status == "committed",
       {"broker_rejects_dict": probe.rejected_dict, "api_status": confirmed_with_broker.status,
        "message": confirmed_with_broker.message})

recovery_repo = SyntheticRepo()
recovery_adapter = PortfolioPlanningAdapter(repo=recovery_repo)
recovery_service = BucketPlanningService(factory, bucket_gen, recovery_adapter, clock, ids)
planning = factory.open().planning
original_save = planning.save_bucket_draft


def reject_applied_save(draft):
    raise RuntimeError("synthetic runtime persistence failure")


planning.save_bucket_draft = reject_applied_save
try:
    recovery_service.apply_bucket_plan(plan.draft_id, "audit-runtime-failure")
except RuntimeError:
    pass
finally:
    planning.save_bucket_draft = original_save
retry_conflicted = False
try:
    recovery_service.apply_bucket_plan(plan.draft_id, "audit-runtime-failure")
except PortfolioConflictError:
    retry_conflicted = True
record("portfolio_committed_runtime_failed_retry_cannot_recover",
       recovery_repo.sequence == 2 and retry_conflicted,
       {"portfolio_sequence": recovery_repo.sequence, "retry_conflicted": retry_conflicted,
        "portfolio_receipt_lookup": recovery_adapter.get_apply_receipt("synthetic-A", "audit")})

evidence_path = Path(__file__).with_name("backend_evidence.json")
evidence_path.write_text(json.dumps({"checks": len(results), "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"checks": len(results), "all_defects_reproduced": True,
                  "cases": [item["case"] for item in results]}, indent=2))
