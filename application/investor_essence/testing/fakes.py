"""Deterministic in-memory fakes for testing the Application Layer."""
from __future__ import annotations

from decimal import Decimal
from typing import Any, Dict, List, Optional

from core.investor_essence.interview import EssenceSession
from core.investor_essence.models import (
    ArtifactRef,
    BucketPlanDraft,
    EssenceSummaryDraft,
    FinancialContextSnapshot,
    InvestmentAxisDraft,
)
from application.investor_essence.dto import (
    AllocationApplyReceipt,
    AllocationPreview,
    ApplyAllocationCommand,
    AxisProposal,
    BucketPlanProposal,
    ContentOptionProposal,
    EssenceSummaryProposal,
    EvidenceSnapshot,
    OperationView,
    PortfolioPlanningSnapshot,
    PreviewAllocationCommand,
    QuestionProposal,
)
from application.investor_essence.ports import (
    AxisGeneratorPort,
    BucketGeneratorPort,
    ClockPort,
    ConfirmedKnowledgeReaderPort,
    EssenceGeneratorPort,
    IdGeneratorPort,
    IntentRepositoryPort,
    InterviewGeneratorPort,
    InvestorRuntimeUow,
    InvestorRuntimeUowFactory,
    KnowledgeWritePort,
    OperationRepositoryPort,
    PlanningRepositoryPort,
    PortfolioPlanningPort,
    SessionRepositoryPort,
)


class FakeClock(ClockPort):
    def __init__(self, initial_iso: str = "2026-10-08T12:00:00Z", initial_epoch: float = 1791460800.0) -> None:
        self._iso = initial_iso
        self._epoch = initial_epoch

    def now_utc(self) -> str:
        return self._iso

    def now_epoch(self) -> float:
        return self._epoch

    def advance(self, seconds: float) -> None:
        self._epoch += seconds


class FakeIdGenerator(IdGeneratorPort):
    def __init__(self, prefix: str = "test-id") -> None:
        self._prefix = prefix
        self._counter = 0

    def new_id(self) -> str:
        self._counter += 1
        return f"{self._prefix}-{self._counter}"


class FakeInterviewGenerator(InterviewGeneratorPort):
    def __init__(self, next_proposals: Optional[List[QuestionProposal]] = None) -> None:
        self.proposals = list(next_proposals or [])
        self.call_history: List[EvidenceSnapshot] = []

    def generate_next_question(self, evidence: EvidenceSnapshot, prompt_version: str) -> QuestionProposal:
        self.call_history.append(evidence)
        if self.proposals:
            return self.proposals.pop(0)
        return QuestionProposal(
            text="คุณอยากให้เงินช่วยอะไรในชีวิตมากที่สุด?",
            options=[
                ContentOptionProposal(key="A", text="อิสระในการใช้ชีวิต"),
                ContentOptionProposal(key="B", text="ความมั่นคงให้ครอบครัว"),
                ContentOptionProposal(key="C", text="รายได้สม่ำเสมอ"),
                ContentOptionProposal(key="D", text="ส่งต่อความมั่งคั่ง"),
            ],
            coverage_topics=["life_goals"],
            evidence_type="self_report",
        )

    def generate_clarification(self, evidence: EvidenceSnapshot, unresolved_topic: str, prompt_version: str) -> QuestionProposal:
        return QuestionProposal(
            text=f"ช่วยอธิบายเพิ่มเติมเกี่ยวกับ {unresolved_topic}",
            options=[
                ContentOptionProposal(key="A", text="ข้อ A"),
                ContentOptionProposal(key="B", text="ข้อ B"),
                ContentOptionProposal(key="C", text="ข้อ C"),
                ContentOptionProposal(key="D", text="ข้อ D"),
            ],
            coverage_topics=[unresolved_topic],
            evidence_type="self_report",
            is_clarification=True,
        )


class InMemorySessionRepository(SessionRepositoryPort):
    def __init__(self) -> None:
        self.sessions: Dict[str, EssenceSession] = {}
        self.summaries: Dict[str, EssenceSummaryDraft] = {}

    def get(self, session_id: str) -> Optional[EssenceSession]:
        return self.sessions.get(session_id)

    def get_current(self, scope: str = "workspace") -> Optional[EssenceSession]:
        for s in reversed(list(self.sessions.values())):
            if s.scope == scope:
                return s
        return None

    def save(self, session: EssenceSession) -> None:
        self.sessions[session.session_id] = session

    def get_summary(self, session_id: str) -> Optional[EssenceSummaryDraft]:
        return self.summaries.get(session_id)

    def save_summary(self, summary: EssenceSummaryDraft) -> None:
        self.summaries[summary.session_id] = summary


class InMemoryPlanningRepository(PlanningRepositoryPort):
    def __init__(self) -> None:
        self.snapshots: Dict[str, FinancialContextSnapshot] = {}
        self.axis_drafts: Dict[str, InvestmentAxisDraft] = {}
        self.bucket_drafts: Dict[str, BucketPlanDraft] = {}
        self.pointers: Dict[str, str] = {}

    def save_context_snapshot(self, snapshot: FinancialContextSnapshot) -> None:
        self.snapshots[snapshot.portfolio_id] = snapshot

    def get_context_snapshot(self, portfolio_id: str) -> Optional[FinancialContextSnapshot]:
        return self.snapshots.get(portfolio_id)

    def save_axis_draft(self, draft: InvestmentAxisDraft) -> None:
        self.axis_drafts[draft.draft_id] = draft

    def get_axis_draft(self, draft_id: str) -> Optional[InvestmentAxisDraft]:
        return self.axis_drafts.get(draft_id)

    def get_latest_axis_draft(self, portfolio_id: str) -> Optional[InvestmentAxisDraft]:
        matches = [d for d in self.axis_drafts.values() if d.portfolio_id == portfolio_id]
        return matches[-1] if matches else None

    def save_bucket_draft(self, draft: BucketPlanDraft) -> None:
        self.bucket_drafts[draft.draft_id] = draft

    def get_bucket_draft(self, draft_id: str) -> Optional[BucketPlanDraft]:
        return self.bucket_drafts.get(draft_id)

    def get_latest_bucket_draft(self, portfolio_id: str) -> Optional[BucketPlanDraft]:
        matches = [d for d in self.bucket_drafts.values() if d.portfolio_id == portfolio_id]
        return matches[-1] if matches else None

    def get_confirmed_pointer(self, scope: str, kind: str) -> Optional[str]:
        return self.pointers.get(f"{scope}:{kind}")

    def get_confirmed_snapshot(self, scope: str, kind: str) -> Optional[Dict[str, Any]]:
        return getattr(self, "_snapshots_dict", {}).get(f"{scope}:{kind}")

    def set_confirmed_pointer(
        self,
        scope: str,
        kind: str,
        artifact_ref: str,
        expected_ref: Optional[str],
        snapshot: Optional[Dict[str, Any]] = None,
    ) -> bool:
        key = f"{scope}:{kind}"
        actual = self.pointers.get(key)
        if actual != expected_ref:
            return False
        self.pointers[key] = artifact_ref
        if not hasattr(self, "_snapshots_dict"):
            self._snapshots_dict = {}
        if snapshot is not None:
            self._snapshots_dict[key] = snapshot
        return True


class InMemoryOperationRepository(OperationRepositoryPort):
    def __init__(self) -> None:
        self.ops: Dict[str, OperationView] = {}
        self.enqueued: List[Dict[str, Any]] = []

    def enqueue(
        self,
        operation_id: str,
        stage: str,
        resource_id: str,
        resource_revision: int,
        input_hash: str,
        prompt_version: str,
        frozen_input: Dict[str, Any],
    ) -> None:
        op = OperationView(
            operation_id=operation_id,
            stage=stage,
            resource_id=resource_id,
            resource_revision=resource_revision,
            status="queued",
            attempt=1,
            poll_url=f"/api/investor/operations/{operation_id}",
        )
        self.ops[operation_id] = op
        self.enqueued.append({"op": op, "frozen_input": frozen_input})

    def get(self, operation_id: str) -> Optional[OperationView]:
        return self.ops.get(operation_id)

    def claim_lease(self, worker_id: str, lease_seconds: int) -> Optional[Dict[str, Any]]:
        for item in self.enqueued:
            if item["op"].status == "queued":
                op = OperationView(
                    operation_id=item["op"].operation_id,
                    stage=item["op"].stage,
                    resource_id=item["op"].resource_id,
                    resource_revision=item["op"].resource_revision,
                    status="running",
                    attempt=item["op"].attempt,
                    poll_url=item["op"].poll_url,
                )
                item["op"] = op
                self.ops[op.operation_id] = op
                return {
                    "operation_id": op.operation_id,
                    "task_type": op.stage,
                    "resource_id": op.resource_id,
                    "payload": {
                        "resource_revision": op.resource_revision,
                        "frozen_input": item.get("frozen_input", {}),
                    },
                    "fencing_token": 1,
                }
        return None

    def complete_with_fence(self, operation_id: str, fence: int, result: Dict[str, Any]) -> bool:
        if operation_id in self.ops:
            op = self.ops[operation_id]
            self.ops[operation_id] = OperationView(
                operation_id=op.operation_id,
                stage=op.stage,
                resource_id=op.resource_id,
                resource_revision=op.resource_revision,
                status="succeeded",
                attempt=op.attempt,
                poll_url=op.poll_url,
                result_ref=result.get("ref"),
                result=result,
            )
            return True
        return False

    def fail_with_fence(self, operation_id: str, fence: int, error_code: str, error_message: str, retryable: bool) -> bool:
        if operation_id in self.ops:
            op = self.ops[operation_id]
            self.ops[operation_id] = OperationView(
                operation_id=op.operation_id,
                stage=op.stage,
                resource_id=op.resource_id,
                resource_revision=op.resource_revision,
                status="failed",
                attempt=op.attempt,
                poll_url=op.poll_url,
                error_code=error_code,
                error_message=error_message,
                retryable=retryable,
            )
            return True
        return False


class InMemoryIntentRepository(IntentRepositoryPort):
    def __init__(self) -> None:
        self.intents: Dict[str, Dict[str, Any]] = {}
        self.command_receipts: Dict[str, Dict[str, Any]] = {}

    def save_intent(self, intent_id: str, scope: str, kind: str, idempotency_key: str, request_hash: str, payload: Dict[str, Any]) -> None:
        self.intents[f"{scope}:{kind}:{idempotency_key}"] = {
            "intent_id": intent_id,
            "scope": scope,
            "kind": kind,
            "idempotency_key": idempotency_key,
            "request_hash": request_hash,
            "payload": payload,
            "status": "pending",
        }

    def get_intent_by_idempotency(self, scope: str, kind: str, idempotency_key: str) -> Optional[Dict[str, Any]]:
        return self.intents.get(f"{scope}:{kind}:{idempotency_key}")

    def update_intent_receipt(self, intent_id: str, receipt: Dict[str, Any], status: str) -> None:
        for it in self.intents.values():
            if it["intent_id"] == intent_id:
                it["receipt"] = receipt
                it["status"] = status
                break

    def save_command_receipt(self, scope: str, use_case: str, idempotency_key: str, request_hash: str, result: Dict[str, Any]) -> None:
        self.command_receipts[f"{scope}:{use_case}:{idempotency_key}"] = {
            "request_hash": request_hash,
            "result": result,
        }

    def get_command_receipt(self, scope: str, use_case: str, idempotency_key: str) -> Optional[Dict[str, Any]]:
        return self.command_receipts.get(f"{scope}:{use_case}:{idempotency_key}")


class FakeInvestorRuntimeUow(InvestorRuntimeUow):
    def __init__(self) -> None:
        self.sessions = InMemorySessionRepository()
        self.planning = InMemoryPlanningRepository()
        self.operations = InMemoryOperationRepository()
        self.intents = InMemoryIntentRepository()
        self.committed = False

    def __enter__(self) -> "FakeInvestorRuntimeUow":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        pass

    def commit(self) -> None:
        self.committed = True

    def rollback(self) -> None:
        pass


class FakeInvestorRuntimeUowFactory(InvestorRuntimeUowFactory):
    def __init__(self, uow: Optional[FakeInvestorRuntimeUow] = None) -> None:
        self._uow = uow or FakeInvestorRuntimeUow()

    def open(self) -> InvestorRuntimeUow:
        return self._uow


class FakeEssenceGenerator(EssenceGeneratorPort):
    def __init__(self, proposal: Optional[EssenceSummaryProposal] = None) -> None:
        self._proposal = proposal
        self.call_history: List[EvidenceSnapshot] = []

    def generate_summary(self, evidence: EvidenceSnapshot, prompt_version: str) -> EssenceSummaryProposal:
        self.call_history.append(evidence)
        if self._proposal:
            return self._proposal

        from application.investor_essence.dto import ClaimProposal
        return EssenceSummaryProposal(
            statement="ลงทุนเพื่อสร้างอิสรภาพทางการเงินและความมั่นคงในระยะยาว",
            claims=[
                ClaimProposal(
                    text="ให้ความสำคัญกับอิสระและผลตอบแทนที่สม่ำเสมอ",
                    source_kind="ai_inferred",
                    supporting_question_ids=[evidence.qa_pairs[0]["question_id"]] if evidence.qa_pairs else [],
                    quote="อิสระในการใช้ชีวิต",
                    evidence_type="self_report",
                ),
                ClaimProposal(
                    text="เน้นการเติบโตแบบยั่งยืนและรักษาวินัย",
                    source_kind="user_stated",
                    supporting_question_ids=[evidence.qa_pairs[1]["question_id"]] if len(evidence.qa_pairs) > 1 else [],
                    quote="ความมั่นคงให้ครอบครัว",
                    evidence_type="self_report",
                ),
            ],
            unresolved_topics=["liquidity_needs"],
            coverage_report=[
                {"topic": "life_goals", "status": "covered", "supporting_answer_ids": ["a1"]},
                {"topic": "risk_tolerance", "status": "covered", "supporting_answer_ids": ["a2"]},
            ],
        )


class FakeAxisGenerator(AxisGeneratorPort):
    def __init__(self, proposal: Optional[AxisProposal] = None) -> None:
        self._proposal = proposal
        self.call_history: List[Dict[str, Any]] = []

    def generate_axis(
        self,
        accepted_claims: List[Dict[str, Any]],
        financial_context: Dict[str, Any],
        prompt_version: str,
    ) -> AxisProposal:
        self.call_history.append({
            "accepted_claims": accepted_claims,
            "financial_context": financial_context,
            "prompt_version": prompt_version,
        })
        if self._proposal:
            return self._proposal

        from application.investor_essence.dto import AllocationRowDTO, NumericPolicyFieldDTO
        return AxisProposal(
            basic_policy="เน้นลงทุนระยะยาวในสินทรัพย์คุณภาพดี กระจายความเสี่ยงอย่างรอบคอบ",
            risk_limits={
                "max_drawdown_percent": NumericPolicyFieldDTO(
                    field_id="max_drawdown_percent",
                    value="15.00",
                    unit="%",
                    calculation_basis="NAV",
                    origin="ai_proposal",
                    assumptions="สมมติฐานความเสี่ยงปานกลาง",
                ),
                "max_single_loss_percent": NumericPolicyFieldDTO(
                    field_id="max_single_loss_percent",
                    value="2.00",
                    unit="%",
                    calculation_basis="trade_capital",
                    origin="ai_proposal",
                ),
            },
            invest_targets=["หุ้นเติบโตคุณภาพสูง", "กองทุนดัชนีโลก"],
            exclude_targets=["หุ้นเก็งกำไรไร้พื้นฐาน", "อนุพันธ์ที่มีเลเวอเรจสูง"],
            primary_methods=["DCA รายเดือนตามแผน", "วิเคราะห์ปัจจัยพื้นฐาน"],
            secondary_methods=["Rebalance เมื่อสัดส่วนเบี่ยงเบนเกิน 5%"],
            investment_horizon="5-10 ปี",
            allocation_basis="purpose",
            allocation_rows=[
                AllocationRowDTO(
                    allocation_id="alloc-1",
                    category_name="เติบโตระยะยาว",
                    target_percent="70.00",
                    role_description="สร้างผลตอบแทนทบต้น",
                ),
                AllocationRowDTO(
                    allocation_id="alloc-2",
                    category_name="สภาพคล่องและเงินสำรอง",
                    target_percent="30.00",
                    role_description="รองรับความผันผวนและค่าใช้จ่าย",
                ),
            ],
            rebalance_frequency="ราย 6 เดือน",
            role_models=["Warren Buffett", "Charlie Munger"],
            non_actions=[
                "ไม่ซื้อขายตามอารมณ์หรือกระแสข่าวระยะสั้น",
                "ไม่กู้ยืมเงินมาลงทุนในสินทรัพย์เสี่ยง",
                "ไม่ลงทุนในธุรกิจที่ไม่เข้าใจโมเดลการสร้างกระแสเงินสด",
            ],
            assumptions=["รายได้มั่นคงไม่มีภาระหนี้เร่งด่วน"],
            clarifications=[],
        )


class FakeBucketGenerator(BucketGeneratorPort):
    def __init__(self, proposal: Optional[BucketPlanProposal] = None) -> None:
        self._proposal = proposal
        self.call_history: List[Dict[str, Any]] = []

    def generate_buckets(self, confirmed_axis: Dict[str, Any], prompt_version: str) -> BucketPlanProposal:
        self.call_history.append({"confirmed_axis": confirmed_axis, "prompt_version": prompt_version})
        if self._proposal:
            return self._proposal

        from application.investor_essence.dto import PurposeBucketProposal
        return BucketPlanProposal(
            purpose_buckets=[
                PurposeBucketProposal(
                    bucket_id="b-growth",
                    name="เติบโตระยะยาว",
                    role="ลงทุนสร้างผลตอบแทนทบต้นระยะยาว",
                    color="#3B82F6",
                    target_percent="70.00",
                    source_axis_allocation_ids=["alloc-1"],
                ),
                PurposeBucketProposal(
                    bucket_id="b-safety",
                    name="เงินพร้อมใช้และปลอดภัย",
                    role="สภาพคล่องและรองรับความเสี่ยง",
                    color="#10B981",
                    target_percent="30.00",
                    source_axis_allocation_ids=["alloc-2"],
                ),
            ],
            allocation_basis="purpose",
            mapping_weights=[
                {"axis_allocation_id": "alloc-1", "bucket_id": "b-growth", "portfolio_weight_percent": "70.00"},
                {"axis_allocation_id": "alloc-2", "bucket_id": "b-safety", "portfolio_weight_percent": "30.00"},
            ],
            constraints=["ห้ามสัดส่วนเงินสำรองต่ำกว่า 20%"],
        )
