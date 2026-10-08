"""Unit tests for InvestmentAxis pure domain validation and confirmation."""
from decimal import Decimal
import pytest

from core.investor_essence.models import (
    AllocationBasis,
    AllocationPlanRow,
    ArtifactRef,
    InvestmentAxisDraft,
    NumericPolicyField,
    NumericPolicyOrigin,
)
from core.investor_essence.investment_axis import (
    build_confirmed_axis,
    validate_axis_completeness,
)


def _make_valid_draft() -> InvestmentAxisDraft:
    risk_limits = {
        "mdd_max_annual": NumericPolicyField(
            field_id="mdd_max_annual",
            value=Decimal("15.00"),
            unit="percent",
            calculation_basis="annual_nav_drawdown",
            origin=NumericPolicyOrigin.USER_INPUT,
            is_confirmed=True,
        ),
        "max_loss_per_trade": NumericPolicyField(
            field_id="max_loss_per_trade",
            value=Decimal("2.00"),
            unit="percent",
            calculation_basis="portfolio_nav_at_entry",
            origin=NumericPolicyOrigin.USER_INPUT,
            is_confirmed=True,
        ),
    }

    return InvestmentAxisDraft(
        draft_id="axis_draft_1",
        portfolio_id="port_main",
        essence_ref=ArtifactRef(
            document_key="investor-essence/current",
            note_id="note_ess_1",
            revision_id="rev_1",
            content_hash="abc",
            artifact_set_hash="set_hash_1",
        ),
        context_ref="fc_snap_1",
        basic_policy="ลงทุนเพื่อสร้างกระแสเงินสดและเติบโตอย่างยั่งยืน",
        risk_limits=risk_limits,
        invest_targets=["หุ้นปันผลคุณภาพสูง", "กองทุนรวมตราสารหนี้"],
        exclude_targets=["หุ้นปั่นที่ไม่มีกำไร", "คริปโตเคอร์เรนซีเก็งกำไร"],
        primary_methods=["คัดเลือกหุ้นคุณค่าที่มี Dividend Yield > 4%"],
        secondary_methods=["DCA รายเดือนเพื่อเฉลี่ยต้นทุน"],
        investment_horizon="ระยะยาวมากกว่า 7 ปี",
        rebalance_frequency="ทุก 6 เดือน หรือเมื่อสัดส่วนเบี่ยงเบนเกิน 5%",
        allocation_basis=AllocationBasis.ASSET_CLASS,
        allocation_rows=[
            AllocationPlanRow(
                allocation_id="cat_stock",
                category_name="หุ้นปันผล",
                target_percent=Decimal("70.00"),
                role_description="สร้างกระแสเงินสด",
            ),
            AllocationPlanRow(
                allocation_id="cat_bond",
                category_name="ตราสารหนี้",
                target_percent=Decimal("30.00"),
                role_description="ลดความผันผวน",
            ),
        ],
        role_models=["Warren Buffett (หลักการลงทุนในธุรกิจที่เข้าใจและมีคูเมือง)"],
        non_actions=[
            "จะไม่ใช้ Leverage หรือกู้ยืมเงินมาลงทุนเด็ดขาด",
            "จะไม่ซื้อหุ้นตามกระแสโซเชียลมีเดียโดยไม่ศึกษาข้อมูล",
            "จะไม่ขายหุ้นเพราะความตื่นตระหนกของตลาดระยะสั้น",
        ],
        numeric_fields=risk_limits,
    )


class TestInvestmentAxisValidation:
    def test_valid_draft_has_no_errors(self):
        draft = _make_valid_draft()
        errors = validate_axis_completeness(draft)
        assert len(errors) == 0

    def test_missing_basic_policy_fails(self):
        draft = _make_valid_draft()
        draft = InvestmentAxisDraft(
            draft_id=draft.draft_id,
            portfolio_id=draft.portfolio_id,
            essence_ref=draft.essence_ref,
            context_ref=draft.context_ref,
            basic_policy="",
            risk_limits=draft.risk_limits,
            invest_targets=draft.invest_targets,
            exclude_targets=draft.exclude_targets,
            primary_methods=draft.primary_methods,
            secondary_methods=draft.secondary_methods,
            investment_horizon=draft.investment_horizon,
            allocation_basis=draft.allocation_basis,
            allocation_rows=draft.allocation_rows,
            rebalance_frequency=draft.rebalance_frequency,
            role_models=draft.role_models,
            non_actions=draft.non_actions,
            numeric_fields=draft.numeric_fields,
        )
        errors = validate_axis_completeness(draft)
        assert any("Section 1" in e for e in errors)

    def test_unconfirmed_risk_limits_fail(self):
        draft = _make_valid_draft()
        unconfirmed_mdd = NumericPolicyField(
            field_id="mdd_max_annual",
            value=Decimal("15.00"),
            unit="percent",
            calculation_basis="annual_nav_drawdown",
            origin=NumericPolicyOrigin.AI_PROPOSAL,
            is_confirmed=False,
        )
        updated_limits = dict(draft.risk_limits)
        updated_limits["mdd_max_annual"] = unconfirmed_mdd
        draft = InvestmentAxisDraft(
            draft_id=draft.draft_id,
            portfolio_id=draft.portfolio_id,
            essence_ref=draft.essence_ref,
            context_ref=draft.context_ref,
            basic_policy=draft.basic_policy,
            risk_limits=updated_limits,
            invest_targets=draft.invest_targets,
            exclude_targets=draft.exclude_targets,
            primary_methods=draft.primary_methods,
            secondary_methods=draft.secondary_methods,
            investment_horizon=draft.investment_horizon,
            allocation_basis=draft.allocation_basis,
            allocation_rows=draft.allocation_rows,
            rebalance_frequency=draft.rebalance_frequency,
            role_models=draft.role_models,
            non_actions=draft.non_actions,
            numeric_fields=draft.numeric_fields,
        )
        errors = validate_axis_completeness(draft)
        assert any("MDD สูงสุดต่อปี" in e for e in errors)

    def test_allocation_sum_not_100_fails(self):
        draft = _make_valid_draft()
        bad_rows = [
            AllocationPlanRow(
                allocation_id="cat_stock",
                category_name="หุ้น",
                target_percent=Decimal("50.00"),
                role_description="หุ้น",
            ),
            AllocationPlanRow(
                allocation_id="cat_bond",
                category_name="ตราสารหนี้",
                target_percent=Decimal("30.00"),
                role_description="ตราสารหนี้",
            ),
        ]
        draft = InvestmentAxisDraft(
            draft_id=draft.draft_id,
            portfolio_id=draft.portfolio_id,
            essence_ref=draft.essence_ref,
            context_ref=draft.context_ref,
            basic_policy=draft.basic_policy,
            risk_limits=draft.risk_limits,
            invest_targets=draft.invest_targets,
            exclude_targets=draft.exclude_targets,
            primary_methods=draft.primary_methods,
            secondary_methods=draft.secondary_methods,
            investment_horizon=draft.investment_horizon,
            allocation_basis=draft.allocation_basis,
            allocation_rows=bad_rows,
            rebalance_frequency=draft.rebalance_frequency,
            role_models=draft.role_models,
            non_actions=draft.non_actions,
            numeric_fields=draft.numeric_fields,
        )
        errors = validate_axis_completeness(draft)
        assert any("สัดส่วนรวมต้องเท่ากับ 100%" in e for e in errors)

    def test_fewer_than_3_non_actions_fails(self):
        draft = _make_valid_draft()
        draft = InvestmentAxisDraft(
            draft_id=draft.draft_id,
            portfolio_id=draft.portfolio_id,
            essence_ref=draft.essence_ref,
            context_ref=draft.context_ref,
            basic_policy=draft.basic_policy,
            risk_limits=draft.risk_limits,
            invest_targets=draft.invest_targets,
            exclude_targets=draft.exclude_targets,
            primary_methods=draft.primary_methods,
            secondary_methods=draft.secondary_methods,
            investment_horizon=draft.investment_horizon,
            allocation_basis=draft.allocation_basis,
            allocation_rows=draft.allocation_rows,
            rebalance_frequency=draft.rebalance_frequency,
            role_models=draft.role_models,
            non_actions=[
                "ไม่กู้เงินมาลงทุน",
                "ไม่ซื้อหุ้นปั่น",
            ],
            numeric_fields=draft.numeric_fields,
        )
        errors = validate_axis_completeness(draft)
        assert any("Section 8 (สิ่งที่จะไม่ทำ)" in e for e in errors)

    def test_duplicate_non_actions_fails(self):
        draft = _make_valid_draft()
        draft = InvestmentAxisDraft(
            draft_id=draft.draft_id,
            portfolio_id=draft.portfolio_id,
            essence_ref=draft.essence_ref,
            context_ref=draft.context_ref,
            basic_policy=draft.basic_policy,
            risk_limits=draft.risk_limits,
            invest_targets=draft.invest_targets,
            exclude_targets=draft.exclude_targets,
            primary_methods=draft.primary_methods,
            secondary_methods=draft.secondary_methods,
            investment_horizon=draft.investment_horizon,
            allocation_basis=draft.allocation_basis,
            allocation_rows=draft.allocation_rows,
            rebalance_frequency=draft.rebalance_frequency,
            role_models=draft.role_models,
            non_actions=[
                "ไม่กู้เงินมาลงทุน",
                "ไม่ซื้อหุ้นปั่น",
                "ไม่กู้เงินมาลงทุน",  # duplicate
            ],
            numeric_fields=draft.numeric_fields,
        )
        errors = validate_axis_completeness(draft)
        assert any("อย่างน้อย 3 ข้อที่ไม่ซ้ำกัน" in e for e in errors)

    def test_build_confirmed_axis_success(self):
        draft = _make_valid_draft()
        confirmed = build_confirmed_axis(
            confirmation_id="axis_conf_1",
            draft=draft,
            confirmed_at_iso="2026-10-08T11:00:00Z",
        )
        assert confirmed.confirmation_id == "axis_conf_1"
        assert confirmed.portfolio_id == "port_main"
        assert confirmed.complete_sections["basic_policy"] == draft.basic_policy
        assert len(confirmed.non_actions) == 3
        assert confirmed.content_hash is not None
        assert len(confirmed.content_hash) == 64

    def test_build_confirmed_axis_incomplete_fails(self):
        draft = _make_valid_draft()
        draft = InvestmentAxisDraft(
            draft_id=draft.draft_id,
            portfolio_id=draft.portfolio_id,
            essence_ref=draft.essence_ref,
            context_ref=draft.context_ref,
            basic_policy="",
            risk_limits=draft.risk_limits,
            invest_targets=draft.invest_targets,
            exclude_targets=draft.exclude_targets,
            primary_methods=draft.primary_methods,
            secondary_methods=draft.secondary_methods,
            investment_horizon=draft.investment_horizon,
            allocation_basis=draft.allocation_basis,
            allocation_rows=draft.allocation_rows,
            rebalance_frequency=draft.rebalance_frequency,
            role_models=draft.role_models,
            non_actions=draft.non_actions,
            numeric_fields=draft.numeric_fields,
        )
        with pytest.raises(ValueError, match="Cannot confirm investment axis with incomplete sections"):
            build_confirmed_axis(
                confirmation_id="axis_conf_1",
                draft=draft,
                confirmed_at_iso="2026-10-08T11:00:00Z",
            )
