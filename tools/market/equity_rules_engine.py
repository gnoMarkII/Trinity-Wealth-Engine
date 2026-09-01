"""Deterministic 4-Pillar Equity Rules Engine & Thesis Scorecard.

คำนวณ Core Conviction Score, Execution Readiness Score, Action Stance, Coverage %, และ Thesis Falsifiers
ด้วย Python 100% (Deterministic Invariant) ปราศจากการสุ่มหรือการประเมินตัวเลขโดย LLM
"""
from typing import Any, Dict, List, Optional, Tuple
from schemas.micro_quant_schemas import (
    DeterministicScorecard,
    EarningsGuidanceContext,
    InsiderConviction,
    PiotroskiFScoreBreakdown,
    QuantSignals,
    ReverseDCFResult,
    TacticalSetup,
    ThesisFalsifier,
)


def compute_deterministic_scorecard(
    ticker: str,
    market: str,
    quant_signals: QuantSignals,
    piotroski: Optional[PiotroskiFScoreBreakdown] = None,
    dcf: Optional[ReverseDCFResult] = None,
    guidance: Optional[EarningsGuidanceContext] = None,
    tactical: Optional[TacticalSetup] = None,
    insider: Optional[InsiderConviction] = None,
) -> Tuple[DeterministicScorecard, List[ThesisFalsifier], List[str]]:
    """คำนวณคะแนน 4 เสาหลักและสร้าง Thesis Falsifiers พร้อม Source References บังคับ"""
    flags: List[str] = []
    data_quality_flags: List[str] = list(quant_signals.data_quality_flags)

    # -------------------------------------------------------------
    # 1. Pillar 1: Fundamental Quality Score (0 - 100)
    # -------------------------------------------------------------
    fund_scores = []
    if piotroski and piotroski.is_eligible and piotroski.f_score is not None:
        fund_scores.append((piotroski.f_score / 9.0) * 100.0)
    elif quant_signals.quality_score is not None:
        fund_scores.append(quant_signals.quality_score)

    if quant_signals.ocf_to_net_income is not None:
        ratio = quant_signals.ocf_to_net_income
        if ratio >= 1.2:
            fund_scores.append(100.0)
        elif ratio >= 0.9:
            fund_scores.append(80.0)
        elif ratio >= 0.5:
            fund_scores.append(50.0)
        else:
            fund_scores.append(20.0)

    if quant_signals.roic_pct is not None:
        roic = quant_signals.roic_pct
        if roic >= 20.0:
            fund_scores.append(95.0)
        elif roic >= 12.0:
            fund_scores.append(80.0)
        elif roic >= 6.0:
            fund_scores.append(55.0)
        else:
            fund_scores.append(30.0)

    fund_quality_score = (sum(fund_scores) / len(fund_scores)) if fund_scores else 50.0

    # -------------------------------------------------------------
    # 2. Pillar 2: Guidance & Expectation Gap Score (0 - 100)
    # -------------------------------------------------------------
    mgmt_scores = []
    if guidance and guidance.status == "available":
        margin_traj = getattr(guidance, "operating_margin_trajectory", None) or getattr(guidance, "margin_guidance_direction", None)
        if margin_traj == "expanding":
            mgmt_scores.append(90.0)
        elif margin_traj == "stable":
            mgmt_scores.append(70.0)
        elif margin_traj == "contracting":
            mgmt_scores.append(35.0)

        rev_g = getattr(guidance, "revenue_growth_guidance_pct", None) or getattr(guidance, "revenue_guidance_yoy_pct", None)
        if rev_g is not None:
            if rev_g >= 20.0:
                mgmt_scores.append(95.0)
            elif rev_g >= 10.0:
                mgmt_scores.append(80.0)
            elif rev_g >= 0.0:
                mgmt_scores.append(60.0)
            else:
                mgmt_scores.append(30.0)

    analyst_scores = []
    if quant_signals.eps_revision_net_30d is not None:
        rev_net = quant_signals.eps_revision_net_30d
        if rev_net > 2:
            analyst_scores.append(90.0)
        elif rev_net >= 0:
            analyst_scores.append(65.0)
        else:
            analyst_scores.append(35.0)
    elif quant_signals.eps_estimate_change_30d_pct is not None:
        chg = quant_signals.eps_estimate_change_30d_pct
        if chg >= 5.0:
            analyst_scores.append(90.0)
        elif chg >= 0.0:
            analyst_scores.append(65.0)
        else:
            analyst_scores.append(35.0)

    management_guidance_score = round(sum(mgmt_scores) / len(mgmt_scores), 1) if mgmt_scores else None
    analyst_expectations_score = round(sum(analyst_scores) / len(analyst_scores), 1) if analyst_scores else None

    if management_guidance_score is not None and analyst_expectations_score is not None:
        guidance_expectation_score = round((management_guidance_score * 0.5) + (analyst_expectations_score * 0.5), 1)
    elif management_guidance_score is not None:
        guidance_expectation_score = management_guidance_score
    elif analyst_expectations_score is not None:
        guidance_expectation_score = analyst_expectations_score
    else:
        guidance_expectation_score = 50.0

    # -------------------------------------------------------------
    # 3. Pillar 3: Valuation Margin of Safety Score (0 - 100)
    # -------------------------------------------------------------
    val_scores = []
    dcf_is_actionable = getattr(dcf, "is_actionable", True) if dcf else True
    if dcf and dcf.is_eligible and dcf.status == "available" and dcf.upside_12m_pct is not None and dcf_is_actionable:
        upside = dcf.upside_12m_pct
        if upside >= 30.0:
            val_scores.append(95.0)
        elif upside >= 15.0:
            val_scores.append(80.0)
        elif upside >= 0.0:
            val_scores.append(60.0)
        elif upside >= -15.0:
            val_scores.append(40.0)
        else:
            val_scores.append(20.0)

        if dcf.market_implied_growth_pct is not None:
            imp_g = dcf.market_implied_growth_pct
            # If market implies < 8% growth for solid company -> attractive expectation gap
            if imp_g <= 8.0:
                val_scores.append(85.0)
            elif imp_g <= 18.0:
                val_scores.append(65.0)
            else:
                val_scores.append(40.0)
    elif quant_signals.value_score is not None and dcf_is_actionable:
        val_scores.append(quant_signals.value_score)

    has_actionable_valuation = len(val_scores) > 0
    valuation_margin_score = round(sum(val_scores) / len(val_scores), 1) if has_actionable_valuation else None

    if not has_actionable_valuation and "valuation_not_actionable:scorecard" not in data_quality_flags:
        data_quality_flags.append("valuation_not_actionable:scorecard")

    # -------------------------------------------------------------
    # Dual Conviction Scores (Business vs Investment)
    # -------------------------------------------------------------
    business_conviction = round(
        max(1.0, min(10.0, ((fund_quality_score * 0.55) + (guidance_expectation_score * 0.45)) / 10.0)),
        1,
    )

    if valuation_margin_score is not None:
        investment_conviction = round(
            max(1.0, min(10.0, ((fund_quality_score * 0.40) + (guidance_expectation_score * 0.30) + (valuation_margin_score * 0.30)) / 10.0)),
            1,
        )
        reweighting_metadata = None
        core_conviction = investment_conviction
    else:
        investment_conviction = None
        reweighting_metadata = {
            "excluded_pillars": ["valuation"],
            "weights_used": {
                "fundamental": 0.55,
                "expectations": 0.45,
            },
            "reason": dcf.actionability_reason if (dcf and getattr(dcf, "actionability_reason", None)) else "non_actionable_macro_anomaly",
        }
        core_conviction = business_conviction

    # -------------------------------------------------------------
    # 4. Pillar 4: Execution Readiness Score (0 - 100 -> 1.0 - 10.0)
    # -------------------------------------------------------------
    setup_readiness_score: Optional[float] = None
    setup_reason: Optional[str] = None

    if tactical and tactical.status == "available":
        pb_status = getattr(tactical, "pullback_entry_status", "unavailable")
        bo_status = getattr(tactical, "breakout_entry_status", "pre_trigger")
        bo_eligible = getattr(tactical, "breakout_entry_eligible", False)
        curr_rr = tactical.current_rr_ratio
        bo_rr = tactical.breakout_current_rr

        # Evaluate actionable / non-actionable setup matrix
        if bo_eligible and bo_rr is not None and bo_rr >= 1.50:
            setup_readiness_score = 90.0
            setup_reason = "breakout_eligible"
        elif pb_status == "in_buy_zone" and curr_rr is not None and curr_rr >= 2.0:
            setup_readiness_score = 90.0
            setup_reason = "in_buy_zone_high_rr"
        elif pb_status == "in_buy_zone" and curr_rr is not None and curr_rr >= 1.50:
            setup_readiness_score = 75.0
            setup_reason = "in_buy_zone_moderate_rr"
        elif bo_status == "pre_trigger" and pb_status not in ("at_or_above_target", "below_stop"):
            setup_readiness_score = 55.0
            setup_reason = "pre_trigger_structure_intact"
        elif pb_status == "between_zone_and_target" and bo_status != "chased":
            setup_readiness_score = 35.0
            setup_reason = "between_buy_zone_and_target"
        elif pb_status == "at_or_above_target" and bo_status == "chased":
            setup_readiness_score = 20.0
            setup_reason = "breakout_chased_and_pullback_above_target"
        elif bo_status == "chased":
            setup_readiness_score = 20.0
            setup_reason = "breakout_chased"
        elif pb_status == "at_or_above_target":
            setup_readiness_score = 25.0
            setup_reason = "pullback_above_target"
        elif bo_status == "expired" or pb_status == "below_stop":
            setup_readiness_score = 10.0
            setup_reason = "setup_expired_or_below_stop"
        else:
            setup_readiness_score = 35.0
            setup_reason = "tactical_non_actionable"

    exec_scores = []
    stage_score = 50.0
    if tactical and tactical.status == "available":
        stage_map = {
            "STAGE_2_MARKUP": 85.0,
            "STAGE_1_BASE": 65.0,
            "STAGE_3_DISTRIBUTION": 45.0,
            "STAGE_4_MARKDOWN": 25.0,
            "UNKNOWN": 50.0,
        }
        stage_score = stage_map.get(tactical.price_stage, 50.0)
        exec_scores.append(stage_score)

        if setup_readiness_score is not None:
            exec_scores.append(setup_readiness_score)

    insider_score = 50.0
    if insider and insider.data_status == "available":
        insider_map = {
            "bullish_cluster": 90.0,
            "moderate_buying": 75.0,
            "neutral_no_signal": 50.0,
            "selling_activity": 50.0,
            "not_applicable": 50.0,
            "unavailable": 50.0,
        }
        insider_score = insider_map.get(insider.status, 50.0)
        exec_scores.append(insider_score)

    execution_readiness = round(
        max(1.0, min(10.0, (sum(exec_scores) / len(exec_scores)) / 10.0)) if exec_scores else 5.0,
        1,
    )

    execution_score_breakdown = {
        "stage": round(stage_score, 1),
        "setup": round(setup_readiness_score, 1) if setup_readiness_score is not None else None,
        "insider": round(insider_score, 1),
        "setup_reason": setup_reason,
    }

    # -------------------------------------------------------------
    # Coverage Calculation (Excludes not_applicable pillars)
    # -------------------------------------------------------------
    total_applicable = 0
    usable_count = 0
    verified_count = 0

    # Pillar 1: Fundamental
    if piotroski is None or piotroski.status != "not_applicable":
        total_applicable += 1
        if (piotroski and piotroski.status in ("available", "partial")) or quant_signals.quality_score is not None:
            usable_count += 1
        if (piotroski and piotroski.status == "available") or quant_signals.quality_score is not None:
            verified_count += 1

    # Pillar 2: Guidance
    total_applicable += 1
    if (guidance and guidance.status in ("available", "partial")) or quant_signals.eps_revision_net_30d is not None:
        usable_count += 1
    if guidance and guidance.status == "available":
        verified_count += 1

    # Pillar 3: Valuation
    if dcf is None or dcf.status != "not_applicable":
        total_applicable += 1
        if (dcf and dcf.status in ("available", "partial")) or quant_signals.value_score is not None:
            usable_count += 1
        if (dcf and dcf.status == "available" and dcf_is_actionable) or (quant_signals.value_score is not None and dcf_is_actionable):
            verified_count += 1

    # Pillar 4: Execution Timing
    total_applicable += 1
    if (tactical and tactical.status in ("available", "partial")) or (quant_signals.beta is not None or quant_signals.adtv_local_currency is not None):
        usable_count += 1
    if tactical and tactical.status == "available":
        verified_count += 1

    usable_coverage_pct = round((usable_count / total_applicable) * 100.0, 1) if total_applicable > 0 else 100.0
    verified_coverage_pct = round((verified_count / total_applicable) * 100.0, 1) if total_applicable > 0 else 100.0
    coverage_pct = usable_coverage_pct

    # -------------------------------------------------------------
    # Action Stance Determination
    # -------------------------------------------------------------
    MIN_ACTIONABLE_RR = 1.50
    freshness_status = "fresh"
    if quant_signals.atomic_market_snapshot:
        freshness_status = getattr(quant_signals.atomic_market_snapshot, "data_freshness_status", "fresh")

    stance_mode: Literal["active", "conditional", "wait", "reduce"] = "wait"

    # -------------------------------------------------------------
    # 5. Deterministic Action Stance Truth Table Gating (Phase 5)
    # -------------------------------------------------------------
    MIN_ACTIONABLE_RR = 1.50
    if coverage_pct < 60.0 or not tactical or tactical.status != "available":
        action_stance = "INSUFFICIENT_DATA"
        stance_mode = "wait"
        stance_reason = "Coverage ต่ำกว่าเกณฑ์ 60% หรือข้อมูล Tactical ไม่พร้อมใช้งาน"
    elif freshness_status == "stale_multiple_sessions":
        action_stance = "HOLD_WAIT"
        stance_mode = "wait"
        stance_reason = f"ข้อมูลตลาดล้าสมัยเกิน 1 trading session ({freshness_status}) — ระงับคำสั่งซื้อทุกประเภท"
    elif business_conviction < 5.0:
        action_stance = "REDUCE"
        stance_mode = "reduce"
        stance_reason = f"Business Conviction ({business_conviction:.1f}/10) ต่ำกว่าเกณฑ์ 5.0 — ปรับลดน้ำหนักการลงทุน"
    elif not has_actionable_valuation:
        # Valuation is non-actionable -> forbid active buy stances (ACCUMULATE_NOW / BREAKOUT_BUY)
        if business_conviction >= 6.5 and tactical.price_stage == "STAGE_2_MARKUP":
            action_stance = "ACCUMULATE_ON_DIP"
            stance_mode = "conditional"
            stance_reason = "Valuation ไม่เป็น Actionable (Macro Anomaly) แต่ Business Conviction แข็งแกร่งและอยู่ใน Stage 2 — กำหนดเป็นแผนตั้งรับเมื่อย่อตัว (Conditional Plan) ไม่ใช่คำสั่งซื้อทันที"
        elif business_conviction >= 5.0:
            action_stance = "HOLD_WAIT"
            stance_mode = "wait"
            stance_reason = "Valuation ไม่เป็น Actionable และ Business Conviction ปานกลาง — ถือหรือรอความชัดเจน"
        else:
            action_stance = "REDUCE"
            stance_mode = "reduce"
            stance_reason = "Business Conviction ต่ำและ Valuation ไม่พร้อมใช้งาน"
    elif (
        investment_conviction is not None
        and investment_conviction >= 7.5
        and execution_readiness >= 7.0
        and tactical.is_in_buy_zone is True
        and tactical.current_rr_ratio is not None
        and tactical.current_rr_ratio >= MIN_ACTIONABLE_RR
        and freshness_status == "fresh"
        and has_actionable_valuation
    ):
        action_stance = "ACCUMULATE_NOW"
        stance_mode = "active"
        stance_reason = f"คุณภาพพื้นฐานและจังหวะราคาพร้อมสมบูรณ์ใน Buy Zone (R:R {tactical.current_rr_ratio:.2f}:1)"
    elif (
        investment_conviction is not None
        and investment_conviction >= 6.5
        and tactical.price_stage == "STAGE_2_MARKUP"
        and bool(getattr(tactical, "breakout_entry_eligible", False))
        and tactical.breakout_current_rr is not None
        and tactical.breakout_current_rr >= MIN_ACTIONABLE_RR
        and freshness_status == "fresh"
        and has_actionable_valuation
        and (getattr(tactical, "breakout_volume_ratio", None) is None or getattr(tactical, "breakout_volume_ratio", 0) >= 1.50)
    ):
        action_stance = "BREAKOUT_BUY"
        stance_mode = "active"
        stance_reason = f"Stage 2 Breakout เหนือแนวต้าน (${tactical.key_resistance_level}) พร้อม Breakout R:R {tactical.breakout_current_rr:.2f}:1"
    elif business_conviction >= 6.5 and tactical.price_stage == "STAGE_2_MARKUP":
        action_stance = "ACCUMULATE_ON_DIP"
        stance_mode = "conditional"
        current_p = tactical.current_price
        if current_p and tactical.tactical_target_price and current_p >= tactical.tactical_target_price:
            stance_reason = f"ราคา (${current_p}) ชนหรือเกินแนวต้าน (${tactical.tactical_target_price}); ควรรอย่อตัวลงมาใน Buy Zone (${tactical.buy_zone_min}-${tactical.buy_zone_max})"
        elif tactical.current_rr_ratio is None:
            stance_reason = f"อยู่ใน Stage 2 แต่ราคา (${current_p}) อยู่นอก Buy Zone — แผนตั้งรับเมื่อย่อตัวเข้ากรอบ Buy Zone"
        else:
            stance_reason = f"อยู่ใน Stage 2 แต่ราคา (${current_p}) ให้ R:R ({tactical.current_rr_ratio:.2f}:1) ต่ำกว่าเกณฑ์ {MIN_ACTIONABLE_RR:.2f}:1 — แผนตั้งรับเมื่อย่อตัว"
    elif business_conviction >= 5.0:
        action_stance = "HOLD_WAIT"
        stance_mode = "wait"
        stance_reason = "Business Conviction ปานกลาง — ถือหรือรอความชัดเจน"
    else:
        action_stance = "REDUCE"
        stance_mode = "reduce"
        stance_reason = "Business Conviction ต่ำกว่าเกณฑ์"

    # DCF Discrepancy & Actionability Check (Generic DCF vs Reverse DCF)
    warnings_list = []
    if dcf and dcf.is_actionable is False:
        warnings_list.append(
            f"Reverse DCF model flagged informational-only: {dcf.actionability_reason or 'Macro parameter anomaly'}"
        )
        if "non_actionable_dcf:valuation" not in data_quality_flags:
            data_quality_flags.append("non_actionable_dcf:valuation")

    if quant_signals.dcf_result and dcf:
        dcf_res = quant_signals.dcf_result
        base_scenario = dcf_res.scenarios.get("base")
        base_upside = base_scenario.upside_pct if base_scenario else 0.0
        rdcf_upside = dcf.upside_12m_pct or 0.0
        if dcf_res.valuation_verdict != dcf.valuation_verdict and (dcf_res.is_actionable is False or abs(base_upside - rdcf_upside) >= 30.0):
            warnings_list.append(
                f"Generic DCF ({dcf_res.valuation_verdict.replace('_', ' ')}) และ Reverse DCF ({dcf.valuation_verdict.replace('_', ' ')}) "
                f"ให้ข้อสรุปต่างกันอย่างมีนัยสำคัญ (Generic Upside {base_upside:+.1f}% vs Reverse DCF Upside {rdcf_upside:+.1f}%)"
            )

    if warnings_list:
        quant_signals.dcf_discrepancy_warning = " | ".join(warnings_list)

    scorecard = DeterministicScorecard(
        core_conviction_score=core_conviction,
        business_conviction_score=business_conviction,
        investment_conviction_score=investment_conviction,
        execution_readiness_score=execution_readiness,
        action_stance=action_stance,
        action_stance_reason=stance_reason,
        stance_mode=stance_mode,
        setup_readiness_score=round(setup_readiness_score, 1) if setup_readiness_score is not None else None,
        execution_score_breakdown=execution_score_breakdown,
        fundamental_quality_score=round(fund_quality_score, 1),
        guidance_expectation_score=round(guidance_expectation_score, 1),
        analyst_expectations_score=round(analyst_expectations_score, 1) if analyst_expectations_score is not None else None,
        management_guidance_score=round(management_guidance_score, 1) if management_guidance_score is not None else None,
        valuation_margin_score=round(valuation_margin_score, 1) if valuation_margin_score is not None else None,
        coverage_pct=coverage_pct,
        usable_coverage_pct=usable_coverage_pct,
        verified_coverage_pct=verified_coverage_pct,
        applicable_pillars_count=total_applicable,
        methodology_version="2.1.0",
        reweighting_metadata=reweighting_metadata,
        data_quality_flags=data_quality_flags,
    )

    # -------------------------------------------------------------
    # Deterministic Thesis Falsifiers Generation with Mandatory Source Ref
    # -------------------------------------------------------------
    falsifiers: List[ThesisFalsifier] = []

    # 1. Operating Margin Floor Falsifier
    if dcf and dcf.fixed_parameters.get("base_ebit_margin_pct"):
        base_m = float(dcf.fixed_parameters["base_ebit_margin_pct"])
        floor_m = round(base_m - 2.5, 1)  # 250 bps margin compression threshold
        falsifiers.append(
            ThesisFalsifier(
                falsifier_id=f"{ticker}_MARGIN_FLOOR",
                metric_name="Operating Margin (EBIT %)",
                condition=f"EBIT margin falls below {floor_m}%",
                threshold_value=floor_m,
                source_basis="forecast_driver_delta",
                source_ref="json-pointer:///fixed_parameters/base_ebit_margin_pct",
                narrative_explanation=f"หากอัตรากำไรจากการดำเนินงานลดลงต่ำกว่า {floor_m}% (ลดลงเกิน 250 bps จากฐานเดิม) แสดงว่าสูญเสีย Pricing Power หรือต้นทุนโครงสร้างเร่งตัวเกินคาด ซึ่งจะลบล้างสมมติฐาน DCF ทันที",
            )
        )

    # 2. Earnings Call Guidance Delivery Falsifier
    if guidance and guidance.status in ("available", "partial") and guidance.source_note_path:
        quotes = getattr(guidance, "key_executive_quotes", None) or getattr(guidance, "guidance_quotes", None) or []
        first_quote = quotes[0] if quotes else None
        falsifiers.append(
            ThesisFalsifier(
                falsifier_id=f"{ticker}_GUIDANCE_MISS",
                metric_name="Management Guidance Delivery",
                condition="Quarterly revenue or billings miss guidance baseline",
                threshold_value=None,
                source_basis="guidance_quote",
                source_ref=guidance.source_note_path,
                source_quote=first_quote,
                narrative_explanation="หากผลประกอบการไตรมาสถัดไปพลาดเป้า Guidance ที่ผู้บริหารให้ไว้ จะส่งผลให้นักวิเคราะห์ปรับลดประมาณการ (Negative Estimate Revisions)",
            )
        )

    # 3. Technical Structure Invalidation Falsifier
    if tactical and tactical.invalidation_stop_loss is not None:
        stop_p = tactical.invalidation_stop_loss
        falsifiers.append(
            ThesisFalsifier(
                falsifier_id=f"{ticker}_STOP_LOSS_BREAK",
                metric_name="Structural Support Breakdown",
                condition=f"Weekly close below ${stop_p:.2f}",
                threshold_value=stop_p,
                source_basis="statement_baseline",
                source_ref="json-pointer:///tactical_setup/invalidation_stop_loss",
                narrative_explanation=f"หากราคาหลุดแนวรับโครงสร้างสำคัญที่ ${stop_p:.2f} จะถือว่ารูปแบบการสร้างฐานล้มเหลวและเข้าสู่แนวโน้มขาลง",
            )
        )

    return scorecard, falsifiers, flags
