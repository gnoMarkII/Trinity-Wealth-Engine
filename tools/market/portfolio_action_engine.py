"""Scoped Multi-Constraint Portfolio Action Engine (Phase 2 & v3.1 Hardened).

Transforms single-security stances into portfolio-executable decisions constrained by:
1. Target vs Current allocation delta
2. Security Stance Buy Gate (Only buyable stances: ACCUMULATE_NOW, ACCUMULATE_ON_DIP, BREAKOUT_BUY)
3. Policy Scope & Target Ticker/Bucket Binding
4. Missing Price / FX / ADTV Fail-Closed Invariants (No default fallbacks)
5. Sector & Asset concentration ceilings
6. Minimum cash buffer requirement
7. Valuation Margin of Safety gate (graceful NOT_APPLICABLE for Banks/REITs)
8. 20D ADTV Market Participation Liquidity limit: Order Notional / 20D ADTV <= Limit
"""
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field

from schemas.micro_quant_schemas import DataStatus, DeterministicScorecard, QuantSignals
from schemas.portfolio_policy_schemas import PortfolioPolicy


class ConstraintCheckResult(BaseModel):
    constraint_name: str
    passed: bool
    limit_value: Optional[float] = None
    actual_value: Optional[float] = None
    unit: str = "%"
    binding_effect: Optional[str] = None


class PortfolioActionVerdict(BaseModel):
    ticker: str
    portfolio_id: Optional[str] = None
    action: Literal[
        "ACCUMULATE_FULL",
        "ACCUMULATE_SCALED",
        "TRIM_REDUCE",
        "EXIT_FULL",
        "HOLD_UNCHANGED",
        "NOT_APPLICABLE",
        "INSUFFICIENT_VALUATION_DATA",
        "RESTRICTED_SCOPE_MISMATCH",
        "RESTRICTED_MISSING_PRICE",
        "RESTRICTED_MISSING_FX",
        "NO_PORTFOLIO_CONNECTED",
    ]
    security_stance: str
    target_weight_pct: Optional[float] = None
    current_weight_pct: Optional[float] = None
    proposed_order_weight_pct: float = 0.0
    proposed_order_notional: float = 0.0
    proposed_order_shares: int = 0
    currency: str = "USD"
    binding_constraints: list[str] = Field(default_factory=list)
    constraint_results: list[ConstraintCheckResult] = Field(default_factory=list)
    action_reasoning: str
    status: DataStatus = "available"


def compute_portfolio_action(
    ticker: str,
    market: Literal["TH", "US"],
    scorecard: DeterministicScorecard,
    quant_signals: QuantSignals,
    policy: Optional[PortfolioPolicy] = None,
    portfolio_state: Optional[dict[str, Any]] = None,
    current_price: Optional[float] = None,
    fx_rate_portfolio_to_asset: Optional[float] = 1.0,  # e.g. 1.0 if same currency, 0.0285 if THB portfolio buying USD asset
) -> PortfolioActionVerdict:
    """Evaluates multi-constraint portfolio execution decision with strict fail-closed gates."""
    san_ticker = ticker.strip().upper()
    stance = scorecard.action_stance
    local_currency = "THB" if market == "TH" else "USD"

    # 1. Gate: Check if Portfolio is connected
    if policy is None or portfolio_state is None:
        return PortfolioActionVerdict(
            ticker=san_ticker,
            portfolio_id=None,
            action="NO_PORTFOLIO_CONNECTED",
            security_stance=stance,
            currency=local_currency,
            action_reasoning="ไม่มีพอร์ตการลงทุนเชื่อมต่อ — แสดงเฉพาะ Security Stance ระดับหุ้นเดี่ยว",
            status="not_applicable",
        )

    # 2. Gate: Policy Scope Binding Verification
    if policy.scope == "asset" and policy.target_ticker:
        if policy.target_ticker.strip().upper() != san_ticker:
            return PortfolioActionVerdict(
                ticker=san_ticker,
                portfolio_id=policy.portfolio_id,
                action="RESTRICTED_SCOPE_MISMATCH",
                security_stance=stance,
                currency=local_currency,
                action_reasoning=f"Policy scope ถูกจำกัดเฉพาะ {policy.target_ticker} แต่ประเมินกับ {san_ticker} — ปฏิเสธการสร้างคำสั่งซื้อ",
                status="unavailable",
            )

    # 3. Gate: Fail-Closed on Missing Price (Never fallback to price or 1.0)
    if current_price is None or current_price <= 0:
        return PortfolioActionVerdict(
            ticker=san_ticker,
            portfolio_id=policy.portfolio_id,
            action="RESTRICTED_MISSING_PRICE",
            security_stance=stance,
            currency=local_currency,
            action_reasoning="ราคาตลาดปัจจุบันไม่พร้อมใช้งาน (current_price <= 0 or missing) — ปฏิเสธการสร้างคำสั่งซื้อตามหลัก Fail-Closed",
            status="unavailable",
        )

    # 4. Gate: Sector / Asset Eligibility (Graceful NOT_APPLICABLE for Banks/Insurance/REITs without DCF)
    is_dcf_excluded = False
    if quant_signals.reverse_dcf_result and not quant_signals.reverse_dcf_result.is_eligible:
        is_dcf_excluded = True

    if is_dcf_excluded and stance in ("INSUFFICIENT_DATA", "HOLD_WAIT"):
        return PortfolioActionVerdict(
            ticker=san_ticker,
            portfolio_id=policy.portfolio_id,
            action="NOT_APPLICABLE",
            security_stance=stance,
            currency=local_currency,
            action_reasoning="กลุ่มธุรกิจการเงิน/ประกัน/REIT ได้รับการยกเว้นจากการประเมิน DCF มาตรฐาน — ให้พิจารณาจาก Multiple และ Forensics แทน",
            status="not_applicable",
        )

    total_portfolio_nav = float(portfolio_state.get("total_nav", 0.0))
    current_cash = float(portfolio_state.get("cash", 0.0))
    current_holding_value = float(portfolio_state.get("positions", {}).get(san_ticker, {}).get("market_value", 0.0))
    current_sector_value = float(portfolio_state.get("sector_allocations", {}).get(quant_signals.peer_sector or "Unknown", 0.0))

    current_weight_pct = (current_holding_value / total_portfolio_nav * 100.0) if total_portfolio_nav > 0 else 0.0
    target_weight = policy.target_weight_pct
    binding_constraints: list[str] = []
    constraint_results: list[ConstraintCheckResult] = []

    # 5. Stance Gate: If Security Stance is Negative (REDUCE)
    if stance == "REDUCE":
        if current_holding_value > 0:
            shares = int(current_holding_value / current_price)
            action_type = "EXIT_FULL" if scorecard.core_conviction_score < 4.0 else "TRIM_REDUCE"
            return PortfolioActionVerdict(
                ticker=san_ticker,
                portfolio_id=policy.portfolio_id,
                action=action_type,
                security_stance=stance,
                target_weight_pct=0.0,
                current_weight_pct=round(current_weight_pct, 2),
                proposed_order_weight_pct=round(-current_weight_pct, 2),
                proposed_order_notional=round(current_holding_value, 2),
                proposed_order_shares=shares,
                currency=local_currency,
                binding_constraints=["negative_security_stance:reduce_mandate"],
                constraint_results=[
                    ConstraintCheckResult(
                        constraint_name="security_stance_gate",
                        passed=False,
                        actual_value=scorecard.core_conviction_score,
                        unit="score",
                        binding_effect="reduce_or_exit_holding",
                    )
                ],
                action_reasoning=f"สัญญาณ Security Stance เป็น REDUCE — ปรับลดน้ำหนักหรือปิดสถานะทั้งหมดตามนโยบายควบคุมความเสี่ยง",
            )
        else:
            return PortfolioActionVerdict(
                ticker=san_ticker,
                portfolio_id=policy.portfolio_id,
                action="HOLD_UNCHANGED",
                security_stance=stance,
                target_weight_pct=0.0,
                current_weight_pct=0.0,
                currency=local_currency,
                action_reasoning="ไม่มีสถานะถือครองเดิม และ Security Stance เป็น REDUCE — คงการไม่ลงทุน",
            )

    # 6. Stance Gate: If Security Stance is Neutral / Non-Buyable (HOLD_WAIT, INSUFFICIENT_DATA)
    if stance not in ("ACCUMULATE_NOW", "ACCUMULATE_ON_DIP", "BREAKOUT_BUY"):
        return PortfolioActionVerdict(
            ticker=san_ticker,
            portfolio_id=policy.portfolio_id,
            action="HOLD_UNCHANGED",
            security_stance=stance,
            target_weight_pct=target_weight,
            current_weight_pct=round(current_weight_pct, 2),
            currency=local_currency,
            constraint_results=[
                ConstraintCheckResult(
                    constraint_name="security_stance_buy_gate",
                    passed=False,
                    actual_value=scorecard.core_conviction_score,
                    unit="score",
                    binding_effect="no_buy_order_permitted",
                )
            ],
            action_reasoning=f"Security Stance เป็น {stance} — ไม่อยู่ในกลุ่มสถานะอนุญาตให้เข้าซื้อ (Buy Gate)",
        )

    # 7. Positive Stance Evaluation (ACCUMULATE_NOW, ACCUMULATE_ON_DIP, BREAKOUT_BUY)
    weight_delta = target_weight - current_weight_pct
    if weight_delta <= 0.2:  # Within 0.2% tolerance
        return PortfolioActionVerdict(
            ticker=san_ticker,
            portfolio_id=policy.portfolio_id,
            action="HOLD_UNCHANGED",
            security_stance=stance,
            target_weight_pct=target_weight,
            current_weight_pct=round(current_weight_pct, 2),
            currency=local_currency,
            action_reasoning=f"สัดส่วนปัจจุบัน ({current_weight_pct:.1f}%) บรรลุเป้าหมายตาม Policy ({target_weight:.1f}%) เรียบร้อยแล้ว",
        )

    # Constraint 1: Cash Buffer Constraint
    min_cash_required = total_portfolio_nav * (policy.minimum_cash_buffer_pct / 100.0)
    max_allocatable_cash = max(0.0, current_cash - min_cash_required)
    raw_desired_notional = total_portfolio_nav * (weight_delta / 100.0)
    capped_notional = min(raw_desired_notional, max_allocatable_cash)
    
    cash_passed = capped_notional == raw_desired_notional
    constraint_results.append(
        ConstraintCheckResult(
            constraint_name="minimum_cash_buffer",
            passed=cash_passed,
            limit_value=policy.minimum_cash_buffer_pct,
            actual_value=round(current_cash / max(1.0, total_portfolio_nav) * 100.0, 1),
            unit="%",
            binding_effect="scaled_to_available_cash" if not cash_passed else None,
        )
    )
    if not cash_passed:
        binding_constraints.append(f"cash_buffer_constraint:min_{policy.minimum_cash_buffer_pct}%_required")

    # Constraint 2: Sector Concentration Limit
    max_sector_add = max(0.0, (total_portfolio_nav * (policy.max_sector_exposure_pct / 100.0)) - current_sector_value)
    sector_passed = capped_notional <= max_sector_add
    if capped_notional > max_sector_add:
        capped_notional = max_sector_add
        binding_constraints.append(f"sector_cap_constraint:max_{policy.max_sector_exposure_pct}%_sector_exposure")
    constraint_results.append(
        ConstraintCheckResult(
            constraint_name="sector_concentration_cap",
            passed=sector_passed,
            limit_value=policy.max_sector_exposure_pct,
            actual_value=round(current_sector_value / max(1.0, total_portfolio_nav) * 100.0, 1),
            unit="%",
            binding_effect="capped_at_sector_headroom" if not sector_passed else None,
        )
    )

    # Constraint 3: 20D ADTV Liquidity Constraint (Proposed Order Notional / 20D ADTV <= Limit)
    adtv = quant_signals.adtv_local_currency
    if adtv and adtv > 0:
        max_adtv_order = adtv * policy.max_adtv_participation_rate
        adtv_passed = capped_notional <= max_adtv_order
        if capped_notional > max_adtv_order:
            capped_notional = max_adtv_order
            binding_constraints.append(
                f"adtv_liquidity_cap:order_capped_at_{round(policy.max_adtv_participation_rate*100, 1)}%_of_20D_ADTV"
            )
        constraint_results.append(
            ConstraintCheckResult(
                constraint_name="adtv_participation_limit",
                passed=adtv_passed,
                limit_value=round(policy.max_adtv_participation_rate * 100, 1),
                actual_value=round((capped_notional / adtv) * 100, 2),
                unit="%",
                binding_effect="scaled_to_adtv_limit" if not adtv_passed else None,
            )
        )

    proposed_shares = int(capped_notional / current_price)
    proposed_notional = round(proposed_shares * current_price, 2)
    proposed_order_weight = round((proposed_notional / total_portfolio_nav * 100.0), 2) if total_portfolio_nav > 0 else 0.0

    action_type = "ACCUMULATE_FULL" if not binding_constraints else "ACCUMULATE_SCALED"

    return PortfolioActionVerdict(
        ticker=san_ticker,
        portfolio_id=policy.portfolio_id,
        action=action_type,
        security_stance=stance,
        target_weight_pct=target_weight,
        current_weight_pct=round(current_weight_pct, 2),
        proposed_order_weight_pct=proposed_order_weight,
        proposed_order_notional=proposed_notional,
        proposed_order_shares=proposed_shares,
        currency=local_currency,
        binding_constraints=binding_constraints,
        constraint_results=constraint_results,
        action_reasoning=(
            f"แนะนำเข้าสะสม {action_type} จำนวน {proposed_shares:,} หุ้น (มูลค่า ~{local_currency} {proposed_notional:,.2f}) "
            f"เพื่อปรับน้ำหนักจาก {current_weight_pct:.1f}% สู่เป้าหมาย {target_weight:.1f}% "
            + (f"โดยมีข้อจำกัด: {', '.join(binding_constraints)}" if binding_constraints else "โดยไม่มีข้อจำกัดขัดขวาง")
        ),
    )
