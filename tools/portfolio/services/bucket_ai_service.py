"""AI-Powered Bucket Auto-Assignment Service.

Analyzes portfolio holdings against strategy bucket definitions and recommends
appropriate bucket placement for unassigned and misallocated assets.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from core.logger import get_logger
import core.llm_factory as llm_factory
from tools.portfolio.domain.models import PortfolioState
from api.schemas.portfolio import (
    HoldingBucketRecommendationItemDTO,
    HoldingsBucketRecommendationResponseDTO,
)

logger = get_logger(__name__)


class HoldingRecommendationLLMItem(BaseModel):
    symbol: str
    recommended_bucket_id: str
    reason: str


class HoldingBucketRecommendationLLMResult(BaseModel):
    recommendations: List[HoldingRecommendationLLMItem] = Field(default_factory=list)
    summary: str = ""


def recommend_holding_buckets(state: PortfolioState) -> HoldingsBucketRecommendationResponseDTO:
    """Analyze all holdings in the portfolio state and generate AI bucket recommendations."""
    if not state.holdings:
        return HoldingsBucketRecommendationResponseDTO(
            recommendations=[],
            summary="ไม่มีสินทรัพย์ในพอร์ตการลงทุน",
            total_unassigned=0,
            total_mismatched=0,
        )

    if not state.allocation_targets:
        return HoldingsBucketRecommendationResponseDTO(
            recommendations=[],
            summary="ยังไม่ได้กำหนดกลุ่ม Strategy Buckets ในพอร์ต",
            total_unassigned=0,
            total_mismatched=0,
        )

    valid_bucket_ids = {t.bucket_id for t in state.allocation_targets}
    default_bucket_id = state.allocation_targets[0].bucket_id

    policy_buckets: List[Dict[str, Any]] = (
        state.allocation_policy.get("purpose_buckets", [])
        if state.allocation_policy and isinstance(state.allocation_policy, dict)
        else []
    )

    buckets_info = []
    for t in state.allocation_targets:
        pb = next((p for p in policy_buckets if isinstance(p, dict) and p.get("bucket_id") == t.bucket_id), None)
        buckets_info.append({
            "bucket_id": t.bucket_id,
            "name": t.name,
            "target_percent": t.target_percent,
            "role": pb.get("role") if pb else "Strategic Role",
            "eligible_assets": pb.get("eligible_assets") if pb else [],
            "linked_goal": pb.get("linked_goal_name") if pb else None,
        })

    holdings_info = []
    for h in state.holdings:
        holdings_info.append({
            "symbol": h.symbol,
            "company_name": h.company_name or h.symbol,
            "asset_type": h.asset_type,
            "current_bucket_id": h.bucket_id,
            "market_value_thb": h.market_value_thb,
            "pe_ratio": h.pe_ratio,
            "dividend_yield": h.dividend_yield,
            "market_cap_tier": h.market_cap_tier,
        })

    prompt_lines = [
        "คุณคือ Senior Portfolio Strategist และ Asset Allocation Expert ของ Trinity Wealth Engine",
        "หน้าที่ของคุณคือ วิเคราะห์สินทรัพย์ทั้งหมดในพอร์ตการลงทุน และจับคู่เข้ากับ Strategy Buckets ที่เหมาะสมที่สุด",
        "",
        "ข้อกำหนดในการวิเคราะห์:",
        "1. สำหรับสินทรัพย์ที่ยังไม่มีกลุ่ม (current_bucket_id เป็น null หรือ unassigned): ให้เลือก Strategy Bucket ที่เหมาะสมที่สุดตาม asset_type, nature ของสินทรัพย์ และเป้าหมายของกลุ่ม",
        "2. สำหรับสินทรัพย์ที่มีกลุ่มเดิมอยู่แล้ว: ให้ตรวจสอบว่าอยู่ในกลุ่มที่ถูกต้องหรือไม่ หากอยู่ผิดกลุ่ม (เช่น เงินสด/ตราสารหนี้ไปอยู่ในกลุ่ม High Growth หรือ หุ้นผันผวนสูงไปอยู่ในกลุ่มเงินสำรองฉุกเฉิน) ให้แนะนำย้ายไปยังกลุ่มที่ถูกต้อง",
        "3. ค่า recommended_bucket_id จะต้องเป็นหนึ่งใน bucket_id ที่ระบุใน Strategy Buckets ด้านล่างนี้เท่านั้น ห้ามสร้าง ID ขึ้นมาใหม่",
        "4. ให้เหตุผลสั้นกระชับเข้าใจง่ายเป็นภาษาไทยในฟิลด์ reason",
        "",
        f"Strategy Buckets ที่มีในระบบ:\n```json\n{json.dumps(buckets_info, ensure_ascii=False, indent=2)}\n```",
        "",
        f"รายการสินทรัพย์ที่ต้องวิเคราะห์ ({len(holdings_info)} รายการ):\n```json\n{json.dumps(holdings_info, ensure_ascii=False, indent=2)}\n```",
    ]

    try:
        llm_result: HoldingBucketRecommendationLLMResult = llm_factory.invoke_structured_llm(
            schema=HoldingBucketRecommendationLLMResult,
            model_env="BUCKET_AUTO_ASSIGN_MODEL",
            prompt_lines=prompt_lines,
            purpose="bucket_auto_assign",
            default_model="gemini-3.1-flash-lite-preview",
            provider="google",
        )
    except Exception as exc:
        logger.error("AI Bucket Auto-assign invocation failed: %s", exc)
        raise RuntimeError(f"ระบบ AI ไม่สามารถวิเคราะห์การจัดกลุ่มได้ในขณะนี้: {exc}") from exc

    rec_map: Dict[str, HoldingRecommendationLLMItem] = {
        item.symbol.strip().upper(): item for item in llm_result.recommendations
    }

    results: List[HoldingBucketRecommendationItemDTO] = []
    total_unassigned = 0
    total_mismatched = 0

    for h in state.holdings:
        sym_key = h.symbol.strip().upper()
        current_b = h.bucket_id
        is_unassigned = not current_b or current_b.lower() == "unassigned"

        if is_unassigned:
            total_unassigned += 1

        rec_item = rec_map.get(sym_key)
        if rec_item and rec_item.recommended_bucket_id in valid_bucket_ids:
            target_b = rec_item.recommended_bucket_id
            reason = rec_item.reason
        else:
            # Fallback: if existing bucket is valid, keep it; otherwise default bucket
            if current_b and current_b in valid_bucket_ids:
                target_b = current_b
                reason = "จัดสรรตามกลุ่มเดิมที่มีความเหมาะสมอยู่แล้ว"
            else:
                target_b = default_bucket_id
                reason = "จัดสรรเข้ากลุ่มหลักตามค่าเริ่มต้น"

        is_changed = target_b != current_b
        if is_changed and not is_unassigned:
            total_mismatched += 1

        results.append(
            HoldingBucketRecommendationItemDTO(
                symbol=h.symbol,
                asset_type=h.asset_type,
                current_bucket_id=current_b,
                recommended_bucket_id=target_b,
                is_changed=is_changed,
                reason=reason,
            )
        )

    return HoldingsBucketRecommendationResponseDTO(
        recommendations=results,
        summary=llm_result.summary or f"วิเคราะห์สินทรัพย์ครบ {len(results)} รายการ",
        total_unassigned=total_unassigned,
        total_mismatched=total_mismatched,
    )
