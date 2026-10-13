"""Tests for AI Bucket Auto-Assignment and Batch Reassignment Endpoints."""
from unittest.mock import patch, MagicMock
import pytest

from tools.portfolio.services.bucket_ai_service import (
    HoldingBucketRecommendationLLMResult,
    HoldingRecommendationLLMItem,
)


@pytest.fixture
def isolated_ai_portfolio(tmp_path, monkeypatch):
    """Set up isolated vault path and reattach modules for portfolio AI tests."""
    from tests.conftest import _reset_portfolio_modules
    mods = _reset_portfolio_modules(tmp_path, monkeypatch)
    core = mods.core

    post, state = core._load_or_init()
    h_cash = core.Holding(
        symbol="CASH_THB",
        asset_type="Cash",
        units=100_000.0,
        current_price_thb=1.0,
        avg_cost_thb=1.0,
        bucket_id="growth",  # Intentionally mismatched: Cash inside growth
    )
    h_stock = core.Holding(
        symbol="NVDA",
        asset_type="Stock",
        units=10.0,
        current_price_thb=4000.0,
        avg_cost_thb=3500.0,
        bucket_id=None,  # Unassigned
    )
    state.holdings = [h_cash, h_stock]
    state.allocation_targets = [
        core.AllocationTarget(bucket_id="growth", name="Core Growth", target_percent=70.0, color="#3B82F6"),
        core.AllocationTarget(bucket_id="reserve", name="Emergency Reserve", target_percent=30.0, color="#10B981"),
    ]
    core._save(post, state)

    return tmp_path


def test_batch_reassign_holding_buckets_success(authed_client, isolated_ai_portfolio):
    payload = {
        "assignments": [
            {"symbol": "NVDA", "bucket_id": "growth"},
            {"symbol": "CASH_THB", "bucket_id": "reserve"},
        ]
    }
    r = authed_client.put("/api/portfolio/actual/holdings/batch-bucket-map", json=payload)
    assert r.status_code == 200
    data = r.json()

    nvda = next((h for h in data["holdings"] if h["symbol"] == "NVDA"), None)
    cash = next((h for h in data["holdings"] if h["symbol"] == "CASH_THB"), None)

    assert nvda is not None
    assert nvda["bucket_id"] == "growth"
    assert cash is not None
    assert cash["bucket_id"] == "reserve"


def test_batch_reassign_holding_buckets_invalid_bucket(authed_client, isolated_ai_portfolio):
    payload = {
        "assignments": [
            {"symbol": "NVDA", "bucket_id": "non_existent_bucket"},
        ]
    }
    r = authed_client.put("/api/portfolio/actual/holdings/batch-bucket-map", json=payload)
    assert r.status_code == 400
    assert "ไม่พบ bucket_id" in r.json()["detail"]


def test_recommend_holding_buckets_success(authed_client, isolated_ai_portfolio):
    mock_llm_result = HoldingBucketRecommendationLLMResult(
        recommendations=[
            HoldingRecommendationLLMItem(
                symbol="NVDA",
                recommended_bucket_id="growth",
                reason="หุ้นเทคโนโลยีเติบโตสูง เหมาะกับกลุ่ม Core Growth",
            ),
            HoldingRecommendationLLMItem(
                symbol="CASH_THB",
                recommended_bucket_id="reserve",
                reason="เงินสดสภาพคล่องสูง ควรอยู่ในกลุ่มเงินสำรองฉุกเฉิน",
            ),
        ],
        summary="วิเคราะห์สินทรัพย์ 2 รายการเรียบร้อย",
    )

    with patch("core.llm_factory.invoke_structured_llm", return_value=mock_llm_result):
        r = authed_client.post("/api/portfolio/actual/holdings/ai-recommend-buckets?portfolio_id=default")
        assert r.status_code == 200
        data = r.json()

        assert len(data["recommendations"]) == 2
        assert data["total_unassigned"] == 1  # NVDA was None
        assert data["total_mismatched"] == 1  # CASH_THB moved from growth to reserve

        nvda_rec = next(rec for rec in data["recommendations"] if rec["symbol"] == "NVDA")
        assert nvda_rec["recommended_bucket_id"] == "growth"
        assert nvda_rec["is_changed"] is True
        assert "Core Growth" in nvda_rec["reason"]

        cash_rec = next(rec for rec in data["recommendations"] if rec["symbol"] == "CASH_THB")
        assert cash_rec["recommended_bucket_id"] == "reserve"
        assert cash_rec["is_changed"] is True


def test_recommend_holding_buckets_llm_failure(authed_client, isolated_ai_portfolio):
    with patch(
        "core.llm_factory.invoke_structured_llm",
        side_effect=Exception("API Rate limit exceeded"),
    ):
        r = authed_client.post("/api/portfolio/actual/holdings/ai-recommend-buckets?portfolio_id=default")
        assert r.status_code == 503
        assert "ระบบ AI แนะนำกลุ่มกลยุทธ์ไม่พร้อมใช้งาน" in r.json()["detail"]
