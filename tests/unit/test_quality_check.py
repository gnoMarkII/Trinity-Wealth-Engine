"""Unit tests for QualityCheckResult in validators/quality_check.py."""
from dataclasses import dataclass
from schemas.warning_registry import (
    COVERAGE_WARNING_INCOMPLETE,
    DEFENSIVE_LOW_SUPPORTING_DATA,
    SINGLE_SOURCE_PENALTY,
    WarningMessage,
)
from validators.quality_check import QualityCheckResult


@dataclass
class MockDirection:
    validation_warnings: list[str]
    asset_allocation: list = None
    pair_trades: list = None
    risk_scenarios: list = None


@dataclass
class MockAsset:
    validation_warnings: list[str]


def test_retryable_critical_id_triggers_retry():
    w = str(WarningMessage(COVERAGE_WARNING_INCOMPLETE, {"count": "3"}))
    direction = MockDirection(validation_warnings=[w])
    quality = QualityCheckResult.from_direction(direction)

    assert COVERAGE_WARNING_INCOMPLETE in quality.critical_ids
    assert COVERAGE_WARNING_INCOMPLETE in quality.retryable_ids
    assert COVERAGE_WARNING_INCOMPLETE not in quality.soft_ids
    assert quality.should_retry is True
    assert "ระบบตรวจพบข้อผิดพลาดเชิงโครงสร้าง" in quality.retry_feedback


def test_non_retryable_critical_id_is_critical_but_does_not_trigger_retry():
    w = str(WarningMessage(DEFENSIVE_LOW_SUPPORTING_DATA))
    direction = MockDirection(validation_warnings=[w])
    quality = QualityCheckResult.from_direction(direction)

    assert DEFENSIVE_LOW_SUPPORTING_DATA in quality.critical_ids
    assert DEFENSIVE_LOW_SUPPORTING_DATA not in quality.retryable_ids
    assert DEFENSIVE_LOW_SUPPORTING_DATA not in quality.soft_ids
    assert quality.should_retry is False
    assert quality.retry_feedback == ""


def test_soft_warning_goes_to_soft_ids():
    w = str(WarningMessage(SINGLE_SOURCE_PENALTY))
    direction = MockDirection(validation_warnings=[w])
    quality = QualityCheckResult.from_direction(direction)

    assert SINGLE_SOURCE_PENALTY in quality.soft_ids
    assert SINGLE_SOURCE_PENALTY not in quality.critical_ids
    assert SINGLE_SOURCE_PENALTY not in quality.retryable_ids
    assert quality.should_retry is False


def test_warnings_collected_from_children_and_deduplicated():
    w1 = str(WarningMessage(COVERAGE_WARNING_INCOMPLETE, {"count": "3"}))
    w2 = str(WarningMessage(DEFENSIVE_LOW_SUPPORTING_DATA))
    w3 = str(WarningMessage(SINGLE_SOURCE_PENALTY))
    
    asset1 = MockAsset(validation_warnings=[w1, w2])
    asset2 = MockAsset(validation_warnings=[w2, w3])  # w2 duplicate
    direction = MockDirection(validation_warnings=[w1], asset_allocation=[asset1, asset2])

    quality = QualityCheckResult.from_direction(direction)

    assert quality.critical_ids.count(COVERAGE_WARNING_INCOMPLETE) == 1
    assert quality.critical_ids.count(DEFENSIVE_LOW_SUPPORTING_DATA) == 1
    assert quality.soft_ids == [SINGLE_SOURCE_PENALTY]
    assert len(quality.retryable_ids) == 1
    assert quality.should_retry is True
