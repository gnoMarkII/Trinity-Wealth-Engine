"""Unit tests for Core Risk-Based Review Sampling Policy."""
from core.portfolio_advisory.review_sampling import (
    ReviewSampleItem,
    SampleType,
    select_review_sample,
)


def test_select_review_sample_normal():
    holdings = [
        {"symbol": "AAPL"},
        {"symbol": "MSFT"},
        {"symbol": "NVDA"},
        {"symbol": "AMZN"},
        {"symbol": "GOOGL"},
    ]
    thesis_exceptions = [
        {
            "ticker": "NVDA",
            "exception_type": "Falsifier Breach",
            "observed_value": "48.2%",
            "expected_value": "52.0%",
        }
    ]
    top_mover = {"ticker": "AAPL", "change_pct": 3.45}

    samples = select_review_sample(
        holdings=holdings,
        thesis_exceptions=thesis_exceptions,
        top_mover=top_mover,
        review_date="2026-11-05",
    )

    # Should have 1 breach, 1 mover, 1 rotation -> 3 items
    assert len(samples) >= 3
    types = [s.sample_type for s in samples]
    assert SampleType.BREACH in types
    assert SampleType.TOP_MOVER in types
    assert SampleType.ROTATION in types

    breach_item = next(s for s in samples if s.sample_type == SampleType.BREACH)
    assert breach_item.symbol == "NVDA"
    assert breach_item.urgency == "CRITICAL"
    assert "48.2%" in breach_item.target_value

    mover_item = next(s for s in samples if s.sample_type == SampleType.TOP_MOVER)
    assert mover_item.symbol == "AAPL"
    assert "+3.45%" in mover_item.target_value


def test_select_review_sample_capped_at_max():
    holdings = [{"symbol": f"SYM_{i}"} for i in range(20)]
    # 5 breaches
    thesis_exceptions = [
        {"ticker": f"SYM_{i}", "exception_type": "Breach", "observed_value": "10"}
        for i in range(5)
    ]
    top_mover = {"ticker": "SYM_10", "change_pct": -5.2}

    samples = select_review_sample(
        holdings=holdings,
        thesis_exceptions=thesis_exceptions,
        top_mover=top_mover,
        review_date="2026-11-05",
        max_sample_size=5,
    )

    # Bounded strictly to max 5
    assert len(samples) <= 5
    # Breaches capped at 3
    breaches = [s for s in samples if s.sample_type == SampleType.BREACH]
    assert len(breaches) == 3


def test_select_review_sample_zero_breaches_pads_to_min():
    holdings = [
        {"symbol": "AAPL"},
        {"symbol": "MSFT"},
        {"symbol": "GOOGL"},
        {"symbol": "AMZN"},
    ]
    top_mover = {"ticker": "AAPL", "change_pct": 1.2}

    samples = select_review_sample(
        holdings=holdings,
        thesis_exceptions=[],
        quarantined_holdings=[],
        pending_decisions=[],
        top_mover=top_mover,
        review_date="2026-11-05",
        min_sample_size=3,
    )

    assert len(samples) == 3
    symbols = [s.symbol for s in samples]
    assert "AAPL" in symbols
    # All symbols must be unique
    assert len(set(symbols)) == len(symbols)


def test_select_review_sample_deterministic_rotation_date():
    holdings = [{"symbol": f"STOCK_{i}"} for i in range(10)]

    samples_day1 = select_review_sample(
        holdings=holdings,
        review_date="2026-11-01",
    )
    samples_day5 = select_review_sample(
        holdings=holdings,
        review_date="2026-11-05",
    )

    rot_symbols_day1 = [s.symbol for s in samples_day1 if s.sample_type == SampleType.ROTATION]
    rot_symbols_day5 = [s.symbol for s in samples_day5 if s.sample_type == SampleType.ROTATION]

    assert rot_symbols_day1 != rot_symbols_day5


def test_select_review_sample_empty_portfolio():
    samples = select_review_sample(
        holdings=[],
        thesis_exceptions=[],
        quarantined_holdings=[],
        pending_decisions=[],
    )
    assert samples == []


def test_select_review_sample_deduplication():
    # If top mover is already quarantined, do not produce duplicate item
    holdings = [{"symbol": "NVDA"}, {"symbol": "AAPL"}]
    quarantined = [{"symbol": "NVDA", "reason": "Missing price feed"}]
    top_mover = {"ticker": "NVDA", "change_pct": -10.5}

    samples = select_review_sample(
        holdings=holdings,
        quarantined_holdings=quarantined,
        top_mover=top_mover,
        review_date="2026-11-05",
    )

    nvda_items = [s for s in samples if s.symbol == "NVDA"]
    assert len(nvda_items) == 1
    assert nvda_items[0].sample_type == SampleType.BREACH

