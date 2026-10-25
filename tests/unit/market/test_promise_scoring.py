"""Unit tests for Core Promise Check scoring, validation, and guidance bias engine."""
import pytest
from core.promise_check.scoring import (
    compute_promise_scores,
    validate_ledger,
    qkey,
    qlabel,
    weight_of,
    score_key,
    FRAMEWORK_VERSION,
)


def _make_8_calls():
    quarters = [
        ("FY2024Q1", "2023-05-24"),
        ("FY2024Q2", "2023-08-23"),
        ("FY2024Q3", "2023-11-21"),
        ("FY2024Q4", "2024-02-21"),
        ("FY2025Q1", "2024-05-22"),
        ("FY2025Q2", "2024-08-28"),
        ("FY2025Q3", "2024-11-20"),
        ("FY2025Q4", "2025-02-26"),
    ]
    return [
        {
            "quarter": q,
            "date": d,
            "status": "ok",
            "source": "stockanalysis",
            "words": 7500,
        }
        for q, d in quarters
    ]


def _make_sample_ledger():
    return {
        "framework_version": FRAMEWORK_VERSION,
        "ticker": "NVDA",
        "company_name": "NVIDIA Corporation",
        "calls": _make_8_calls(),
        "promises": [
            {
                "id": "P-NVDA-FY2024Q1-01",
                "made_in": "FY2024Q1",
                "speaker": "Colette Kress, CFO",
                "section": "prepared",
                "type": "A_numeric",
                "category": "revenue",
                "scope": "company",
                "quote": "We expect Q2 revenue to be $11.00 billion.",
                "metric": "Q2 FY24 Revenue",
                "target_kind": "point",
                "target_point": 11.0,
                "unit": "USD bn",
                "basis": "GAAP",
                "due": "FY2024Q2",
                "due_precision": "explicit",
                "status": "Delivered",
                "resolved_in": "FY2024Q2",
                "evidence_quote": "Revenue for the second quarter was $13.51 billion.",
                "actual_numeric": 13.51,
                "delta_pct": 0.228,
                "notes": "Delivered well ahead of guidance.",
            },
            {
                "id": "P-NVDA-FY2025Q3-01",
                "made_in": "FY2025Q3",
                "speaker": "Jensen Huang, CEO",
                "section": "prepared",
                "type": "B_milestone",
                "category": "product",
                "scope": "company",
                "quote": "Blackwell production shipments will begin in Q4 and ramp into next year.",
                "metric": "Blackwell Q4 Volume Shipment",
                "target_kind": "milestone",
                "due": "FY2025Q4",
                "due_precision": "explicit",
                "status": "Delivered",
                "resolved_in": "FY2025Q4",
                "evidence_quote": "Blackwell is in full production and shipping to cloud customers.",
                "slip_quarters": 0,
                "notes": "Shipped on schedule.",
            },
            {
                "id": "P-NVDA-FY2025Q4-01",
                "made_in": "FY2025Q4",
                "speaker": "Colette Kress, CFO",
                "section": "prepared",
                "type": "A_numeric",
                "category": "revenue",
                "scope": "company",
                "quote": "Revenue for the first quarter of fiscal 2026 is expected to be $37.5 billion.",
                "metric": "Q1 FY26 Revenue Outlook",
                "target_kind": "point",
                "target_point": 37.5,
                "unit": "USD bn",
                "basis": "GAAP",
                "due": "FY2026Q1",  # Future quarter beyond latest FY2025Q4
                "due_precision": "explicit",
                "status": "Pending",
                "notes": "Pending next earnings report.",
            },
        ],
    }


def test_qkey_and_qlabel():
    assert qkey("FY2025Q1") == 2025 * 4 + 0
    assert qkey("FY2025Q4") == 2025 * 4 + 3
    assert qkey("FY2026Q2") == 2026 * 4 + 1
    assert qkey("invalid") is None
    assert qkey("") is None

    assert qlabel("FY2025Q1") == "FY25 Q1"
    assert qlabel("FY2026Q4") == "FY26 Q4"
    assert qlabel("invalid") == "invalid"
    assert qlabel(None) == ""


def test_weight_of():
    numeric_comp_explicit = {
        "type": "A_numeric",
        "target_kind": "point",
        "scope": "company",
        "due_precision": "explicit",
    }
    numeric_seg_inferred = {
        "type": "A_numeric",
        "target_kind": "range",
        "scope": "segment",
        "due_precision": "inferred",
    }
    milestone_comp_explicit = {
        "type": "B_milestone",
        "target_kind": "milestone",
        "scope": "company",
        "due_precision": "explicit",
    }

    assert weight_of(numeric_comp_explicit) == 2.0
    assert weight_of(numeric_seg_inferred) == 1.5 * 0.75  # 1.125
    assert weight_of(milestone_comp_explicit) == 1.0


def test_score_key():
    assert score_key({"status": "Delivered"}) == "Delivered"
    assert score_key({"status": "Partial"}) == "Partial"
    assert score_key({"status": "Missed"}) == "Missed"
    assert score_key({"status": "Quietly dropped"}) == "Quietly dropped"
    assert score_key({"status": "Restated", "restated_direction": "lowered"}) == "Restated-lowered"
    assert score_key({"status": "Restated", "restated_direction": "raised"}) is None
    assert score_key({"status": "Pending"}) is None
    assert score_key({"status": "Unverifiable"}) is None


def test_validate_ledger_happy_path():
    ledger = _make_sample_ledger()
    errors, warnings = validate_ledger(ledger)
    assert errors == []
    assert warnings == []


def test_validate_ledger_quote_length_warning():
    ledger = _make_sample_ledger()
    # 45 words quote (> 40 max allowed) generates a warning
    long_quote = " ".join(["word"] * 45)
    ledger["promises"][0]["quote"] = long_quote
    errors, warnings = validate_ledger(ledger)
    assert errors == []
    assert any("quote is 45 words (>40)" in w for w in warnings)


def test_validate_ledger_resolved_requirements():
    ledger = _make_sample_ledger()
    # Delivered status must have evidence_quote
    ledger["promises"][0]["evidence_quote"] = ""
    errors, _ = validate_ledger(ledger)
    assert any("without evidence_quote" in e for e in errors)


def test_validate_ledger_quietly_dropped_rules():
    ledger = _make_sample_ledger()
    # Promise due in FY2025Q4 cannot be quietly dropped if latest call is only FY2025Q4 (grace period due+1 not reached)
    dropped_p = {
        "id": "P-NVDA-FY2024Q1-02",
        "made_in": "FY2024Q1",
        "speaker": "Colette Kress, CFO",
        "section": "prepared",
        "type": "B_milestone",
        "category": "product",
        "scope": "segment",
        "quote": "We plan new software portal launch by Q4.",
        "metric": "Software Portal Launch",
        "target_kind": "milestone",
        "due": "FY2025Q4",
        "due_precision": "explicit",
        "status": "Quietly dropped",
        "resolved_in": "FY2025Q4",
        "notes": "Searched calls but item was omitted.",
    }
    ledger["promises"].append(dropped_p)
    errors, _ = validate_ledger(ledger)
    assert any("grace period (due+1) has not expired" in e for e in errors)

    # If due is FY2024Q4 (which is well before latest FY2025Q4), it passes if notes are present
    dropped_p["due"] = "FY2024Q3"
    dropped_p["resolved_in"] = "FY2024Q4"
    dropped_p["notes"] = "Searched calls FY2024Q3, FY2024Q4, FY2025Q1 transcripts but item was dropped."
    errors_cleared, _ = validate_ledger(ledger)
    assert not any("Quietly dropped" in e for e in errors_cleared)


def test_compute_promise_scores_and_transparency():
    ledger = _make_sample_ledger()
    scores = compute_promise_scores(ledger)

    # 2 delivered promises with weight 2.0 (numeric) and 1.0 (milestone), 1 pending
    assert scores["delivery"] == 100.0
    assert scores["transparency"] == 100.0
    assert scores["credibility"] == 100.0
    assert scores["grade"] == "A"
    assert scores["n_resolved"] == 2
    assert scores["counts"]["Delivered"] == 2
    assert scores["counts"]["Pending"] == 1
    # Less than 3 milestone slips -> None (insufficient samples)
    assert scores["milestone_slip_quarters"] is None
    assert scores["milestone_slip_n"] == 1


def test_milestone_slip_calculation():
    ledger = _make_sample_ledger()
    # Add 2 more milestone promises with slips: 0, 1, 2 quarters -> mean = 1.0
    for i, slip in enumerate([1, 2], start=2):
        ledger["promises"].append({
            "id": f"P-NVDA-FY2024Q1-0{i}",
            "made_in": "FY2024Q1",
            "speaker": "Jensen Huang, CEO",
            "section": "prepared",
            "type": "B_milestone",
            "category": "product",
            "scope": "company",
            "quote": f"Milestone quote {i}.",
            "metric": f"Milestone {i}",
            "target_kind": "milestone",
            "due": "FY2024Q3",
            "due_precision": "explicit",
            "status": "Delivered",
            "resolved_in": "FY2024Q4",
            "evidence_quote": f"Delivered milestone {i}.",
            "slip_quarters": slip,
            "notes": f"Slipped {slip} quarters.",
        })
    scores = compute_promise_scores(ledger)
    assert scores["milestone_slip_n"] == 3
    assert scores["milestone_slip_quarters"] == 1.0  # (0 + 1 + 2) / 3


def test_guidance_bias_classification():
    ledger = _make_sample_ledger()
    # Add 5 numeric promises with beats to test 'Sandbagger' classification (>= 80% beats, median > 0.02)
    for i in range(1, 6):
        ledger["promises"].append({
            "id": f"P-NVDA-FY2024Q{i % 4 + 1}-1{i}",
            "made_in": f"FY2024Q{i % 4 + 1}",
            "speaker": "Colette Kress, CFO",
            "section": "prepared",
            "type": "A_numeric",
            "category": "revenue",
            "scope": "company",
            "quote": f"Guidance target {i} quote under 40 words.",
            "metric": f"Metric {i}",
            "target_kind": "range",
            "target_low": 10.0,
            "target_high": 12.0,
            "unit": "USD bn",
            "basis": "GAAP",
            "due": "FY2025Q1",
            "due_precision": "explicit",
            "status": "Delivered",
            "resolved_in": "FY2025Q1",
            "evidence_quote": f"Actual {i} was 13.0 above range.",
            "actual_numeric": 13.0,
            "delta_pct": 0.08,  # +8% beat
            "notes": "Beat high end.",
        })
    scores = compute_promise_scores(ledger)
    bias = scores["bias"]
    assert bias["label"] == "Sandbagger"
    assert bias["n"] >= 5
    assert bias["beat_rate"] >= 80.0
    assert bias["median_delta_pct"] > 2.0
