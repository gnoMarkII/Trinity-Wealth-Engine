"""API integration tests for PM Review and Thesis routers."""
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def isolate_thesis_store(monkeypatch, tmp_path):
    d_dir = tmp_path / "data_theses"
    v_dir = tmp_path / "vault_theses"
    d_dir.mkdir(parents=True, exist_ok=True)
    v_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("THESIS_DATA_DIR", str(d_dir))
    monkeypatch.setenv("THESIS_VAULT_DIR", str(v_dir))


def test_unauthenticated_requests_rejected(client: TestClient):
    resp1 = client.get("/api/manager/pm-review/today")
    assert resp1.status_code == 401

    resp2 = client.get("/api/thesis/NVDA")
    assert resp2.status_code == 401


def test_get_today_morning_packet(authed_client: TestClient):
    resp = authed_client.get("/api/manager/pm-review/today")
    assert resp.status_code == 200
    data = resp.json()
    assert "review_date" in data
    assert "status" in data
    assert "portfolio_delta" in data
    assert "macro_delta" in data
    assert "thesis_exceptions" in data
    assert "pending_decisions" in data
    assert "packet_sha256" in data
    assert "review_sample" in data
    assert isinstance(data["review_sample"], list)


def test_sign_off_morning_review_endpoint(authed_client: TestClient):
    # 1. Get today's packet
    get_resp = authed_client.get("/api/manager/pm-review/today")
    assert get_resp.status_code == 200
    packet = get_resp.json()
    r_date = packet["review_date"]
    sha = packet["packet_sha256"]

    # 2. Submit the samples and every pending exception acknowledgement.
    sample_ids = [s["sample_id"] for s in packet.get("review_sample", [])]
    exception_ids = [
        row["exception_id"]
        for row in packet.get("thesis_exceptions", [])
        if str(row.get("status") or "PENDING").upper() not in ("ACKNOWLEDGED", "REVISED")
    ]
    sign_resp = authed_client.post(
        "/api/manager/pm-review/sign-off",
        json={
            "review_date": r_date,
            "signed_off_by": "CHINO_PM",
            "notes": "Morning allocations verified.",
            "packet_sha256": sha,
            "verified_sample_ids": sample_ids,
            "acknowledged_exception_ids": exception_ids,
        },
    )
    assert sign_resp.status_code == 200, sign_resp.text
    signoff = sign_resp.json()
    assert signoff["status"] == "SIGNED_OFF"
    assert signoff["packet_sha256"] == sha
    assert signoff["verified_sample_ids"] == sample_ids
    assert signoff["acknowledged_exception_ids"] == exception_ids

    # The acknowledged evidence survives a read-back from the repository.
    after_resp = authed_client.get("/api/manager/pm-review/today")
    assert after_resp.status_code == 200
    assert after_resp.json()["status"] == "SIGNED_OFF"

    hist_resp = authed_client.get("/api/manager/pm-review/history")
    assert hist_resp.status_code == 200
    assert any(
        row["review_date"] == r_date
        and row["packet_sha256"] == sha
        and row["verified_sample_ids"] == sample_ids
        for row in hist_resp.json()
    )


def test_create_and_get_thesis_api(authed_client: TestClient):
    ticker = "NVDA"
    payload = {
        "claim_statement": "NVDA maintains datacenter revenue > 35B and gross margin >= 70%",
        "falsifiers": [
            {
                "falsifier_id": "FAL-NVDA-REV",
                "metric_name": "datacenter_revenue",
                "operator": "<",
                "threshold_value": "35000000000",
                "unit": "USD",
                "period_basis": "QUARTERLY",
                "invalidation_impact": "BREAKS_BUY_CASE",
                "narrative_explanation": "Revenue drop below 35B signals slowing AI capex",
            }
        ],
        "catalysts": [
            {
                "catalyst_id": "CAT-NVDA-Q3",
                "event_name": "Q3 FY27 Earnings",
                "expected_date_str": "2026-11-20",
                "confidence_window_days": 5,
                "source_locator": "SEC 8-K",
                "status": "PENDING",
            }
        ],
        "evidence_tier": "DOCUMENTED",
        "author_role": "HUMAN_PM",
        "as_of_utc": "2026-11-01T00:00:00Z",
    }

    # 1. Post new thesis
    post_resp = authed_client.post(f"/api/thesis/{ticker}", json=payload)
    assert post_resp.status_code == 200
    data = post_resp.json()
    assert data["ticker"] == "NVDA"
    assert data["version"] == 1
    assert len(data["falsifiers"]) == 1
    assert data["checksum_sha256"] != ""

    # 2. Get active thesis
    get_resp = authed_client.get(f"/api/thesis/{ticker}")
    assert get_resp.status_code == 200
    active = get_resp.json()
    assert active["ticker"] == "NVDA"
    assert active["claim_id"] == data["claim_id"]

    # 3. Time travel query
    as_of_resp = authed_client.get(f"/api/thesis/{ticker}/as-of?as_of_utc=2026-11-02T00:00:00Z")
    assert as_of_resp.status_code == 200
    assert as_of_resp.json()["claim_id"] == data["claim_id"]

    # Past date before record creation
    past_resp = authed_client.get(f"/api/thesis/{ticker}/as-of?as_of_utc=2026-01-01T00:00:00Z")
    assert past_resp.status_code == 200
    assert past_resp.json() is None


def test_watchdog_evaluate_endpoint(authed_client: TestClient):
    resp = authed_client.post("/api/thesis/evaluate")
    assert resp.status_code == 200
    data = resp.json()
    assert "exceptions_detected" in data
    assert isinstance(data["exceptions"], list)


def test_synthesize_thesis_endpoint(authed_client: TestClient):
    resp = authed_client.post(
        "/api/thesis/synthesize/GOOGL",
        json={"dry_run": True, "custom_stop_loss": 140.0, "custom_growth_floor": 10.0, "custom_margin_floor": 25.0},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["ticker"] == "GOOGL"
    assert data["dry_run"] is True
    assert len(data["falsifiers"]) == 3
    assert data["stop_loss"] == 140.0


def test_batch_synthesize_endpoint(authed_client: TestClient):
    resp = authed_client.post(
        "/api/thesis/batch-synthesize",
        json={"portfolio_id": "default", "dry_run": True},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "portfolio_id" in data
    assert "prior_coverage_pct" in data
    assert "newly_synthesized_count" in data


def test_earnings_bridge_endpoint(authed_client: TestClient):
    resp = authed_client.post(
        "/api/thesis/earnings-bridge",
        json={
            "ticker": "GOOGL",
            "period": "Q3-2026",
            "as_of_date": "2026-11-04",
            "revenue_reported": "1050.0",
            "revenue_consensus": "1000.0",
            "eps_reported": "2.10",
            "eps_consensus": "1.95",
            "revenue_unit": "USD_M",
            "eps_unit": "USD",
            "revenue_drivers": [
                {
                    "name": "Cloud Volume",
                    "category": "VOLUME",
                    "impact_value": "30.0",
                    "unit": "USD_M",
                    "narrative": "Organic expansion",
                },
                {
                    "name": "Search Price Mix",
                    "category": "PRICE_MIX",
                    "impact_value": "20.0",
                    "unit": "USD_M",
                    "narrative": "Higher monetization",
                },
            ],
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["ticker"] == "GOOGL"
    assert data["is_mathematically_balanced"] is True
    assert data["sum_drivers"] == "50.0"
    assert data["revenue_surprise"]["direction"] == "BEAT"
    assert data["thesis_reconciliation_verdict"] == "THESIS_CONFIRMED"


def test_sign_off_missing_checksum_rejected(authed_client):
    """P1 Gate (J01): Sign-off without packet_sha256 must return 422 Unprocessable Entity."""
    resp = authed_client.post(
        "/api/manager/pm-review/sign-off",
        json={
            "review_date": "2026-11-06",
            "signed_off_by": "CHINO_PM",
            "notes": "Missing checksum test",
        },
    )
    assert resp.status_code == 422


def test_sign_off_empty_samples_rejected_when_packet_has_samples(authed_client):
    """P1 Gate (J01): Sign-off with empty verified_sample_ids is rejected if morning packet contains review samples."""
    get_resp = authed_client.get("/api/manager/pm-review/today")
    assert get_resp.status_code == 200
    pkt = get_resp.json()
    if pkt.get("review_sample"):
        resp = authed_client.post(
            "/api/manager/pm-review/sign-off",
            json={
                "review_date": pkt["review_date"],
                "packet_sha256": pkt["packet_sha256"],
                "verified_sample_ids": [],
            },
        )
        assert resp.status_code == 422


