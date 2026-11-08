"""API Router tests for Earnings Reported-vs-Expected Bridge (Chapter 5)."""
from fastapi.testclient import TestClient
import pytest


def test_get_earnings_bridge_us(authed_client: TestClient):
    response = authed_client.get("/api/equity/GOOGL/earnings-bridge")
    assert response.status_code == 200
    data = response.json()

    assert data["ticker"] == "GOOGL"
    assert data["period"] == "Q3 2026"
    assert data["is_mathematically_balanced"] is True
    assert data["revenue_surprise"]["direction"] == "BEAT"
    assert data["eps_surprise"]["direction"] == "BEAT"
    assert len(data["revenue_drivers"]) == 3

    # Check driver sum matches net delta
    delta = float(data["revenue_surprise"]["delta"])
    sum_drivers = float(data["sum_drivers"])
    assert abs(delta - sum_drivers) < 0.01
    assert data["thesis_reconciliation_verdict"] == "THESIS_CONFIRMED"


def test_get_earnings_bridge_thai(authed_client: TestClient):
    response = authed_client.get("/api/equity/DELTA.BK/earnings-bridge?period=Q2%202026")
    assert response.status_code == 200
    data = response.json()

    assert data["ticker"] == "DELTA.BK"
    assert data["period"] == "Q2 2026"
    assert data["is_mathematically_balanced"] is True
    assert data["revenue_surprise"]["unit"] == "THB_M"
    assert len(data["revenue_drivers"]) == 3

