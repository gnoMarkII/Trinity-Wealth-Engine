"""Phase 0 Gate: Read-Only Contract Test for FastAPI OpenAPI Schema & Route Precedence.

Asserts zero OpenAPI schema drift and validates route precedence and auth dependency.
"""
import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from api.main import app

MANIFEST_PATH = Path(__file__).resolve().parent.parent / "fixtures" / "manifest_openapi_schema.json"


@pytest.fixture(scope="module")
def golden_manifest():
    assert MANIFEST_PATH.exists(), f"Missing baseline manifest: {MANIFEST_PATH}"
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def test_openapi_paths_and_operations(golden_manifest):
    actual_openapi = app.openapi()
    expected_paths = golden_manifest["paths"]
    actual_paths = actual_openapi.get("paths", {})

    # Assert exact paths match (Zero Drift: no extra routes and no missing routes)
    assert set(actual_paths.keys()) == set(expected_paths.keys()), (
        f"OpenAPI paths mismatch.\n"
        f"Missing: {set(expected_paths.keys()) - set(actual_paths.keys())}\n"
        f"Unexpected extra: {set(actual_paths.keys()) - set(expected_paths.keys())}"
    )

    for path, expected_methods in expected_paths.items():
        assert path in actual_paths, f"Missing OpenAPI path: {path}"
        actual_methods = actual_paths[path]

        assert set(actual_methods.keys()) == set(expected_methods.keys()), (
            f"Methods mismatch on path '{path}':\n"
            f"Expected: {set(expected_methods.keys())}\n"
            f"Actual:   {set(actual_methods.keys())}"
        )

        for method, expected_op in expected_methods.items():
            actual_op = actual_methods[method]

            assert (
                actual_op.get("operationId") == expected_op.get("operationId")
            ), f"OperationId drift on {method} {path}"
            assert (
                sorted(actual_op.get("tags", [])) == sorted(expected_op.get("tags", []))
            ), f"Tags mismatch on {method} {path}"

            # Check request body presence
            actual_has_body = "requestBody" in actual_op
            expected_has_body = expected_op.get("has_request_body", False)
            assert (
                actual_has_body == expected_has_body
            ), f"Request body mismatch on {method} {path}: Expected has_request_body={expected_has_body}, got {actual_has_body}"

            # Check parameters
            expected_params = expected_op.get("parameters", [])
            actual_params = actual_op.get("parameters", [])
            expected_param_names = sorted([p["name"] for p in expected_params])
            actual_param_names = sorted([p["name"] for p in actual_params])
            assert (
                actual_param_names == expected_param_names
            ), f"Parameter mismatch on {method} {path}: Expected {expected_param_names}, got {actual_param_names}"

            # Check responses
            expected_responses = sorted(expected_op.get("responses", []))
            actual_responses = sorted(list(actual_op.get("responses", {}).keys()))
            assert (
                actual_responses == expected_responses
            ), f"Response codes mismatch on {method} {path}: Expected {expected_responses}, got {actual_responses}"


def test_equity_route_precedence():
    """Verify that static route /api/equity/notes/content resolves before /{ticker}."""
    routes = [route for route in app.routes if getattr(route, "path", "").startswith("/api/equity")]
    paths = [route.path for route in routes]

    if "/api/equity/notes/content" in paths and "/api/equity/{ticker}" in paths:
        idx_content = paths.index("/api/equity/notes/content")
        idx_ticker = paths.index("/api/equity/{ticker}")
        assert idx_content < idx_ticker, (
            f"Route precedence error: /api/equity/notes/content (index {idx_content}) "
            f"must be registered before /api/equity/{{ticker}} (index {idx_ticker})"
        )


def test_protected_routes_require_authentication():
    """Verify that unauthenticated requests to protected endpoints receive 401 Unauthorized."""
    client = TestClient(app)
    protected_test_endpoints = [
        ("GET", "/api/portfolio/list"),
        ("GET", "/api/portfolio/actual/state"),
        ("GET", "/api/kanban/cards"),
        ("GET", "/api/agents/active"),
        ("GET", "/api/notebooklm/available-sources"),
    ]

    for method, endpoint in protected_test_endpoints:
        if method == "GET":
            response = client.get(endpoint)
        else:
            response = client.post(endpoint)
        # Should return 401 Unauthorized when session is missing
        assert response.status_code == 401, (
            f"Expected endpoint {method} {endpoint} to require authentication (401), got {response.status_code}"
        )
