"""Phase 0 Gate: Read-Only Contract Test for FastAPI OpenAPI Schema & Route Precedence.

Asserts zero OpenAPI schema drift and validates route precedence.
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

    for path, expected_methods in expected_paths.items():
        assert path in actual_openapi.get("paths", {}), f"Missing OpenAPI path: {path}"
        actual_methods = actual_openapi["paths"][path]

        for method, expected_op in expected_methods.items():
            assert method in actual_methods, f"Missing method {method} on path {path}"
            actual_op = actual_methods[method]

            assert (
                actual_op.get("operationId") == expected_op.get("operationId")
            ), f"OperationId drift on {method} {path}"
            assert (
                sorted(actual_op.get("tags", [])) == sorted(expected_op.get("tags", []))
            ), f"Tags mismatch on {method} {path}"

            expected_param_names = sorted([p["name"] for p in expected_op.get("parameters", [])])
            actual_param_names = sorted([p["name"] for p in actual_op.get("parameters", [])])
            assert (
                actual_param_names == expected_param_names
            ), f"Parameter mismatch on {method} {path}: Expected {expected_param_names}, got {actual_param_names}"

            expected_responses = sorted(expected_op.get("responses", []))
            actual_responses = sorted(list(actual_op.get("responses", {}).keys()))
            assert (
                actual_responses == expected_responses
            ), f"Response codes mismatch on {method} {path}: Expected {expected_responses}, got {actual_responses}"


def test_equity_route_precedence():
    """Verify that static route /api/equity/notes/content resolves before /{ticker}."""
    client = TestClient(app)
    # Even unauthenticated, we check if route router matching works or returns 401 (auth) vs 404 (not matched)
    # The key is checking the route pattern matched by FastAPI router
    routes = [route for route in app.routes if getattr(route, "path", "").startswith("/api/equity")]
    paths = [route.path for route in routes]

    if "/api/equity/notes/content" in paths and "/api/equity/{ticker}" in paths:
        idx_content = paths.index("/api/equity/notes/content")
        idx_ticker = paths.index("/api/equity/{ticker}")
        assert idx_content < idx_ticker, (
            f"Route precedence error: /api/equity/notes/content (index {idx_content}) "
            f"must be registered before /api/equity/{{ticker}} (index {idx_ticker})"
        )
