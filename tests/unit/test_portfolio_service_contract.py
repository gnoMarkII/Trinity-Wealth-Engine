"""Phase 0 Gate: Read-Only Contract Test for PortfolioService.

Asserts that PortfolioService public methods, parameter names, kinds, defaults,
return type annotations, and docstrings match the immutable golden manifest exactly.
"""
import inspect
import json
from pathlib import Path

import pytest
from tools.portfolio.service import PortfolioService

MANIFEST_PATH = Path(__file__).resolve().parent.parent / "fixtures" / "manifest_portfolio_service.json"


def _format_annotation(ann):
    if ann is inspect.Parameter.empty:
        return "empty"
    if hasattr(ann, "__name__"):
        return ann.__name__
    return str(ann)


def _format_default(default):
    if default is inspect.Parameter.empty:
        return "<EMPTY>"
    if default is None:
        return None
    if isinstance(default, (int, float, bool, str, list, dict)):
        return default
    return str(default)


@pytest.fixture(scope="module")
def golden_manifest():
    assert MANIFEST_PATH.exists(), f"Missing baseline manifest: {MANIFEST_PATH}"
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def test_portfolio_service_methods_exist(golden_manifest):
    expected_methods = set(golden_manifest["methods"].keys())
    actual_methods = {
        name
        for name, method in inspect.getmembers(PortfolioService, predicate=inspect.isfunction)
        if not name.startswith("_") or name == "__init__" or name in expected_methods
    }
    missing = expected_methods - actual_methods
    extra = actual_methods - expected_methods
    assert not missing, f"PortfolioService is missing public methods: {missing}"
    assert not extra, f"PortfolioService has unexpected unmanifested methods: {extra}"


def test_portfolio_service_method_signatures(golden_manifest):
    for method_name, expected_info in golden_manifest["methods"].items():
        assert hasattr(PortfolioService, method_name), f"Missing method {method_name}"
        method = getattr(PortfolioService, method_name)
        sig = inspect.signature(method)

        expected_params = expected_info["parameters"]
        actual_params = [
            {
                "name": p_name,
                "kind": str(param.kind),
                "default": _format_default(param.default),
                "annotation": _format_annotation(param.annotation),
            }
            for p_name, param in sig.parameters.items()
        ]

        assert (
            actual_params == expected_params
        ), f"Signature mismatch for method '{method_name}':\nExpected: {expected_params}\nActual:   {actual_params}"

        actual_return = _format_annotation(sig.return_annotation)
        expected_return = expected_info["return_annotation"]
        assert (
            actual_return == expected_return
        ), f"Return annotation mismatch for '{method_name}': Expected {expected_return}, got {actual_return}"

        # Docstring check (if present in expected manifest)
        expected_doc = expected_info.get("docstring")
        if expected_doc:
            actual_doc = inspect.getdoc(method) or (method.__doc__.strip() if method.__doc__ else "")
            # Normalize whitespace
            assert (
                actual_doc.strip() == expected_doc.strip()
            ), f"Docstring mismatch on '{method_name}':\nExpected: {expected_doc}\nActual: {actual_doc}"
