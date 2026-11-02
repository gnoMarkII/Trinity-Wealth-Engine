"""Tests for core/sandbox/ast_guard.py — Scratchpad Python security gate."""
import pytest
from core.sandbox.ast_guard import validate_scratch_python


def test_valid_financial_math_is_safe():
    code = """
import math
from decimal import Decimal

principal = Decimal('1000000.00')
rate = Decimal('0.07')
years = 10
future_value = principal * ((1 + rate) ** years)
print(future_value)
"""
    result = validate_scratch_python(code)
    assert result.is_safe is True
    assert len(result.violations) == 0


def test_syntax_error_is_caught():
    code = "def broken_syntax(:"
    result = validate_scratch_python(code)
    assert result.is_safe is False
    assert any("SyntaxError" in v for v in result.violations)


def test_forbidden_module_import_is_blocked():
    code = """
import os
import subprocess
os.system('whoami')
"""
    result = validate_scratch_python(code)
    assert result.is_safe is False
    assert any("Forbidden import: 'os'" in v for v in result.violations)
    assert any("Forbidden import: 'subprocess'" in v for v in result.violations)


def test_forbidden_from_import_is_blocked():
    code = "from sys import exit\nexit(1)"
    result = validate_scratch_python(code)
    assert result.is_safe is False
    assert any("Forbidden import from: 'sys'" in v for v in result.violations)


def test_forbidden_builtin_eval_is_blocked():
    code = "result = eval('2 + 2')"
    result = validate_scratch_python(code)
    assert result.is_safe is False
    assert any("Forbidden builtin call: 'eval()'" in v for v in result.violations)


def test_forbidden_builtin_open_is_blocked():
    code = "with open('/etc/passwd', 'r') as f: data = f.read()"
    result = validate_scratch_python(code)
    assert result.is_safe is False
    assert any("Forbidden builtin call: 'open()'" in v for v in result.violations)


def test_dunder_attribute_traversal_is_blocked():
    code = "x = ().__class__.__bases__[0].__subclasses__()"
    result = validate_scratch_python(code)
    assert result.is_safe is False
    assert any("Forbidden dunder attribute access" in v for v in result.violations)

