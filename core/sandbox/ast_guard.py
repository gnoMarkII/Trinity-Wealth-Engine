"""AST Whitelist Validator for manager_sandbox/scratch/ scripts."""
from __future__ import annotations

import ast
from dataclasses import dataclass

ALLOWED_MODULES: frozenset[str] = frozenset({
    "math",
    "decimal",
    "statistics",
    "json",
    "datetime",
    "collections",
    "itertools",
    "functools",
    "typing",
    "dataclasses",
})

FORBIDDEN_BUILTINS: frozenset[str] = frozenset({
    "eval",
    "exec",
    "compile",
    "open",
    "__import__",
    "input",
    "breakpoint",
    "globals",
    "locals",
    "getattr",
    "setattr",
    "delattr",
})


@dataclass(frozen=True)
class AstValidationResult:
    is_safe: bool
    violations: tuple[str, ...]


def validate_scratch_python(source_code: str) -> AstValidationResult:
    """Parse and inspect Python code before allowing execution in manager_sandbox/scratch/."""
    try:
        tree = ast.parse(source_code)
    except SyntaxError as exc:
        return AstValidationResult(
            is_safe=False,
            violations=(f"SyntaxError: {exc.msg} (line {exc.lineno})",),
        )

    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root_mod = alias.name.split(".")[0]
                if root_mod not in ALLOWED_MODULES:
                    violations.append(f"Forbidden import: '{alias.name}'")
        elif isinstance(node, ast.ImportFrom):
            root_mod = (node.module or "").split(".")[0]
            if root_mod not in ALLOWED_MODULES:
                violations.append(f"Forbidden import from: '{node.module}'")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in FORBIDDEN_BUILTINS:
                violations.append(f"Forbidden builtin call: '{node.func.id}()'")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            violations.append(f"Forbidden dunder attribute access: '{node.attr}'")

    return AstValidationResult(is_safe=len(violations) == 0, violations=tuple(violations))

