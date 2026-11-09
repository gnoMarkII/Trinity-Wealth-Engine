import ast
import os
from pathlib import Path
from typing import Iterator, Union

_FuncNode = Union[ast.FunctionDef, ast.AsyncFunctionDef]


def _is_pytest_fixture(node: _FuncNode) -> bool:
    for decorator in node.decorator_list:
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        if isinstance(target, ast.Name) and target.id == "fixture":
            return True
        if isinstance(target, ast.Attribute) and target.attr == "fixture":
            return True
    return False


def _collect_test_functions(tree: ast.AST) -> list[_FuncNode]:
    """Collects top-level and class-method test_* functions, excluding fixtures and nested defs."""
    tests: list[_FuncNode] = []
    for child in ast.iter_child_nodes(tree):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if child.name.startswith("test_") and not _is_pytest_fixture(child):
                tests.append(child)
        elif isinstance(child, ast.ClassDef):
            tests.extend(_collect_test_functions(child))
    return tests


def _iter_body_without_nested_defs(fn: _FuncNode) -> Iterator[ast.AST]:
    """Yields AST nodes inside fn.body without descending into nested defs or classes."""
    stack: list[ast.AST] = list(fn.body)
    while stack:
        node = stack.pop()
        yield node
        for child in ast.iter_child_nodes(node):
            if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                stack.append(child)


def _has_effective_assertion(fn: _FuncNode) -> bool:
    for node in _iter_body_without_nested_defs(fn):
        if isinstance(node, ast.Assert):
            return True
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute):
                if func.attr.startswith("assert"):
                    return True
                if (
                    isinstance(func.value, ast.Name)
                    and func.value.id == "pytest"
                    and func.attr in {"raises", "warns", "fail", "approx"}
                ):
                    return True
            elif isinstance(func, ast.Name):
                if func.id.startswith(("assert", "_assert")) or func.id in {"raises", "warns", "fail"}:
                    return True
    return False


def _iter_test_files() -> list[Path]:
    project_root = Path(__file__).parent.parent
    tests_dir = project_root / "tests"
    result: list[Path] = []
    for root, dirs, files in os.walk(tests_dir):
        dirs[:] = [d for d in dirs if d != "__pycache__"]
        for file in sorted(files):
            if file.startswith("test_") and file.endswith(".py"):
                result.append(Path(root) / file)
    return result


def test_no_empty_test_functions():
    empty_tests = []
    project_root = Path(__file__).parent.parent

    for file_path in _iter_test_files():
        content = file_path.read_text(encoding="utf-8-sig")
        tree = ast.parse(content, filename=str(file_path))
        rel = file_path.relative_to(project_root).as_posix()
        for node in _collect_test_functions(tree):
            if len(node.body) == 1:
                stmt = node.body[0]
                if isinstance(stmt, ast.Pass):
                    empty_tests.append(f"{rel}:{node.name} (pass)")
                elif (
                    isinstance(stmt, ast.Expr)
                    and isinstance(stmt.value, ast.Constant)
                    and stmt.value.value is Ellipsis
                ):
                    empty_tests.append(f"{rel}:{node.name} (...)")

    assert not empty_tests, "Found empty test functions (pass or ...):\n" + "\n".join(empty_tests)


def test_all_test_functions_have_assertions():
    """Every test_* function (excluding fixtures) must assert or check at least one expectation."""
    missing_assertions: list[str] = []
    project_root = Path(__file__).parent.parent

    for file_path in _iter_test_files():
        content = file_path.read_text(encoding="utf-8-sig")
        tree = ast.parse(content, filename=str(file_path))
        rel = file_path.relative_to(project_root).as_posix()
        for fn in _collect_test_functions(tree):
            if not _has_effective_assertion(fn):
                missing_assertions.append(f"{rel}:{fn.lineno}:{fn.name}")

    assert not missing_assertions, (
        "Found test functions with no assertions / pytest.raises / mock assert_* calls:\n"
        + "\n".join(missing_assertions)
    )
