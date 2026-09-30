"""Architecture and Hexagonal Boundary Tests for Terminal V2 (Unified Canonical Architecture).

Enforces:
1. Pure standard library isolation in domain (zero external imports).
2. Application service isolation (ports & domain only; no adapters, requests, or web frameworks).
3. Ports isolation (standard library and domain models only).
4. Interface Segregation Principle (ISP): TerminalDataServicePort composed of domain sub-interfaces.
5. Preservation of Phase 1 driving port contract (no broken backward compatibility).
6. Adapters do not import peer adapters or application services.
7. Zero phase-split files exist in terminal_v2 directory.
8. Zero runtime dependency on temp/ directory.
"""
import ast
import inspect
from pathlib import Path
import sys
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TERMINAL_ROOT = PROJECT_ROOT / "tools" / "market" / "terminal_v2"


def _get_ast_imports(file_path: Path) -> list[tuple[int, str]]:
    with open(file_path, "r", encoding="utf-8-sig") as f:
        tree = ast.parse(f.read(), filename=str(file_path))
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append((node.lineno, alias.name))
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append((node.lineno, node.module))
    return imports


def test_domain_pure_standard_library():
    """Domain models and calculations must use Python standard library only."""
    domain_files = [
        TERMINAL_ROOT / "domain" / "models.py",
        TERMINAL_ROOT / "domain" / "calculations.py",
    ]
    stdlib_names = set(sys.stdlib_module_names)

    for fpath in domain_files:
        assert fpath.is_file(), f"Missing domain file: {fpath}"
        for lineno, mod in _get_ast_imports(fpath):
            root_mod = mod.split(".")[0]
            if mod.startswith("tools.market.terminal_v2.domain"):
                continue
            assert root_mod in stdlib_names, (
                f"{fpath.name}:{lineno} imports external/framework module '{mod}'. "
                "Domain must use standard library only!"
            )


def test_application_service_layer_isolation():
    """Application service must only depend on domain and ports, never concrete adapters or frameworks."""
    app_service = TERMINAL_ROOT / "application" / "terminal_data_service.py"
    assert app_service.is_file(), f"Missing terminal_data_service: {app_service}"

    forbidden_patterns = [
        "adapters",
        "requests",
        "fastapi",
        "pydantic",
        "langchain",
        "sqlite3",
        "yfinance",
    ]

    for lineno, mod in _get_ast_imports(app_service):
        for pat in forbidden_patterns:
            assert pat not in mod, (
                f"terminal_data_service.py:{lineno} imports '{mod}' which violates hexagonal architecture "
                f"(forbidden pattern: '{pat}')."
            )


def test_ports_isolation():
    """Ports must only depend on domain models or standard library."""
    port_files = [
        TERMINAL_ROOT / "ports" / "driving_ports.py",
        TERMINAL_ROOT / "ports" / "driven_ports.py",
    ]
    stdlib_names = set(sys.stdlib_module_names)

    for fpath in port_files:
        assert fpath.is_file(), f"Missing port file: {fpath}"
        for lineno, mod in _get_ast_imports(fpath):
            if mod.startswith("tools.market.terminal_v2.domain"):
                continue
            root_mod = mod.split(".")[0]
            assert root_mod in stdlib_names, (
                f"{fpath.name}:{lineno} imports '{mod}'. "
                "Ports may only import domain models and standard library!"
            )


def test_interface_segregation_in_driving_ports():
    """TerminalDataServicePort must compose 5 granular domain sub-interfaces (ISP)."""
    from tools.market.terminal_v2.ports.driving_ports import (
        DerivativesDrivingPort,
        EquityDataDrivingPort,
        FixedIncomeDrivingPort,
        FundFlowDrivingPort,
        MacroDataDrivingPort,
        TerminalDataServicePort,
    )

    expected_bases = {
        MacroDataDrivingPort,
        FixedIncomeDrivingPort,
        EquityDataDrivingPort,
        DerivativesDrivingPort,
        FundFlowDrivingPort,
    }
    actual_bases = set(TerminalDataServicePort.__bases__)
    assert expected_bases.issubset(actual_bases), (
        f"TerminalDataServicePort must inherit from all 5 ISP sub-interfaces. "
        f"Expected: {expected_bases}, Got: {actual_bases}"
    )


def test_phase1_driving_port_unmodified():
    """MarketTerminalServicePort must remain dedicated to real-time routing."""
    from tools.market.terminal_v2.ports.driving_ports import MarketTerminalServicePort

    expected_phase1_methods = {
        "get_investor_flow",
        "get_market_valuation",
        "get_market_breadth",
        "get_retail_gold",
        "get_macro_series",
        "get_perp_quote",
        "query_by_capability",
    }

    actual_methods = {
        name
        for name, member in inspect.getmembers(MarketTerminalServicePort, predicate=inspect.isfunction)
        if getattr(member, "__isabstractmethod__", False)
    }

    assert actual_methods == expected_phase1_methods, (
        f"Phase 1 MarketTerminalServicePort contract was altered! "
        f"Expected: {expected_phase1_methods}, Got: {actual_methods}"
    )


def test_zero_phase_split_files_remain():
    """Strictly assert that no phase-split files remain in terminal_v2."""
    for py_file in TERMINAL_ROOT.rglob("*.py"):
        filename = py_file.name.lower()
        assert not (
            ("phase2" in filename or "phase3" in filename or "phase4" in filename)
            and filename != "phase4"
        ), f"Lingering phase file found: {py_file}"


def test_adapters_do_not_import_peer_adapters():
    """Adapters must be isolated and must not import concrete peer adapters."""
    adapters_dir = TERMINAL_ROOT / "adapters"
    adapter_files = list(adapters_dir.glob("*.py"))
    assert len(adapter_files) >= 15, f"Expected at least 15 adapter files, got {len(adapter_files)}"

    for fpath in adapter_files:
        if fpath.name == "__init__.py":
            continue
        for lineno, mod in _get_ast_imports(fpath):
            if "tools.market.terminal_v2.adapters" in mod:
                imported_adapter = mod.split(".")[-1]
                assert imported_adapter in (fpath.stem, "adapters"), (
                    f"{fpath.name}:{lineno} imports peer adapter '{mod}'. Adapters must remain decoupled!"
                )


def test_zero_runtime_dependency_on_temp():
    """Verify that no production files in tools, api, or agents import from temp/."""
    scan_dirs = [
        PROJECT_ROOT / "tools" / "market" / "terminal_v2",
        PROJECT_ROOT / "api" / "routers",
        PROJECT_ROOT / "api" / "schemas",
        PROJECT_ROOT / "agents",
    ]

    for d in scan_dirs:
        for py_file in d.rglob("*.py"):
            with open(py_file, "r", encoding="utf-8") as f:
                content = f.read()
            assert "temp.zframes" not in content and "zframes-main" not in content, (
                f"Production file {py_file} has illegal reference to temporary study directory 'temp'!"
            )
