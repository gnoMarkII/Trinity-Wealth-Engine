"""Phase 0 Gate: Architecture AST Dependency Rules & Layer Invariants.

Asserts layer boundaries and dependency directions using AST parsing:
- Domain isolation (no framework/adapter/api/sqlite/yfinance/langchain imports)
- Application isolation (no concrete adapter imports)
- Lower layer isolation (tools/agents/application must not import api.* unless tracked)
- Raw DAO isolation (no commit/rollback calls)
- No unallowed sys.modules service locator lookups in production code
- All exceptions are strictly tracked in allowlists with owners, removal phases, and removal conditions
"""
import ast
import os
from pathlib import Path
import pytest

from tests.architecture.allowlists import ARCHITECTURE_ALLOWLIST, is_allowed

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _get_imports(file_path: Path) -> list[tuple[int, str]]:
    """Parse python file and extract all imported module names with line numbers."""
    with open(file_path, "r", encoding="utf-8-sig") as f:
        content = f.read()

    tree = ast.parse(content, filename=str(file_path))
    imports = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append((node.lineno, alias.name))
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append((node.lineno, node.module))
    return imports


def _find_sys_modules_lookups(file_path: Path) -> list[tuple[int, str]]:
    """Detect AST nodes accessing sys.modules."""
    with open(file_path, "r", encoding="utf-8-sig") as f:
        content = f.read()

    tree = ast.parse(content, filename=str(file_path))
    lookups = []

    for node in ast.walk(tree):
        # Match sys.modules.get("...") or sys.modules[...]
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and node.func.attr == "get":
                if isinstance(node.func.value, ast.Attribute) and isinstance(node.func.value.value, ast.Name):
                    if node.func.value.value.id == "sys" and node.func.value.attr == "modules":
                        arg_str = ""
                        if node.args and isinstance(node.args[0], ast.Constant):
                            arg_str = str(node.args[0].value)
                        lookups.append((node.lineno, arg_str))
        elif isinstance(node, ast.Subscript):
            if isinstance(node.value, ast.Attribute) and isinstance(node.value.value, ast.Name):
                if node.value.value.id == "sys" and node.value.attr == "modules":
                    arg_str = ""
                    if isinstance(node.slice, ast.Constant):
                        arg_str = str(node.slice.value)
                    lookups.append((node.lineno, arg_str))
    return lookups


def test_domain_layer_isolation():
    """Domain layers must not import infrastructure, frameworks, adapters, or api."""
    domain_dirs = [
        PROJECT_ROOT / "core" / "investor_essence",
        PROJECT_ROOT / "tools" / "portfolio" / "domain",
        PROJECT_ROOT / "tools" / "market" / "financials" / "domain",
        PROJECT_ROOT / "tools" / "market" / "ohlcv" / "domain",
        PROJECT_ROOT / "tools" / "market" / "terminal_v2" / "domain",
    ]

    forbidden_patterns = [
        "fastapi",
        "sqlite3",
        "yfinance",
        "langchain",
        "api",
        "tools.portfolio.adapters",
        "tools.market.financials.adapters",
        "tools.market.ohlcv.adapters",
        "tools.market.terminal_v2.adapters",
        "tools.investor_essence.adapters",
    ]

    violations = []
    for domain_dir in domain_dirs:
        if not domain_dir.is_dir():
            continue
        for root, _, files in os.walk(domain_dir):
            for file in files:
                if file.endswith(".py"):
                    file_path = Path(root) / file
                    rel_path = file_path.relative_to(PROJECT_ROOT).as_posix()
                    for lineno, imported in _get_imports(file_path):
                        for forbidden in forbidden_patterns:
                            if imported == forbidden or imported.startswith(forbidden + "."):
                                if not is_allowed(rel_path, imported, "forbidden_domain_import"):
                                    violations.append(f"{rel_path}:{lineno} imports '{imported}' (forbidden in domain)")

    assert not violations, "Domain layer violations found:\n" + "\n".join(violations)


def test_portfolio_services_do_not_import_adapters():
    """Application services must not import concrete adapters directly."""
    services_dir = PROJECT_ROOT / "tools" / "portfolio" / "services"
    assert services_dir.is_dir(), "Missing portfolio services directory"

    violations = []
    for root, _, files in os.walk(services_dir):
        for file in files:
            if file.endswith(".py"):
                file_path = Path(root) / file
                rel_path = file_path.relative_to(PROJECT_ROOT).as_posix()
                for lineno, imported in _get_imports(file_path):
                    if "tools.portfolio.adapters" in imported:
                        if not is_allowed(rel_path, imported, "adapter_in_application_service"):
                            violations.append(f"{rel_path}:{lineno} imports adapter '{imported}'")
                    if imported == "tools.portfolio.journal" or imported.startswith("tools.portfolio.journal."):
                        if not is_allowed(rel_path, imported, "forbidden_service_import"):
                            violations.append(f"{rel_path}:{lineno} imports legacy module '{imported}'")

    assert not violations, "Portfolio services importing concrete adapters:\n" + "\n".join(violations)


def test_application_and_router_layers_do_not_import_infrastructure():
    """Application code depends on ports; inbound routers never reach DB/providers."""
    application_dirs = [PROJECT_ROOT / "application"]
    application_dirs.extend(
        [
            PROJECT_ROOT / "tools" / "market" / "ohlcv" / "application",
            PROJECT_ROOT / "tools" / "market" / "financials" / "application",
            PROJECT_ROOT / "tools" / "market" / "terminal_v2" / "application",
        ]
    )
    application_files = [
        PROJECT_ROOT / "tools" / "market" / "financials" / "service.py",
        # The legacy OHLCV facade may delegate to its bootstrap factory, but
        # must not import or construct concrete adapters itself.
        PROJECT_ROOT / "tools" / "market" / "ohlcv" / "service.py",
    ]
    application_forbidden = (
        "api.db",
        "api.state_db",
        "sqlite3",
        "yfinance",
        "fastapi",
        ".adapters",
        "tools.content.notebooklm.adapter",
    )
    violations = []
    for base_dir in application_dirs:
        if not base_dir.is_dir():
            continue
        for root, _, files in os.walk(base_dir):
            for file in files:
                if not file.endswith(".py"):
                    continue
                path = Path(root) / file
                rel = path.relative_to(PROJECT_ROOT).as_posix()
                for lineno, imported in _get_imports(path):
                    if any(imported == pattern or imported.startswith(pattern + ".") for pattern in application_forbidden):
                        if not is_allowed(rel, imported, "infrastructure_import"):
                            violations.append(f"{rel}:{lineno} imports '{imported}'")
    for path in application_files:
        if not path.is_file():
            continue
        rel = path.relative_to(PROJECT_ROOT).as_posix()
        for lineno, imported in _get_imports(path):
            if any(imported == pattern or imported.startswith(pattern + ".") for pattern in application_forbidden):
                if not is_allowed(rel, imported, "infrastructure_import"):
                    violations.append(f"{rel}:{lineno} imports '{imported}'")
    assert not violations, "Application layer infrastructure imports found:\n" + "\n".join(violations)

    router_forbidden = (
        "api.db",
        "api.state_db",
        "api.routes_portfolio",
        "api.routes_equity",
        "api.routes_notebooklm",
        "api.routes_ohlcv",
        "api.compatibility",
        "api.news_funnel_cards",
        "sqlite3",
        "yfinance",
        "sys",
        "tools.archivist",
        "tools.market.asset_resolver",
        "tools.market.calendar",
        "tools.market.earnings",
    )
    for root, _, files in os.walk(PROJECT_ROOT / "api" / "routers"):
        for file in files:
            if not file.endswith(".py"):
                continue
            path = Path(root) / file
            rel = path.relative_to(PROJECT_ROOT).as_posix()
            for lineno, imported in _get_imports(path):
                if any(imported == pattern or imported.startswith(pattern + ".") for pattern in router_forbidden):
                    if not is_allowed(rel, imported, "router_infrastructure_import"):
                        violations.append(f"{rel}:{lineno} imports '{imported}'")
    assert not violations, "Router infrastructure imports found:\n" + "\n".join(violations)


def test_routers_are_http_only_and_do_not_perform_filesystem_io():
    """Routers parse/map requests; files and providers belong to adapters."""
    forbidden_calls = {"open", "read_text", "write_text", "glob", "rglob", "iterdir", "mkdir", "unlink"}
    violations = []
    for root, _, files in os.walk(PROJECT_ROOT / "api" / "routers"):
        for file in files:
            if not file.endswith(".py"):
                continue
            path = Path(root) / file
            rel = path.relative_to(PROJECT_ROOT).as_posix()
            with open(path, "r", encoding="utf-8-sig") as handle:
                tree = ast.parse(handle.read(), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    func = node.func
                    if isinstance(func, ast.Name) and func.id in forbidden_calls:
                        violations.append(f"{rel}:{node.lineno} calls filesystem function '{func.id}()'")
                    elif isinstance(func, ast.Attribute) and func.attr in forbidden_calls:
                        violations.append(f"{rel}:{node.lineno} calls filesystem method '{func.attr}()'")
    assert not violations, "Routers performing filesystem I/O:\n" + "\n".join(violations)


def test_background_workers_use_compatibility_adapter_not_state_db_directly():
    """Worker entry points must depend on an outbound adapter boundary."""
    worker_files = (
        PROJECT_ROOT / "api" / "jobs.py",
        PROJECT_ROOT / "api" / "notebooklm_worker.py",
        PROJECT_ROOT / "api" / "news_funnel_cards.py",
    )
    violations = []
    for path in worker_files:
        rel = path.relative_to(PROJECT_ROOT).as_posix()
        for lineno, imported in _get_imports(path):
            if imported == "api.state_db" or imported == "api":
                violations.append(f"{rel}:{lineno} imports '{imported}' directly")
    assert not violations, "Background workers importing state_db directly:\n" + "\n".join(violations)


def test_job_entrypoint_has_one_workflow_owner():
    """The API queue must delegate graph execution to the agent driver only once."""
    path = PROJECT_ROOT / "api" / "jobs.py"
    source = path.read_text(encoding="utf-8-sig")
    assert "_legacy_default_run_fn" not in source
    imports = _get_imports(path)
    forbidden = [
        imported
        for _, imported in imports
        if imported == "langgraph"
        or imported.startswith("langgraph.")
            or imported in {
            "agents.news_youtube_flow",
            "agents.news_funnel_flow",
            "agents.youtube_pitch_flow",
        }
    ]
    assert not forbidden, "api/jobs.py still owns graph execution: " + ", ".join(forbidden)


def test_insider_sync_pipeline_has_provider_boundary():
    """Legacy SEC parsing may remain, but external fetching belongs to its adapter."""
    pipeline = (PROJECT_ROOT / "tools" / "market" / "sec_form4_pipeline.py").read_text(
        encoding="utf-8-sig"
    )
    provider = (PROJECT_ROOT / "tools" / "market" / "adapters" / "insider_provider.py").read_text(
        encoding="utf-8-sig"
    )
    assert "import yfinance" not in pipeline
    assert "import yfinance" in provider
    assert "YFinanceInsiderHistoryAdapter" in pipeline


def test_composition_root_does_not_import_route_facades():
    """Concrete wiring must not depend backwards on inbound route modules."""
    path = PROJECT_ROOT / "api" / "dependencies.py"
    imports = _get_imports(path)
    forbidden = [
        imported
        for _, imported in imports
        if imported.startswith("api.routes_") or imported == "api.news_funnel_cards"
    ]
    assert not forbidden, "Composition root imports inbound compatibility modules: " + ", ".join(forbidden)


def test_main_uses_canonical_routers_only():
    """The production app must not assemble itself through route facades."""
    path = PROJECT_ROOT / "api" / "main.py"
    imports = _get_imports(path)
    forbidden = [
        imported
        for _, imported in imports
        if imported.startswith("api.routes_")
    ]
    assert not forbidden, "api/main.py imports compatibility route facades: " + ", ".join(forbidden)


def test_canonical_router_packages_do_not_import_compatibility_modules():
    """Canonical router package initializers must stay free of legacy seams."""
    package_files = (
        PROJECT_ROOT / "api" / "routers" / "portfolio" / "__init__.py",
        PROJECT_ROOT / "api" / "routers" / "equity" / "__init__.py",
    )
    violations = []
    for path in package_files:
        rel = path.relative_to(PROJECT_ROOT).as_posix()
        for lineno, imported in _get_imports(path):
            if imported.startswith("api.compatibility") or imported.startswith("api.routes_"):
                violations.append(f"{rel}:{lineno} imports '{imported}'")
    assert not violations, "Canonical router packages import compatibility modules:\n" + "\n".join(violations)


def test_state_db_does_not_register_content_outbox_at_import_time():
    """The compatibility facade must remain inert until the composition root wires ports."""
    path = PROJECT_ROOT / "api" / "state_db.py"
    imports = _get_imports(path)
    forbidden = [
        imported
        for _, imported in imports
        if imported == "tools.content.parking_lot_outbox"
        or imported.startswith("tools.content.parking_lot_outbox.")
    ]
    assert not forbidden, "state_db.py performs import-time outbox wiring: " + ", ".join(forbidden)


def test_legacy_state_adapter_has_explicit_operations():
    """Compatibility bridges must not become untyped service locators."""
    source = (PROJECT_ROOT / "api" / "db" / "legacy_adapter.py").read_text(encoding="utf-8-sig")
    assert "def __getattr__" not in source


def test_lower_layers_do_not_import_api():
    """Tools, Agents, and Application must not import API layer unless tracked in allowlist."""
    check_dirs = [
        PROJECT_ROOT / "tools",
        PROJECT_ROOT / "agents",
        PROJECT_ROOT / "application",
    ]

    violations = []
    for base_dir in check_dirs:
        for root, _, files in os.walk(base_dir):
            for file in files:
                if file.endswith(".py"):
                    file_path = Path(root) / file
                    rel_path = file_path.relative_to(PROJECT_ROOT).as_posix()
                    for lineno, imported in _get_imports(file_path):
                        if imported == "api" or imported.startswith("api."):
                            if not is_allowed(rel_path, imported, "lower_layer_imports_api"):
                                violations.append(
                                    f"{rel_path}:{lineno} imports '{imported}' (lower layer -> api violation)"
                                )

    assert not violations, "Lower layer importing API layer:\n" + "\n".join(violations)


def test_dao_no_commit_or_rollback():
    """Raw DAO files in api/db/repositories must not call .commit() or .rollback() directly."""
    repo_dir = PROJECT_ROOT / "api" / "db" / "repositories"
    assert repo_dir.is_dir(), "Missing repositories directory"

    violations = []
    for file_path in repo_dir.glob("*.py"):
        rel_path = file_path.relative_to(PROJECT_ROOT).as_posix()
        with open(file_path, "r", encoding="utf-8-sig") as f:
            content = f.read()
        tree = ast.parse(content, filename=str(file_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute) and node.func.attr in ("commit", "rollback"):
                    violations.append(f"{rel_path}:{node.lineno} calls forbidden method '{node.func.attr}()' on connection")

    assert not violations, "Raw DAOs calling transaction methods directly:\n" + "\n".join(violations)


def test_raw_daos_do_not_construct_connections():
    """Raw SQL repositories must receive a connection from their caller."""
    repo_dir = PROJECT_ROOT / "api" / "db" / "repositories"
    violations = []
    for file_path in repo_dir.glob("*.py"):
        rel_path = file_path.relative_to(PROJECT_ROOT).as_posix()
        for lineno, imported in _get_imports(file_path):
            if imported == "api.db.connection" or imported.startswith("api.db.connection."):
                violations.append(f"{rel_path}:{lineno} imports connection factory '{imported}'")

    assert not violations, "Raw DAOs importing the connection factory:\n" + "\n".join(violations)


def test_db_uow_exposes_typed_workflow_boundary_without_dynamic_dao_proxy():
    """The UoW must expose a concrete workflow adapter, not a dynamic DAO service locator."""
    path = PROJECT_ROOT / "api" / "db" / "uow.py"
    source = path.read_text(encoding="utf-8-sig")
    assert "class _ConnectionBoundDao" not in source
    assert "def __getattr__" not in source
    assert "def earnings_call_workflow" in source
    assert "SqliteEarningsCallWorkflowAdapter" in source


def test_insider_sync_adapter_receives_provider_from_composition_root():
    """SQLite infrastructure must not instantiate the yfinance provider itself."""
    adapter_path = PROJECT_ROOT / "api" / "db" / "adapters.py"
    imports = _get_imports(adapter_path)
    forbidden = [
        imported
        for _, imported in imports
        if imported == "tools.market.adapters.insider_provider"
        or imported.startswith("tools.market.adapters.insider_provider.")
    ]
    assert not forbidden, "SQLite adapter constructs a concrete insider provider: " + ", ".join(forbidden)

    source = adapter_path.read_text(encoding="utf-8-sig")
    assert "YFinanceInsiderHistoryAdapter" not in source


def test_no_unallowed_sys_modules_in_production():
    """Production code must not use sys.modules as a dynamic service locator unless tracked in allowlist."""
    check_dirs = [
        PROJECT_ROOT / "tools",
        PROJECT_ROOT / "agents",
        PROJECT_ROOT / "application",
        PROJECT_ROOT / "core",
        PROJECT_ROOT / "api",
    ]

    violations = []
    for base_dir in check_dirs:
        for root, _, files in os.walk(base_dir):
            for file in files:
                if file.endswith(".py"):
                    file_path = Path(root) / file
                    rel_path = file_path.relative_to(PROJECT_ROOT).as_posix()
                    lookups = _find_sys_modules_lookups(file_path)
                    for lineno, target_mod in lookups:
                        if not is_allowed(rel_path, target_mod, "sys_modules_lookup"):
                            violations.append(
                                f"{rel_path}:{lineno} uses sys.modules lookup for '{target_mod}'"
                            )

    assert not violations, "Unallowed sys.modules lookups in production code:\n" + "\n".join(violations)


def test_allowlist_entries_are_valid():
    """All allowlist entries must point to existing files with owners, removal phases, and conditions."""
    for entry in ARCHITECTURE_ALLOWLIST:
        file_path = PROJECT_ROOT / entry.source_file
        assert file_path.is_file(), f"Allowlist entry file does not exist: {entry.source_file}"
        assert entry.owner, f"Allowlist entry missing owner: {entry.source_file}"
        assert entry.reason, f"Allowlist entry missing reason: {entry.source_file}"
        assert entry.removal_phase.startswith("Phase "), f"Invalid removal phase: {entry.removal_phase}"
        assert entry.removal_condition, f"Allowlist entry missing removal condition: {entry.source_file}"
