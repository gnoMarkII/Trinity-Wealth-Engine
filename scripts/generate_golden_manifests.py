#!/usr/bin/env python3
"""Phase 0: Generate Immutable Golden Baseline Manifests for Backend Refactoring.

Extracts:
1. PortfolioService public method signatures & typing
2. FastAPI OpenAPI routes, schemas, and parameters
3. LangChain Agent Tools schemas and definitions
4. api.state_db exported functions and signatures

Saves JSON snapshots to tests/fixtures/manifest_*.json
"""
import inspect
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("WEBUI_PASSWORD", "test-password-for-manifest")
os.environ.setdefault("SESSION_SECRET_KEY", "test-session-secret-key-32-chars-long!!")
os.environ.setdefault("UNVERIFIED_DRAFT_SIGNING_KEY", "test-draft-signing-key-32-chars-long!!")

FIXTURES_DIR = PROJECT_ROOT / "tests" / "fixtures"
FIXTURES_DIR.mkdir(parents=True, exist_ok=True)


def format_annotation(ann: Any) -> str:
    if ann is inspect.Parameter.empty:
        return "empty"
    if hasattr(ann, "__name__"):
        return ann.__name__
    return str(ann)


def format_default(default: Any) -> Any:
    if default is inspect.Parameter.empty:
        return "<EMPTY>"
    if default is None:
        return None
    if isinstance(default, (int, float, bool, str, list, dict)):
        return default
    return str(default)


def generate_portfolio_service_manifest() -> Dict[str, Any]:
    from tools.portfolio.service import PortfolioService

    manifest: Dict[str, Any] = {
        "class": "PortfolioService",
        "module": "tools.portfolio.service",
        "methods": {},
    }

    # Inspect all public methods and __init__
    for name, method in inspect.getmembers(PortfolioService, predicate=inspect.isfunction):
        if name.startswith("_") and name != "__init__" and not name.startswith("_execute") and not name.startswith("_edit") and not name.startswith("_manage") and not name.startswith("_record"):
            continue
        sig = inspect.signature(method)
        params_info = []
        for p_name, param in sig.parameters.items():
            params_info.append({
                "name": p_name,
                "kind": str(param.kind),
                "default": format_default(param.default),
                "annotation": format_annotation(param.annotation),
            })
        
        manifest["methods"][name] = {
            "name": name,
            "docstring": inspect.getdoc(method),
            "parameters": params_info,
            "return_annotation": format_annotation(sig.return_annotation),
        }

    return manifest


def generate_openapi_schema_manifest() -> Dict[str, Any]:
    from api.main import app

    openapi = app.openapi()
    # Normalize paths and route operations
    manifest: Dict[str, Any] = {
        "openapi_version": openapi.get("openapi"),
        "title": openapi.get("info", {}).get("title"),
        "paths": {},
    }

    for path, methods in sorted(openapi.get("paths", {}).items()):
        manifest["paths"][path] = {}
        for method, op_info in sorted(methods.items()):
            manifest["paths"][path][method] = {
                "operationId": op_info.get("operationId"),
                "tags": op_info.get("tags", []),
                "summary": op_info.get("summary"),
                "parameters": [
                    {
                        "name": p.get("name"),
                        "in": p.get("in"),
                        "required": p.get("required"),
                        "schema": p.get("schema"),
                    }
                    for p in op_info.get("parameters", [])
                ],
                "has_request_body": "requestBody" in op_info,
                "responses": sorted(list(op_info.get("responses", {}).keys())),
            }

    return manifest


def generate_langchain_tools_manifest() -> Dict[str, Any]:
    import tools.portfolio.agent_tools as agent_tools

    tools_manifest: Dict[str, Any] = {
        "module": "tools.portfolio.agent_tools",
        "tools": {},
    }

    for name in dir(agent_tools):
        obj = getattr(agent_tools, name)
        # Check if it's a langchain BaseTool
        if hasattr(obj, "name") and hasattr(obj, "description") and hasattr(obj, "args_schema"):
            schema_json = obj.args_schema.model_json_schema() if obj.args_schema else {}
            tools_manifest["tools"][obj.name] = {
                "name": obj.name,
                "description": obj.description,
                "func_name": name,
                "args_schema": schema_json,
                "return_direct": getattr(obj, "return_direct", False),
            }

    return tools_manifest


def generate_state_db_exports_manifest() -> Dict[str, Any]:
    import api.state_db as state_db

    manifest: Dict[str, Any] = {
        "module": "api.state_db",
        "functions": {},
        "constants": [],
    }

    for name in dir(state_db):
        if name.startswith("__"):
            continue
        obj = getattr(state_db, name)
        if inspect.isfunction(obj):
            if obj.__module__ == "api.state_db" or (hasattr(state_db, "__all__") and name in state_db.__all__):
                sig = inspect.signature(obj)
                manifest["functions"][name] = {
                    "name": name,
                    "parameters": [
                        {
                            "name": p_name,
                            "kind": str(param.kind),
                            "default": format_default(param.default),
                            "annotation": format_annotation(param.annotation),
                        }
                        for p_name, param in sig.parameters.items()
                    ],
                    "return_annotation": format_annotation(sig.return_annotation),
                    "docstring": inspect.getdoc(obj),
                }
        elif isinstance(obj, str) and (name.startswith("_SCHEMA") or name.isupper()):
            manifest["constants"].append(name)

    return manifest


def main():
    print("Generating Phase 0 Golden Baseline Manifests...")

    # 1. PortfolioService Manifest
    ps_manifest = generate_portfolio_service_manifest()
    ps_file = FIXTURES_DIR / "manifest_portfolio_service.json"
    ps_file.write_text(json.dumps(ps_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  [OK] PortfolioService Manifest: {len(ps_manifest['methods'])} methods -> {ps_file}")

    # 2. OpenAPI Schema Manifest
    openapi_manifest = generate_openapi_schema_manifest()
    openapi_file = FIXTURES_DIR / "manifest_openapi_schema.json"
    openapi_file.write_text(json.dumps(openapi_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  [OK] OpenAPI Schema Manifest: {len(openapi_manifest['paths'])} paths -> {openapi_file}")

    # 3. LangChain Tools Manifest
    tools_manifest = generate_langchain_tools_manifest()
    tools_file = FIXTURES_DIR / "manifest_langchain_tools.json"
    tools_file.write_text(json.dumps(tools_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  [OK] LangChain Tools Manifest: {len(tools_manifest['tools'])} tools -> {tools_file}")

    # 4. State DB Exports Manifest
    state_db_manifest = generate_state_db_exports_manifest()
    state_db_file = FIXTURES_DIR / "manifest_state_db_exports.json"
    state_db_file.write_text(json.dumps(state_db_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  [OK] State DB Exports Manifest: {len(state_db_manifest['functions'])} functions -> {state_db_file}")

    print("\nAll baseline manifests successfully generated!")


if __name__ == "__main__":
    main()
