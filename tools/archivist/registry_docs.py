"""Render human-readable documentation from the machine-readable Vault registry."""
from __future__ import annotations

from tools.archivist.schema_registry import SchemaRegistry, load_default_registry


def render_registry_documentation(registry: SchemaRegistry | None = None) -> str:
    registry = registry or load_default_registry()
    lines = [
        "# Vault profile and policy registry",
        "",
        f"Registry digest: `{registry.digest()}`  ",
        f"Policy digest: `{registry.policy_digest()}`",
        "",
        "| Profile | Entity types | Source of truth | Manual edit | Search default | Retention |",
        "|---|---|---|---|---|---|",
    ]
    for profile_id in sorted(registry.profiles):
        profile = registry.profiles[profile_id]
        entities = ", ".join(profile.entity_types) or "(none)"
        lines.append(
            f"| `{profile_id}` | {entities} | `{profile.source_of_truth}` | "
            f"`{profile.manual_edit}` | `{profile.search_scope_default}` | "
            f"`{profile.retention_class_default}` |"
        )
    lines.extend(
        [
            "",
            "Aliases are read-compatible only; new canonical writes use the entity type listed in `entity_profile`.",
            "",
        ]
    )
    return "\n".join(lines)
