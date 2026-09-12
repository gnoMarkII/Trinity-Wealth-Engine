"""Declarative Vault profile, ownership, retention, and indexing policies.

The registry is deliberately independent from the Markdown parser and from
filesystem writers.  It answers policy questions; callers decide whether a
policy failure should reject, quarantine, or route a note to capture scope.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Optional


class SchemaRegistryError(ValueError):
    """Raised when a registry is malformed or internally inconsistent."""


@dataclass(frozen=True)
class RegistryProfile:
    profile_id: str
    profile_version: int
    entity_types: tuple[str, ...]
    required: tuple[str, ...]
    forbidden: tuple[str, ...]
    required_values: Mapping[str, Any]
    source_of_truth: str
    manual_edit: str
    search_scope_default: str
    index_policy: str
    retention_class_default: str
    path_policy: str
    identity_policy: str

    @classmethod
    def from_dict(cls, profile_id: str, raw: Mapping[str, Any]) -> "RegistryProfile":
        required_values = raw.get("required_values") or {}
        if not isinstance(required_values, Mapping):
            raise SchemaRegistryError(f"profile {profile_id!r} required_values must be an object")
        version = raw.get("profile_version", 1)
        if not isinstance(version, int) or version < 1:
            raise SchemaRegistryError(f"profile {profile_id!r} has invalid profile_version")
        return cls(
            profile_id=profile_id,
            profile_version=version,
            entity_types=tuple(str(item) for item in raw.get("entity_types") or ()),
            required=tuple(str(item) for item in raw.get("required") or ()),
            forbidden=tuple(str(item) for item in raw.get("forbidden") or ()),
            required_values=dict(required_values),
            source_of_truth=str(raw.get("source_of_truth") or ""),
            manual_edit=str(raw.get("manual_edit") or ""),
            search_scope_default=str(raw.get("search_scope_default") or "excluded"),
            index_policy=str(raw.get("index_policy") or "never"),
            retention_class_default=str(raw.get("retention_class_default") or "ephemeral"),
            path_policy=str(raw.get("path_policy") or ""),
            identity_policy=str(raw.get("identity_policy") or ""),
        )


class SchemaRegistry:
    """Immutable-in-use view of schemas/vault/registry.json and its policies."""

    def __init__(
        self,
        payload: Mapping[str, Any],
        *,
        source_path: Optional[Path] = None,
        indexing_policy: Optional[Mapping[str, Any]] = None,
        ownership_policy: Optional[Mapping[str, Any]] = None,
        retention_policy: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.source_path = source_path
        self._payload = json.loads(json.dumps(payload, ensure_ascii=False))
        self._validate_shape()
        self.registry_id = str(self._payload["registry_id"])
        self.registry_version = int(self._payload["registry_version"])
        self.note_schema_version = int(self._payload["note_schema_version"])
        self.global_policy = dict(self._payload["global"])
        self._profiles = {
            profile_id: RegistryProfile.from_dict(profile_id, raw)
            for profile_id, raw in self._payload["profiles"].items()
        }
        self._aliases = {
            str(key).strip().lower(): str(value).strip().lower()
            for key, value in (self._payload.get("entity_aliases") or {}).items()
        }
        self._entity_profiles = {
            str(key).strip().lower(): str(value).strip()
            for key, value in (self._payload.get("entity_profile") or {}).items()
        }
        self.indexing_policy = dict(indexing_policy or {})
        self.ownership_policy = dict(ownership_policy or {})
        self.retention_policy = dict(retention_policy or {})
        self._validate_consistency()

    @classmethod
    def load(cls, path: str | Path | None = None) -> "SchemaRegistry":
        registry_path = Path(path) if path is not None else default_registry_path()
        registry_path = registry_path.resolve()
        try:
            payload = json.loads(registry_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SchemaRegistryError(f"Cannot load schema registry {registry_path}: {exc}") from exc
        if not isinstance(payload, Mapping):
            raise SchemaRegistryError("Schema registry root must be an object")
        policy_root = registry_path.parent / "policies"
        return cls(
            payload,
            source_path=registry_path,
            indexing_policy=_load_optional_json(policy_root / "indexing.json"),
            ownership_policy=_load_optional_json(policy_root / "field-ownership.json"),
            retention_policy=_load_optional_json(policy_root / "retention.json"),
        )

    def _validate_shape(self) -> None:
        required = ("registry_id", "registry_version", "note_schema_version", "global", "profiles")
        missing = [field for field in required if field not in self._payload]
        if missing:
            raise SchemaRegistryError(f"Registry missing required fields: {', '.join(missing)}")
        if not isinstance(self._payload["profiles"], Mapping):
            raise SchemaRegistryError("Registry profiles must be an object")
        if not isinstance(self._payload["global"], Mapping):
            raise SchemaRegistryError("Registry global policy must be an object")

    def _validate_consistency(self) -> None:
        if self.registry_version < 1 or self.note_schema_version < 1:
            raise SchemaRegistryError("Registry versions must be positive")
        profile_types: dict[str, str] = {}
        for profile_id, profile in self._profiles.items():
            if not profile.source_of_truth or not profile.path_policy or not profile.identity_policy:
                raise SchemaRegistryError(f"Profile {profile_id!r} is missing ownership/path/identity policy")
            if set(profile.required) & set(profile.forbidden):
                raise SchemaRegistryError(f"Profile {profile_id!r} has required and forbidden overlap")
            for entity_type in profile.entity_types:
                previous = profile_types.get(entity_type)
                if previous and previous != profile_id:
                    raise SchemaRegistryError(
                        f"Entity type {entity_type!r} is assigned to both {previous!r} and {profile_id!r}"
                    )
                profile_types[entity_type] = profile_id
        for entity_type, profile_id in self._entity_profiles.items():
            if profile_id not in self._profiles:
                raise SchemaRegistryError(
                    f"Entity type {entity_type!r} references unknown profile {profile_id!r}"
                )
            profile = self._profiles[profile_id]
            if profile.entity_types and entity_type not in profile.entity_types:
                raise SchemaRegistryError(
                    f"Entity type {entity_type!r} is not declared by profile {profile_id!r}"
                )
        for alias, target in self._aliases.items():
            if target not in self._entity_profiles and target not in profile_types:
                raise SchemaRegistryError(f"Alias {alias!r} points to unknown entity type {target!r}")

    @property
    def profiles(self) -> Mapping[str, RegistryProfile]:
        return dict(self._profiles)

    @property
    def entity_profiles(self) -> Mapping[str, str]:
        return dict(self._entity_profiles)

    def canonical_entity_type(self, entity_type: Any) -> str:
        current = str(entity_type or "").strip().lower()
        if not current:
            return ""
        seen: set[str] = set()
        while current in self._aliases:
            if current in seen:
                raise SchemaRegistryError(f"Entity alias cycle includes {current!r}")
            seen.add(current)
            current = self._aliases[current]
        return current

    def profile_for(
        self,
        entity_type: Any,
        *,
        profile_id: Optional[str] = None,
    ) -> Optional[RegistryProfile]:
        if profile_id is not None:
            return self._profiles.get(str(profile_id))
        canonical = self.canonical_entity_type(entity_type)
        selected = self._entity_profiles.get(canonical)
        return self._profiles.get(selected) if selected else None

    def validate_metadata(
        self,
        metadata: Mapping[str, Any],
        *,
        profile_id: Optional[str] = None,
        require_schema_version: bool = True,
        allow_identity_allocation: bool = False,
    ) -> tuple[bool, list[dict[str, str]]]:
        """Validate declarative profile rules without mutating metadata."""
        issues: list[dict[str, str]] = []
        raw_type = metadata.get("entity_type")
        canonical_type = self.canonical_entity_type(raw_type)
        profile = self.profile_for(canonical_type, profile_id=profile_id)
        if profile is None:
            issues.append({
                "code": "unknown_profile",
                "field": "entity_type",
                "reason": f"No registered profile for entity_type {raw_type!r}.",
            })
            return False, issues

        if require_schema_version and metadata.get("schema_version") != self.note_schema_version:
            issues.append({
                "code": "schema_version",
                "field": "schema_version",
                "reason": f"Registry requires schema_version {self.note_schema_version}.",
            })
        for field in profile.required:
            value = metadata.get(field)
            if allow_identity_allocation and field in {"note_id", "document_key"}:
                continue
            if value is None or (isinstance(value, str) and not value.strip()):
                issues.append({
                    "code": "missing_required",
                    "field": field,
                    "reason": f"Profile {profile.profile_id} requires {field}.",
                })
        for field, expected in profile.required_values.items():
            if metadata.get(field) != expected:
                issues.append({
                    "code": "required_value",
                    "field": field,
                    "reason": f"Profile {profile.profile_id} requires {field}={expected!r}.",
                })
        for field in profile.forbidden:
            value = metadata.get(field)
            if value not in (None, ""):
                issues.append({
                    "code": "forbidden_field",
                    "field": field,
                    "reason": f"Profile {profile.profile_id} forbids non-empty {field}.",
                })
        if canonical_type != str(raw_type or "").strip().lower() and raw_type:
            issues.append({
                "code": "legacy_alias",
                "field": "entity_type",
                "reason": f"Use canonical entity_type {canonical_type!r}; alias is read-compatible only.",
            })
        return not issues, issues

    def field_owner(self, field: str) -> Optional[str]:
        classes = self.ownership_policy.get("classes") or {}
        for owner, fields in classes.items():
            if field in fields:
                return str(owner)
        return None

    def retention_class_for(self, profile_id: str) -> str:
        defaults = self.retention_policy.get("default_by_profile") or {}
        profile = self._profiles.get(profile_id)
        return str(defaults.get(profile_id) or (profile.retention_class_default if profile else "ephemeral"))

    def is_index_eligible(
        self,
        metadata: Mapping[str, Any],
        *,
        profile_id: Optional[str] = None,
        vector: bool = True,
    ) -> bool:
        profile = self.profile_for(metadata.get("entity_type"), profile_id=profile_id)
        if profile is None:
            return False
        overrides = self.indexing_policy.get("profile_overrides") or {}
        override = overrides.get(profile.profile_id) or {}
        policy_key = "vector" if vector else "catalog"
        if policy_key in override:
            return bool(override[policy_key])
        if str(metadata.get("search_scope") or profile.search_scope_default).lower() != "included":
            return False if vector else True
        if not vector:
            return True
        if metadata.get("indexing_allowed") is False:
            return False
        lifecycle = str(metadata.get("content_status") or "published").lower()
        excluded_lifecycle = set(self.indexing_policy.get("lifecycle_excluded_for_vector") or ())
        if lifecycle in excluded_lifecycle:
            return False
        verification = str(
            metadata.get("verification_status")
            or metadata.get("content_verification_status")
            or ""
        ).lower()
        if verification in set(self.indexing_policy.get("verification_excluded_for_vector") or ()):
            return False
        sensitivity = str(metadata.get("sensitivity") or "internal").lower()
        if sensitivity in set(self.indexing_policy.get("sensitivity_excluded_for_vector") or ()):
            return False
        return True

    def digest(self) -> str:
        canonical = json.dumps(self._payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def policy_digest(self) -> str:
        payload = {
            "registry": self._payload,
            "indexing": self.indexing_policy,
            "ownership": self.ownership_policy,
            "retention": self.retention_policy,
        }
        canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _load_optional_json(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SchemaRegistryError(f"Cannot load registry policy {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise SchemaRegistryError(f"Registry policy {path} must be an object")
    return value


def default_registry_path() -> Path:
    return Path(__file__).resolve().parents[2] / "schemas" / "vault" / "registry.json"


@lru_cache(maxsize=1)
def load_default_registry() -> SchemaRegistry:
    return SchemaRegistry.load(default_registry_path())
