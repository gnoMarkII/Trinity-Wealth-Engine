"""Strict, read-only resolver for immutable Obsidian Vault V2 artifacts.

The current Markdown projection is a cache. Historical reads resolve a
registered committed revision and verify the exact bytes named by its manifest.
This module deliberately has no repair or cleanup side effects.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from tools.archivist.metadata import parse_note
from tools.archivist.vault_paths import VaultPaths


class ArtifactError(Exception):
    """Base class for trusted-artifact resolution failures."""


class ArtifactNotFoundError(ArtifactError):
    """The requested committed revision or required file does not exist."""


class ArtifactIntegrityError(ArtifactError):
    """The artifact set is incomplete, unregistered, or internally inconsistent."""


class ManifestDigestMismatchError(ArtifactIntegrityError):
    """A manifest digest does not match the exact bytes on disk."""


@dataclass
class StoredArtifact:
    note_id: str
    revision_id: str
    revision: int
    metadata: dict[str, Any]
    body: str
    primary_file: Path
    companions: dict[str, bytes] = field(default_factory=dict)
    manifest: dict[str, Any] = field(default_factory=dict)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class DurableArtifactStore:
    """Resolves one registered revision and verifies its complete artifact set."""

    SUPPORTED_MANIFEST_VERSIONS = frozenset({1})

    def __init__(self, vault_paths: Optional[VaultPaths] = None) -> None:
        self._vault_paths = vault_paths or VaultPaths()

    @property
    def vault_paths(self) -> VaultPaths:
        return self._vault_paths

    def _revision_dir(self, note_id: str, revision_id: str) -> Path:
        # VaultPaths performs containment and Windows device-name checks. Do not
        # concatenate untrusted IDs and resolve them later.
        return self._vault_paths.revision_path(note_id, revision_id, "manifest.json").parent

    @staticmethod
    def _safe_member(rev_dir: Path, name: str) -> Path:
        if not isinstance(name, str) or not name or Path(name).name != name:
            raise ArtifactIntegrityError(f"Manifest member path is not a safe filename: {name!r}")
        path = (rev_dir / name).resolve()
        if not path.is_relative_to(rev_dir.resolve()):
            raise ArtifactIntegrityError(f"Manifest member escapes revision directory: {name!r}")
        return path

    def _load_manifest(self, rev_dir: Path, note_id: str, revision_id: str) -> dict[str, Any]:
        manifest_file = rev_dir / "manifest.json"
        if not manifest_file.is_file():
            raise ArtifactNotFoundError(
                f"Missing required manifest.json for revision '{revision_id}' of note '{note_id}'"
            )
        try:
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ArtifactIntegrityError(f"Corrupt manifest at {manifest_file}: {exc}") from exc
        if not isinstance(manifest, dict):
            raise ArtifactIntegrityError(f"Manifest at {manifest_file} must be an object")
        version = manifest.get("manifest_version", 1)
        if version not in self.SUPPORTED_MANIFEST_VERSIONS:
            raise ArtifactIntegrityError(f"Unsupported artifact manifest_version={version!r}")
        if manifest.get("note_id") != note_id or manifest.get("revision_id") != revision_id:
            raise ArtifactIntegrityError(
                "Manifest identity does not match requested note_id/revision_id"
            )
        return manifest

    def _registration_contains(self, note_id: str, revision_id: str, manifest: dict[str, Any]) -> bool:
        """Check durable registration when the writer has created the registry.

        Legacy manifests without a registry remain readable only as explicitly
        marked compatibility evidence. New writer output always has a registry.
        """
        registry = self._vault_paths.root / ".system" / "artifacts" / "registry.json"
        if not registry.exists():
            return bool(manifest.get("legacy_read_only", False))
        try:
            data = json.loads(registry.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ArtifactIntegrityError(f"Corrupt artifact registry: {exc}") from exc
        refs = data.get("revisions", {}) if isinstance(data, dict) else {}
        key = f"{note_id}:{revision_id}"
        entry = refs.get(key)
        expected = _sha256(
            json.dumps(
                manifest,
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        return isinstance(entry, dict) and entry.get("manifest_digest") == expected

    def get_revision_artifact(
        self,
        note_id: str,
        revision_id: str,
        verify_digests: bool = True,
    ) -> StoredArtifact:
        """Return a complete immutable revision or raise a typed integrity error.

        ``verify_digests=False`` is retained for explicitly requested diagnostics;
        production callers should leave it enabled. Missing required members are
        always errors regardless of that flag.
        """
        rev_dir = self._revision_dir(note_id, revision_id)
        if not rev_dir.is_dir():
            raise ArtifactNotFoundError(
                f"Revision {revision_id} for note {note_id} not found at {rev_dir}"
            )
        manifest = self._load_manifest(rev_dir, note_id, revision_id)

        primary_name = manifest.get("primary_file")
        if not primary_name:
            raise ArtifactIntegrityError("Manifest is missing designated primary_file")
        primary_file = self._safe_member(rev_dir, primary_name)
        if primary_file.suffix.lower() != ".md" or not primary_file.is_file():
            raise ArtifactNotFoundError(f"Designated primary artifact is missing: {primary_name}")
        primary_bytes = primary_file.read_bytes()

        expected_primary = manifest.get("primary_sha256")
        if verify_digests:
            if not isinstance(expected_primary, str):
                raise ArtifactIntegrityError("Manifest is missing exact primary_sha256")
            actual_primary = _sha256(primary_bytes)
            if actual_primary != expected_primary:
                raise ManifestDigestMismatchError(
                    f"Primary file SHA-256 mismatch for {primary_name}: "
                    f"expected {expected_primary}, got {actual_primary}"
                )

        declared = manifest.get("companions", {})
        if declared is None:
            declared = {}
        if not isinstance(declared, dict):
            raise ArtifactIntegrityError("Manifest companions must be an object")
        required = manifest.get("required_companions")
        if required is None:
            required = list(declared.keys())
        if not isinstance(required, list) or any(not isinstance(n, str) for n in required):
            raise ArtifactIntegrityError("Manifest required_companions must be a list of filenames")
        if set(required) != set(declared):
            raise ArtifactIntegrityError(
                "Manifest required_companions and companions do not describe the same artifact set"
            )

        companions: dict[str, bytes] = {}
        for name in required:
            path = self._safe_member(rev_dir, name)
            if not path.is_file():
                raise ArtifactNotFoundError(
                    f"Required companion '{name}' missing from revision '{revision_id}'"
                )
            data = path.read_bytes()
            if verify_digests:
                expected = declared.get(name)
                actual = _sha256(data)
                if not isinstance(expected, str) or actual != expected:
                    raise ManifestDigestMismatchError(
                        f"Companion {name} SHA-256 mismatch: expected {expected}, got {actual}"
                    )
            companions[name] = data

        allowed = {primary_name, "manifest.json", *required}
        extras = sorted(
            item.name for item in rev_dir.iterdir() if item.is_file() and item.name not in allowed
        )
        if extras:
            raise ArtifactIntegrityError(f"Unregistered files in revision artifact set: {extras}")

        # A complete directory is not a commit.  New manifests must carry an
        # explicit committed marker; compatibility evidence is permitted only
        # when it is explicitly labelled legacy_read_only and has no durable
        # registry (handled below).
        if "committed" not in manifest:
            if not manifest.get("legacy_read_only", False):
                raise ArtifactIntegrityError("Manifest is missing required committed marker")
        elif manifest.get("committed") is not True:
            raise ArtifactIntegrityError("Revision is not registered as committed")

        if not self._registration_contains(note_id, revision_id, manifest):
            raise ArtifactIntegrityError(
                f"Revision '{revision_id}' is not registered as a committed artifact"
            )

        try:
            metadata, body, issues = parse_note(primary_bytes.decode("utf-8"))
        except UnicodeDecodeError as exc:
            raise ArtifactIntegrityError(f"Primary artifact is not valid UTF-8: {exc}") from exc
        if issues:
            raise ArtifactIntegrityError(f"Primary artifact metadata is malformed: {issues}")
        revision = manifest.get("revision", metadata.get("revision", 1))
        try:
            revision = int(revision)
        except (TypeError, ValueError) as exc:
            raise ArtifactIntegrityError("Revision number is not an integer") from exc

        return StoredArtifact(
            note_id=note_id,
            revision_id=revision_id,
            revision=revision,
            metadata=metadata,
            body=body,
            primary_file=primary_file,
            companions=companions,
            manifest=manifest,
        )

    def list_revisions(self, note_id: str) -> list[str]:
        note_dir = self._revision_dir(note_id, "_list").parent
        if not note_dir.is_dir():
            return []
        return sorted(item.name for item in note_dir.iterdir() if item.is_dir())
