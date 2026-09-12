"""Capture repository-host branch protection evidence for the R9 release gate."""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REQUIRED_CHECK = "Vault architecture and write-boundary contracts"


def _get_json(url: str, token: str | None) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "vault-r9-branch-protection-audit",
            **({"Authorization": f"Bearer {token}"} if token else {}),
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
            return {"http_status": int(response.status), "payload": payload if isinstance(payload, dict) else {}}
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            payload = json.loads(body)
        except ValueError:
            payload = {"message": body[:500]}
        return {"http_status": int(exc.code), "payload": payload if isinstance(payload, dict) else {}}
    except (OSError, ValueError) as exc:
        return {"http_status": None, "payload": {}, "error": str(exc)}


def audit(repository: str, branch: str) -> dict[str, Any]:
    encoded_repository = urllib.parse.quote(repository, safe="/")
    encoded_branch = urllib.parse.quote(branch, safe="")
    base = f"https://api.github.com/repos/{encoded_repository}/branches/{encoded_branch}"
    token = os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")
    branch_result = _get_json(base, token)
    protection_result = _get_json(f"{base}/protection", token)
    branch_payload = branch_result.get("payload") or {}
    branch_protection = branch_payload.get("protection") or {}
    branch_checks = branch_protection.get("required_status_checks") or {}
    protection_payload = protection_result.get("payload") or {}
    contexts = list(branch_checks.get("contexts") or [])
    checks = [item.get("context") or item.get("name") for item in (branch_checks.get("checks") or []) if isinstance(item, dict)]
    required_checks = [str(item) for item in contexts + checks]
    required_check_present = REQUIRED_CHECK in required_checks
    protected = branch_payload.get("protected") is True and branch_protection.get("enabled") is not False
    passes = protected and required_check_present and branch_checks.get("enforcement_level") not in {"off", "disabled"}
    return {
        "schema": "vault-r9-github-branch-protection-evidence-v1",
        "status": "PASS" if passes else "FAIL",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "repository": repository,
        "branch": branch,
        "protected": bool(protected),
        "required_check": REQUIRED_CHECK,
        "required_check_present": required_check_present,
        "required_checks": required_checks,
        "enforcement_level": branch_checks.get("enforcement_level"),
        "branch_api": {"http_status": branch_result.get("http_status"), "error": branch_result.get("error")},
        "protection_api": {"http_status": protection_result.get("http_status"), "error": protection_result.get("error"), "message": protection_payload.get("message")},
        "limitations": [
            "A PASS requires the repository host to report a protected branch and the exact required check.",
            "The audit never writes repository settings and never records authentication material.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", default="gnoMarkII/Trinity-Wealth-Engine")
    parser.add_argument("--branch", default="main")
    parser.add_argument("--output", type=Path, default=Path("scratch/vault-r9/acceptance/branch-protection-evidence.json"))
    args = parser.parse_args()
    report = audit(args.repository, args.branch)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "protected": report["protected"], "required_check_present": report["required_check_present"], "output": str(args.output.resolve())}, ensure_ascii=True))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
