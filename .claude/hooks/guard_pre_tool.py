"""PreToolUse Guard Hook for Coding Agent.

Intercepts and blocks dangerous commands before execution:
- git push --force / git push -f
- git reset --hard
- Destructive deletions (rm -rf / or equivalents)
- Direct dumping/reading of .env secrets via shell
- Direct manual overwriting of golden baseline manifest fixtures via shell redirection

Exits with code 2 to block the tool call with an explanatory message on stderr.
Exits with code 0 to allow safe tool calls.
"""
import json
import re
import sys

DANGEROUS_PATTERNS = [
    (
        re.compile(r"\bgit\s+push\b.*(-f\b|--force\b)", re.IGNORECASE),
        "BLOCKED: 'git push --force' is forbidden. Force-pushing can destroy commit history.",
    ),
    (
        re.compile(r"\bgit\s+reset\b.*--hard\b", re.IGNORECASE),
        "BLOCKED: 'git reset --hard' is forbidden. Hard reset can discard uncommitted work.",
    ),
    (
        re.compile(r"\brm\s+-rf\s+[/~]", re.IGNORECASE),
        "BLOCKED: Destructive root/home folder deletion is forbidden.",
    ),
    (
        re.compile(r"\b(cat|type|head|tail|Get-Content)\s+[^\n|><]*\.env\b", re.IGNORECASE),
        "BLOCKED: Dumping or reading '.env' secrets directly via shell command is forbidden.",
    ),
    (
        re.compile(r"(>|>>)\s*.*tests[/\\]fixtures[/\\]manifest_\w+\.json", re.IGNORECASE),
        "BLOCKED: Direct shell redirection into golden baseline manifests is forbidden. Use generate_golden_manifests.py instead.",
    ),
]


def check_command(command: str) -> None:
    for pattern, reason in DANGEROUS_PATTERNS:
        if pattern.search(command):
            sys.stderr.write(f"[GUARD BLOCKED] {reason}\n")
            sys.exit(2)


def main() -> None:
    try:
        raw = sys.stdin.read()
        if not raw.strip():
            sys.exit(0)

        command = ""
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                # Check typical tool call structures
                tool_input = data.get("tool_input") or data.get("input") or data
                if isinstance(tool_input, dict):
                    command = str(tool_input.get("command") or tool_input.get("cmd") or "")
                elif isinstance(tool_input, str):
                    command = tool_input
        except Exception:
            command = raw

        if command:
            check_command(command)

    except SystemExit:
        raise
    except Exception as exc:
        # On hook runtime error, fail-safe or warn
        sys.stderr.write(f"[GUARD HOOK WARNING] Failed to parse input: {exc}\n")
        sys.exit(0)

    sys.exit(0)


if __name__ == "__main__":
    main()
