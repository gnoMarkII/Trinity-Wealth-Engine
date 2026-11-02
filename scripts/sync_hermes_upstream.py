"""Utility script to clone or update NousResearch/hermes-agent in services/hermes_agent."""
import subprocess
import sys
from pathlib import Path

HERMES_REPO_URL = "https://github.com/NousResearch/hermes-agent.git"
TARGET_DIR = Path("services/hermes_agent")


def sync_hermes() -> None:
    TARGET_DIR.parent.mkdir(parents=True, exist_ok=True)

    if not (TARGET_DIR / ".git").is_dir():
        print(f"Cloning {HERMES_REPO_URL} into {TARGET_DIR}...")
        res = subprocess.run(
            ["git", "clone", "--depth", "1", HERMES_REPO_URL, str(TARGET_DIR)],
            check=False,
        )
        if res.returncode != 0:
            print("Failed to clone hermes-agent.")
            sys.exit(res.returncode)
        print("Clone complete.")
    else:
        print(f"Updating {TARGET_DIR} (git pull --ff-only)...")
        res = subprocess.run(
            ["git", "-C", str(TARGET_DIR), "pull", "--ff-only"],
            check=False,
        )
        if res.returncode != 0:
            print("Failed to pull latest changes from upstream.")
            sys.exit(res.returncode)

    # Print current commit
    rev = subprocess.run(
        ["git", "-C", str(TARGET_DIR), "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    commit = rev.stdout.strip()
    print(f"Hermes Agent is at upstream commit: {commit}")


if __name__ == "__main__":
    sync_hermes()

