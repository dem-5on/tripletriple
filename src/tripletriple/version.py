"""
TripleTriple Version & Self-Update

Provides version tracking, update detection (via git), and
self-update execution (git pull + pip install).
"""

import asyncio
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger("tripletriple.version")

__version__ = "0.1.0"


def get_version() -> str:
    """Return the current version string."""
    return __version__


def get_project_root() -> Path:
    """Return the root of the tripletriple package (where pyproject.toml lives)."""
    # Walk up from this file to find pyproject.toml
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    # Fallback: assume standard layout
    return Path(__file__).resolve().parent.parent.parent


def _is_git_repo(path: Path) -> bool:
    """Check if the given path is inside a git repository."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=str(path),
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _git_run(args: list, cwd: Path, timeout: int = 30) -> subprocess.CompletedProcess:
    """Run a git command and return the result."""
    return subprocess.run(
        ["git"] + args,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def check_for_updates() -> dict:
    """
    Check if updates are available.

    Returns dict with:
      - available: bool
      - current_version: str
      - current_commit: str (short hash)
      - remote_commit: str (short hash)
      - behind_count: int (commits behind remote)
      - message: str (human-readable status)
    """
    root = get_project_root()

    result = {
        "available": False,
        "current_version": __version__,
        "current_commit": None,
        "remote_commit": None,
        "behind_count": 0,
        "message": "",
    }

    if not _is_git_repo(root):
        result["message"] = "Not a git repository. Cannot check for updates."
        return result

    # Fetch latest from remote (non-blocking)
    fetch = _git_run(["fetch", "--quiet"], cwd=root, timeout=15)
    if fetch.returncode != 0:
        result["message"] = f"Could not reach remote: {fetch.stderr.strip()}"
        return result

    # Get current commit
    local = _git_run(["rev-parse", "--short", "HEAD"], cwd=root)
    if local.returncode == 0:
        result["current_commit"] = local.stdout.strip()

    # Get current branch
    branch = _git_run(["rev-parse", "--abbrev-ref", "HEAD"], cwd=root)
    branch_name = branch.stdout.strip() if branch.returncode == 0 else "main"

    # Get remote commit
    remote = _git_run(["rev-parse", "--short", f"origin/{branch_name}"], cwd=root)
    if remote.returncode == 0:
        result["remote_commit"] = remote.stdout.strip()

    # Count commits behind
    behind = _git_run(
        ["rev-list", "--count", f"HEAD..origin/{branch_name}"],
        cwd=root,
    )
    if behind.returncode == 0:
        count = int(behind.stdout.strip())
        result["behind_count"] = count
        result["available"] = count > 0

    if result["available"]:
        # Get changelog (commit messages for new commits)
        log = _git_run(
            ["log", "--oneline", f"HEAD..origin/{branch_name}", "--no-decorate"],
            cwd=root,
        )
        changelog = log.stdout.strip() if log.returncode == 0 else ""
        result["message"] = (
            f"🔄 Update available! {count} new commit(s).\n"
            f"   Local:  {result['current_commit']}\n"
            f"   Remote: {result['remote_commit']}\n"
        )
        if changelog:
            result["changelog"] = changelog
    else:
        result["message"] = f"✅ Already up to date (v{__version__}, {result['current_commit'] or 'unknown'})"

    return result


def perform_update(force: bool = False) -> dict:
    """
    Perform the self-update.

    Steps:
    1. git pull (fast-forward)
    2. pip install -e . (reinstall with new code)
    3. Return status

    Returns dict with:
      - success: bool
      - message: str
      - needs_restart: bool
    """
    root = get_project_root()

    result = {
        "success": False,
        "message": "",
        "needs_restart": False,
        "steps": [],
    }

    if not _is_git_repo(root):
        result["message"] = "Not a git repository. Cannot update."
        return result

    # Check if updates available (unless forced)
    if not force:
        check = check_for_updates()
        if not check["available"]:
            result["success"] = True
            result["message"] = check["message"]
            return result

    # Step 1: git pull
    pull = _git_run(["pull", "--ff-only"], cwd=root, timeout=60)
    if pull.returncode != 0:
        # Try rebase if ff-only fails
        pull = _git_run(["pull", "--rebase"], cwd=root, timeout=60)

    if pull.returncode != 0:
        result["message"] = f"Git pull failed: {pull.stderr.strip()}"
        result["steps"].append({"step": "git pull", "status": "failed", "error": pull.stderr.strip()})
        return result

    result["steps"].append({"step": "git pull", "status": "ok", "output": pull.stdout.strip()})

    # Step 2: pip install -e .
    pip_install = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", ".", "--quiet"],
        cwd=str(root),
        capture_output=True,
        text=True,
        timeout=120,
    )

    if pip_install.returncode != 0:
        result["message"] = f"pip install failed: {pip_install.stderr.strip()}"
        result["steps"].append({"step": "pip install", "status": "failed", "error": pip_install.stderr.strip()})
        return result

    result["steps"].append({"step": "pip install", "status": "ok"})

    # Step 3: Success
    result["success"] = True
    result["needs_restart"] = True
    result["message"] = (
        f"✅ Updated successfully!\n"
        f"   {pull.stdout.strip()}\n"
        f"   Restart required to apply changes."
    )

    return result


def restart_process():
    """Restart the current process to apply updates."""
    logger.info("Restarting TripleTriple process...")
    os.execv(sys.executable, [sys.executable] + sys.argv)
