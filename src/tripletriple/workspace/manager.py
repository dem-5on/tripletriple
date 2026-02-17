"""
Workspace Manager — Runtime workspace file operations.

Handles reading, writing, and managing workspace files at runtime.
Complements SystemPromptBuilder.init_workspace() which handles initial setup.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict

logger = logging.getLogger("tripletriple.workspace.manager")


class WorkspaceManager:
    """
    Runtime workspace manager.

    Provides read/write access to workspace identity files,
    daily memory logs, and workspace state inspection.
    """

    def __init__(self, workspace_root: str = "~/.tripletriple/workspace"):
        self.root = Path(workspace_root).expanduser()
        self.memory_dir = self.root / "memory"

    # ── File Operations ──────────────────────────────────────

    def read_file(self, name: str) -> Optional[str]:
        """Read a workspace file by name. Returns None if missing."""
        path = self.root / name
        if path.exists():
            try:
                return path.read_text(encoding="utf-8").strip()
            except Exception as e:
                logger.warning(f"Error reading {name}: {e}")
        return None

    def write_file(self, name: str, content: str) -> bool:
        """Write content to a workspace file. Creates parent dirs if needed."""
        try:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            logger.info(f"Wrote workspace file: {name}")
            return True
        except Exception as e:
            logger.error(f"Failed to write {name}: {e}")
            return False

    def delete_file(self, name: str) -> bool:
        """Delete a workspace file."""
        path = self.root / name
        if path.exists():
            try:
                path.unlink()
                logger.info(f"Deleted workspace file: {name}")
                return True
            except Exception as e:
                logger.error(f"Failed to delete {name}: {e}")
        return False

    def file_exists(self, name: str) -> bool:
        """Check if a workspace file exists."""
        return (self.root / name).exists()

    # ── Daily Memory Logs ────────────────────────────────────

    def ensure_daily_log(self) -> Path:
        """Create today's daily log file if it doesn't exist. Returns the path."""
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        today = datetime.now().strftime("%Y-%m-%d")
        path = self.memory_dir / f"{today}.md"
        if not path.exists():
            path.write_text(
                f"## {today}\n\n",
                encoding="utf-8",
            )
            logger.info(f"Created daily log: {today}.md")
        return path

    def append_daily_log(self, content: str) -> bool:
        """Append content to today's daily memory log."""
        try:
            path = self.ensure_daily_log()
            timestamp = datetime.now().strftime("%H:%M")
            line = f"- [{timestamp}] {content}\n"
            with open(path, "a", encoding="utf-8") as f:
                f.write(line)
            return True
        except Exception as e:
            logger.error(f"Failed to append daily log: {e}")
            return False

    def read_daily_log(self, date_str: str = None) -> Optional[str]:
        """Read a daily log by date string (YYYY-MM-DD). Defaults to today."""
        if date_str is None:
            date_str = datetime.now().strftime("%Y-%m-%d")
        return self.read_file(f"memory/{date_str}.md")

    # ── Session Reset Memory Flush ───────────────────────────

    def flush_session_summary(
        self,
        session_key: str,
        message_count: int,
        topics: List[str],
    ) -> bool:
        """
        Write a session summary to the daily log before reset.
        Called by SessionManager.reset_session().
        """
        summary = (
            f"Session `{session_key}` reset "
            f"({message_count} messages). "
            f"Topics: {', '.join(topics)}"
        )
        return self.append_daily_log(summary)

    # ── Workspace Status ─────────────────────────────────────

    def list_files(self) -> List[str]:
        """List all files in the workspace root (non-recursive)."""
        if not self.root.exists():
            return []
        return [f.name for f in self.root.iterdir() if f.is_file()]

    def get_status(self) -> Dict[str, any]:
        """Get workspace status as a dictionary."""
        identity_files = [
            "AGENTS.md", "SOUL.md", "IDENTITY.md",
            "USER.md", "MEMORY.md", "HEARTBEAT.md",
        ]
        status = {
            "root": str(self.root),
            "exists": self.root.exists(),
            "files": {},
            "daily_logs": 0,
        }

        if self.root.exists():
            for fname in identity_files:
                path = self.root / fname
                status["files"][fname] = {
                    "exists": path.exists(),
                    "size": path.stat().st_size if path.exists() else 0,
                }

            if self.memory_dir.exists():
                status["daily_logs"] = len(list(self.memory_dir.glob("*.md")))

        return status

    def format_status(self) -> str:
        """Format workspace status as a human-readable string."""
        s = self.get_status()
        lines = [
            f"📂 Workspace: {s['root']}",
            f"   {'Exists' if s['exists'] else '⚠️  Not initialized'}",
        ]

        if s["exists"]:
            for fname, info in s["files"].items():
                icon = "✅" if info["exists"] else "❌"
                size = f" ({info['size']:,} bytes)" if info["exists"] else " (missing)"
                lines.append(f"  {icon} {fname}{size}")

            lines.append(f"\n  📓 Daily notes: {s['daily_logs']} entries")

        return "\n".join(lines)
