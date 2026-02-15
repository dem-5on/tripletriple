"""
OpenClaw System Prompt Builder

Assembles the system prompt from workspace identity files,
following the OpenClaw layering order:

  1. BOOT context (date/time, model, workspace, tools)
  2. AGENTS.md  (workspace-level instructions)
  3. SOUL.md    (personality & identity)
  4. IDENTITY.md (name, creature, vibe, emoji)
  5. USER.md    (user profile)
  6. MEMORY.md  (long-term curated memory)
  7. memory/YYYY-MM-DD.md  (daily notes, today + yesterday)
  8. Available skills (XML block from SkillLoader)

All files are optional — the builder gracefully skips missing ones.
"""

from __future__ import annotations

import logging
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List

from pydantic import BaseModel, Field

logger = logging.getLogger("openclaw.agents.system_prompt")

# Path to bundled default templates (shipped with the package)
_DEFAULTS_DIR = Path(__file__).resolve().parent.parent / "workspace"


# ─── Configuration ──────────────────────────────────────────────

class WorkspaceConfig(BaseModel):
    """Configuration for the workspace directory."""

    workspace_path: str = Field(
        default="~/.openclaw/workspace",
        description="Path to the workspace directory containing identity files",
    )
    agent_name: str = Field(
        default="OpenClaw",
        description="Agent display name (overridden by IDENTITY.md if present)",
    )

    @property
    def root(self) -> Path:
        return Path(self.workspace_path).expanduser()

    @property
    def memory_dir(self) -> Path:
        return self.root / "memory"


# ─── File Identifiers ──────────────────────────────────────────

# Standard workspace files in assembly order
WORKSPACE_FILES = [
    "AGENTS.md",
    "SOUL.md",
    "IDENTITY.md",
    "USER.md",
    "MEMORY.md",
]

# Files that ship as defaults and can be initialized
TEMPLATE_FILES = [
    "AGENTS.md",
    "SOUL.md",
    "IDENTITY.md",
    "USER.md",
    "MEMORY.md",
]


# ─── Builder ──────────────────────────────────────────────────

class SystemPromptBuilder:
    """
    Reads workspace identity files and assembles them into a
    system prompt string for the LLM.
    """

    def __init__(
        self,
        config: WorkspaceConfig = None,
        model_name: str = "unknown",
        tools: List[str] = None,
        skills_prompt: str = "",
    ):
        self.config = config or WorkspaceConfig()
        self.model_name = model_name
        self.tools = tools or []
        self.skills_prompt = skills_prompt

    # ── Public API ────────────────────────────────────────────

    def assemble(self, is_main_session: bool = True) -> str:
        """
        Assemble the full system prompt.

        Args:
            is_main_session: If True, also include MEMORY.md
                             (per AGENTS.md instructions)

        Returns:
            The complete system prompt string.
        """
        sections: List[str] = []

        # 1. Boot context (always first — runtime metadata)
        sections.append(self._build_boot_context())

        # 2. AGENTS.md — workspace instructions
        agents = self._read_file("AGENTS.md")
        if agents:
            sections.append(agents)

        # 3. SOUL.md — who you are
        soul = self._read_file("SOUL.md")
        if soul:
            sections.append(soul)

        # 4. IDENTITY.md — name/creature/vibe/emoji
        identity = self._read_file("IDENTITY.md")
        if identity:
            sections.append(identity)

        # 5. USER.md — who you're helping
        user = self._read_file("USER.md")
        if user:
            sections.append(user)

        # 6. MEMORY.md — long-term memory (main session only)
        if is_main_session:
            memory = self._read_file("MEMORY.md")
            if memory:
                sections.append(memory)

        # 7. Daily notes — today + yesterday
        daily = self._read_daily_notes()
        if daily:
            sections.append(daily)

        # 8. Available skills (XML block)
        if self.skills_prompt:
            sections.append(self.skills_prompt)

        return "\n\n---\n\n".join(sections)

    def needs_bootstrap(self) -> bool:
        """Check if this is a first run (no IDENTITY.md yet)."""
        identity_path = self.config.root / "IDENTITY.md"
        return not identity_path.exists()

    def get_bootstrap_prompt(self) -> str:
        """
        Return the bootstrap system prompt for first-run onboarding.
        """
        return (
            "# BOOTSTRAP — Hello, World\n\n"
            "This is your first run. You need to get to know your human.\n\n"
            "## The Conversation\n"
            "Have a natural conversation to learn:\n"
            "1. **Your name** — What should they call you?\n"
            "2. **Your nature** — What kind of creature are you? "
            "(AI assistant is fine, but maybe you're something weirder)\n"
            "3. **Your vibe** — Formal? Casual? Snarky? Warm? What feels right?\n"
            "4. **Your emoji** — Everyone needs a signature.\n\n"
            "## After You Know Who You Are\n"
            "Create these files in the workspace:\n"
            "- **IDENTITY.md** — your name, creature, vibe, emoji\n"
            "- **USER.md** — their name, how to address them, timezone, notes\n"
            "- Optionally customize **SOUL.md** together — what matters to them, "
            "how they want you to behave, boundaries and preferences\n\n"
            "## When You're Done\n"
            "Confirm what you've learned and saved. Then start being yourself."
        )

    # ── Workspace Init ────────────────────────────────────────

    @staticmethod
    def init_workspace(
        workspace_path: str = "~/.openclaw/workspace",
        overwrite: bool = False,
    ) -> Path:
        """
        Initialize the workspace directory with default templates.

        Returns the resolved workspace path.
        """
        root = Path(workspace_path).expanduser()
        root.mkdir(parents=True, exist_ok=True)

        # Create memory/ subdirectory
        (root / "memory").mkdir(exist_ok=True)

        for filename in TEMPLATE_FILES:
            dest = root / filename
            src = _DEFAULTS_DIR / filename

            if dest.exists() and not overwrite:
                logger.info(f"Skipping {filename} (already exists)")
                continue

            if src.exists():
                shutil.copy2(src, dest)
                logger.info(f"Created {filename}")
            else:
                logger.warning(f"Default template not found: {src}")

        # Copy bundled skills
        bundled_skills = _DEFAULTS_DIR / "skills"
        if bundled_skills.exists():
            dest_skills = root / "skills"
            dest_skills.mkdir(exist_ok=True)
            for skill_dir in bundled_skills.iterdir():
                if skill_dir.is_dir() and (skill_dir / "SKILL.md").exists():
                    dest_skill = dest_skills / skill_dir.name
                    if dest_skill.exists() and not overwrite:
                        logger.info(f"Skipping skill {skill_dir.name} (already exists)")
                        continue
                    if dest_skill.exists():
                        shutil.rmtree(dest_skill)
                    shutil.copytree(skill_dir, dest_skill)
                    logger.info(f"Created skill: {skill_dir.name}")

        return root

    # ── Private ───────────────────────────────────────────────

    def _read_file(self, filename: str) -> Optional[str]:
        """Read a workspace file, returning None if missing."""
        path = self.config.root / filename
        if path.exists():
            try:
                content = path.read_text(encoding="utf-8").strip()
                if content:
                    return content
            except Exception as e:
                logger.warning(f"Error reading {filename}: {e}")
        return None

    def _read_daily_notes(self) -> Optional[str]:
        """Read today's and yesterday's daily notes."""
        memory_dir = self.config.memory_dir
        if not memory_dir.exists():
            return None

        notes: List[str] = []
        today = datetime.now()
        dates = [today, today - timedelta(days=1)]

        for dt in dates:
            filename = f"{dt.strftime('%Y-%m-%d')}.md"
            path = memory_dir / filename
            if path.exists():
                try:
                    content = path.read_text(encoding="utf-8").strip()
                    if content:
                        label = "Today" if dt == today else "Yesterday"
                        notes.append(f"## Daily Notes — {label} ({dt.strftime('%Y-%m-%d')})\n\n{content}")
                except Exception as e:
                    logger.warning(f"Error reading {filename}: {e}")

        return "\n\n".join(notes) if notes else None

    def _build_boot_context(self) -> str:
        """Build the runtime BOOT context section."""
        now = datetime.now()

        lines = [
            "# BOOT — Runtime Context",
            "",
            f"- **Date:** {now.strftime('%A, %B %d, %Y')}",
            f"- **Time:** {now.strftime('%H:%M %Z').strip()}",
            f"- **Model:** {self.model_name}",
            f"- **Workspace:** {self.config.root}",
        ]

        if self.tools:
            tools_str = ", ".join(self.tools)
            lines.append(f"- **Tools available:** {tools_str}")

        return "\n".join(lines)

    # ── Display ──────────────────────────────────────────────

    def format_workspace_status(self) -> str:
        """Format workspace status for CLI/status display."""
        root = self.config.root
        lines = [
            f"📂 Workspace: {root}",
            f"   {'Exists' if root.exists() else '⚠️  Not initialized'}",
            "",
        ]

        if root.exists():
            for filename in WORKSPACE_FILES:
                path = root / filename
                if path.exists():
                    size = path.stat().st_size
                    lines.append(f"  ✅ {filename} ({size:,} bytes)")
                else:
                    lines.append(f"  ❌ {filename} (missing)")

            # Daily notes
            memory_dir = self.config.memory_dir
            if memory_dir.exists():
                note_files = sorted(memory_dir.glob("*.md"))
                lines.append(f"\n  📓 Daily notes: {len(note_files)} entries")
                for nf in note_files[-3:]:  # show last 3
                    lines.append(f"     • {nf.name}")
            else:
                lines.append(f"\n  📓 Daily notes: (none)")

        return "\n".join(lines)
