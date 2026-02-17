"""
TripleTriple System Prompt Builder

Assembles the system prompt from workspace identity files,
following the TripleTriple layering order:

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

logger = logging.getLogger("tripletriple.agents.system_prompt")

# Path to bundled default templates (shipped with the package)
_DEFAULTS_DIR = Path(__file__).resolve().parent.parent / "workspace"


# ─── Configuration ──────────────────────────────────────────────

class WorkspaceConfig(BaseModel):
    """Configuration for the workspace directory."""

    workspace_path: str = Field(
        default="~/.tripletriple/workspace",
        description="Path to the workspace directory containing identity files",
    )
    agent_name: str = Field(
        default="TripleTriple",
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
    "active-tasks.md",
    "lessons.md",
    "projects.md",
    "self-review.md",
]

# Files that ship as defaults and can be initialized
TEMPLATE_FILES = [
    "AGENTS.md",
    "SOUL.md",
    "IDENTITY.md",
    "USER.md",
    "MEMORY.md",
    "BOOTSTRAP.md",
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

        # 0. Bootstrap Mode Check
        # If we need bootstrap, IGNORE everything else and return the bootstrap prompt.
        if self.needs_bootstrap():
            return self.get_bootstrap_prompt()

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

        # 6. Structured Memory (Split Files)
        if is_main_session:
            # 6a. active-tasks.md (Critical Context — Read First)
            active_tasks = self._read_file("active-tasks.md")
            if active_tasks:
                sections.append(f"# ACTIVE TASKS (CRITICAL)\n{active_tasks}")

            # 6b. lessons.md (Mistakes & Learnings)
            lessons = self._read_file("lessons.md")
            if lessons:
                sections.append(f"# LESSONS & LEARNINGS\n{lessons}")

            # 6c. projects.md (High-Level State)
            projects = self._read_file("projects.md")
            if projects:
                sections.append(f"# PROJECTS STATE\n{projects}")

            # 6d. self-review.md (Periodic Critiques)
            self_review = self._read_file("self-review.md")
            if self_review:
                sections.append(f"# SELF REVIEW LOG\n{self_review}")

            # 6e. MEMORY.md (Legacy/General - Deprecated but kept for transition)
            memory = self._read_file("MEMORY.md")
            if memory:
                sections.append(f"# GENERAL MEMORY\n{memory}")

        # 7. Daily notes — today + yesterday
        daily = self._read_daily_notes()
        if daily:
            sections.append(daily)

        # 8. Available skills (XML block)
        if self.skills_prompt:
            sections.append(self.skills_prompt)

        return "\n\n---\n\n".join(sections)

    def needs_bootstrap(self) -> bool:
        """
        Check if we are in bootstrap mode.
        Condition: IDENTITY.md is missing AND BOOTSTRAP.md is present.
        """
        identity_path = self.config.root / "IDENTITY.md"
        bootstrap_path = self.config.root / "BOOTSTRAP.md"
        
        # If identity exists, we are done.
        if identity_path.exists():
            return False
            
        # If identity is missing, do we have the bootstrap instructions?
        return bootstrap_path.exists()

    def get_bootstrap_prompt(self) -> str:
        """
        Return the bootstrap system prompt for first-run onboarding.
        Reads from BOOTSTRAP.md in the workspace.
        """
        bootstrap_path = self.config.root / "BOOTSTRAP.md"
        if bootstrap_path.exists():
            return bootstrap_path.read_text(encoding="utf-8").strip()
            
        # Fallback if file missing (shouldn't happen if needs_bootstrap checked it)
        return (
            "# BOOTSTRAP MODE\n"
            "I am in bootstrap mode but can't find BOOTSTRAP.md. "
            "Please ask the user to restore the file or create IDENTITY.md manually."
        )

    # ── Workspace Init ────────────────────────────────────────

    @staticmethod
    def init_workspace(
        workspace_path: str = "~/.tripletriple/workspace",
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
            "",
            "## Critical Tool Rules",
            "1. **Silent Execution**: Do NOT announce that you are going to use a tool. Just use it. Avoid phrases like 'I will now check...' or 'Let me see...'.",
            "2. **Direct Answers**: After using a tool, provide the answer directly. Do not narrate your actions (e.g. 'I have read the file. It says...'). Just say what it says.",
            "3. **No Stuttering**: Do not repeat the same information. If you found the answer, give it.",
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
