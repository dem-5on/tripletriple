"""
TripleTriple Agent Skills System

Skills are SKILL.md files with YAML frontmatter that teach the agent
how to use specific tools, CLIs, and workflows.

Loading order (higher precedence wins on name collision):
  1. Bundled skills   — shipped with the package
  2. Managed skills   — ~/.tripletriple/skills/
  3. Workspace skills — <workspace>/skills/

Each skill folder contains:
  - SKILL.md  (required)  — frontmatter + instructions
  - scripts/  (optional)  — helper scripts
  - examples/ (optional)  — reference implementations
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
from html import escape as html_escape
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger("tripletriple.agents.skills")

# Path to bundled default skills (shipped with the package)
_BUNDLED_SKILLS_DIR = Path(__file__).resolve().parent.parent / "workspace" / "skills"


# ─── Models ─────────────────────────────────────────────────────

class SkillRequirements(BaseModel):
    """Gating requirements for a skill."""

    bins: List[str] = Field(default_factory=list, description="Binaries that must exist on PATH")
    any_bins: List[str] = Field(default_factory=list, description="At least one must exist on PATH")
    env: List[str] = Field(default_factory=list, description="Env vars that must be set")
    config: List[str] = Field(default_factory=list, description="Config paths that must be truthy")


class SkillEntry(BaseModel):
    """A loaded and parsed skill."""

    name: str
    description: str = ""
    emoji: str = ""
    homepage: str = ""
    location: str = ""  # tier label: "bundled", "managed", or "workspace"
    base_dir: str = ""  # absolute path to the skill folder
    content: str = ""   # full SKILL.md body (after frontmatter)
    requires: SkillRequirements = Field(default_factory=SkillRequirements)
    primary_env: str = ""
    user_invocable: bool = True
    disable_model_invocation: bool = False
    enabled: bool = True

    # TripleTriple metadata bag
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SkillConfigEntry(BaseModel):
    """Per-skill config override from tripletriple.json."""

    enabled: Optional[bool] = None
    api_key: Optional[str] = None
    env: Dict[str, str] = Field(default_factory=dict)
    config: Dict[str, Any] = Field(default_factory=dict)


class SkillConfig(BaseModel):
    """Top-level skills configuration."""

    workspace_path: str = Field(
        default="~/.tripletriple/workspace",
        description="Workspace directory",
    )
    managed_path: str = Field(
        default="~/.tripletriple/skills",
        description="Managed (shared) skills directory",
    )
    extra_dirs: List[str] = Field(
        default_factory=list,
        description="Additional skill directories (lowest precedence)",
    )
    entries: Dict[str, SkillConfigEntry] = Field(
        default_factory=dict,
        description="Per-skill overrides (enabled, apiKey, env)",
    )

    @property
    def workspace_skills_dir(self) -> Path:
        return Path(self.workspace_path).expanduser() / "skills"

    @property
    def managed_skills_dir(self) -> Path:
        return Path(self.managed_path).expanduser()


# ─── SKILL.md Parser ────────────────────────────────────────────

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def _parse_frontmatter(raw: str) -> tuple[Dict[str, Any], str]:
    """
    Parse YAML-like frontmatter from a SKILL.md file.
    Returns (frontmatter_dict, body_content).

    We use a lightweight parser instead of requiring PyYAML:
    - Supports: name, description, homepage, user-invocable, disable-model-invocation
    - Parses metadata as inline JSON
    """
    match = _FRONTMATTER_RE.match(raw)
    if not match:
        return {}, raw

    fm_text = match.group(1)
    body = raw[match.end():]

    result: Dict[str, Any] = {}

    # Extract metadata JSON block first (can span multiple lines)
    meta_match = re.search(
        r"^metadata:\s*(.+?)(?=\n\w|\n---|\Z)",
        fm_text,
        re.DOTALL | re.MULTILINE,
    )
    if meta_match:
        json_str = meta_match.group(1).strip()
        try:
            result["metadata"] = json.loads(json_str)
        except json.JSONDecodeError:
            # Try to fix common issues: single quotes, trailing commas
            fixed = json_str.replace("'", '"')
            fixed = re.sub(r",\s*([}\]])", r"\1", fixed)
            try:
                result["metadata"] = json.loads(fixed)
            except json.JSONDecodeError:
                logger.warning(f"Could not parse metadata JSON: {json_str[:100]}")

    # Parse simple key: value lines
    for line in fm_text.split("\n"):
        line = line.strip()
        if not line or line.startswith("metadata:") or line.startswith("{"):
            continue

        colon_idx = line.find(":")
        if colon_idx == -1:
            continue

        key = line[:colon_idx].strip()
        value = line[colon_idx + 1:].strip()

        # Strip surrounding quotes
        if (value.startswith('"') and value.endswith('"')) or \
           (value.startswith("'") and value.endswith("'")):
            value = value[1:-1]

        if key in ("name", "description", "homepage"):
            result[key] = value
        elif key == "user-invocable":
            result["user_invocable"] = value.lower() == "true"
        elif key == "disable-model-invocation":
            result["disable_model_invocation"] = value.lower() == "true"
        elif key == "command-dispatch":
            result["command_dispatch"] = value
        elif key == "command-tool":
            result["command_tool"] = value

    return result, body


def _parse_skill_md(path: Path, location: str) -> Optional[SkillEntry]:
    """Parse a single SKILL.md file into a SkillEntry."""
    try:
        raw = path.read_text(encoding="utf-8")
    except Exception as e:
        logger.warning(f"Error reading {path}: {e}")
        return None

    fm, body = _parse_frontmatter(raw)
    if not fm.get("name"):
        # Use directory name as fallback
        fm["name"] = path.parent.name

    # Extract tripletriple metadata
    oc_meta = fm.get("metadata", {}).get("tripletriple", {})

    # Build requirements
    req_data = oc_meta.get("requires", {})
    requires = SkillRequirements(
        bins=req_data.get("bins", []),
        any_bins=req_data.get("anyBins", []),
        env=req_data.get("env", []),
        config=req_data.get("config", []),
    )

    # Substitute {baseDir} in body
    base_dir = str(path.parent)
    body = body.replace("{baseDir}", base_dir)

    return SkillEntry(
        name=fm.get("name", path.parent.name),
        description=fm.get("description", ""),
        emoji=oc_meta.get("emoji", ""),
        homepage=fm.get("homepage", oc_meta.get("homepage", "")),
        location=location,
        base_dir=base_dir,
        content=body.strip(),
        requires=requires,
        primary_env=oc_meta.get("primaryEnv", ""),
        user_invocable=fm.get("user_invocable", True),
        disable_model_invocation=fm.get("disable_model_invocation", False),
        metadata=oc_meta,
    )


# ─── Skill Loader ───────────────────────────────────────────────

class SkillLoader:
    """
    Discovers, loads, and gates skills from all tiers.
    """

    def __init__(self, config: SkillConfig = None):
        self.config = config or SkillConfig()
        self._skills: List[SkillEntry] = []
        self._loaded = False

    # ── Public API ────────────────────────────────────────────

    def load_all(self) -> List[SkillEntry]:
        """
        Load skills from all tiers, apply gating, return eligible skills.
        Skills with the same name from higher-precedence tiers override lower ones.
        """
        raw: Dict[str, SkillEntry] = {}

        # Tier 1: Extra dirs (lowest precedence)
        for extra_dir in self.config.extra_dirs:
            self._scan_dir(Path(extra_dir).expanduser(), "extra", raw)

        # Tier 2: Bundled skills
        if _BUNDLED_SKILLS_DIR.exists():
            self._scan_dir(_BUNDLED_SKILLS_DIR, "bundled", raw)

        # Tier 3: Managed skills (~/.tripletriple/skills/)
        managed_dir = self.config.managed_skills_dir
        if managed_dir.exists():
            self._scan_dir(managed_dir, "managed", raw)

        # Tier 4: Workspace skills (highest precedence)
        ws_dir = self.config.workspace_skills_dir
        if ws_dir.exists():
            self._scan_dir(ws_dir, "workspace", raw)

        # Apply config overrides
        for name, entry in self.config.entries.items():
            if name in raw:
                if entry.enabled is not None:
                    raw[name].enabled = entry.enabled

        # Filter: gating + enabled + model-invocation
        eligible = []
        for skill in raw.values():
            if not skill.enabled:
                logger.debug(f"Skill {skill.name}: disabled by config")
                continue

            if not self._check_requirements(skill.requires):
                logger.debug(f"Skill {skill.name}: requirements not met")
                continue

            eligible.append(skill)

        self._skills = sorted(eligible, key=lambda s: s.name)
        self._loaded = True
        logger.info(f"Loaded {len(self._skills)} skills from {len(raw)} discovered")
        return self._skills

    def get_skills(self) -> List[SkillEntry]:
        """Return loaded skills (loads if not yet loaded)."""
        if not self._loaded:
            self.load_all()
        return self._skills

    def get_skill(self, name: str) -> Optional[SkillEntry]:
        """Get a specific skill by name."""
        for s in self.get_skills():
            if s.name == name:
                return s
        return None

    def get_prompt_eligible(self) -> List[SkillEntry]:
        """Return skills eligible for model prompt injection."""
        return [s for s in self.get_skills() if not s.disable_model_invocation]

    def get_user_invocable(self) -> List[SkillEntry]:
        """Return skills exposed as user slash commands."""
        return [s for s in self.get_skills() if s.user_invocable]

    # ── Prompt Formatting ─────────────────────────────────────

    @staticmethod
    def format_for_prompt(skills: List[SkillEntry]) -> str:
        """
        Format skills as XML for system prompt injection,
        matching TripleTriple's `formatSkillsForPrompt` format.

        Output example:
        <available-skills>
        Below are the skills available to you. ...
        <skill>
          <name>weather</name>
          <description>Get current weather and forecasts</description>
          <location>bundled</location>
        </skill>
        ...
        </available-skills>
        """
        if not skills:
            return ""

        lines = [
            "<available-skills>",
            "Below are the skills available to you. To use a skill, read the skill "
            "file at the location shown to learn detailed instructions.",
            "",
        ]

        for skill in skills:
            name_esc = html_escape(skill.name)
            desc_esc = html_escape(skill.description)
            loc = html_escape(f"{skill.base_dir}/SKILL.md")

            lines.append("<skill>")
            lines.append(f"  <name>{name_esc}</name>")
            lines.append(f"  <description>{desc_esc}</description>")
            lines.append(f"  <location>{loc}</location>")
            lines.append("</skill>")
            lines.append("")

        lines.append("</available-skills>")
        return "\n".join(lines)

    # ── Display ───────────────────────────────────────────────

    def format_skills_list(self) -> str:
        """Format skills list for CLI/chat display."""
        skills = self.get_skills()
        if not skills:
            return "No skills loaded."

        lines = [f"🧩 Loaded Skills ({len(skills)}):", ""]

        for s in skills:
            emoji = s.emoji or "📦"
            status = "✅" if s.enabled else "❌"
            location_tag = f"[{s.location}]"
            lines.append(f"  {status} {emoji} {s.name} — {s.description}  {location_tag}")

        return "\n".join(lines)

    def format_skill_detail(self, name: str) -> str:
        """Format a single skill's details for display."""
        skill = self.get_skill(name)
        if not skill:
            return f"Skill '{name}' not found."

        emoji = skill.emoji or "📦"
        lines = [
            f"{emoji} {skill.name}",
            f"   Description: {skill.description}",
            f"   Location: {skill.location} ({skill.base_dir})",
        ]

        if skill.homepage:
            lines.append(f"   Homepage: {skill.homepage}")
        if skill.requires.bins:
            lines.append(f"   Requires bins: {', '.join(skill.requires.bins)}")
        if skill.requires.env:
            lines.append(f"   Requires env: {', '.join(skill.requires.env)}")
        if skill.primary_env:
            lines.append(f"   Primary env: {skill.primary_env}")

        lines.append(f"   User-invocable: {skill.user_invocable}")
        lines.append(f"   Model prompt: {not skill.disable_model_invocation}")

        if skill.content:
            lines.append("")
            lines.append("─── Content ───")
            lines.append(skill.content[:2000])
            if len(skill.content) > 2000:
                lines.append(f"... ({len(skill.content) - 2000} more chars)")

        return "\n".join(lines)

    # ── Env Injection ─────────────────────────────────────────

    def inject_env(self) -> Dict[str, str]:
        """
        Inject environment variables from skill config entries.
        Returns a dict of original values for restoration.
        """
        original: Dict[str, str] = {}

        for name, entry in self.config.entries.items():
            # apiKey → primaryEnv
            if entry.api_key:
                skill = self.get_skill(name)
                if skill and skill.primary_env:
                    key = skill.primary_env
                    if key not in os.environ:
                        original[key] = os.environ.get(key, "")
                        os.environ[key] = entry.api_key

            # env overrides
            for key, value in entry.env.items():
                if key not in os.environ:
                    original[key] = os.environ.get(key, "")
                    os.environ[key] = value

        return original

    @staticmethod
    def restore_env(original: Dict[str, str]) -> None:
        """Restore environment variables after a run."""
        for key, value in original.items():
            if value:
                os.environ[key] = value
            else:
                os.environ.pop(key, None)

    # ── Private ───────────────────────────────────────────────

    def _scan_dir(
        self,
        directory: Path,
        location: str,
        target: Dict[str, SkillEntry],
    ) -> None:
        """Scan a directory for skill folders containing SKILL.md."""
        if not directory.exists():
            return

        for child in sorted(directory.iterdir()):
            if not child.is_dir():
                continue

            skill_md = child / "SKILL.md"
            if not skill_md.exists():
                continue

            skill = _parse_skill_md(skill_md, location)
            if skill:
                target[skill.name] = skill  # higher-precedence overwrites

    @staticmethod
    def _check_requirements(req: SkillRequirements) -> bool:
        """Check if all requirements are satisfied."""
        # All bins must exist on PATH
        for binary in req.bins:
            if not shutil.which(binary):
                return False

        # At least one of any_bins must exist
        if req.any_bins:
            if not any(shutil.which(b) for b in req.any_bins):
                return False

        # All env vars must be set
        for var in req.env:
            if not os.environ.get(var):
                return False

        return True
