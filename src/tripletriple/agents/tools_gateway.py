"""
Gateway Tool — System-level actions for the agent.

Provides actions like:
  - update.check  → Check if a new version is available
  - update.run    → Pull updates and reinstall
  - update.restart → Restart the process to apply updates
  - system.version → Return current version info
"""

import asyncio
import json
import logging
from typing import Any, Optional

from pydantic import BaseModel, Field

from .tools import Tool
from ..version import (
    get_version,
    check_for_updates,
    perform_update,
    restart_process,
    get_project_root,
    _is_git_repo,
)

logger = logging.getLogger("tripletriple.agents.gateway_tool")


class GatewaySchema(BaseModel):
    action: str = Field(
        ...,
        description=(
            'Action to perform. Available actions: '
            '"update.check" (check for updates), '
            '"update.run" (download and install updates), '
            '"update.restart" (restart process after update), '
            '"system.version" (show current version info)'
        ),
    )
    force: bool = Field(
        False,
        description="Force the action even if not needed (e.g., force update even if up-to-date)",
    )


class GatewayTool(Tool):
    name = "gateway"
    description = (
        "System-level gateway actions. Use action='update.check' to check for updates, "
        "action='update.run' to download and install updates, "
        "action='update.restart' to restart the process after updating, "
        "or action='system.version' to get version info."
    )
    args_schema = GatewaySchema

    async def run(self, **kwargs) -> str:
        # Strip _context if present (we don't need it for this tool)
        kwargs.pop("_context", None)

        action = kwargs.get("action", "")
        force = kwargs.get("force", False)

        if action == "system.version":
            return self._version_info()

        elif action == "update.check":
            return await self._update_check()

        elif action == "update.run":
            return await self._update_run(force=force)

        elif action == "update.restart":
            return self._update_restart()

        else:
            return json.dumps({
                "error": f"Unknown action: {action}",
                "available_actions": [
                    "update.check",
                    "update.run",
                    "update.restart",
                    "system.version",
                ],
            })

    def _version_info(self) -> str:
        root = get_project_root()
        return json.dumps({
            "version": get_version(),
            "project_root": str(root),
            "is_git_repo": _is_git_repo(root),
        }, indent=2)

    async def _update_check(self) -> str:
        """Run update check in a thread to avoid blocking the event loop."""
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, check_for_updates)
        return json.dumps(result, indent=2)

    async def _update_run(self, force: bool = False) -> str:
        """Run the update in a thread to avoid blocking."""
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, perform_update, force)
        return json.dumps(result, indent=2)

    def _update_restart(self) -> str:
        """Schedule a restart. Returns immediately, process restarts after."""
        logger.info("Restart requested via gateway tool")
        
        # Create a marker file to indicate we are restarting after an update
        # We can put this in the workspace root or local dir
        root = get_project_root()
        marker = root / ".update_success"
        try:
            marker.touch()
        except Exception as e:
            logger.warning(f"Could not create update marker: {e}")

        # Schedule restart after a short delay to allow response to be sent
        asyncio.get_event_loop().call_later(2.0, restart_process)
        return json.dumps({
            "status": "restarting",
            "message": "♻️ Restarting TripleTriple in 2 seconds...",
        })
