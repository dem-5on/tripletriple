"""
Heartbeat Manager
Triggers periodic health checks in active sessions.
Reference: "3. Set up a HEARTBEAT.md checklist"
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Set

from ..session import SessionManager, Session
from .agent_runner import run_session_turn

logger = logging.getLogger("tripletriple.heartbeat")

class HeartbeatManager:
    def __init__(
        self, 
        session_manager: SessionManager, 
        interval_seconds: int = 1800,
        workspace_root: Path = None,
    ):
        self.session_manager = session_manager
        self.interval = interval_seconds
        self.workspace_root = workspace_root
        self.running = False
        self.task = None
        self.agent = None

    def set_agent(self, agent):
        """Inject agent instance for running autonomic turns."""
        self.agent = agent

    async def start(self):
        """Start the heartbeat loop."""
        if self.running:
            return
        self.running = True
        self.task = asyncio.create_task(self._loop())
        logger.info(f"💓 Heartbeat started (interval: {self.interval}s)")

    async def stop(self):
        """Stop the heartbeat loop."""
        self.running = False
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass

    async def _loop(self):
        while self.running:
            await asyncio.sleep(self.interval)
            try:
                await self._pulse()
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")

    async def _pulse(self):
        """Send heartbeat to valid sessions and perform hygiene."""
        # 1. Hygiene checks (always run)
        await self._check_hygiene()

        # 2. Agent heartbeat (if agent is set)
        if not self.agent:
            return

        # Target sessions active in the last 2 hours (120 minutes)
        active_entries = self.session_manager.list_sessions(active_minutes=120)
        
        for entry in active_entries:
            session = self.session_manager.get_session_by_key(entry.session_key)
            if session:
                await self._trigger_heartbeat(session)

    async def _check_hygiene(self):
        """
        Enforce session limits and rotate logs.
        Ref: '7. Session hygiene — archive aggressively'
        """
        # A. Session Size Check
        for entry in self.session_manager.list_sessions(active_minutes=1440): # 24h
            if entry.tokens.total_tokens > 32000:
                 # Just log for now
                 pass

        # B. Daily Log Rotation (>7 days)
        if self.workspace_root:
            memory_dir = self.workspace_root / "memory"
            if memory_dir.exists():
                cutoff = time.time() - (7 * 86400)
                for f in memory_dir.glob("????-??-??.md"):
                    try:
                        if f.stat().st_mtime < cutoff:
                            logger.info(f"🗑️ Deleting old daily log: {f.name}")
                            f.unlink()
                    except Exception as e:
                        logger.error(f"Failed to rotate log {f}: {e}")

    async def _trigger_heartbeat(self, session: Session):
        """Inject the heartbeat prompt and run the agent."""
        logger.info(f"💓 Sending heartbeat to session {session.id}")
        
        # Read HEARTBEAT.md checklist from workspace
        checklist = ""
        if self.workspace_root:
            hb_path = self.workspace_root / "HEARTBEAT.md"
            if hb_path.exists():
                try:
                    checklist = hb_path.read_text(encoding="utf-8").strip()
                except Exception as e:
                    logger.warning(f"Failed to read HEARTBEAT.md: {e}")
        
        prompt = "⏰ **SYSTEM HEARTBEAT**\n"
        if checklist:
            prompt += f"\n{checklist}\n\n"
        else:
            prompt += (
                "Please perform health checks "
                "(active tasks, session size, self-review).\n"
            )
        prompt += "If everything is fine, reply with `HEARTBEAT_OK`."
        
        # Run the agent turn autonomously
        await run_session_turn(self.agent, session, prompt, self.session_manager)
