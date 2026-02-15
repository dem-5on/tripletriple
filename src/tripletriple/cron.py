import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from pydantic import BaseModel

from .agent_runner import run_session_turn
from .gateway.session import SessionManager, InboundContext, ChatType

logger = logging.getLogger("tripletriple.cron")

class CronJob(BaseModel):
    id: str
    schedule: str  # Simplified cron expression or "SS MM HH * * *"
    command: str
    last_run: float = 0.0
    enabled: bool = True

class CronManager:
    def __init__(self, workspace_root: Path, session_manager: SessionManager):
        self.workspace_root = workspace_root
        self.session_manager = session_manager
        self.crontab_path = workspace_root / "crontab.json"
        
        self.running = False
        self.task = None
        self.agent = None
        self._jobs: Dict[str, CronJob] = {}

    def set_agent(self, agent):
        self.agent = agent

    async def start(self):
        self._load_jobs()
        if self.running:
            return
        self.running = True
        self.task = asyncio.create_task(self._loop())
        logger.info(f"⏳ Cron manager started ({len(self._jobs)} jobs loaded)")

    async def stop(self):
        self.running = False
        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass

    def add_job(self, schedule: str, command: str) -> str:
        """Register a new job via tool."""
        import uuid
        job_id = f"cron-{str(uuid.uuid4())[:8]}"
        job = CronJob(id=job_id, schedule=schedule, command=command)
        self._jobs[job_id] = job
        self._save_jobs()
        return job_id

    def delete_job(self, job_id: str) -> bool:
        """Delete a job by ID."""
        if job_id in self._jobs:
            del self._jobs[job_id]
            self._save_jobs()
            return True
        return False

    def list_jobs(self) -> List[Dict[str, Any]]:
        """Return list of all jobs."""
        return [j.model_dump() for j in self._jobs.values()]

    async def _loop(self):
        while self.running:
            # Check every minute
            await asyncio.sleep(60) 
            try:
                await self._check_jobs()
            except Exception as e:
                logger.error(f"Cron loop error: {e}")

    async def _check_jobs(self):
        if not self.agent:
            return

        now = time.localtime()
        # Simple cron check logic (HH:MM matching for now to keep it robust without extra deps)
        # Format: "HH:MM" for daily jobs or we can parse "M H * * *" if needed.
        # The user example: "6 AM content research", "8 AM tech news".
        # Let's support "HH:MM" format mainly, or standard 5-field cron if we implement parser.
        # For simplicity without 'croniter', let's match "H M" locally.
        
        current_hm = f"{now.tm_hour}:{now.tm_min:02d}"
        
        for job in self._jobs.values():
            if not job.enabled:
                continue
                
            # Basic matching: if schedule == "06:00" and current == "6:00"
            # Or if schedule is complex.
            # Let's assume the schedule string IS the time "HH:MM".
            # If the user provides valid Cron "0 6 * * *", we'd need a parser.
            # I'll implement a VERY basic parser match for "M H * * *" (standard cron).
            
            if self._is_due(job, now):
                # Don't run twice in same minute (last_run check)
                if (time.time() - job.last_run) > 60:
                    await self._execute_job(job)

    def _is_due(self, job: CronJob, now_struct) -> bool:
        # 1. Simple HH:MM check
        s = job.schedule.strip()
        current_hm = f"{now_struct.tm_hour}:{now_struct.tm_min:02d}"
        if s == current_hm or s == f"{now_struct.tm_hour:02d}:{now_struct.tm_min:02d}":
            return True
            
        # 2. Basic Cron Parser (Minute Hour * * *)
        parts = s.split()
        if len(parts) == 5:
            min_field, hour_field = parts[0], parts[1]
            # Check Match
            min_match = (min_field == "*") or (min_field == str(now_struct.tm_min))
            hour_match = (hour_field == "*") or (hour_field == str(now_struct.tm_hour))
            # Ignore DOM/Month/Day for now (assume daily)
            return min_match and hour_match
            
        return False

    async def _execute_job(self, job: CronJob):
        logger.info(f"⏳ Executing cron job {job.id}: {job.command}")
        job.last_run = time.time()
        self._save_jobs()

        # Create isolated session for this job
        # session key: cron:job_id:timestamp -> unique every run
        # User said "Each cron runs in its own isolated session. No context bleed."
        timestamp = int(time.time())
        session_key = f"cron:{job.id}:{timestamp}"
        
        ctx = InboundContext(
            channel="cron",
            sender_id="system", 
            display_name=f"Cron: {job.command[:20]}...",
            is_dm=True
        )
        
        session = self.session_manager.get_or_create(ctx)
        # Type CRON
        session.entry.chat_type = ChatType.CRON
        
        prompt = f"⏰ **CRON EXECUTION**\nCommand: `{job.command}`\n\nPlease execute this task now."
        
        await run_session_turn(self.agent, session, prompt, self.session_manager)

    def _load_jobs(self):
        if self.crontab_path.exists():
            try:
                data = json.loads(self.crontab_path.read_text())
                self._jobs = {j["id"]: CronJob(**j) for j in data}
            except Exception as e:
                logger.warning(f"Failed to load crontab: {e}")

    def _save_jobs(self):
        try:
            data = [j.model_dump() for j in self._jobs.values()]
            self.crontab_path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.error(f"Failed to save crontab: {e}")
