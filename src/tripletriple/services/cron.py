import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
from pydantic import BaseModel

from .agent_runner import run_session_turn
from ..session import SessionManager, InboundContext, ChatType

logger = logging.getLogger("tripletriple.cron")

class CronDelivery(BaseModel):
    mode: str = "announce"  # "announce" or "none"
    channel: str = "last"   # "telegram", "last", etc.
    to: Optional[str] = None # channel ID, chat ID, phone, etc.

class CronJob(BaseModel):
    id: str
    schedule: str
    command: str
    last_run: float = 0.0
    enabled: bool = True
    delivery: CronDelivery

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

    def add_job(
        self, 
        schedule: str, 
        command: str, 
        delivery_channel: str = "last",
        delivery_to: Optional[str] = None,
        delivery_mode: str = "announce"
    ) -> str:
        """Register a new job via tool."""
        import uuid
        job_id = f"cron-{str(uuid.uuid4())[:8]}"
        
        # Construct delivery object
        delivery = CronDelivery(
            mode=delivery_mode,
            channel=delivery_channel,
            to=delivery_to
        )
        
        job = CronJob(
            id=job_id, 
            schedule=schedule, 
            command=command,
            delivery=delivery
        )
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
        current_hm = f"{now.tm_hour}:{now.tm_min:02d}"
        
        # KEY FIX: Iterate over a copy of values to avoid "dictionary changed size" error
        current_jobs = list(self._jobs.values())
        
        for job in current_jobs:
            if not job.enabled:
                continue
            
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
            return min_match and hour_match
            
        return False

    async def _execute_job(self, job: CronJob):
        logger.info(f"⏳ Executing cron job {job.id}: {job.command}")
        job.last_run = time.time()
        self._save_jobs()

        if job.delivery.mode == "none":
            pass

        # Resolve target
        target_channel = job.delivery.channel
        target_recipient = job.delivery.to
        
        ctx = InboundContext(
            channel=target_channel if target_channel != "last" else "cron",
            sender_id=target_recipient or "system", 
            display_name=f"Cron: {job.command[:20]}...",
            is_dm=True
        )
        
        session = self.session_manager.get_or_create(ctx)
        session.entry.chat_type = ChatType.CRON
        
        # Inject context about where to reply
        prompt = (
            f"⏰ **CRON EXECUTION**\n"
            f"Command: `{job.command}`\n"
            f"Context: Scheduled task. Delivery: {job.delivery.mode} to {target_channel}:{target_recipient}.\n\n"
            f"**INSTRUCTION**: Execute task.\n"
            f"- If implied message, **output it**.\n"
            f"- Response routed to: {target_channel} (user: {target_recipient})\n"
            f"- Speaking IS sending."
        )
        
        await run_session_turn(self.agent, session, prompt, self.session_manager)

    def _load_jobs(self):
        if self.crontab_path.exists():
            try:
                data = json.loads(self.crontab_path.read_text())
                migrated_jobs = {}
                for j in data:
                    # Migration: Handle legacy jobs without 'delivery'
                    if "delivery" not in j:
                        # Map old flat fields if present, else default
                        old_channel = j.pop("target_channel", None)
                        old_recipient = j.pop("target_recipient", None)
                        
                        j["delivery"] = {
                            "mode": "announce",
                            "channel": old_channel if old_channel else "last",
                            "to": old_recipient
                        }
                    
                    try:
                        job = CronJob(**j)
                        migrated_jobs[job.id] = job
                    except Exception as e:
                        logger.error(f"Skipping invalid job {j.get('id')}: {e}")
                
                self._jobs = migrated_jobs
                
                # Save immediately to persist migration
                if self._jobs:
                    self._save_jobs()
                    
            except Exception as e:
                logger.warning(f"Failed to load crontab: {e}")

    def _save_jobs(self):
        try:
            data = [j.model_dump() for j in self._jobs.values()]
            self.crontab_path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.error(f"Failed to save crontab: {e}")
