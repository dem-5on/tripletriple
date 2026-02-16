import json
from typing import Any, Optional
from pydantic import BaseModel, Field
from .tools import Tool

class CronScheduleSchema(BaseModel):
    schedule: str = Field(..., description="Cron expression (e.g., '0 6 * * *' or '06:00')")
    command: str = Field(..., description="Command or task description to run")

class CronScheduleTool(Tool):
    name = "cron_schedule"
    description = "Schedule a recurring task. Saves to crontab.json."
    args_schema = CronScheduleSchema

    def __init__(self, manager: Any):
        self.manager = manager

    async def run(self, schedule: str, command: str) -> str:
        # Capture context from the session creating this job
        target_channel = None
        target_recipient = None
        
        # Tools initialized with tool_context get it injected
        if hasattr(self, "tool_context") and self.tool_context:
            session = self.tool_context.get("session")
            if session:
                target_channel = session.entry.channel
                # For DM, recipient is the user. For group, it's the group ID (often same path).
                # We use origin.from_id as the primary target for DMs.
                target_recipient = session.entry.origin.from_id

        job_id = self.manager.add_job(
            schedule, 
            command, 
            target_channel=target_channel, 
            target_recipient=target_recipient
        )
        dest = f" (target: {target_channel})" if target_channel else ""
        return f"✅ Scheduled task '{command}' with ID {job_id}{dest}"

class CronListSchema(BaseModel):
    pass

class CronListTool(Tool):
    name = "cron_list"
    description = "List all scheduled cron jobs."
    args_schema = CronListSchema

    def __init__(self, manager: Any):
        self.manager = manager

    async def run(self) -> str:
        jobs = self.manager.list_jobs()
        if not jobs:
            return "No cron jobs scheduled."
        
        lines = ["📅 **Scheduled Jobs**"]
        for j in jobs:
            lines.append(f"- `[{j['id']}]` **{j['schedule']}**: {j['command']}")
        return "\n".join(lines)

class CronDeleteSchema(BaseModel):
    job_id: str = Field(..., description="ID of the job to delete")

class CronDeleteTool(Tool):
    name = "cron_delete"
    description = "Delete a scheduled cron job by ID."
    args_schema = CronDeleteSchema

    def __init__(self, manager: Any):
        self.manager = manager

    async def run(self, job_id: str) -> str:
        if self.manager.delete_job(job_id):
            return f"🗑️ Deleted job {job_id}"
        return f"❌ Job {job_id} not found."
