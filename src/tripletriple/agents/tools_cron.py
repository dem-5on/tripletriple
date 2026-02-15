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
        job_id = self.manager.add_job(schedule, command)
        return f"✅ Scheduled task '{command}' with ID {job_id} (schedule: '{schedule}')"

class CronListTool(Tool):
    name = "cron_list"
    description = "List all scheduled cron jobs."

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
