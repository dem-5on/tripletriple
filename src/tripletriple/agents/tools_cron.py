import json
from typing import Any, Optional
from pydantic import BaseModel, Field
from .tools import Tool

class CronScheduleSchema(BaseModel):
    schedule: str = Field(..., description="Cron expression (e.g., '0 6 * * *' or '06:00')")
    command: str = Field(..., description="Command or task description to run")
    delivery_channel: str = Field("last", description="Where to send output (e.g. 'telegram', 'last')")
    delivery_to: Optional[str] = Field(None, description="Specific target ID if not 'last'")

class CronScheduleTool(Tool):
    name = "cron_schedule"
    description = "Schedule a recurring task. Saves to crontab.json. Supports delivery configuration."
    args_schema = CronScheduleSchema

    def __init__(self, manager: Any):
        self.manager = manager

    async def run(
        self, 
        schedule: str, 
        command: str, 
        delivery_channel: str = "last",
        delivery_to: str = None
    ) -> str:
        # Resolve 'last' channel from current session context if not explicitly set
        if delivery_channel == "last" and hasattr(self, "tool_context") and self.tool_context:
            try:
                session = self.tool_context.get("session")
                if session and session.entry.channel:
                    # Update config to be explicit about "last"
                    delivery_channel = session.entry.channel
                    if not delivery_to:
                        delivery_to = session.entry.origin.from_id
            except Exception:
                pass # Fallback to generic "last" logic in manager

        job_id = self.manager.add_job(
            schedule, 
            command, 
            delivery_channel=delivery_channel, 
            delivery_to=delivery_to
        )
        dest = f" (target: {delivery_channel})"
        return f"✅ Scheduled task '{command}' {dest} with ID {job_id}"

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
