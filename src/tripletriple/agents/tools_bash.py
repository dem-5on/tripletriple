import asyncio
from pydantic import BaseModel, Field
from .tools import Tool

from typing import Optional

class BashSchema(BaseModel):
    command: str = Field(..., description="The bash command to execute")
    cwd: Optional[str] = Field(None, description="Current working directory for the command. Use absolute paths.")

class BashTool(Tool):
    name = "bash"
    description = "Full system access. Execute bash commands on the host. Can manage files, install packages, and control the OS."
    args_schema = BashSchema

    async def run(self, command: str, cwd: Optional[str] = None) -> str:
        try:
            process = await asyncio.create_subprocess_shell(
                command,
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            result = ""
            if stdout:
                result += f"STDOUT:\n{stdout.decode().strip()}\n"
            if stderr:
                result += f"STDERR:\n{stderr.decode().strip()}\n"
            if not result:
                result = "Command executed with no output."
            return result
        except Exception as e:
            return f"Error executing command: {str(e)}"
