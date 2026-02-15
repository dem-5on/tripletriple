import os
from pydantic import BaseModel, Field
from .tools import Tool

class ReadFileSchema(BaseModel):
    path: str = Field(..., description="Path to the file to read")

class ReadFileTool(Tool):
    name = "read_file"
    description = "Read content from a file."
    args_schema = ReadFileSchema

    async def run(self, path: str) -> str:
        try:
            if not os.path.exists(path):
                return "Error: File does not exist."
            with open(path, "r") as f:
                return f.read()
        except Exception as e:
            return f"Error reading file: {e}"

class WriteFileSchema(BaseModel):
    path: str = Field(..., description="Path to the file to write")
    content: str = Field(..., description="Content to write to the file")

class WriteFileTool(Tool):
    name = "write_file"
    description = "Write content to a file."
    args_schema = WriteFileSchema

    async def run(self, path: str, content: str) -> str:
        try:
            with open(path, "w") as f:
                f.write(content)
            return f"Successfully wrote to {path}"
        except Exception as e:
            return f"Error writing file: {e}"
