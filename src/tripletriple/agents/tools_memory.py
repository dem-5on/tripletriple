from pydantic import BaseModel, Field
from typing import Optional
from ..agents.tools import Tool
from ..memory.store import MemoryStore, MemoryEntry


class MemorySaveSchema(BaseModel):
    content: str = Field(..., description="The information to memorize")
    category: str = Field(
        "general",
        description=(
            "Category for routing: 'active_task' (crash recovery), 'lesson' (mistakes), "
            "'project' (state), 'self_review', 'daily' (log), 'general' (legacy)"
        ),
    )
    metadata: Optional[str] = Field(None, description="Optional context or tags")


class MemorySaveTool(Tool):
    """Tool for saving information into structured memory categories."""

    name = "memory_save"
    description = (
        "Save information to long-term memory. "
        "Choose a category: active_task, lesson, project, self_review, daily, or general."
    )
    args_schema = MemorySaveSchema

    def __init__(self, store: MemoryStore):
        self.store = store

    async def run(
        self, content: str, category: str = "general", metadata: Optional[str] = None
    ) -> str:
        entry = MemoryEntry(
            content=content,
            category=category,
            metadata={"tags": metadata} if metadata else {},
        )
        entry_id = await self.store.add(entry)
        return f"Saved to {category} memory (id: {entry_id})"


class MemorySearchSchema(BaseModel):
    query: str = Field(..., description="Search query to find relevant memories")
    top_k: int = Field(5, description="Number of results to return")


class MemorySearchTool(Tool):
    """Tool for searching long-term memory using semantic similarity."""

    name = "memory_search"
    description = "Search long-term memory for previously stored information."
    args_schema = MemorySearchSchema

    def __init__(self, store: MemoryStore):
        self.store = store

    async def run(self, query: str, top_k: int = 5) -> str:
        results = await self.store.search(query, top_k=top_k)
        if not results:
            return "No relevant memories found."

        output = "Found memories:\n\n"
        for i, r in enumerate(results, 1):
            output += f"{i}. (score: {r.score:.2f}) {r.entry.content}\n"
        return output
