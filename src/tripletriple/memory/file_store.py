"""
File-backed memory store that dual-writes to Markdown and a vector store.

This implements OpenClaw's "Dual Memory" architecture:
1. Source of Truth: Markdown files in the workspace (readable, editable).
2. Index: Vector store (LanceDB) for semantic search.
"""

import time
import uuid
import logging
from pathlib import Path
from typing import List, Optional

from .store import MemoryStore, MemoryEntry, SearchResult

logger = logging.getLogger(__name__)


class FileBackedMemoryStore(MemoryStore):
    """
    A memory store that persists entries to a Markdown file
    AND indexes them in a backing vector store.
    """

    def __init__(
        self,
        workspace_root: Path,
        vector_store: MemoryStore,
        main_file: str = "MEMORY.md",
    ):
        self.workspace_root = workspace_root
        self.vector_store = vector_store
        self.main_file = main_file
        self.memory_path = self.workspace_root / self.main_file

        # Ensure workspace exists
        if not self.workspace_root.exists():
            self.workspace_root.mkdir(parents=True, exist_ok=True)

        # Ensure MEMORY.md exists
        if not self.memory_path.exists():
            self.memory_path.touch()

    async def add(self, entry: MemoryEntry) -> str:
        """
        Add a memory entry.
        1. Append to appropriate Markdown file based on category.
        2. Add to vector store.
        """
        # 1. Determine target file
        target_file = self.main_file  # Default: MEMORY.md
        
        if entry.category == "active_task":
            target_file = "active-tasks.md"
        elif entry.category == "lesson":
            target_file = "lessons.md"
        elif entry.category == "project":
            target_file = "projects.md"
        elif entry.category == "self_review":
            target_file = "self-review.md"
        elif entry.category == "daily":
            # memory/YYYY-MM-DD.md
            date_str = time.strftime("%Y-%m-%d", time.localtime(entry.timestamp))
            target_file = f"memory/{date_str}.md"
            # Ensure memory dir exists
            (self.workspace_root / "memory").mkdir(exist_ok=True)

        target_path = self.workspace_root / target_file

        # 2. Append to Markdown file
        try:
            timestamp_str = time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(entry.timestamp)
            )
            # Format: - [YYYY-MM-DD HH:MM:SS] <content> <!-- id: <uuid> -->
            line = f"- [{timestamp_str}] {entry.content} <!-- id: {entry.id} -->\n"
            
            # Ensure file exists before writing (for non-daily files mainly)
            if not target_path.exists():
                target_path.touch()

            with open(target_path, "a", encoding="utf-8") as f:
                f.write(line)
        except Exception as e:
            logger.error(f"Failed to write to {target_file}: {e}")
            # We still proceed to vector store

        # 2. Add to vector store
        return await self.vector_store.add(entry)

    async def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Delegate search to vector store."""
        return await self.vector_store.search(query, top_k)

    async def delete(self, entry_id: str) -> bool:
        """
        Delete from vector store.
        TODO: Remove from Markdown file (complex due to file locking/parsing).
        For now, we just remove from index. The Markdown remains as a log.
        """
        return await self.vector_store.delete(entry_id)

    async def list_all(self, session_id: Optional[str] = None) -> List[MemoryEntry]:
        """Delegate list to vector store."""
        return await self.vector_store.list_all(session_id)
