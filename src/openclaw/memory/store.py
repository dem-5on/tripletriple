"""
RAG/Memory system for OpenClaw.

Provides an abstract MemoryStore interface and a LanceDB-based implementation
for vector-based retrieval-augmented generation.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import time
import uuid


class MemoryEntry(BaseModel):
    """A single memory entry that can be stored and retrieved."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    metadata: Dict[str, Any] = {}
    timestamp: float = Field(default_factory=time.time)
    session_id: Optional[str] = None


class SearchResult(BaseModel):
    """A ranked search result from memory retrieval."""

    entry: MemoryEntry
    score: float  # similarity score (0-1, higher is better)


class MemoryStore(ABC):
    """
    Abstract memory store interface.

    Implementations should handle:
    - Adding new memories (with optional embeddings)
    - Searching by semantic similarity
    - Deleting memories
    """

    @abstractmethod
    async def add(self, entry: MemoryEntry) -> str:
        """Store a memory entry. Returns the entry ID."""
        pass

    @abstractmethod
    async def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Search for relevant memories using semantic similarity."""
        pass

    @abstractmethod
    async def delete(self, entry_id: str) -> bool:
        """Delete a memory entry by ID."""
        pass

    @abstractmethod
    async def list_all(self, session_id: Optional[str] = None) -> List[MemoryEntry]:
        """List all entries, optionally filtered by session."""
        pass
