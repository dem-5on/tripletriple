"""
LanceDB-based vector memory store for RAG.

Uses LanceDB for persistent vector search with automatic embedding
generation via the built-in embedding functions.
"""

import os
from typing import List, Optional
from .store import MemoryStore, MemoryEntry, SearchResult

try:
    import lancedb
    from lancedb.pydantic import LanceModel, Vector
    from lancedb.embeddings import get_registry

    LANCEDB_AVAILABLE = True
except ImportError:
    LANCEDB_AVAILABLE = False


class LanceDBMemoryStore(MemoryStore):
    """
    Memory store backed by LanceDB for vector similarity search.

    Uses sentence-transformers for embedding generation and LanceDB
    for fast approximate nearest-neighbor retrieval.
    """

    def __init__(
        self,
        db_path: str = None,
        table_name: str = "memories",
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        if not LANCEDB_AVAILABLE:
            raise ImportError(
                "lancedb is required for LanceDBMemoryStore. "
                "Install with: pip install lancedb"
            )

        self.db_path = db_path or os.path.expanduser("~/.tripletriple/state/memory")
        self.table_name = table_name
        self.db = lancedb.connect(self.db_path)

        # Use sentence-transformers for embeddings
        self._embedding_fn = (
            get_registry()
            .get("sentence-transformers")
            .create(name=embedding_model)
        )

        self._ensure_table()

    def _ensure_table(self):
        """Create the table if it does not exist."""
        if self.table_name not in self.db.table_names():
            import pyarrow as pa

            schema = pa.schema(
                [
                    pa.field("id", pa.string()),
                    pa.field("content", pa.string()),
                    pa.field("metadata", pa.string()),
                    pa.field("timestamp", pa.float64()),
                    pa.field("session_id", pa.string()),
                    pa.field(
                        "vector",
                        pa.list_(pa.float32(), 384),
                    ),  # MiniLM-L6 dim
                ]
            )
            self.table = self.db.create_table(self.table_name, schema=schema)
        else:
            self.table = self.db.open_table(self.table_name)

    async def add(self, entry: MemoryEntry) -> str:
        """Embed and store a memory entry."""
        import json

        # Generate embedding
        embedding = self._embedding_fn.compute_source_embeddings([entry.content])[0]

        data = [
            {
                "id": entry.id,
                "content": entry.content,
                "metadata": json.dumps(entry.metadata),
                "timestamp": entry.timestamp,
                "session_id": entry.session_id or "",
                "vector": embedding,
            }
        ]
        self.table.add(data)
        return entry.id

    async def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Semantic search over stored memories."""
        import json

        query_embedding = self._embedding_fn.compute_source_embeddings([query])[0]

        results = (
            self.table.search(query_embedding)
            .limit(top_k)
            .to_pandas()
        )

        search_results = []
        for _, row in results.iterrows():
            entry = MemoryEntry(
                id=row["id"],
                content=row["content"],
                metadata=json.loads(row.get("metadata", "{}")),
                timestamp=row.get("timestamp", 0),
                session_id=row.get("session_id") or None,
            )
            score = 1.0 - row.get("_distance", 0)  # LanceDB returns distance
            search_results.append(SearchResult(entry=entry, score=score))

        return search_results

    async def delete(self, entry_id: str) -> bool:
        """Delete a memory entry by ID."""
        try:
            self.table.delete(f"id = '{entry_id}'")
            return True
        except Exception:
            return False

    async def list_all(self, session_id: Optional[str] = None) -> List[MemoryEntry]:
        """List all entries, optionally filtered by session."""
        import json

        df = self.table.to_pandas()
        if session_id:
            df = df[df["session_id"] == session_id]

        entries = []
        for _, row in df.iterrows():
            entries.append(
                MemoryEntry(
                    id=row["id"],
                    content=row["content"],
                    metadata=json.loads(row.get("metadata", "{}")),
                    timestamp=row.get("timestamp", 0),
                    session_id=row.get("session_id") or None,
                )
            )
        return entries


class InMemoryStore(MemoryStore):
    """
    Simple in-memory store for testing and development.
    No vector search — uses naive substring matching.
    """

    def __init__(self):
        self._entries: dict[str, MemoryEntry] = {}

    async def add(self, entry: MemoryEntry) -> str:
        self._entries[entry.id] = entry
        return entry.id

    async def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        results = []
        query_lower = query.lower()
        for entry in self._entries.values():
            if query_lower in entry.content.lower():
                results.append(SearchResult(entry=entry, score=0.8))
        return sorted(results, key=lambda r: r.score, reverse=True)[:top_k]

    async def delete(self, entry_id: str) -> bool:
        if entry_id in self._entries:
            del self._entries[entry_id]
            return True
        return False

    async def list_all(self, session_id: Optional[str] = None) -> List[MemoryEntry]:
        entries = list(self._entries.values())
        if session_id:
            entries = [e for e in entries if e.session_id == session_id]
        return entries
