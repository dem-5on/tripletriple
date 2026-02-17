"""
Session Persistence — JSONL transcripts, JSON store, and helpers.

Handles loading/saving session entries and transcript files.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from .models import Message, Session, SessionEntry

logger = logging.getLogger("tripletriple.session.persistence")


# ─── Store Operations ─────────────────────────────────────────────


def resolve_store_path(store_path_template: str, agent_id: str) -> Path:
    """Resolve the session store file path from config template."""
    raw = store_path_template.format(agent_id=agent_id)
    return Path(os.path.expanduser(raw))


def load_store(
    store_path: Path,
    sessions: Dict[str, Session],
    entries: Dict[str, SessionEntry],
) -> None:
    """Load session entries from disk into the provided dicts."""
    if store_path.exists():
        try:
            data = json.loads(store_path.read_text())
            for key, entry_data in data.items():
                entry = SessionEntry(**entry_data)
                entries[key] = entry
                # Create session shell (messages restored from transcript if needed)
                sessions[key] = Session(entry=entry)
            logger.info(f"Loaded {len(entries)} sessions from store")
        except Exception as e:
            logger.warning(f"Could not load session store: {e}")


def save_store(
    store_path: Path,
    entries: Dict[str, SessionEntry],
) -> None:
    """Persist session entries to disk."""
    try:
        store_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            key: entry.model_dump(mode="json")
            for key, entry in entries.items()
        }
        store_path.write_text(json.dumps(data, indent=2))
    except Exception as e:
        logger.warning(f"Could not save session store: {e}")


def write_transcript(store_path: Path, session: Session, message: Message) -> None:
    """Append a message to the JSONL transcript file."""
    try:
        transcript_dir = store_path.parent
        transcript_path = transcript_dir / f"{session.id}.jsonl"
        transcript_dir.mkdir(parents=True, exist_ok=True)
        with open(transcript_path, "a") as f:
            line = message.model_dump(mode="json")
            f.write(json.dumps(line) + "\n")
    except Exception as e:
        logger.warning(f"Could not write transcript: {e}")


# ─── Formatting Helpers ──────────────────────────────────────────


def format_age(ts: float) -> str:
    """Format a timestamp as a human-readable age string."""
    delta = time.time() - ts
    if delta < 60:
        return "just now"
    if delta < 3600:
        return f"{int(delta / 60)}m ago"
    if delta < 86400:
        return f"{int(delta / 3600)}h ago"
    return f"{int(delta / 86400)}d ago"


def format_timestamp(ts: float) -> str:
    """Format a Unix timestamp as ISO string."""
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def extract_topics(messages: List[Message]) -> List[str]:
    """Extract rough topic hints from messages for compaction summaries."""
    topics = set()
    for m in messages:
        if m.role == "user" and len(m.content) > 10:
            # Take first few words as a rough topic
            words = m.content.split()[:4]
            topics.add(" ".join(words) + "...")
        if len(topics) >= 3:
            break
    return list(topics) or ["general conversation"]
