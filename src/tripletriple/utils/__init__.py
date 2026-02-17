"""
Common utility functions used across TripleTriple modules.
"""

import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List


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


def ensure_dir(path: Path) -> Path:
    """Create a directory (and parents) if it doesn't exist. Returns the path."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def expand_path(raw: str) -> Path:
    """Expand ~ and env vars in a path string."""
    return Path(os.path.expanduser(os.path.expandvars(raw)))
