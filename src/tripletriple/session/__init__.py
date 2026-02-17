"""
TripleTriple Session Package

Re-exports all public session types for clean imports:
    from tripletriple.session import SessionManager, Session, ChatCommandHandler
"""

from .models import (
    ChatType,
    DMScope,
    InboundContext,
    Message,
    ResetMode,
    ResetPolicy,
    SendAction,
    SendPolicy,
    SendPolicyRule,
    Session,
    SessionConfig,
    SessionEntry,
    SessionOrigin,
    TokenUsage,
)
from .manager import SessionManager
from .commands import ChatCommandHandler
from .persistence import (
    format_age,
    format_timestamp,
    extract_topics,
)

__all__ = [
    # Enums
    "ChatType",
    "DMScope",
    "ResetMode",
    "SendAction",
    # Config
    "ResetPolicy",
    "SendPolicy",
    "SendPolicyRule",
    "SessionConfig",
    # Data Models
    "InboundContext",
    "Message",
    "Session",
    "SessionEntry",
    "SessionOrigin",
    "TokenUsage",
    # Manager
    "SessionManager",
    # Commands
    "ChatCommandHandler",
    # Helpers
    "format_age",
    "format_timestamp",
    "extract_topics",
]
