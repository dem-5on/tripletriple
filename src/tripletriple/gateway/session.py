"""
BACKWARD-COMPATIBILITY SHIM

The session module has been refactored into the `tripletriple.session` package.
This file re-exports everything so existing imports continue to work.

Prefer importing directly from `tripletriple.session` in new code.
"""

# Re-export everything from the new session package
from ..session import (  # noqa: F401
    ChatCommandHandler,
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
    SessionManager,
    SessionOrigin,
    TokenUsage,
    extract_topics,
    format_age,
    format_timestamp,
)
