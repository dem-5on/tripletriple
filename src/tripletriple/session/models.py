"""
Session Models — Enums, Config, and Data Models

All session-related Pydantic models and enums live here.
"""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


# ─── Enums ────────────────────────────────────────────────────────


class DMScope(str, Enum):
    """How DM sessions are isolated."""
    MAIN = "main"
    PER_PEER = "per-peer"
    PER_CHANNEL_PEER = "per-channel-peer"
    PER_ACCOUNT_CHANNEL_PEER = "per-account-channel-peer"


class ResetMode(str, Enum):
    """Session reset strategy."""
    DAILY = "daily"
    IDLE = "idle"


class ChatType(str, Enum):
    """Type of conversation."""
    DIRECT = "direct"
    GROUP = "group"
    THREAD = "thread"
    CRON = "cron"
    HOOK = "hook"
    NODE = "node"


class SendAction(str, Enum):
    ALLOW = "allow"
    DENY = "deny"


# ─── Config Models ────────────────────────────────────────────────


class ResetPolicy(BaseModel):
    """Session reset policy configuration."""
    mode: ResetMode = ResetMode.DAILY
    at_hour: int = Field(default=4, ge=0, le=23, description="Hour for daily reset (local time)")
    idle_minutes: Optional[int] = Field(default=None, description="Idle timeout in minutes")


class SendPolicyRule(BaseModel):
    """A single send-policy matching rule."""
    action: SendAction = SendAction.DENY
    match: Dict[str, str] = {}  # channel, chatType, keyPrefix


class SendPolicy(BaseModel):
    """Controls whether the agent replies in specific contexts."""
    rules: List[SendPolicyRule] = []
    default: SendAction = SendAction.ALLOW


class SessionConfig(BaseModel):
    """
    Full session configuration — mirrors tripletriple.json → session block.
    """
    dm_scope: DMScope = DMScope.MAIN
    main_key: str = "main"
    agent_id: str = "default"
    store_path: str = "~/.tripletriple/agents/{agent_id}/sessions/sessions.json"

    # Identity links: canonical_name → [provider-prefixed ids]
    identity_links: Dict[str, List[str]] = {}

    # Reset policies
    reset: ResetPolicy = Field(default_factory=ResetPolicy)
    reset_by_type: Dict[str, ResetPolicy] = {}       # direct / group / thread
    reset_by_channel: Dict[str, ResetPolicy] = {}     # discord / telegram / etc.

    # Triggers that force a new session
    reset_triggers: List[str] = Field(default_factory=lambda: ["/new", "/reset"])

    # Send policy
    send_policy: SendPolicy = Field(default_factory=SendPolicy)


# ─── Data Models ──────────────────────────────────────────────────


class SessionOrigin(BaseModel):
    """Where a session was created from (routing metadata)."""
    label: Optional[str] = None
    provider: Optional[str] = None          # normalized channel id
    from_id: Optional[str] = Field(None, alias="from")
    to_id: Optional[str] = Field(None, alias="to")
    account_id: Optional[str] = None
    thread_id: Optional[str] = None


class TokenUsage(BaseModel):
    """Token counts tracked per session."""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    context_tokens: int = 0

    def add(self, input_t: int = 0, output_t: int = 0):
        self.input_tokens += input_t
        self.output_tokens += output_t
        self.total_tokens = self.input_tokens + self.output_tokens


class Message(BaseModel):
    """A single message in the session conversation history."""
    role: str                                  # user, assistant, system, tool
    content: Union[str, List[Any]]
    timestamp: float = Field(default_factory=time.time)
    token_count: Optional[int] = None
    metadata: Dict[str, Any] = {}


class SessionEntry(BaseModel):
    """
    Represents a session stored in sessions.json.
    Maps session_key → metadata (without full message history).
    """
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_key: str
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    chat_type: ChatType = ChatType.DIRECT
    channel: Optional[str] = None
    display_name: Optional[str] = None
    subject: Optional[str] = None
    room: Optional[str] = None
    space: Optional[str] = None
    model_override: Optional[str] = None
    origin: SessionOrigin = Field(default_factory=SessionOrigin)
    tokens: TokenUsage = Field(default_factory=TokenUsage)
    send_override: Optional[SendAction] = None    # /send on|off override


class Session(BaseModel):
    """
    Full in-memory session with conversation history.
    """
    entry: SessionEntry
    messages: List[Message] = []

    # ── Convenience properties ──
    @property
    def id(self) -> str:
        return self.entry.session_id

    @property
    def key(self) -> str:
        return self.entry.session_key

    @property
    def channel_id(self) -> str:
        return self.entry.channel or ""

    @property
    def user_id(self) -> str:
        return self.entry.origin.from_id or ""

    def add_message(self, role: str, content: Union[str, List[Any]], **kwargs):
        self.messages.append(Message(role=role, content=content, **kwargs))
        self.entry.updated_at = time.time()

    def add_tokens(self, input_t: int = 0, output_t: int = 0):
        self.entry.tokens.add(input_t, output_t)

    def context_summary(self) -> str:
        n = len(self.messages)
        t = self.entry.tokens
        return (
            f"Session `{self.key}` — {n} messages, "
            f"{t.total_tokens} tokens ({t.input_tokens} in / {t.output_tokens} out)"
        )

    def prune_context(self, keep_last: int = 20) -> int:
        """
        Prune message history to keep only recent messages.
        Keeps the first message (if it's a system/boot prompt equivalent) 
        and the last `keep_last` messages.
        Returns number of messages removed.
        """
        if len(self.messages) <= keep_last:
            return 0

        # Simple strategy: Keep last N
        # TODO: More sophisticated summarization or keeping initial system prompt if stored in messages
        original_count = len(self.messages)
        self.messages = self.messages[-keep_last:]
        removed = original_count - len(self.messages)
        return removed


# ─── Inbound Context ─────────────────────────────────────────────


class InboundContext(BaseModel):
    """
    Normalized inbound message context used for key resolution.
    Channels fill this in before passing to the SessionManager.
    """
    channel: str                             # "telegram", "discord", etc.
    sender_id: str                           # platform-specific user id
    account_id: str = "default"              # for multi-account setups
    group_id: Optional[str] = None           # set if message is from a group
    thread_id: Optional[str] = None          # forum topic / thread
    is_dm: bool = True
    sender_name: Optional[str] = None
    display_name: Optional[str] = None       # conversation label
    group_subject: Optional[str] = None
    group_channel: Optional[str] = None      # Slack channel name, etc.
    group_space: Optional[str] = None
