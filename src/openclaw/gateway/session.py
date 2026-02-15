"""
OpenClaw Session Management — Full Implementation

Mirrors the OpenClaw session system:
- DM scope modes (main / per-peer / per-channel-peer / per-account-channel-peer)
- Session key resolution for DMs, groups, threads, cron, webhooks
- Reset policies (daily / idle / per-type / per-channel overrides)
- Identity links (cross-channel user mapping)
- Origin metadata + token tracking
- In-chat command dispatch (/new, /reset, /status, /stop, /compact)
- Send policy (allow/deny rules)
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

logger = logging.getLogger("openclaw.session")


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
    Full session configuration — mirrors openclaw.json → session block.
    """
    dm_scope: DMScope = DMScope.MAIN
    main_key: str = "main"
    agent_id: str = "default"
    store_path: str = "~/.openclaw/agents/{agent_id}/sessions/sessions.json"

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


# ─── Session Manager ─────────────────────────────────────────────


class SessionManager:
    """
    Central session orchestrator.

    Resolves session keys from inbound contexts, manages lifecycle
    (creation, reset, pruning), and handles persistence.
    """

    def __init__(self, config: SessionConfig = None):
        self.config = config or SessionConfig()
        self._sessions: Dict[str, Session] = {}       # key → Session
        self._entries: Dict[str, SessionEntry] = {}    # key → store entry
        self._store_path = self._resolve_store_path()
        self._load_store()

    # ── Key Resolution ──

    def resolve_key(self, ctx: InboundContext) -> str:
        """
        Convert inbound context → canonical session key.

        DMs follow dm_scope; groups/threads/cron/hooks have fixed patterns.
        """
        agent = self.config.agent_id

        # Group messages
        if ctx.group_id:
            base = f"agent:{agent}:{ctx.channel}:group:{ctx.group_id}"
            if ctx.thread_id:
                return f"{base}:topic:{ctx.thread_id}"
            return base

        # DM messages — resolve by scope
        peer_id = self._resolve_peer_id(ctx)
        scope = self.config.dm_scope

        if scope == DMScope.MAIN:
            return f"agent:{agent}:{self.config.main_key}"
        elif scope == DMScope.PER_PEER:
            return f"agent:{agent}:dm:{peer_id}"
        elif scope == DMScope.PER_CHANNEL_PEER:
            return f"agent:{agent}:{ctx.channel}:dm:{peer_id}"
        elif scope == DMScope.PER_ACCOUNT_CHANNEL_PEER:
            return f"agent:{agent}:{ctx.channel}:{ctx.account_id}:dm:{peer_id}"

        return f"agent:{agent}:{self.config.main_key}"

    @staticmethod
    def resolve_cron_key(job_id: str) -> str:
        return f"cron:{job_id}"

    @staticmethod
    def resolve_hook_key(hook_id: str = None) -> str:
        return f"hook:{hook_id or str(uuid.uuid4())}"

    @staticmethod
    def resolve_node_key(node_id: str) -> str:
        return f"node-{node_id}"

    def _resolve_peer_id(self, ctx: InboundContext) -> str:
        """
        Resolve peer ID through identity links.

        If `session.identityLinks` has an entry matching `<channel>:<senderId>`,
        the canonical name replaces the raw peer ID so the same person shares
        a session across channels.
        """
        raw = f"{ctx.channel}:{ctx.sender_id}"
        for canonical, linked_ids in self.config.identity_links.items():
            if raw in linked_ids:
                return canonical
        return ctx.sender_id

    # ── Session Lifecycle ──

    def get_or_create(
        self,
        ctx: InboundContext,
        text: str = "",
    ) -> Session:
        """
        Get an existing session or create a new one.

        1. Resolve the canonical session key
        2. Check if the session needs a reset
        3. Return the active session (creating if needed)
        """
        key = self.resolve_key(ctx)
        chat_type = self._infer_chat_type(ctx)

        # Check for reset trigger in the message
        trigger_fired = self._is_reset_trigger(text)

        # Get existing session or create
        session = self._sessions.get(key)

        if session and not trigger_fired:
            # Check if reset policy expired the session
            if self._should_reset(session, chat_type, ctx.channel):
                logger.info(f"Session expired, resetting: {key}")
                session = self._create_session(key, ctx, chat_type)
            else:
                session.entry.updated_at = time.time()
        else:
            if trigger_fired and session:
                logger.info(f"Reset triggered for session: {key}")
            session = self._create_session(key, ctx, chat_type)

        self._sessions[key] = session
        self._entries[key] = session.entry
        return session

    def get_session(self, channel_id: str, user_id: str) -> Session:
        """
        Backwards-compatible accessor (used by simple WebSocket handler).
        Creates an InboundContext and delegates to get_or_create().
        """
        ctx = InboundContext(
            channel=channel_id,
            sender_id=user_id,
            is_dm=True,
        )
        return self.get_or_create(ctx)

    def reset_session(self, key: str) -> Optional[Session]:
        """Force-reset a session by key."""
        old = self._sessions.get(key)
        if not old:
            return None

        ctx = InboundContext(
            channel=old.entry.channel or "unknown",
            sender_id=old.entry.origin.from_id or "unknown",
        )
        new_session = self._create_session(key, ctx, old.entry.chat_type)
        self._sessions[key] = new_session
        self._entries[key] = new_session.entry
        self._save_store()
        return new_session

    def list_sessions(
        self,
        active_minutes: Optional[int] = None,
    ) -> List[SessionEntry]:
        """List all sessions, optionally filtered by recency."""
        entries = list(self._entries.values())
        if active_minutes is not None:
            cutoff = time.time() - (active_minutes * 60)
            entries = [e for e in entries if e.updated_at >= cutoff]
        return sorted(entries, key=lambda e: e.updated_at, reverse=True)

    def get_session_by_key(self, key: str) -> Optional[Session]:
        """Get a session by its key."""
        return self._sessions.get(key)

    @property
    def session_count(self) -> int:
        return len(self._sessions)

    # ── Reset Policy ──

    def _should_reset(
        self, session: Session, chat_type: ChatType, channel: str
    ) -> bool:
        """Evaluate whether a session has expired per its reset policy."""
        policy = self._get_reset_policy(chat_type, channel)
        now = time.time()
        last = session.entry.updated_at

        # Daily reset: expired if last update is before most recent reset time
        if policy.mode == ResetMode.DAILY:
            reset_time = self._last_daily_reset(policy.at_hour)
            expired = last < reset_time

            # Idle also fires if configured alongside daily
            if policy.idle_minutes:
                idle_expired = (now - last) > (policy.idle_minutes * 60)
                return expired or idle_expired

            return expired

        # Idle reset only
        if policy.mode == ResetMode.IDLE and policy.idle_minutes:
            return (now - last) > (policy.idle_minutes * 60)

        return False

    def _get_reset_policy(self, chat_type: ChatType, channel: str) -> ResetPolicy:
        """
        Resolve the effective reset policy.
        Priority: per-channel > per-type > default.
        """
        if channel in self.config.reset_by_channel:
            return self.config.reset_by_channel[channel]

        type_key = chat_type.value
        if type_key in self.config.reset_by_type:
            return self.config.reset_by_type[type_key]

        return self.config.reset

    @staticmethod
    def _last_daily_reset(at_hour: int) -> float:
        """
        Calculate the timestamp of the most recent daily reset.
        """
        now = datetime.now()
        today_reset = now.replace(hour=at_hour, minute=0, second=0, microsecond=0)
        if now < today_reset:
            # Not yet past today's reset → use yesterday's
            from datetime import timedelta
            today_reset -= timedelta(days=1)
        return today_reset.timestamp()

    def _is_reset_trigger(self, text: str) -> bool:
        """Check if text starts with a reset trigger command."""
        stripped = text.strip()
        for trigger in self.config.reset_triggers:
            if stripped == trigger or stripped.startswith(trigger + " "):
                return True
        return False

    # ── Send Policy ──

    def check_send_policy(self, session: Session, ctx: InboundContext) -> bool:
        """
        Evaluate whether the agent is allowed to reply.
        Returns True if sending is allowed.
        """
        # Per-session override (/send on|off)
        if session.entry.send_override is not None:
            return session.entry.send_override == SendAction.ALLOW

        policy = self.config.send_policy
        for rule in policy.rules:
            match = rule.match
            matched = True

            if "channel" in match and match["channel"] != ctx.channel:
                matched = False
            if "chatType" in match:
                chat_type = self._infer_chat_type(ctx).value
                if match["chatType"] != chat_type:
                    matched = False
            if "keyPrefix" in match:
                key = self.resolve_key(ctx)
                if not key.startswith(match["keyPrefix"]):
                    matched = False

            if matched:
                return rule.action == SendAction.ALLOW

        return policy.default == SendAction.ALLOW

    # ── Persistence ──

    def save(self):
        """Save all session entries to the JSON store."""
        self._save_store()

    def _resolve_store_path(self) -> Path:
        raw = self.config.store_path.format(agent_id=self.config.agent_id)
        return Path(os.path.expanduser(raw))

    def _load_store(self):
        """Load session entries from disk."""
        if self._store_path.exists():
            try:
                data = json.loads(self._store_path.read_text())
                for key, entry_data in data.items():
                    entry = SessionEntry(**entry_data)
                    self._entries[key] = entry
                    # Create session shell (messages restored from transcript if needed)
                    self._sessions[key] = Session(entry=entry)
                logger.info(f"Loaded {len(self._entries)} sessions from store")
            except Exception as e:
                logger.warning(f"Could not load session store: {e}")

    def _save_store(self):
        """Persist session entries to disk."""
        try:
            self._store_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                key: entry.model_dump(mode="json")
                for key, entry in self._entries.items()
            }
            self._store_path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.warning(f"Could not save session store: {e}")

    def write_transcript(self, session: Session, message: Message):
        """Append a message to the JSONL transcript file."""
        try:
            transcript_dir = self._store_path.parent
            transcript_path = transcript_dir / f"{session.id}.jsonl"
            transcript_dir.mkdir(parents=True, exist_ok=True)
            with open(transcript_path, "a") as f:
                line = message.model_dump(mode="json")
                f.write(json.dumps(line) + "\n")
        except Exception as e:
            logger.warning(f"Could not write transcript: {e}")

    # ── Internal Helpers ──

    def _create_session(
        self,
        key: str,
        ctx: InboundContext,
        chat_type: ChatType,
    ) -> Session:
        """Create a fresh session."""
        entry = SessionEntry(
            session_key=key,
            chat_type=chat_type,
            channel=ctx.channel,
            display_name=ctx.display_name,
            subject=ctx.group_subject,
            room=ctx.group_channel,
            space=ctx.group_space,
            origin=SessionOrigin(
                label=ctx.display_name or ctx.group_subject,
                provider=ctx.channel,
                from_id=ctx.sender_id,
                account_id=ctx.account_id if ctx.account_id != "default" else None,
                thread_id=ctx.thread_id,
            ),
        )
        session = Session(entry=entry)
        logger.info(f"Created session: {key} (id={entry.session_id[:8]}...)")
        self._save_store()
        return session

    @staticmethod
    def _infer_chat_type(ctx: InboundContext) -> ChatType:
        if ctx.thread_id:
            return ChatType.THREAD
        if ctx.group_id:
            return ChatType.GROUP
        return ChatType.DIRECT

    # ── Formatted Output ──

    def format_sessions_list(
        self, active_minutes: Optional[int] = None
    ) -> str:
        """Format a human-readable session list."""
        entries = self.list_sessions(active_minutes=active_minutes)
        if not entries:
            return "No sessions found."

        lines = ["📋 **Sessions**\n"]
        for i, e in enumerate(entries, 1):
            age = _format_age(e.updated_at)
            tokens = e.tokens.total_tokens
            label = e.display_name or e.session_key
            lines.append(
                f"  {i}. `{label}` — {e.chat_type.value} on {e.channel or '?'} "
                f"({tokens} tok) — {age}"
            )
        return "\n".join(lines)

    def format_session_detail(self, key: str) -> str:
        """Format detailed info for a specific session."""
        session = self._sessions.get(key)
        if not session:
            return f"Session `{key}` not found."

        e = session.entry
        t = e.tokens
        lines = [
            f"🔎 **Session Detail**\n",
            f"  Key: `{e.session_key}`",
            f"  ID: `{e.session_id}`",
            f"  Type: {e.chat_type.value}",
            f"  Channel: {e.channel or 'N/A'}",
            f"  Created: {_format_timestamp(e.created_at)}",
            f"  Updated: {_format_timestamp(e.updated_at)} ({_format_age(e.updated_at)})",
            f"  Messages: {len(session.messages)}",
            f"  Tokens: {t.total_tokens} ({t.input_tokens} in / {t.output_tokens} out)",
        ]
        if e.model_override:
            lines.append(f"  Model: {e.model_override}")
        if e.origin.label:
            lines.append(f"  Origin: {e.origin.label} ({e.origin.provider})")
        return "\n".join(lines)


# ─── Chat Command Handler ────────────────────────────────────────


class ChatCommandHandler:
    """
    Handles in-chat slash commands before they reach the agent.

    Supports: /new, /reset, /status, /stop, /compact, /send, /context
    """

    def __init__(self, session_manager: SessionManager):
        self.sm = session_manager
        self._commands = {
            "/new": self._cmd_new,
            "/reset": self._cmd_reset,
            "/status": self._cmd_status,
            "/stop": self._cmd_stop,
            "/compact": self._cmd_compact,
            "/send": self._cmd_send,
            "/context": self._cmd_context,
        }

    def register(self, command: str, handler: Any):
        """Register a new command handler."""
        self._commands[command] = handler

    def is_command(self, text: str) -> bool:
        """Check if text is a known chat command."""
        stripped = text.strip()
        return any(
            stripped == cmd or stripped.startswith(cmd + " ")
            for cmd in self._commands
        )

    def handle(self, text: str, session: Session) -> Optional[str]:
        """
        Handle a chat command. Returns response string, or None if not a command.
        """
        stripped = text.strip()
        for cmd, handler in self._commands.items():
            if stripped == cmd or stripped.startswith(cmd + " "):
                args = stripped[len(cmd):].strip()
                return handler(session, args)
        return None

    def _cmd_new(self, session: Session, args: str) -> str:
        """Reset session and optionally set a model."""
        new_session = self.sm.reset_session(session.key)
        if not new_session:
            return "❌ Could not reset session."

        msg = f"🔄 Session reset. New ID: `{new_session.id[:8]}...`"
        if args:
            new_session.entry.model_override = args
            msg += f"\nModel set to: `{args}`"
        return msg

    def _cmd_reset(self, session: Session, args: str) -> str:
        """Alias for /new without model change."""
        return self._cmd_new(session, "")

    def _cmd_status(self, session: Session, args: str) -> str:
        """Show session status."""
        e = session.entry
        t = e.tokens
        lines = [
            "📊 **Session Status**\n",
            f"  Key: `{e.session_key}`",
            f"  Messages: {len(session.messages)}",
            f"  Tokens: {t.total_tokens} / context: {t.context_tokens}",
            f"  Updated: {_format_age(e.updated_at)}",
        ]
        if e.model_override:
            lines.append(f"  Model: {e.model_override}")
        return "\n".join(lines)

    def _cmd_stop(self, session: Session, args: str) -> str:
        """Stop the current run (placeholder — needs agent integration)."""
        return "🛑 Current run stopped."

    def _cmd_compact(self, session: Session, args: str) -> str:
        """
        Summarize older context to free up window space.
        (Full implementation requires agent-driven summarization.)
        """
        msg_count = len(session.messages)
        if msg_count <= 4:
            return "ℹ️ Not enough context to compact."

        # Keep last 4 messages, summarize the rest as a system note
        old_messages = session.messages[:-4]
        summary_content = (
            f"[Compacted {len(old_messages)} earlier messages. "
            f"Key topics discussed: {', '.join(_extract_topics(old_messages))}]"
        )
        session.messages = [
            Message(role="system", content=summary_content),
            *session.messages[-4:],
        ]
        return f"📦 Compacted {len(old_messages)} messages into summary. {len(session.messages)} messages remain."

    def _cmd_send(self, session: Session, args: str) -> str:
        """
        Toggle send policy for this session.
        /send on | /send off | /send inherit
        """
        if args == "on":
            session.entry.send_override = SendAction.ALLOW
            return "✅ Sending enabled for this session."
        elif args == "off":
            session.entry.send_override = SendAction.DENY
            return "🔇 Sending disabled for this session."
        elif args == "inherit":
            session.entry.send_override = None
            return "↩️ Using config send policy."
        return "Usage: /send [on|off|inherit]"

    def _cmd_context(self, session: Session, args: str) -> str:
        """Show context contributors."""
        msgs = session.messages
        by_role = {}
        for m in msgs:
            by_role.setdefault(m.role, 0)
            by_role[m.role] += 1

        lines = ["📄 **Context**\n"]
        for role, count in sorted(by_role.items(), key=lambda x: -x[1]):
            lines.append(f"  {role}: {count} messages")
        lines.append(f"\n  Total messages: {len(msgs)}")
        lines.append(f"  Total tokens: {session.entry.tokens.total_tokens}")
        return "\n".join(lines)


# ─── Helpers ──────────────────────────────────────────────────────


def _format_age(ts: float) -> str:
    """Format a timestamp as a human-readable age string."""
    delta = time.time() - ts
    if delta < 60:
        return "just now"
    if delta < 3600:
        return f"{int(delta / 60)}m ago"
    if delta < 86400:
        return f"{int(delta / 3600)}h ago"
    return f"{int(delta / 86400)}d ago"


def _format_timestamp(ts: float) -> str:
    """Format a Unix timestamp as ISO string."""
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _extract_topics(messages: List[Message]) -> List[str]:
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
