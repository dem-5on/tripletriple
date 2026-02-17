"""
Session Manager — Central session orchestrator.

Resolves session keys from inbound contexts, manages lifecycle
(creation, reset, pruning), and handles persistence.
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from .models import (
    ChatType,
    DMScope,
    InboundContext,
    ResetMode,
    ResetPolicy,
    SendAction,
    Session,
    SessionConfig,
    SessionEntry,
    SessionOrigin,
    Message,
)
from .persistence import (
    format_age,
    format_timestamp,
    load_store,
    resolve_store_path,
    save_store,
    write_transcript as _write_transcript,
)

logger = logging.getLogger("tripletriple.session.manager")


class SessionManager:
    """
    Central session orchestrator.

    Resolves session keys from inbound contexts, manages lifecycle
    (creation, reset, pruning), and handles persistence.
    """

    def __init__(self, config: SessionConfig = None, workspace_manager=None):
        self.config = config or SessionConfig()
        self.workspace_manager = workspace_manager
        self._sessions: Dict[str, Session] = {}       # key → Session
        self._entries: Dict[str, SessionEntry] = {}    # key → store entry
        self._store_path = resolve_store_path(
            self.config.store_path, self.config.agent_id
        )
        load_store(self._store_path, self._sessions, self._entries)

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
        """Force-reset a session by key. Flushes summary to daily log first."""
        old = self._sessions.get(key)
        if not old:
            return None

        # Flush session summary to daily memory log
        if self.workspace_manager and old.messages:
            from .persistence import extract_topics
            topics = extract_topics(old.messages) or ["general"]
            self.workspace_manager.flush_session_summary(
                session_key=key,
                message_count=len(old.messages),
                topics=topics,
            )

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

    def _save_store(self):
        """Persist session entries to disk."""
        save_store(self._store_path, self._entries)

    def write_transcript(self, session: Session, message: Message):
        """Append a message to the JSONL transcript file."""
        _write_transcript(self._store_path, session, message)

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
            age = format_age(e.updated_at)
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
            f"  Created: {format_timestamp(e.created_at)}",
            f"  Updated: {format_timestamp(e.updated_at)} ({format_age(e.updated_at)})",
            f"  Messages: {len(session.messages)}",
            f"  Tokens: {t.total_tokens} ({t.input_tokens} in / {t.output_tokens} out)",
        ]
        if e.model_override:
            lines.append(f"  Model: {e.model_override}")
        if e.origin.label:
            lines.append(f"  Origin: {e.origin.label} ({e.origin.provider})")
        return "\n".join(lines)
