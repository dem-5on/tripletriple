"""
Chat Command Handler — In-chat slash commands.

Handles /new, /reset, /status, /stop, /compact, /send, /context
before messages reach the agent.
"""

from __future__ import annotations

from typing import Any, Optional

from .models import Message, SendAction, Session
from .persistence import extract_topics, format_age


class ChatCommandHandler:
    """
    Handles in-chat slash commands before they reach the agent.

    Supports: /new, /reset, /status, /stop, /compact, /send, /context
    """

    def __init__(self, session_manager):
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
            f"  Updated: {format_age(e.updated_at)}",
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
            f"Key topics discussed: {', '.join(extract_topics(old_messages))}]"
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
