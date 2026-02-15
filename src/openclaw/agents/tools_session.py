"""
Session Tools — Full Suite

sessions_list    → Discover active sessions
sessions_history → Fetch conversation transcript for a session
sessions_send    → Message another session (sync or fire-and-forget)
sessions_spawn   → Spawn a subagent for a background task
"""

import asyncio
import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .tools import Tool

logger = logging.getLogger("openclaw.agents.session_tools")


# ─── Schemas ──────────────────────────────────────────────────────


class SessionsListSchema(BaseModel):
    kinds: Optional[List[str]] = Field(
        None,
        description='Filter by session kind: "main", "group", "cron", "hook", "node", "other"',
    )
    limit: Optional[int] = Field(
        20, description="Max sessions to return (default 20)"
    )
    active_minutes: Optional[int] = Field(
        None, description="Only sessions updated within N minutes"
    )


class SessionsHistorySchema(BaseModel):
    session_key: str = Field(
        ..., description="Session key or session ID to fetch history for"
    )
    limit: Optional[int] = Field(
        50, description="Max messages to return (default 50)"
    )
    include_tools: bool = Field(
        False, description="Include tool result messages (default false)"
    )


class SessionsSendSchema(BaseModel):
    session_key: str = Field(
        ..., description="Session key or session ID to send message to"
    )
    message: str = Field(..., description="The message to send")
    timeout_seconds: int = Field(
        30,
        description="Seconds to wait for reply. 0 = fire-and-forget.",
    )


class SessionsSpawnSchema(BaseModel):
    task: str = Field(..., description="Task description for the subagent")
    label: Optional[str] = Field(
        None, description="Label for logs/UI"
    )
    model: Optional[str] = Field(
        None, description="Override model for the subagent"
    )
    run_timeout_seconds: int = Field(
        120, description="Abort subagent after N seconds (default 120)"
    )
    cleanup: str = Field(
        "keep",
        description='"delete" or "keep" the subagent session after completion',
    )


# ─── Helpers ─────────────────────────────────────────────────────


def _get_context(kwargs: dict) -> dict:
    """Extract and remove _context from kwargs."""
    ctx = kwargs.pop("_context", {})
    if not ctx:
        raise RuntimeError("Session tools require runtime context (session_manager, dock)")
    return ctx


def _kind_from_key(key: str) -> str:
    """Infer session kind from its key string."""
    if ":group:" in key:
        return "group"
    if key.startswith("cron:"):
        return "cron"
    if key.startswith("hook:"):
        return "hook"
    if key.startswith("node-"):
        return "node"
    if ":subagent:" in key:
        return "subagent"
    if ":dm:" in key or key.endswith(":main") or key == "main":
        return "main"
    return "other"


# ─── sessions_list ────────────────────────────────────────────────


class SessionsListTool(Tool):
    name = "sessions_list"
    description = (
        "List active sessions. Returns session keys, kinds, channels, "
        "token counts, and last activity. Use to discover what sessions exist."
    )
    args_schema = SessionsListSchema
    needs_context = True

    async def run(self, **kwargs) -> str:
        ctx = _get_context(kwargs)
        sm = ctx["session_manager"]

        kinds = kwargs.get("kinds")
        limit = kwargs.get("limit", 20)
        active_minutes = kwargs.get("active_minutes")

        entries = sm.list_sessions(active_minutes=active_minutes)

        # Filter by kind
        if kinds:
            entries = [e for e in entries if _kind_from_key(e.session_key) in kinds]

        # Apply limit
        entries = entries[:limit]

        results = []
        for e in entries:
            results.append({
                "key": e.session_key,
                "session_id": e.session_id,
                "kind": _kind_from_key(e.session_key),
                "channel": e.channel or "unknown",
                "display_name": e.display_name,
                "updated_at": e.updated_at,
                "model": e.model_override,
                "tokens": {
                    "input": e.tokens.input_tokens,
                    "output": e.tokens.output_tokens,
                    "total": e.tokens.total_tokens,
                },
            })

        return json.dumps({"sessions": results, "count": len(results)}, indent=2)


# ─── sessions_history ─────────────────────────────────────────────


class SessionsHistoryTool(Tool):
    name = "sessions_history"
    description = (
        "Fetch conversation history for a session. "
        "Returns messages with role, content, and timestamp."
    )
    args_schema = SessionsHistorySchema
    needs_context = True

    async def run(self, **kwargs) -> str:
        ctx = _get_context(kwargs)
        sm = ctx["session_manager"]

        session_key = kwargs["session_key"]
        limit = kwargs.get("limit", 50)
        include_tools = kwargs.get("include_tools", False)

        # Try by key first, then by session_id
        session = sm.get_session_by_key(session_key)
        if not session:
            # Search by session_id
            for s in sm._sessions.values():
                if s.id == session_key:
                    session = s
                    break

        if not session:
            return json.dumps({"error": f"Session not found: {session_key}"})

        messages = session.messages
        if not include_tools:
            messages = [m for m in messages if m.role != "tool"]

        # Take last N messages
        messages = messages[-limit:]

        result = []
        for m in messages:
            entry = {
                "role": m.role,
                "content": m.content if isinstance(m.content, str) else "[multimodal]",
                "timestamp": m.timestamp,
            }
            result.append(entry)

        return json.dumps({
            "session_key": session.key,
            "session_id": session.id,
            "messages": result,
            "count": len(result),
        }, indent=2)


# ─── sessions_send ────────────────────────────────────────────────


class SessionsSendTool(Tool):
    name = "sessions_send"
    description = (
        "Send a message to another session. If timeout_seconds > 0, "
        "waits for the agent to finish and returns the reply. "
        "If timeout_seconds == 0, fires and forgets."
    )
    args_schema = SessionsSendSchema
    needs_context = True

    async def run(self, **kwargs) -> str:
        ctx = _get_context(kwargs)
        sm = ctx["session_manager"]
        agent = ctx.get("agent")
        dock = ctx.get("dock")

        session_key = kwargs["session_key"]
        message = kwargs["message"]
        timeout = kwargs.get("timeout_seconds", 30)

        # Find or resolve the target session
        session = sm.get_session_by_key(session_key)
        if not session:
            # Search by session_id
            for s in sm._sessions.values():
                if s.id == session_key:
                    session = s
                    break

        if not session:
            return json.dumps({"error": f"Session not found: {session_key}"})

        if not agent:
            return json.dumps({"error": "No agent available to process the message"})

        run_id = str(uuid.uuid4())[:8]

        async def _run_agent():
            """Run the agent on the target session."""
            session.add_message("user", message, metadata={"provenance": "inter_session"})
            response = ""
            async for chunk in agent.process_message(session, message):
                response += chunk
            session.add_message("assistant", response)
            return response

        if timeout == 0:
            # Fire-and-forget
            asyncio.create_task(_run_agent())
            return json.dumps({
                "status": "accepted",
                "run_id": run_id,
                "session_key": session_key,
            })

        # Wait for completion
        try:
            reply = await asyncio.wait_for(_run_agent(), timeout=timeout)
            return json.dumps({
                "status": "ok",
                "run_id": run_id,
                "reply": reply,
            })
        except asyncio.TimeoutError:
            return json.dumps({
                "status": "timeout",
                "run_id": run_id,
                "error": f"Timed out after {timeout}s. Use sessions_history to check later.",
            })
        except Exception as e:
            return json.dumps({
                "status": "error",
                "run_id": run_id,
                "error": str(e),
            })


# ─── sessions_spawn ───────────────────────────────────────────────


class SessionsSpawnTool(Tool):
    name = "sessions_spawn"
    description = (
        "Spawn a subagent to perform a background task. "
        "Returns immediately with a run ID and child session key. "
        "The subagent runs asynchronously and announces results when done. "
        "Subagents have all tools except session tools (no recursive spawning)."
    )
    args_schema = SessionsSpawnSchema
    needs_context = True

    async def run(self, **kwargs) -> str:
        ctx = _get_context(kwargs)
        sm = ctx["session_manager"]
        agent = ctx.get("agent")
        dock = ctx.get("dock")
        current_session = ctx.get("current_session")

        task = kwargs["task"]
        label = kwargs.get("label", "subagent")
        model_override = kwargs.get("model")
        run_timeout = kwargs.get("run_timeout_seconds", 120)
        cleanup = kwargs.get("cleanup", "keep")

        if not agent:
            return json.dumps({"error": "No agent available for subagent spawning"})

        # Create child session
        agent_id = sm.config.agent_id
        child_uuid = str(uuid.uuid4())[:12]
        child_key = f"agent:{agent_id}:subagent:{child_uuid}"
        run_id = str(uuid.uuid4())[:8]

        from ..gateway.session import (
            Session, SessionEntry, SessionOrigin, ChatType, InboundContext,
        )

        child_entry = SessionEntry(
            session_key=child_key,
            chat_type=ChatType.NODE,
            channel="internal",
            display_name=label,
            origin=SessionOrigin(
                label=label,
                provider="internal",
            ),
        )
        child_session = Session(entry=child_entry)
        sm._sessions[child_key] = child_session
        sm._entries[child_key] = child_entry

        # Build a child agent WITHOUT session tools (no recursive spawning)
        from .core import ReActAgent

        SESSION_TOOL_NAMES = {
            "sessions_list", "sessions_history", "sessions_send", "sessions_spawn",
        }

        # Copy non-session tools from the parent agent
        child_agent = ReActAgent(
            llm=agent.llm,
            tools=[],  # We'll register manually
            prompt_builder=agent.prompt_builder,
            tool_context=agent.tool_context,
        )
        for name, tool in agent.tool_registry.get_all().items():
            if name not in SESSION_TOOL_NAMES:
                child_agent.tool_registry.register(tool)

        async def _run_subagent():
            """Execute the subagent task and announce results."""
            start_time = time.time()
            try:
                child_session.add_message("user", task)
                response = ""
                async for chunk in child_agent.process_message(child_session, task):
                    response += chunk
                child_session.add_message("assistant", response)

                elapsed = round(time.time() - start_time, 1)
                tokens = child_session.entry.tokens

                # Announce result to parent session's channel
                announce = (
                    f"🤖 **Subagent Complete** — `{label}`\n\n"
                    f"**Status:** ✅ Done\n"
                    f"**Result:** {response[:2000]}\n\n"
                    f"⏱️ {elapsed}s · {tokens.total_tokens} tokens · "
                    f"Session: `{child_key}`"
                )

                # Deliver announce to parent's channel
                if dock and current_session:
                    channel_name = current_session.entry.channel
                    channel = dock.channels.get(channel_name) if channel_name else None
                    if channel:
                        # Extract chat_id from session key
                        chat_id = current_session.entry.origin.from_id
                        if chat_id:
                            await channel.send_message(chat_id, announce)

                logger.info(
                    f"Subagent '{label}' completed in {elapsed}s "
                    f"({tokens.total_tokens} tokens)"
                )

            except asyncio.CancelledError:
                logger.warning(f"Subagent '{label}' cancelled")
            except Exception as e:
                logger.error(f"Subagent '{label}' error: {e}", exc_info=True)

                # Announce error
                if dock and current_session:
                    error_msg = (
                        f"🤖 **Subagent Failed** — `{label}`\n\n"
                        f"**Status:** ❌ Error\n"
                        f"**Error:** {str(e)[:500]}"
                    )
                    channel_name = current_session.entry.channel
                    channel = dock.channels.get(channel_name) if channel_name else None
                    if channel:
                        chat_id = current_session.entry.origin.from_id
                        if chat_id:
                            await channel.send_message(chat_id, error_msg)
            finally:
                # Cleanup if requested
                if cleanup == "delete":
                    sm._sessions.pop(child_key, None)
                    sm._entries.pop(child_key, None)
                    logger.info(f"Subagent session '{child_key}' cleaned up")

        # Launch with timeout
        task_coro = _run_subagent()
        if run_timeout > 0:
            task_coro = asyncio.wait_for(task_coro, timeout=run_timeout)

        asyncio.create_task(task_coro)

        return json.dumps({
            "status": "accepted",
            "run_id": run_id,
            "child_session_key": child_key,
            "label": label,
        })
