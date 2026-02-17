"""
Gateway Startup — Initialization and wiring of all core services.

Creates and wires: SessionManager, ChatCommandHandler, ModelSelector,
LLM Provider, SystemPromptBuilder, Agent, ChannelDock, Cron, Heartbeat.
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from ..agents.core import ReActAgent
from ..agents.tools_bash import BashTool
from ..agents.tools_browser import BrowserNavigateTool
from ..agents.tools_file import ReadFileTool, WriteFileTool
from ..agents.tools_cron import CronScheduleTool, CronListTool, CronDeleteTool
from ..agents.tools_memory import MemorySaveTool, MemorySearchTool
from ..agents.tools_session import (
    SessionsListTool,
    SessionsHistoryTool,
    SessionsSendTool,
    SessionsSpawnTool,
)
from ..agents.tools_gateway import GatewayTool
from ..agents.tools_search import WebSearchTool
from ..version import get_version, check_for_updates
from ..agents.llm import create_provider_with_fallback
from ..agents.model_catalog import ModelSelector
from ..agents.system_prompt import SystemPromptBuilder, WorkspaceConfig
from ..agents.skills import SkillLoader, SkillConfig
from ..channels.dock import ChannelDock
from ..session import (
    SessionManager,
    SessionConfig,
    ChatCommandHandler,
)
from ..memory.lancedb_store import LanceDBMemoryStore
from ..memory.file_store import FileBackedMemoryStore
from ..services.heartbeat import HeartbeatManager
from ..services.cron import CronManager

logger = logging.getLogger("tripletriple.gateway")


# ─── Shared State ─────────────────────────────────────────────────

session_manager = SessionManager(config=SessionConfig())
chat_commands = ChatCommandHandler(session_manager=session_manager)
heartbeat_manager = HeartbeatManager(session_manager=session_manager)
cron_manager = CronManager(
    workspace_root=WorkspaceConfig().root,
    session_manager=session_manager,
)

# Persistent Memory Store
try:
    vector_store = LanceDBMemoryStore(
        db_path="~/.tripletriple/state/memory",
        table_name="memories",
    )
except ImportError:
    logger.warning("LanceDB not available, falling back to InMemoryStore (non-persistent).")
    from ..memory.lancedb_store import InMemoryStore
    vector_store = InMemoryStore()

memory_store = FileBackedMemoryStore(
    workspace_root=WorkspaceConfig().root,
    vector_store=vector_store,
)

model_selector = ModelSelector()

# Create provider from model catalog (auto-detects available API keys)
llm_provider = create_provider_with_fallback(selector=model_selector)

# Load agent skills
skill_loader = SkillLoader(config=SkillConfig())
loaded_skills = skill_loader.load_all()
skills_prompt = SkillLoader.format_for_prompt(skill_loader.get_prompt_eligible())

# Initialize the workspace (copies defaults if not already present)
SystemPromptBuilder.init_workspace()

# Build the system prompt assembler
primary = model_selector.get_primary()
prompt_builder = SystemPromptBuilder(
    config=WorkspaceConfig(),
    model_name=f"{llm_provider.provider_name}/{llm_provider.model_id}",
    tools=[
        "bash", "browser", "read_file", "write_file", "web_search",
        "cron_schedule", "cron_list", "cron_delete",
        "memory_save", "memory_search",
        "sessions_list", "sessions_history", "sessions_send", "sessions_spawn",
        "gateway",
    ],
    skills_prompt=skills_prompt,
)

# Tool context — shared services that context-aware tools need
tool_context = {
    "session_manager": session_manager,
}

# Create the agent with all tools + prompt builder
agent = ReActAgent(
    llm=llm_provider,
    tools=[
        BashTool,
        BrowserNavigateTool,
        ReadFileTool,
        WriteFileTool,
        WebSearchTool,
    ],
    prompt_builder=prompt_builder,
    tool_context=tool_context,
)
agent.tool_registry.register(MemorySaveTool(store=memory_store))
agent.tool_registry.register(MemorySearchTool(store=memory_store))

# Register session tools
agent.tool_registry.register(SessionsListTool())
agent.tool_registry.register(SessionsHistoryTool())
agent.tool_registry.register(SessionsSendTool())
agent.tool_registry.register(SessionsSpawnTool())
agent.tool_registry.register(GatewayTool())
agent.tool_registry.register(CronScheduleTool(manager=cron_manager))
agent.tool_registry.register(CronListTool(manager=cron_manager))
agent.tool_registry.register(CronDeleteTool(manager=cron_manager))

# Inject agent into heartbeat manager
heartbeat_manager.set_agent(agent)
cron_manager.set_agent(agent)

# Create the channel dock
dock = ChannelDock(
    session_manager=session_manager,
    agent=agent,
    chat_commands=chat_commands,
)

# Wire dock and agent into tool_context (circular ref, needs to happen after creation)
tool_context["dock"] = dock
tool_context["agent"] = agent


# ─── Chat Command Registration ───────────────────────────────────

def _handle_model_command(text: str) -> str:
    """Handle /model slash command (like TripleTriple's in-chat model picker)."""
    parts = text.split()

    # /model or /model list → show numbered picker
    if len(parts) == 1 or (len(parts) == 2 and parts[1] == "list"):
        return model_selector.format_model_list()

    # /model status → show current selection
    if len(parts) == 2 and parts[1] == "status":
        return model_selector.format_status()

    # /model <number> → select by number
    if len(parts) == 2:
        ref = parts[1]

        # Try as a number (picker index)
        try:
            idx = int(ref)
            all_models = model_selector.list_models()
            if 1 <= idx <= len(all_models):
                model = all_models[idx - 1]
                model_selector.set_model(model.full_id)
                if model.provider == llm_provider.provider_name:
                    llm_provider.switch_model(model.id)
                return f"✅ Switched to: {model.full_id} ({model.name})"
            return f"❌ Invalid number. Use /model to see the list."
        except ValueError:
            pass

        # Try as ref/alias
        model = model_selector.get_model(ref)
        if model:
            model_selector.set_model(model.full_id)
            if model.provider == llm_provider.provider_name:
                llm_provider.switch_model(model.id)
            return f"✅ Switched to: {model.full_id} ({model.name})"

        return f'❌ Model "{ref}" not found. Use /model to see available models.'

    return "Usage: /model [list|status|<number>|<provider/model>]"


def _cmd_model_wrapper(session, args: str) -> str:
    cmd = "/model"
    if args:
        cmd += f" {args}"
    return _handle_model_command(cmd)

chat_commands.register("/model", _cmd_model_wrapper)


def _cmd_status_wrapper(session, args: str) -> str:
    e = session.entry
    t = e.tokens
    primary = model_selector.get_primary() if model_selector else None
    model_name = e.model_override or (primary.full_id if primary else "unknown")
    lines = [
        "📊 **Session Status**\n",
        f"  Key: `{e.session_key}`",
        f"  Messages: {len(session.messages)}",
        f"  Tokens: {t.total_tokens} ({t.input_tokens} in / {t.output_tokens} out)",
        f"  Model: {model_name}",
    ]
    return "\n".join(lines)

chat_commands.register("/status", _cmd_status_wrapper)


def _cmd_help(session, args: str) -> str:
    return (
        "📖 Available Commands\n\n"
        "  /model — List available AI models\n"
        "  /model [N] — Switch to model number N\n"
        "  /model status — Show active model\n"
        "  /status — Session info and current model\n"
        "  /new — Start a fresh session\n"
        "  /reset — Reset current session\n"
        "  /context — View context breakdown\n"
        "  /help — Show this message"
    )

chat_commands.register("/help", _cmd_help)


# ─── Channel Loading ─────────────────────────────────────────────

channel_tasks = []

async def _load_channels():
    """Load and start channels configured in .env."""
    # Telegram
    if os.getenv("TELEGRAM_BOT_TOKEN"):
        try:
            from ..channels.plugins.telegram.adapter import TelegramChannel
            logger.info("🔌 Loading Telegram channel...")
            channel = TelegramChannel(on_message=dock.handle_incoming_message)
            dock.register_channel("telegram", channel)
            task = asyncio.create_task(channel.start())
            channel_tasks.append(task)
        except Exception as e:
            logger.error(f"❌ Failed to load Telegram: {e}")

    # Discord
    if os.getenv("DISCORD_BOT_TOKEN"):
        try:
            from ..channels.plugins.discord.adapter import DiscordChannel
            logger.info("🔌 Loading Discord channel...")
            channel = DiscordChannel(on_message=dock.handle_incoming_message)
            dock.register_channel("discord", channel)
            task = asyncio.create_task(channel.start())
            channel_tasks.append(task)
        except Exception as e:
            logger.error(f"❌ Failed to load Discord: {e}")

    # Slack
    if os.getenv("SLACK_BOT_TOKEN"):
        try:
            from ..channels.plugins.slack.adapter import SlackChannel
            logger.info("🔌 Loading Slack channel...")
            channel = SlackChannel(on_message=dock.handle_incoming_message)
            dock.register_channel("slack", channel)
            task = asyncio.create_task(channel.start())
            channel_tasks.append(task)
        except Exception as e:
            logger.error(f"❌ Failed to load Slack: {e}")


# ─── Lifespan ─────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic."""
    logger.info("OpenClaw Gateway starting up...")
    logger.info(f"Version: {get_version()}")
    primary = model_selector.get_primary()
    logger.info(f"Primary model: {primary.full_id if primary else 'none'} ({primary.name if primary else ''})")

    # Auto-load channels
    await _load_channels()

    # Start heartbeat
    await heartbeat_manager.start()
    await cron_manager.start()

    logger.info(f"Active model: {llm_provider.provider_name}/{llm_provider.model_id}")
    logger.info(f"Registered tools: {list(agent.tool_registry._tools.keys())}")
    logger.info(f"Registered channels: {list(dock.channels.keys())}")
    logger.info(f"Sessions loaded: {session_manager.session_count}")

    # Non-blocking update check
    try:
        loop = asyncio.get_event_loop()
        update_info = await loop.run_in_executor(None, check_for_updates)
        if update_info.get("available"):
            logger.warning(
                f"🔄 Update available! {update_info['behind_count']} new commit(s). "
                f"Run 'openclaw update' or use the gateway tool."
            )
    except Exception:
        pass  # Non-critical, don't block startup

    # Check for post-update marker
    root = WorkspaceConfig().root
    marker = root / ".update_success"
    if marker.exists():
        try:
            logger.info("✨ Update detected! Broadcasting success message...")
            version = get_version()
            msg = f"🚀 **Update Complete!**\nTripleTriple is now running version `{version}`."

            recent_sessions = session_manager.list_sessions(active_minutes=1440)
            count = 0
            for entry in recent_sessions:
                session = session_manager.get_session_by_key(entry.session_key)
                if session:
                    session.add_message("system", msg)
                    try:
                        await dock.send_outbound(
                            channel=entry.channel,
                            recipient_id=entry.origin.sender_id,
                            text=msg,
                            session=session,
                        )
                        count += 1
                    except Exception as e:
                        logger.warning(f"Failed to broadcast to {entry.session_key}: {e}")

            logger.info(f"Broadcast update notification to {count} sessions.")
            marker.unlink()
        except Exception as e:
            logger.error(f"Failed to process update marker: {e}")

    yield

    # Shutdown logic
    logger.info("OpenClaw Gateway shutting down...")

    # Stop all channels
    for name, channel in dock.channels.items():
        try:
            logger.info(f"Stopping {name} channel...")
            await asyncio.wait_for(channel.stop(), timeout=3.0)
        except asyncio.TimeoutError:
            logger.warning(f"Timeout stopping {name}, forcing...")
        except Exception as e:
            logger.error(f"Error stopping {name}: {e}")

    # Cancel tasks
    for task in channel_tasks:
        if not task.done():
            task.cancel()

    if channel_tasks:
        await asyncio.gather(*channel_tasks, return_exceptions=True)

    # Stop heartbeat
    await heartbeat_manager.stop()
    await cron_manager.stop()

    # Persist sessions on shutdown
    session_manager.save()
    logger.info("OpenClaw Gateway shutdown complete.")
