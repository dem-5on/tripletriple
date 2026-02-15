"""
OpenClaw Gateway Server — Full Integration

Wires together the FastAPI application, Agent runtime, Channel Dock,
Model Catalog, Session Manager, Chat Commands, and WebSocket protocol
into a single runnable server.
"""

from fastapi import FastAPI, WebSocket, Request
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
import json

from ..agents.core import ReActAgent
from ..agents.tools_bash import BashTool
from ..agents.tools_browser import BrowserNavigateTool
from ..agents.tools_file import ReadFileTool, WriteFileTool
from ..agents.tools_search import WebSearchTool
from ..agents.tools_cron import CronTool
from ..agents.tools_memory import MemorySaveTool, MemorySearchTool
from ..agents.tools_session import (
    SessionsListTool,
    SessionsHistoryTool,
    SessionsSendTool,
    SessionsSpawnTool,
)
from ..agents.tools_gateway import GatewayTool
from ..version import get_version, check_for_updates
from ..agents.llm import create_provider_with_fallback
from ..agents.model_catalog import ModelSelector
from ..agents.system_prompt import SystemPromptBuilder, WorkspaceConfig
from ..agents.skills import SkillLoader, SkillConfig
from ..channels.dock import ChannelDock
from ..gateway.session import (
    SessionManager,
    SessionConfig,
    ChatCommandHandler,
    InboundContext,
)
from ..memory.lancedb_store import InMemoryStore

logger = logging.getLogger("openclaw.gateway")

# ─── Shared State ─────────────────────────────────────────────────

session_manager = SessionManager(config=SessionConfig())
chat_commands = ChatCommandHandler(session_manager=session_manager)

# Register /model command handler (defined below)
def _cmd_model_wrapper(session, args: str) -> str:
    # Reconstruct command for handler
    cmd = "/model"
    if args:
        cmd += f" {args}"
    return _handle_model_command(cmd)

chat_commands.register("/model", _cmd_model_wrapper)

# Override /status to include active model info
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

# Add /help command
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

memory_store = InMemoryStore()
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
        "bash", "browser", "read_file", "write_file", "web_search", "cron",
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
        CronTool,
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

# Create the channel dock
dock = ChannelDock(
    session_manager=session_manager,
    agent=agent,
    chat_commands=chat_commands,
)

# Wire dock and agent into tool_context (circular ref, needs to happen after creation)
tool_context["dock"] = dock
tool_context["agent"] = agent


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic."""
    logger.info("OpenClaw Gateway starting up...")
    logger.info(f"Version: {get_version()}")
    primary = model_selector.get_primary()
    logger.info(f"Primary model: {primary.full_id if primary else 'none'} ({primary.name if primary else ''})")
    logger.info(f"Active model: {llm_provider.provider_name}/{llm_provider.model_id}")
    logger.info(f"Registered tools: {list(agent.tool_registry._tools.keys())}")
    logger.info(f"Registered channels: {list(dock.channels.keys())}")
    logger.info(f"Sessions loaded: {session_manager.session_count}")

    # Non-blocking update check
    try:
        import asyncio
        loop = asyncio.get_event_loop()
        update_info = await loop.run_in_executor(None, check_for_updates)
        if update_info.get("available"):
            logger.warning(
                f"🔄 Update available! {update_info['behind_count']} new commit(s). "
                f"Run 'openclaw update' or use the gateway tool."
            )
    except Exception:
        pass  # Non-critical, don't block startup

    yield
    # Persist sessions on shutdown
    session_manager.save()
    logger.info("OpenClaw Gateway shutting down...")


app = FastAPI(
    title="OpenClaw Gateway",
    description="AI Agent Gateway — Control Plane",
    version=get_version(),
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── HTTP Endpoints ──────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "status": "ok",
        "service": "openclaw-gateway",
        "version": "0.1.0",
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model": f"{llm_provider.provider_name}/{llm_provider.model_id}",
        "channels": list(dock.channels.keys()),
        "tools": list(agent.tool_registry._tools.keys()),
    }


@app.get("/status")
async def system_status():
    primary = model_selector.get_primary()
    return {
        "gateway": "running",
        "model": f"{llm_provider.provider_name}/{llm_provider.model_id}",
        "model_name": primary.name if primary else "unknown",
        "sessions": session_manager.session_count,
        "channels": {name: "connected" for name in dock.channels},
        "tools": list(agent.tool_registry._tools.keys()),
    }


# ─── Models API ──────────────────────────────────────────────────

@app.get("/models")
async def list_models(provider: str = None):
    """List all available models, optionally filtered by provider."""
    models = model_selector.list_models(provider=provider)
    return {
        "primary": model_selector.selection.primary,
        "models": [
            {
                "id": m.full_id,
                "name": m.name,
                "alias": m.alias,
                "provider": m.provider,
                "context_window": m.context_window,
                "max_output_tokens": m.max_output_tokens,
                "capabilities": [c.value for c in m.capabilities],
            }
            for m in models
        ],
    }


@app.post("/models/set")
async def set_model(request: Request):
    """Switch the active model at runtime."""
    body = await request.json()
    ref = body.get("model", "")

    model_info = model_selector.set_model(ref)
    if not model_info:
        return {"error": f'Model "{ref}" not found'}, 404

    # Hot-swap if same provider, otherwise need a restart
    if model_info.provider == llm_provider.provider_name:
        llm_provider.switch_model(model_info.id)
        return {
            "status": "switched",
            "model": model_info.full_id,
            "name": model_info.name,
        }
    else:
        return {
            "status": "pending_restart",
            "model": model_info.full_id,
            "name": model_info.name,
            "message": f"Provider changed to {model_info.provider}. Restart the gateway to activate.",
        }


@app.get("/models/status")
async def model_status():
    """Current model selection status."""
    primary = model_selector.get_primary()
    fallbacks = model_selector.get_fallback_chain()
    return {
        "primary": {
            "id": primary.full_id if primary else None,
            "name": primary.name if primary else None,
        },
        "active": f"{llm_provider.provider_name}/{llm_provider.model_id}",
        "fallbacks": [
            {"id": m.full_id, "name": m.name}
            for m in fallbacks[1:]  # skip primary
        ],
        "aliases": model_selector.selection.aliases,
    }


# ─── Sessions API ────────────────────────────────────────────────

@app.get("/sessions")
async def list_sessions(active: int = None):
    """List all sessions, optionally filtered by recency (minutes)."""
    entries = session_manager.list_sessions(active_minutes=active)
    return {
        "count": len(entries),
        "sessions": [
            {
                "key": e.session_key,
                "id": e.session_id,
                "type": e.chat_type.value,
                "channel": e.channel,
                "display_name": e.display_name,
                "tokens": e.tokens.model_dump(),
                "updated_at": e.updated_at,
                "created_at": e.created_at,
            }
            for e in entries
        ],
    }


@app.get("/sessions/{key:path}")
async def get_session_detail(key: str):
    """Get detailed info for a specific session."""
    session = session_manager.get_session_by_key(key)
    if not session:
        return {"error": f"Session '{key}' not found"}, 404

    e = session.entry
    return {
        "key": e.session_key,
        "id": e.session_id,
        "type": e.chat_type.value,
        "channel": e.channel,
        "messages": len(session.messages),
        "tokens": e.tokens.model_dump(),
        "origin": e.origin.model_dump(by_alias=True),
        "model_override": e.model_override,
        "created_at": e.created_at,
        "updated_at": e.updated_at,
    }


@app.post("/sessions/{key:path}/reset")
async def reset_session_endpoint(key: str):
    """Reset a specific session."""
    new = session_manager.reset_session(key)
    if not new:
        return {"error": f"Session '{key}' not found"}, 404
    return {
        "status": "reset",
        "new_session_id": new.id,
        "key": key,
    }


# ─── WebSocket Endpoint ──────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time agent interaction.

    Protocol: JSON messages with {"type": "...", "data": "..."}
    Supports chat commands (/new, /reset, /status, /model, etc.)
    """
    await websocket.accept()
    logger.info("New WebSocket connection accepted")

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                msg = {"type": "chat", "data": raw}

            if msg.get("type") == "chat":
                text = msg.get("data", "")

                # Resolve session from WS context
                ctx = InboundContext(
                    channel="websocket",
                    sender_id=msg.get("user_id", "ws-user"),
                    is_dm=True,
                )
                session = session_manager.get_or_create(ctx, text=text)

                # Handle /model command
                if text.strip().startswith("/model"):
                    result = _handle_model_command(text.strip())
                    await websocket.send_text(
                        json.dumps({"type": "done", "data": result})
                    )
                    continue

                # Handle session chat commands (/new, /reset, /status, etc.)
                if chat_commands.is_command(text):
                    result = chat_commands.handle(text, session)
                    if result:
                        await websocket.send_text(
                            json.dumps({"type": "done", "data": result})
                        )
                        continue

                # Check send policy
                if not session_manager.check_send_policy(session, ctx):
                    await websocket.send_text(
                        json.dumps({"type": "done", "data": "🔇 Sending is disabled for this session."})
                    )
                    continue

                # Normal agent interaction
                session.add_message("user", text)
                session_manager.write_transcript(
                    session,
                    session.messages[-1],
                )

                full_response = ""
                async for chunk in agent.process_message(session, text):
                    full_response += chunk
                    await websocket.send_text(
                        json.dumps({"type": "stream", "data": chunk})
                    )

                session.add_message("assistant", full_response)
                session_manager.write_transcript(
                    session,
                    session.messages[-1],
                )
                session_manager.save()

                await websocket.send_text(
                    json.dumps({"type": "done", "data": full_response})
                )
            else:
                await websocket.send_text(
                    json.dumps({"type": "error", "data": f"Unknown type: {msg.get('type')}"})
                )

    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        session_manager.save()
        logger.info("WebSocket connection closed")


def _handle_model_command(text: str) -> str:
    """Handle /model slash command (like OpenClaw's in-chat model picker)."""
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


# ─── WhatsApp Webhook ─────────────────────────────────────────────

@app.post("/webhooks/whatsapp")
async def whatsapp_webhook(request: Request):
    """Receive WhatsApp webhook events."""
    body = await request.json()
    whatsapp_channel = dock.channels.get("whatsapp")
    if whatsapp_channel and hasattr(whatsapp_channel, "handle_webhook"):
        await whatsapp_channel.handle_webhook(body)
    return {"status": "ok"}


@app.get("/webhooks/whatsapp")
async def whatsapp_verify(request: Request):
    """WhatsApp webhook verification challenge."""
    mode = request.query_params.get("hub.mode")
    token = request.query_params.get("hub.verify_token")
    challenge = request.query_params.get("hub.challenge")

    verify_token = "openclaw-whatsapp-verify"
    if mode == "subscribe" and token == verify_token:
        return int(challenge)
    return {"error": "Verification failed"}, 403
