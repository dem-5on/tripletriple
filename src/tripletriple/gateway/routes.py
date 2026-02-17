"""
Gateway HTTP Routes — REST API endpoints.

Provides: health check, system status, model management,
session listing, WhatsApp webhook, and generic webhook handler.
"""

from fastapi import APIRouter, Request

from .startup import (
    agent,
    dock,
    llm_provider,
    model_selector,
    session_manager,
)
from ..version import get_version

router = APIRouter()


# ─── Core Endpoints ──────────────────────────────────────────────

@router.get("/")
async def root():
    return {
        "status": "ok",
        "service": "tripletriple-gateway",
        "version": get_version(),
    }


@router.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model": f"{llm_provider.provider_name}/{llm_provider.model_id}",
        "channels": list(dock.channels.keys()),
        "tools": list(agent.tool_registry._tools.keys()),
    }


@router.get("/status")
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

@router.get("/models")
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


@router.post("/models/set")
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


@router.get("/models/status")
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

@router.get("/sessions")
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


@router.get("/sessions/{key:path}")
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


@router.post("/sessions/{key:path}/reset")
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


# ─── WhatsApp Webhook ─────────────────────────────────────────────

@router.post("/webhooks/whatsapp")
async def whatsapp_webhook(request: Request):
    """Receive WhatsApp webhook events."""
    body = await request.json()
    whatsapp_channel = dock.channels.get("whatsapp")
    if whatsapp_channel and hasattr(whatsapp_channel, "handle_webhook"):
        await whatsapp_channel.handle_webhook(body)
    return {"status": "ok"}


@router.get("/webhooks/whatsapp")
async def whatsapp_verify(request: Request):
    """WhatsApp webhook verification challenge."""
    mode = request.query_params.get("hub.mode")
    token = request.query_params.get("hub.verify_token")
    challenge = request.query_params.get("hub.challenge")

    verify_token = "tripletriple-whatsapp-verify"
    if mode == "subscribe" and token == verify_token:
        return int(challenge)
    return {"error": "Verification failed"}, 403


# ─── Generic Webhook ──────────────────────────────────────────────

@router.post("/webhook/{hook_id}")
async def generic_webhook(hook_id: str, request: Request):
    """
    Generic webhook endpoint.
    Creates a hook session and runs the agent with the payload.
    """
    from ..services.agent_runner import run_session_turn
    from ..session.models import InboundContext

    body = await request.json()
    payload_text = body.get("text", body.get("message", str(body)))

    # Create a hook session
    ctx = InboundContext(
        channel="webhook",
        sender_id=hook_id,
        is_dm=True,
    )
    session = session_manager.get_or_create(ctx, text=payload_text)

    # Run the agent
    response = await run_session_turn(
        agent, session, payload_text, session_manager
    )

    return {
        "status": "processed",
        "hook_id": hook_id,
        "response": response,
    }
