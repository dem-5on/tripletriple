"""
Gateway WebSocket — Real-time agent interaction endpoint.

Protocol: JSON messages with {"type": "...", "data": "..."}
Supports chat commands (/new, /reset, /status, /model, etc.)
"""

import json
import logging

from fastapi import APIRouter, WebSocket

from .startup import (
    agent,
    chat_commands,
    _handle_model_command,
    llm_provider,
    session_manager,
)
from ..session import InboundContext

logger = logging.getLogger("tripletriple.gateway.ws")

router = APIRouter()


@router.websocket("/ws")
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
