"""
Slack Channel Adapter using slack_bolt.

Connects Slack Events API to the OpenClaw Channel Dock.
"""

import os
import logging
from typing import Optional
from slack_bolt.async_app import AsyncApp
from slack_bolt.adapter.fastapi.async_handler import AsyncSlackRequestHandler

from ...dock import Channel

logger = logging.getLogger("openclaw.channels.slack")


class SlackChannel(Channel):
    """
    Slack channel adapter using slack_bolt.

    Uses Slack's Events API (Socket Mode or HTTP) to receive
    messages and the Web API to send replies.
    """

    def __init__(
        self,
        bot_token: Optional[str] = None,
        app_token: Optional[str] = None,
        on_message=None,
    ):
        self.bot_token = bot_token or os.getenv("SLACK_BOT_TOKEN")
        self.app_token = app_token or os.getenv("SLACK_APP_TOKEN")

        if not self.bot_token:
            raise ValueError("SLACK_BOT_TOKEN is required")

        self.slack_app = AsyncApp(token=self.bot_token)
        self.handler = AsyncSlackRequestHandler(self.slack_app)
        self._on_message = on_message
        self._register_handlers()

    def _register_handlers(self):
        @self.slack_app.event("app_mention")
        async def handle_mention(event, say):
            """Handle @mention messages in channels."""
            text = event.get("text", "")
            user_id = event.get("user", "unknown")
            chat_id = event.get("channel", "unknown")

            # Strip bot mention from text
            # Slack format: <@BOTID> message
            import re

            text = re.sub(r"<@[A-Z0-9]+>", "", text).strip()

            if not text:
                return

            logger.info(f"Slack mention from {user_id} in {chat_id}: {text[:50]}...")

            if self._on_message:
                await self._on_message(
                    channel_name="slack",
                    chat_id=chat_id,
                    user_id=user_id,
                    text=text,
                )

        @self.slack_app.event("message")
        async def handle_dm(event, say):
            """Handle direct messages."""
            # Only handle DMs (no subtype = regular message)
            if event.get("channel_type") != "im" or event.get("subtype"):
                return

            text = event.get("text", "")
            user_id = event.get("user", "unknown")
            chat_id = event.get("channel", "unknown")

            if not text:
                return

            logger.info(f"Slack DM from {user_id}: {text[:50]}...")

            if self._on_message:
                await self._on_message(
                    channel_name="slack",
                    chat_id=chat_id,
                    user_id=user_id,
                    text=text,
                )

    async def start(self):
        """Start in Socket Mode (requires SLACK_APP_TOKEN)."""
        if self.app_token:
            from slack_bolt.adapter.socket_mode.async_handler import (
                AsyncSocketModeHandler,
            )

            logger.info("Starting Slack channel (Socket Mode)")
            socket_handler = AsyncSocketModeHandler(self.slack_app, self.app_token)
            await socket_handler.start_async()
        else:
            logger.info(
                "Slack channel ready — register the handler with your FastAPI app"
            )

    async def stop(self):
        """Stop the Slack channel."""
        logger.info("Stopping Slack channel")

    async def send_message(self, chat_id: str, text: str):
        """Send a message to a Slack channel or DM."""
        try:
            await self.slack_app.client.chat_postMessage(
                channel=chat_id,
                text=text,
            )
            logger.info(f"Slack message sent to {chat_id}")
        except Exception as e:
            logger.error(f"Failed to send Slack message: {e}")
