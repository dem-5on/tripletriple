"""
Discord Channel Adapter using discord.py

Connects the Discord Bot API to the TripleTriple Channel Dock.
"""

import os
import logging
from typing import Optional
import discord

from ...dock import Channel

logger = logging.getLogger("tripletriple.channels.discord")


class DiscordChannel(Channel):
    """
    Discord channel adapter using discord.py.

    Listens for messages in allowed guilds/channels and routes
    them through the Channel Dock.
    """

    def __init__(
        self,
        token: Optional[str] = None,
        on_message=None,
    ):
        self.token = token or os.getenv("DISCORD_BOT_TOKEN")
        if not self.token:
            raise ValueError("DISCORD_BOT_TOKEN is required")

        intents = discord.Intents.default()
        intents.message_content = True

        self.client = discord.Client(intents=intents)
        self._on_message = on_message
        self._register_handlers()

    def _register_handlers(self):
        @self.client.event
        async def on_ready():
            logger.info(f"Discord bot logged in as {self.client.user}")

        @self.client.event
        async def on_message(message: discord.Message):
            # Ignore bot's own messages
            if message.author == self.client.user:
                return

            # Ignore messages that don't mention the bot (in servers)
            if message.guild and self.client.user not in message.mentions:
                return

            chat_id = str(message.channel.id)
            user_id = str(message.author.id)
            text = message.content

            # Strip bot mention from text
            if self.client.user:
                text = text.replace(f"<@{self.client.user.id}>", "").strip()

            if not text:
                return

            logger.info(f"Discord message from {user_id} in {chat_id}: {text[:50]}...")

            if self._on_message:
                await self._on_message(
                    channel_name="discord",
                    chat_id=chat_id,
                    user_id=user_id,
                    text=text,
                )

    async def start(self):
        """Start the Discord bot."""
        logger.info("Starting Discord channel")
        await self.client.start(self.token)

    async def stop(self):
        """Stop the Discord bot."""
        logger.info("Stopping Discord channel")
        await self.client.close()

    async def send_message(self, chat_id: str, text: str):
        """Send a message to a Discord channel."""
        channel = self.client.get_channel(int(chat_id))
        if channel and isinstance(channel, discord.TextChannel):
            # Split long messages (Discord limit: 2000 chars)
            max_len = 2000
            for i in range(0, len(text), max_len):
                chunk = text[i : i + max_len]
                await channel.send(chunk)
        else:
            logger.warning(f"Channel {chat_id} not found or not a text channel")
