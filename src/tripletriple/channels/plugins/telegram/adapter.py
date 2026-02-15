"""
Telegram Channel Adapter using aiogram 3.x

Connects the Telegram Bot API to the TripleTriple Channel Dock.
"""

import os
import logging
from typing import Optional
from aiogram import Bot, Dispatcher, Router
from aiogram.client.default import DefaultBotProperties
from aiogram.types import Message as TGMessage
from aiogram.enums import ParseMode
import io
from aiogram.filters import CommandStart
from aiogram.exceptions import TelegramBadRequest
from aiogram.types import BotCommand

from ...dock import Channel

logger = logging.getLogger("tripletriple.channels.telegram")
router = Router()


class TelegramChannel(Channel):
    """
    Telegram channel adapter.

    Uses aiogram for async Telegram Bot API communication.
    Supports both polling and webhook modes.
    """

    def __init__(
        self,
        token: Optional[str] = None,
        on_message=None,
    ):
        self.token = token or os.getenv("TELEGRAM_BOT_TOKEN")
        if not self.token:
            raise ValueError("TELEGRAM_BOT_TOKEN is required")

        self.bot = Bot(token=self.token, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
        self.dp = Dispatcher()
        self.dp.include_router(router)
        self._on_message = on_message

        # Register handlers
        self._register_handlers()

    def _register_handlers(self):
        @self.dp.message(CommandStart())
        async def handle_start(message: TGMessage):
            await message.reply(
                "Hello! I'm your TripleTriple assistant. How can I help you?"
            )

        @self.dp.message()
        async def handle_message(message: TGMessage):
            text = message.text or message.caption or ""
            attachments = []

            # Handle Photos
            if message.photo:
                photo = message.photo[-1]  # Largest size
                f = io.BytesIO()
                await self.bot.download(photo, destination=f)
                attachments.append({
                    "filename": f"photo_{photo.file_id}.jpg",
                    "data": f.getvalue(),
                    "mime": "image/jpeg"
                })

            # Handle Documents
            if message.document:
                doc = message.document
                f = io.BytesIO()
                await self.bot.download(doc, destination=f)
                attachments.append({
                    "filename": doc.file_name or "document",
                    "data": f.getvalue(),
                    "mime": doc.mime_type or "application/octet-stream"
                })

            if not text and not attachments:
                return

            chat_id = str(message.chat.id)
            user_id = str(message.from_user.id) if message.from_user else "unknown"

            logger.info(f"Telegram message from {user_id} in {chat_id}: {text[:50]}... (attachments: {len(attachments)})")

            # Show "typing..." indicator while processing
            await self.bot.send_chat_action(chat_id=message.chat.id, action="typing")

            if self._on_message:
                await self._on_message(
                    channel_name="telegram",
                    chat_id=chat_id,
                    user_id=user_id,
                    text=text,
                    attachments=attachments
                )

    async def start(self):
        """Start the Telegram bot in polling mode."""
        logger.info("Starting Telegram channel (polling mode)")
        # Register bot commands in the menu
        commands = [
            BotCommand(command="start", description="Start the bot"),
            BotCommand(command="model", description="View or switch AI model"),
            BotCommand(command="status", description="Check session status"),
            BotCommand(command="new", description="Start a new session"),
            BotCommand(command="reset", description="Reset current session"),
            BotCommand(command="context", description="View context info"),
            BotCommand(command="help", description="Show help message"),
        ]
        await self.bot.set_my_commands(commands)
        # Remove any existing webhook so polling works
        await self.bot.delete_webhook(drop_pending_updates=True)
        await self.dp.start_polling(self.bot)

    async def stop(self):
        """Stop the Telegram bot."""
        logger.info("Stopping Telegram channel")
        await self.dp.stop_polling()
        await self.bot.session.close()

    async def send_message(self, chat_id: str, text: str):
        """Send a message to a Telegram chat."""
        # Split long messages (Telegram limit: 4096 chars)
        max_len = 4096
        for i in range(0, len(text), max_len):
            chunk = text[i : i + max_len]
            try:
                await self.bot.send_message(chat_id=int(chat_id), text=chunk)
            except TelegramBadRequest as e:
                # If HTML parsing fails, try sending as plain text
                if "can't parse entities" in str(e):
                    logger.warning(f"Telegram HTML parse error: {e}. Falling back to plain text.")
                    await self.bot.send_message(
                        chat_id=int(chat_id),
                        text=chunk,
                        parse_mode=None
                    )
                else:
                    raise e
