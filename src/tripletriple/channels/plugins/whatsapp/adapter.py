"""
WhatsApp Channel Adapter using webhook-based approach.

Uses a FastAPI router to receive incoming webhook messages from a
WhatsApp Business API provider and sends replies via HTTP.
"""

import os
import logging
from typing import Optional
import httpx

from ...dock import Channel

logger = logging.getLogger("tripletriple.channels.whatsapp")


class WhatsAppChannel(Channel):
    """
    WhatsApp channel adapter using webhooks.

    Works with WhatsApp Business API providers (Meta Cloud API,
    GreenAPI, Twilio, etc.) by receiving webhooks and sending
    messages via their HTTP API.
    """

    def __init__(
        self,
        api_url: Optional[str] = None,
        api_token: Optional[str] = None,
        phone_number_id: Optional[str] = None,
        on_message=None,
    ):
        self.api_url = api_url or os.getenv(
            "WHATSAPP_API_URL",
            "https://graph.facebook.com/v18.0",
        )
        self.api_token = api_token or os.getenv("WHATSAPP_API_TOKEN")
        self.phone_number_id = phone_number_id or os.getenv("WHATSAPP_PHONE_NUMBER_ID")
        self._on_message = on_message

        if not self.api_token:
            logger.warning("WHATSAPP_API_TOKEN is not set")

    async def start(self):
        """
        Start receiving messages.

        This channel works via webhooks — the start method is a no-op.
        Register the webhook router with your FastAPI app instead.
        """
        logger.info(
            "WhatsApp channel started — register the webhook router to receive messages"
        )

    async def stop(self):
        """Stop the WhatsApp channel."""
        logger.info("WhatsApp channel stopped")

    async def send_message(self, chat_id: str, text: str):
        """
        Send a text message via the WhatsApp Business API.

        Args:
            chat_id: The recipient phone number (with country code, no +)
            text: The message text
        """
        url = f"{self.api_url}/{self.phone_number_id}/messages"
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
        }
        payload = {
            "messaging_product": "whatsapp",
            "to": chat_id,
            "type": "text",
            "text": {"body": text},
        }

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(url, json=payload, headers=headers, timeout=10)
                resp.raise_for_status()
                logger.info(f"WhatsApp message sent to {chat_id}")
        except Exception as e:
            logger.error(f"Failed to send WhatsApp message: {e}")

    async def handle_webhook(self, body: dict):
        """
        Process an incoming webhook payload from Meta Cloud API.

        Call this from your FastAPI webhook endpoint.
        """
        try:
            for entry in body.get("entry", []):
                for change in entry.get("changes", []):
                    value = change.get("value", {})
                    for message in value.get("messages", []):
                        if message.get("type") == "text":
                            chat_id = message["from"]
                            text = message["text"]["body"]
                            user_id = message["from"]

                            logger.info(
                                f"WhatsApp message from {user_id}: {text[:50]}..."
                            )

                            if self._on_message:
                                await self._on_message(
                                    channel_name="whatsapp",
                                    chat_id=chat_id,
                                    user_id=user_id,
                                    text=text,
                                )
        except Exception as e:
            logger.error(f"Error processing WhatsApp webhook: {e}")
