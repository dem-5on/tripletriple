import os
import time
import logging
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union
from ..gateway.session import SessionManager, ChatCommandHandler
from ..session.models import Message
from ..agents.base import Agent

logger = logging.getLogger("tripletriple.channels.dock")

class Channel(ABC):
    @abstractmethod
    async def start(self):
        """Start receiving messages."""
        pass

    @abstractmethod
    async def stop(self):
        """Stop the channel."""
        pass

    @abstractmethod
    async def send_message(self, chat_id: str, text: str):
        """Send a message to a chat."""
        pass

class ChannelDock:
    def __init__(
        self,
        session_manager: SessionManager,
        agent: Agent,
        chat_commands: ChatCommandHandler = None,
    ):
        self.session_manager = session_manager
        self.agent = agent
        self.chat_commands = chat_commands
        self.channels: Dict[str, Channel] = {}

    def _save_attachment(self, session_id: str, att: Dict[str, Any]) -> str:
        """Save attachment to disk and return absolute path."""
        base = Path(os.path.expanduser("~/.tripletriple/data/uploads"))
        sess_dir = base / session_id
        sess_dir.mkdir(parents=True, exist_ok=True)
        
        fname = att.get("filename") or "unknown"
        ts = int(time.time() * 1000)
        final_name = f"{ts}_{fname}"
        
        path = sess_dir / final_name
        path.write_bytes(att["data"])
        return str(path)

    def register_channel(self, name: str, channel: Channel):
        self.channels[name] = channel

    async def handle_incoming_message(
        self,
        channel_name: str,
        chat_id: str,
        user_id: str,
        text: str,
        attachments: List[Dict[str, Any]] = None,
    ):
        """
        Central point for all incoming messages.
        """
        session = self.session_manager.get_session(f"{channel_name}:{chat_id}", user_id)
        
        # Determine channel to reply to
        channel = self.channels.get(channel_name)
        if not channel:
            print(f"Error: Channel {channel_name} not found")
            return

        # Handle Chat Commands (e.g. /model, /status)
        if self.chat_commands and self.chat_commands.is_command(text):
            response_text = self.chat_commands.handle(text, session)
            if response_text:
                await channel.send_message(chat_id, response_text)
                return
        
        # Build structured content if attachments exist
        if attachments:
            content = []
            if text:
                content.append({"type": "text", "text": text})
            
            for att in attachments:
                path = self._save_attachment(session.id, att)
                content.append({
                    "type": "image" if att.get("mime", "").startswith("image/") else "file",
                    "path": path,
                    "mime_type": att.get("mime", "application/octet-stream")
                })
            
            # Use structured list
            session.add_message("user", content)
            self.session_manager.write_transcript(session, session.messages[-1])
            message_payload = content
        else:
            # Use simple string
            session.add_message("user", text)
            self.session_manager.write_transcript(session, session.messages[-1])
            message_payload = text
        
        # Determine channel to reply to


        # Process with Agent
        response_text = ""
        async for chunk in self.agent.process_message(session, message_payload):
            response_text += chunk
            # TODO: Stream to channel if supported
        
        session.add_message("assistant", response_text)
        self.session_manager.write_transcript(session, session.messages[-1])
        self.session_manager.save()
        await channel.send_message(chat_id, response_text)

    async def send_outbound(self, channel: str, recipient_id: str, text: str, session: Any = None):
        """
        Send a proactive message (outbound) to a channel.
        For persistent channels (Telegram/Discord), this works if the channel implements send_message.
        For WebSocket, we might need the socket object (which we don't store here yet).
        """
        chan = self.channels.get(channel)
        if not chan:
            print(f"Warning: Channel {channel} not found for outbound message")
            return

        # Special handling for WebSocket? 
        # For now, just try send_message if the channel supports it.
        # Most adapters (Telegram, Discord) take (chat_id, text).
        try:
            await chan.send_message(recipient_id, text)
        except Exception as e:
            print(f"Error sending outbound to {channel}/{recipient_id}: {e}")
