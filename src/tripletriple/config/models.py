from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Literal

class GatewayConfig(BaseModel):
    token: str = Field(..., description="Authentication token for the gateway")
    port: int = Field(18789, description="Port to listen on")
    host: str = Field("127.0.0.1", description="Host to bind to")

class ChannelConfig(BaseModel):
    enabled: bool = True
    provider: Literal["telegram", "discord", "whatsapp", "slack"]
    credentials: Dict[str, str]

class AgentConfig(BaseModel):
    name: str = "TripleTriple"
    system_prompt: str = "You are a helpful AI assistant."
    model: str = "gpt-4-turbo"

class TripleTripleConfig(BaseModel):
    gateway: GatewayConfig
    channels: List[ChannelConfig] = []
    agent: AgentConfig = AgentConfig()
