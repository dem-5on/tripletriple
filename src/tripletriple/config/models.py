"""
Configuration Models — Full config hierarchy for TripleTriple.

Mirrors OpenClaw's configuration layers:
env vars > tripletriple.json > defaults
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Literal, Optional


# ─── Gateway Config ──────────────────────────────────────────────

class GatewayConfig(BaseModel):
    """HTTP/WS gateway configuration."""
    token: str = Field("", description="Authentication token for the gateway")
    port: int = Field(18789, description="Port to listen on")
    host: str = Field("127.0.0.1", description="Host to bind to")
    cors_origins: List[str] = Field(default_factory=lambda: ["*"])


# ─── Channel Config ──────────────────────────────────────────────

class ChannelConfig(BaseModel):
    """Per-channel configuration."""
    enabled: bool = True
    provider: Literal["telegram", "discord", "whatsapp", "slack"]
    credentials: Dict[str, str] = Field(default_factory=dict)


# ─── Agent Config ────────────────────────────────────────────────

class AgentConfig(BaseModel):
    """Agent runtime configuration."""
    name: str = "TripleTriple"
    system_prompt: str = "You are a helpful AI assistant."
    model: str = "gpt-4-turbo"
    max_tool_iterations: int = Field(10, description="Max ReAct loop iterations")


# ─── Tool Config ─────────────────────────────────────────────────

class ToolConfig(BaseModel):
    """Tool system configuration."""
    profile: str = Field("full", description="Tool profile: full, coding, messaging")
    allow: List[str] = Field(default_factory=list, description="Allowed tools (overrides profile)")
    deny: List[str] = Field(default_factory=list, description="Denied tools")


# ─── Workspace Config ────────────────────────────────────────────

class WorkspaceConfigModel(BaseModel):
    """Workspace directory configuration."""
    root: str = Field("~/.tripletriple/workspace", description="Workspace root directory")
    identity_files: List[str] = Field(
        default_factory=lambda: ["AGENTS.md", "SOUL.md", "PERSONA.md"],
        description="Identity files to scan for system prompt",
    )


# ─── Memory Config ───────────────────────────────────────────────

class MemoryConfig(BaseModel):
    """Memory/RAG backend configuration."""
    backend: str = Field("lancedb", description="Memory backend: lancedb, file, memory")
    db_path: str = Field("~/.tripletriple/state/memory", description="Vector DB storage path")
    table_name: str = Field("memories", description="LanceDB table name")


# ─── Logging Config ──────────────────────────────────────────────

class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: str = Field("INFO", description="Log level")
    format: str = Field(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        description="Log format string",
    )


# ─── Top-Level Config ────────────────────────────────────────────

class TripleTripleConfig(BaseModel):
    """
    Master configuration model — mirrors tripletriple.json.

    Hierarchy: env vars > tripletriple.json > defaults
    """
    gateway: GatewayConfig = Field(default_factory=GatewayConfig)
    channels: List[ChannelConfig] = Field(default_factory=list)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    tools: ToolConfig = Field(default_factory=ToolConfig)
    workspace: WorkspaceConfigModel = Field(default_factory=WorkspaceConfigModel)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
