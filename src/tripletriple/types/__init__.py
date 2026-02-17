"""
Shared type aliases and constants used across TripleTriple modules.
"""

from typing import Any, Dict, List, Union

# ─── Type Aliases ─────────────────────────────────────────────────

# Message content can be a plain string or a list of multimodal parts
MessageContent = Union[str, List[Any]]

# Tool parameters are always a string→any dictionary
ToolParams = Dict[str, Any]

# Tool result can be a string or structured data
ToolResult = Union[str, Dict[str, Any]]

# Provider name literals
ProviderName = str  # "openai", "anthropic", "gemini"

# Session key is always a string
SessionKey = str
