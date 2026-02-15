"""
Configuration Manager

Handles reading and writing configuration to the .env file.
Used by the onboarding wizard to save user credentials.
"""

import os
from pathlib import Path
from typing import Dict, Optional, List

from dotenv import dotenv_values, set_key


class ConfigManager:
    def __init__(self, env_path: Optional[Path] = None):
        if env_path:
            self.env_path = env_path
        else:
            # Default to src/tripletriple/.env (where the code expects it)
            # But checking if we are in editable mode, maybe finding the root properties
            # Current structure: src/tripletriple/.env
            current = Path(__file__).resolve().parent  # src/tripletriple/config
            self.env_path = current.parent / ".env"

    def get_all(self) -> Dict[str, Optional[str]]:
        """Return all config as a dictionary."""
        if not self.env_path.exists():
            return {}
        return dotenv_values(self.env_path)

    def get(self, key: str) -> Optional[str]:
        """Get a specific config value."""
        return os.getenv(key) or self.get_all().get(key)

    def set(self, key: str, value: str):
        """Write a config value to the .env file."""
        if not self.env_path.exists():
            self.env_path.touch()
        
        # set_key handles quoting and updating existing keys
        set_key(str(self.env_path), key, value, quote_mode="never")
        
        # Also update os.environ for the current process
        os.environ[key] = value

    def is_configured(self) -> bool:
        """Check if minimum required configuration exists."""
        # We need at least one LLM key
        keys = ["GEMINI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"]
        current = self.get_all()
        return any(current.get(k) for k in keys)

    def get_path(self) -> Path:
        return self.env_path
