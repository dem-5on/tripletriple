"""
Configuration Loader — Priority-based config resolution.

Loading priority (highest wins):
  1. TRIPLETRIPLE_CONFIG_PATH env var (explicit path)
  2. ~/.tripletriple/tripletriple.json (default location)
  3. Built-in defaults
"""

import os
import json
import logging
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from .models import TripleTripleConfig

logger = logging.getLogger("tripletriple.config.loader")

# Default config file locations (checked in order)
_DEFAULT_CONFIG_PATHS = [
    "~/.tripletriple/tripletriple.json",
    "~/.tripletriple/config.json",
]


def _resolve_config_path(explicit_path: Optional[str] = None) -> Optional[Path]:
    """
    Resolve configuration file path using priority chain:
    1. Explicit path argument
    2. TRIPLETRIPLE_CONFIG_PATH env var
    3. Default locations (~/.tripletriple/tripletriple.json)
    """
    # Priority 1: Explicit argument
    if explicit_path:
        p = Path(explicit_path).expanduser()
        if p.exists():
            return p
        logger.warning(f"Explicit config path not found: {explicit_path}")

    # Priority 2: Environment variable
    env_path = os.getenv("TRIPLETRIPLE_CONFIG_PATH")
    if env_path:
        p = Path(env_path).expanduser()
        if p.exists():
            return p
        logger.warning(f"TRIPLETRIPLE_CONFIG_PATH not found: {env_path}")

    # Priority 3: Default locations
    for default in _DEFAULT_CONFIG_PATHS:
        p = Path(default).expanduser()
        if p.exists():
            logger.info(f"Found config at default location: {p}")
            return p

    return None


def _load_file_config(path: Path) -> dict:
    """Load config data from a JSON or YAML file."""
    try:
        with open(path, "r") as f:
            if path.suffix in [".yaml", ".yml"]:
                import yaml
                return yaml.safe_load(f) or {}
            elif path.suffix == ".json":
                return json.load(f)
            else:
                logger.warning(f"Unsupported config format: {path.suffix}")
                return {}
    except Exception as e:
        logger.error(f"Failed to load config from {path}: {e}")
        return {}


def load_config(config_path: Optional[str] = None) -> TripleTripleConfig:
    """
    Load configuration with priority: env vars > config file > defaults.
    """
    load_dotenv()

    # Start with env-var overrides for gateway
    config_data = {
        "gateway": {
            "token": os.getenv("OPENCLAW_GATEWAY_TOKEN", "change-me"),
            "port": int(os.getenv("PORT", "18789")),
            "host": os.getenv("HOST", "127.0.0.1"),
        },
        "channels": [],
        "agent": {},
    }

    # Load from config file (if found)
    resolved_path = _resolve_config_path(config_path)
    if resolved_path:
        file_data = _load_file_config(resolved_path)
        if file_data:
            # Merge: file config fills in values not set by env vars
            for key, value in file_data.items():
                if key in config_data and isinstance(config_data[key], dict) and isinstance(value, dict):
                    # Shallow merge for dict sections
                    merged = {**value}  # file values as base
                    merged.update({k: v for k, v in config_data[key].items()
                                   if v != config_data.get(key, {}).get(k)})
                    config_data[key] = merged
                else:
                    config_data[key] = value
            logger.info(f"Loaded config from: {resolved_path}")
    else:
        logger.info("No config file found, using defaults + env vars")

    return TripleTripleConfig(**config_data)


# Singleton instance
current_config = load_config()
