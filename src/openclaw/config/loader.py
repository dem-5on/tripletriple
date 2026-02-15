import os
import json
import yaml
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from .models import OpenClawConfig

def load_config(config_path: Optional[str] = None) -> OpenClawConfig:
    load_dotenv()
    
    # Defaults
    config_data = {
        "gateway": {
            "token": os.getenv("OPENCLAW_GATEWAY_TOKEN", "change-me"),
            "port": int(os.getenv("PORT", 18789)),
            "host": os.getenv("HOST", "127.0.0.1"),
        },
        "channels": [],
        "agent": {}
    }

    # Load from file if provided
    if config_path:
        path = Path(config_path)
        if path.exists():
            with open(path, "r") as f:
                if path.suffix in [".yaml", ".yml"]:
                    file_data = yaml.safe_load(f)
                elif path.suffix == ".json":
                    file_data = json.load(f)
                else:
                    raise ValueError("Unsupported config format")
                
                # Simple merge (deep merge would be better in prod)
                config_data.update(file_data)
    
    return OpenClawConfig(**config_data)

# Singleton instance
current_config = load_config()
