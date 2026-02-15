from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Type

class Tool(ABC):
    name: str
    description: str
    args_schema: Type[BaseModel]
    needs_context: bool = False  # Set True for tools needing SessionManager etc.

    @abstractmethod
    async def run(self, **kwargs) -> Any:
        pass

    def to_schema(self) -> Dict[str, Any]:
        """
        Convert tool to OpenAI-compatible function schema.
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.args_schema.model_json_schema()
            }
        }

class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool):
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        return self._tools.get(name)

    def get_all_schemas(self) -> list[Dict[str, Any]]:
        return [t.to_schema() for t in self._tools.values()]

    def list_names(self) -> List[str]:
        return list(self._tools.keys())

    def get_all(self) -> Dict[str, Tool]:
        return dict(self._tools)
