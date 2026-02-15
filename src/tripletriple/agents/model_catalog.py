"""
Model Catalog & Selection System for TripleTriple.

Mirrors the original TripleTriple model system:
- Provider-based model catalog (select a provider -> see all its models)
- Primary model + fallback chain
- Per-session model override
- Model format: "provider/model" (e.g., "openai/gpt-4o")
- Aliases for quick switching
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from enum import Enum
import os


class ModelCapability(str, Enum):
    TEXT = "text"
    IMAGE_INPUT = "image_input"
    TOOL_USE = "tool_use"
    STREAMING = "streaming"
    VISION = "vision"


class ModelInfo(BaseModel):
    """Metadata about a single model."""
    id: str
    provider: str
    name: str
    alias: Optional[str] = None
    context_window: int = 128000
    max_output_tokens: int = 4096
    capabilities: List[ModelCapability] = [ModelCapability.TEXT, ModelCapability.STREAMING]

    @property
    def full_id(self) -> str:
        return f"{self.provider}/{self.id}"


class ProviderInfo(BaseModel):
    """Metadata about an LLM provider."""
    name: str
    display_name: str
    env_key: str
    models: List[ModelInfo] = []


# ─── Full Model Catalog ──────────────────────────────────────────

MODEL_CATALOG: Dict[str, ProviderInfo] = {
    "openai": ProviderInfo(
        name="openai",
        display_name="OpenAI",
        env_key="OPENAI_API_KEY",
        models=[
            # ── GPT-5 Family (Flagship) ──────────────────────────
            ModelInfo(id="gpt-5.2", provider="openai", name="GPT-5.2", alias="5.2",
                      context_window=1047576, max_output_tokens=65536,
                      capabilities=[ModelCapability.TEXT, ModelCapability.STREAMING, ModelCapability.TOOL_USE, ModelCapability.VISION]),
            ModelInfo(id="gpt-5.2-pro", provider="openai", name="GPT-5.2 Pro", alias="5.2pro",
                      context_window=1047576, max_output_tokens=65536,
                      capabilities=[ModelCapability.TEXT, ModelCapability.STREAMING, ModelCapability.TOOL_USE, ModelCapability.VISION]),
            ModelInfo(id="gpt-5.1", provider="openai", name="GPT-5.1", alias="5.1",
                      context_window=1047576, max_output_tokens=65536,
                      capabilities=[ModelCapability.TEXT, ModelCapability.STREAMING, ModelCapability.TOOL_USE, ModelCapability.VISION]),
            ModelInfo(id="gpt-5", provider="openai", name="GPT-5", alias="5",
                      context_window=1047576, max_output_tokens=65536,
                      capabilities=[ModelCapability.TEXT, ModelCapability.STREAMING, ModelCapability.TOOL_USE, ModelCapability.VISION]),
            ModelInfo(id="gpt-5-pro", provider="openai", name="GPT-5 Pro", alias="5pro",
                      context_window=1047576, max_output_tokens=65536,
                      capabilities=[ModelCapability.TEXT, ModelCapability.STREAMING, ModelCapability.TOOL_USE, ModelCapability.VISION]),
            ModelInfo(id="gpt-5-mini", provider="openai", name="GPT-5 Mini", alias="5m",
                      context_window=1047576, max_output_tokens=32768,
                      capabilities=[ModelCapability.TEXT, ModelCapability.STREAMING, ModelCapability.TOOL_USE, ModelCapability.VISION]),
            ModelInfo(id="gpt-5-nano", provider="openai", name="GPT-5 Nano", alias="5n",
                      context_window=1047576, max_output_tokens=16384,
                      capabilities=[ModelCapability.TEXT, ModelCapability.STREAMING, ModelCapability.TOOL_USE]),

            # ── GPT-5 Codex (Coding-Optimized) ───────────────────
            ModelInfo(id="gpt-5.2-codex", provider="openai", name="GPT-5.2 Codex", alias="5.2cx",
                      context_window=1047576, max_output_tokens=65536,
                      capabilities=[ModelCapability.TEXT, ModelCapability.STREAMING, ModelCapability.TOOL_USE]),
            ModelInfo(id="gpt-5.1-codex", provider="openai", name="GPT-5.1 Codex", alias="5.1cx",
                      context_window=1047576, max_output_tokens=65536,
                      capabilities=[ModelCapability.TEXT, ModelCapability.STREAMING, ModelCapability.TOOL_USE]),
            ModelInfo(id="gpt-5.1-codex-max", provider="openai", name="GPT-5.1 Codex Max", alias="5.1cxmax",
                      context_window=1047576, max_output_tokens=65536,
                      capabilities=[ModelCapability.TEXT, ModelCapability.STREAMING, ModelCapability.TOOL_USE]),
            ModelInfo(id="gpt-5.1-codex-mini", provider="openai", name="GPT-5.1 Codex Mini", alias="5.1cxm",
                      context_window=1047576, max_output_tokens=32768,
                      capabilities=[ModelCapability.TEXT, ModelCapability.STREAMING, ModelCapability.TOOL_USE]),
            ModelInfo(id="gpt-5-codex", provider="openai", name="GPT-5 Codex", alias="5cx",
                      context_window=1047576, max_output_tokens=65536,
                      capabilities=[ModelCapability.TEXT, ModelCapability.STREAMING, ModelCapability.TOOL_USE]),

            # ── GPT-4.1 Family ───────────────────────────────────
            ModelInfo(id="gpt-4.1", provider="openai", name="GPT-4.1", alias="4.1",
                      context_window=1047576, max_output_tokens=32768,
                      capabilities=[ModelCapability.TEXT, ModelCapability.STREAMING, ModelCapability.TOOL_USE, ModelCapability.VISION]),
            ModelInfo(id="gpt-4.1-mini", provider="openai", name="GPT-4.1 Mini", alias="4.1m",
                      context_window=1047576, max_output_tokens=32768,
                      capabilities=[ModelCapability.TEXT, ModelCapability.STREAMING, ModelCapability.TOOL_USE, ModelCapability.VISION]),
            ModelInfo(id="gpt-4.1-nano", provider="openai", name="GPT-4.1 Nano", alias="4.1n",
                      context_window=1047576, max_output_tokens=32768,
                      capabilities=[ModelCapability.TEXT, ModelCapability.STREAMING, ModelCapability.TOOL_USE]),

            # ── GPT-4o Family ────────────────────────────────────
            ModelInfo(id="gpt-4o", provider="openai", name="GPT-4o", alias="4o",
                      context_window=128000, max_output_tokens=16384,
                      capabilities=[ModelCapability.TEXT, ModelCapability.STREAMING, ModelCapability.TOOL_USE, ModelCapability.VISION]),
            ModelInfo(id="gpt-4o-mini", provider="openai", name="GPT-4o Mini", alias="4om",
                      context_window=128000, max_output_tokens=16384,
                      capabilities=[ModelCapability.TEXT, ModelCapability.STREAMING, ModelCapability.TOOL_USE, ModelCapability.VISION]),
            ModelInfo(id="gpt-4o-search-preview", provider="openai", name="GPT-4o Search Preview", alias="4os",
                      context_window=128000, max_output_tokens=16384,
                      capabilities=[ModelCapability.TEXT, ModelCapability.STREAMING, ModelCapability.TOOL_USE, ModelCapability.VISION]),
            ModelInfo(id="gpt-4o-mini-search-preview", provider="openai", name="GPT-4o Mini Search", alias="4oms",
                      context_window=128000, max_output_tokens=16384,
                      capabilities=[ModelCapability.TEXT, ModelCapability.STREAMING, ModelCapability.TOOL_USE]),

            # ── GPT-4 Turbo ──────────────────────────────────────
            ModelInfo(id="gpt-4-turbo", provider="openai", name="GPT-4 Turbo", alias="4t",
                      context_window=128000, max_output_tokens=4096,
                      capabilities=[ModelCapability.TEXT, ModelCapability.STREAMING, ModelCapability.TOOL_USE, ModelCapability.VISION]),

            # ── Reasoning Models (o-series) ──────────────────────
            ModelInfo(id="o3", provider="openai", name="o3", alias="o3",
                      context_window=200000, max_output_tokens=100000,
                      capabilities=[ModelCapability.TEXT, ModelCapability.STREAMING, ModelCapability.TOOL_USE, ModelCapability.VISION]),
            ModelInfo(id="o3-pro", provider="openai", name="o3 Pro", alias="o3pro",
                      context_window=200000, max_output_tokens=100000,
                      capabilities=[ModelCapability.TEXT, ModelCapability.STREAMING, ModelCapability.TOOL_USE, ModelCapability.VISION]),
            ModelInfo(id="o3-mini", provider="openai", name="o3 Mini", alias="o3m",
                      context_window=200000, max_output_tokens=100000,
                      capabilities=[ModelCapability.TEXT, ModelCapability.STREAMING, ModelCapability.TOOL_USE]),
            ModelInfo(id="o4-mini", provider="openai", name="o4 Mini", alias="o4m",
                      context_window=200000, max_output_tokens=100000,
                      capabilities=[ModelCapability.TEXT, ModelCapability.STREAMING, ModelCapability.TOOL_USE, ModelCapability.VISION]),

            # ── Deep Research Models ─────────────────────────────
            ModelInfo(id="o3-deep-research", provider="openai", name="o3 Deep Research", alias="o3dr",
                      context_window=200000, max_output_tokens=100000,
                      capabilities=[ModelCapability.TEXT, ModelCapability.STREAMING, ModelCapability.TOOL_USE, ModelCapability.VISION]),
            ModelInfo(id="o4-mini-deep-research", provider="openai", name="o4 Mini Deep Research", alias="o4mdr",
                      context_window=200000, max_output_tokens=100000,
                      capabilities=[ModelCapability.TEXT, ModelCapability.STREAMING, ModelCapability.TOOL_USE, ModelCapability.VISION]),

            # ── Specialized ──────────────────────────────────────
            ModelInfo(id="computer-use-preview", provider="openai", name="Computer Use Preview", alias="cup",
                      context_window=128000, max_output_tokens=16384,
                      capabilities=[ModelCapability.TEXT, ModelCapability.STREAMING, ModelCapability.TOOL_USE, ModelCapability.VISION]),

            # ── Open-Weight Models ───────────────────────────────
            ModelInfo(id="gpt-oss-120b", provider="openai", name="GPT-OSS 120B", alias="oss120",
                      context_window=128000, max_output_tokens=16384,
                      capabilities=[ModelCapability.TEXT, ModelCapability.STREAMING]),
            ModelInfo(id="gpt-oss-20b", provider="openai", name="GPT-OSS 20B", alias="oss20",
                      context_window=128000, max_output_tokens=16384,
                      capabilities=[ModelCapability.TEXT, ModelCapability.STREAMING]),
        ],
    ),
    "anthropic": ProviderInfo(
        name="anthropic",
        display_name="Anthropic",
        env_key="ANTHROPIC_API_KEY",
        models=[
            # ── Claude 4.6 (Latest Flagship) ─────────────────────
            ModelInfo(id="claude-opus-4-6", provider="anthropic", name="Claude Opus 4.6", alias="opus4.6",
                      context_window=1000000, max_output_tokens=32768,
                      capabilities=[ModelCapability.TEXT, ModelCapability.STREAMING, ModelCapability.TOOL_USE, ModelCapability.VISION]),

            # ── Claude 4.5 Family ────────────────────────────────
            ModelInfo(id="claude-opus-4-5-20251101", provider="anthropic", name="Claude Opus 4.5", alias="opus4.5",
                      context_window=200000, max_output_tokens=16384,
                      capabilities=[ModelCapability.TEXT, ModelCapability.STREAMING, ModelCapability.TOOL_USE, ModelCapability.VISION]),
            ModelInfo(id="claude-sonnet-4-5-20250929", provider="anthropic", name="Claude Sonnet 4.5", alias="sonnet",
                      context_window=200000, max_output_tokens=16384,
                      capabilities=[ModelCapability.TEXT, ModelCapability.STREAMING, ModelCapability.TOOL_USE, ModelCapability.VISION]),
            ModelInfo(id="claude-haiku-4-5-20251001", provider="anthropic", name="Claude Haiku 4.5", alias="haiku4.5",
                      context_window=200000, max_output_tokens=8192,
                      capabilities=[ModelCapability.TEXT, ModelCapability.STREAMING, ModelCapability.TOOL_USE, ModelCapability.VISION]),

            # ── Claude 4 Family ──────────────────────────────────
            ModelInfo(id="claude-opus-4-1-20250805", provider="anthropic", name="Claude Opus 4.1", alias="opus4.1",
                      context_window=200000, max_output_tokens=16384,
                      capabilities=[ModelCapability.TEXT, ModelCapability.STREAMING, ModelCapability.TOOL_USE, ModelCapability.VISION]),
            ModelInfo(id="claude-sonnet-4-20250514", provider="anthropic", name="Claude Sonnet 4", alias="sonnet4",
                      context_window=200000, max_output_tokens=16384,
                      capabilities=[ModelCapability.TEXT, ModelCapability.STREAMING, ModelCapability.TOOL_USE, ModelCapability.VISION]),
            ModelInfo(id="claude-opus-4-20250514", provider="anthropic", name="Claude Opus 4", alias="opus4",
                      context_window=200000, max_output_tokens=16384,
                      capabilities=[ModelCapability.TEXT, ModelCapability.STREAMING, ModelCapability.TOOL_USE, ModelCapability.VISION]),

            # ── Claude 3.5 (Legacy) ──────────────────────────────
            ModelInfo(id="claude-3-5-sonnet-20241022", provider="anthropic", name="Claude 3.5 Sonnet", alias="3.5sonnet",
                      context_window=200000, max_output_tokens=8192,
                      capabilities=[ModelCapability.TEXT, ModelCapability.STREAMING, ModelCapability.TOOL_USE, ModelCapability.VISION]),
            ModelInfo(id="claude-3-5-haiku-20241022", provider="anthropic", name="Claude 3.5 Haiku", alias="haiku",
                      context_window=200000, max_output_tokens=8192,
                      capabilities=[ModelCapability.TEXT, ModelCapability.STREAMING, ModelCapability.TOOL_USE]),
        ],
    ),
    "gemini": ProviderInfo(
        name="gemini",
        display_name="Google Gemini",
        env_key="GEMINI_API_KEY",
        models=[
            # ── Gemini 3 Family (Flagship) ───────────────────────
            ModelInfo(id="gemini-3-pro-preview", provider="gemini", name="Gemini 3 Pro", alias="3pro",
                      context_window=1048576, max_output_tokens=65536,
                      capabilities=[ModelCapability.TEXT, ModelCapability.STREAMING, ModelCapability.TOOL_USE, ModelCapability.VISION]),
            ModelInfo(id="gemini-3-pro-image-preview", provider="gemini", name="Gemini 3 Pro Image", alias="3proimg",
                      context_window=65536, max_output_tokens=32768,
                      capabilities=[ModelCapability.TEXT, ModelCapability.STREAMING, ModelCapability.IMAGE_INPUT, ModelCapability.VISION]),
            ModelInfo(id="gemini-3-flash-preview", provider="gemini", name="Gemini 3 Flash", alias="3flash",
                      context_window=1048576, max_output_tokens=65536,
                      capabilities=[ModelCapability.TEXT, ModelCapability.STREAMING, ModelCapability.TOOL_USE, ModelCapability.VISION]),

            # ── Gemini 2.5 Family ────────────────────────────────
            ModelInfo(id="gemini-2.5-pro", provider="gemini", name="Gemini 2.5 Pro", alias="2.5pro",
                      context_window=1048576, max_output_tokens=65536,
                      capabilities=[ModelCapability.TEXT, ModelCapability.STREAMING, ModelCapability.TOOL_USE, ModelCapability.VISION]),
            ModelInfo(id="gemini-2.5-flash", provider="gemini", name="Gemini 2.5 Flash", alias="2.5flash",
                      context_window=1048576, max_output_tokens=65536,
                      capabilities=[ModelCapability.TEXT, ModelCapability.STREAMING, ModelCapability.TOOL_USE, ModelCapability.VISION]),
            ModelInfo(id="gemini-2.5-flash-image", provider="gemini", name="Gemini 2.5 Flash Image", alias="2.5flashimg",
                      context_window=65536, max_output_tokens=32768,
                      capabilities=[ModelCapability.TEXT, ModelCapability.STREAMING, ModelCapability.IMAGE_INPUT, ModelCapability.VISION]),
            ModelInfo(id="gemini-2.5-flash-lite", provider="gemini", name="Gemini 2.5 Flash-Lite", alias="2.5flashlite",
                      context_window=1048576, max_output_tokens=65536,
                      capabilities=[ModelCapability.TEXT, ModelCapability.STREAMING, ModelCapability.TOOL_USE, ModelCapability.VISION]),

            # ── Gemini 2.0 (Deprecated — shutdown March 2026) ────
            ModelInfo(id="gemini-2.0-flash", provider="gemini", name="Gemini 2.0 Flash ⚠️", alias="2flash",
                      context_window=1048576, max_output_tokens=8192,
                      capabilities=[ModelCapability.TEXT, ModelCapability.STREAMING, ModelCapability.TOOL_USE, ModelCapability.VISION]),
            ModelInfo(id="gemini-2.0-flash-lite", provider="gemini", name="Gemini 2.0 Flash Lite ⚠️", alias="2flashlite",
                      context_window=1048576, max_output_tokens=8192,
                      capabilities=[ModelCapability.TEXT, ModelCapability.STREAMING, ModelCapability.TOOL_USE]),
        ],
    ),
}


# ─── Model Selection Engine ──────────────────────────────────────

class ModelSelection(BaseModel):
    """Tracks the user's model selection and fallback chain."""
    primary: str = Field(default_factory=lambda: os.getenv("TRIPLETREBLE_MODEL", "openai/gpt-5.2"))
    fallbacks: List[str] = ["openai/gpt-5-mini", "anthropic/claude-sonnet-4-5-20250929", "gemini/gemini-2.5-flash"]
    image_model: Optional[str] = "openai/gpt-4o"
    aliases: Dict[str, str] = {}


class ModelSelector:
    """
    Handles model resolution, catalog browsing, and selection.

    Resolution chain: Config model -> Session override -> Fallback chain -> Error
    """

    def __init__(self, selection: ModelSelection = None):
        self.selection = selection or ModelSelection()
        self.catalog = MODEL_CATALOG

    def list_providers(self) -> List[ProviderInfo]:
        return list(self.catalog.values())

    def list_models(self, provider: str = None) -> List[ModelInfo]:
        models = []
        for p_name, p_info in self.catalog.items():
            if provider and p_name != provider:
                continue
            models.extend(p_info.models)
        return models

    def get_model(self, ref: str) -> Optional[ModelInfo]:
        """
        Resolve a model reference. Accepts:
        - Full ref: "openai/gpt-4o"
        - Alias: "4o"
        - Bare model ID: "gpt-4o"
        """
        # User aliases first
        if ref in self.selection.aliases:
            ref = self.selection.aliases[ref]

        # Catalog aliases
        for provider in self.catalog.values():
            for model in provider.models:
                if model.alias == ref:
                    return model

        # Full ref: provider/model
        if "/" in ref:
            provider_name, model_id = ref.split("/", 1)
            provider = self.catalog.get(provider_name)
            if provider:
                for model in provider.models:
                    if model.id == model_id:
                        return model

        # Bare model ID
        for provider in self.catalog.values():
            for model in provider.models:
                if model.id == ref:
                    return model

        return None

    def get_primary(self) -> Optional[ModelInfo]:
        return self.get_model(self.selection.primary)

    def get_fallback_chain(self) -> List[ModelInfo]:
        chain = []
        primary = self.get_primary()
        if primary:
            chain.append(primary)
        for fb_ref in self.selection.fallbacks:
            model = self.get_model(fb_ref)
            if model:
                chain.append(model)
        return chain

    def set_model(self, ref: str) -> Optional[ModelInfo]:
        model = self.get_model(ref)
        if model:
            self.selection.primary = model.full_id
        return model

    def add_fallback(self, ref: str) -> Optional[ModelInfo]:
        model = self.get_model(ref)
        if model and model.full_id not in self.selection.fallbacks:
            self.selection.fallbacks.append(model.full_id)
        return model

    def remove_fallback(self, ref: str) -> bool:
        model = self.get_model(ref)
        if model and model.full_id in self.selection.fallbacks:
            self.selection.fallbacks.remove(model.full_id)
            return True
        return False

    def add_alias(self, alias: str, ref: str) -> Optional[ModelInfo]:
        model = self.get_model(ref)
        if model:
            self.selection.aliases[alias] = model.full_id
        return model

    def format_model_list(self, provider: str = None) -> str:
        """Numbered model list grouped by provider (like /model list)."""
        output = ""
        idx = 1
        for p_name, p_info in self.catalog.items():
            if provider and p_name != provider:
                continue
            output += f"\n{'─' * 45}\n  {p_info.display_name}\n{'─' * 45}\n"
            for model in p_info.models:
                marker = " ◆" if model.full_id == self.selection.primary else "  "
                alias_str = f" ({model.alias})" if model.alias else ""
                caps = ", ".join(c.value for c in model.capabilities if c != ModelCapability.TEXT)
                output += (
                    f"{marker} {idx:>2}. {model.name}{alias_str}\n"
                    f"      {model.full_id}  [{caps}]  ctx:{model.context_window:,}\n"
                )
                idx += 1
        return output

    def format_status(self) -> str:
        primary = self.get_primary()
        output = "Model Status:\n\n"
        output += f"  Primary: {self.selection.primary}"
        output += f" ({primary.name})\n" if primary else " (not found!)\n"
        output += "\n  Fallbacks:\n"
        for i, fb in enumerate(self.selection.fallbacks, 1):
            model = self.get_model(fb)
            name = model.name if model else "unknown"
            output += f"    {i}. {fb} ({name})\n"
        if self.selection.aliases:
            output += "\n  Aliases:\n"
            for alias, ref in self.selection.aliases.items():
                output += f"    {alias} -> {ref}\n"
        return output
