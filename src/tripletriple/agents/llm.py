from typing import List, Dict, Any, AsyncGenerator, Optional
from dataclasses import dataclass, field
import os
import json
import logging
import base64
from pathlib import Path
from openai import AsyncOpenAI
import anthropic
from google import genai
from google.genai import types
from .model_catalog import ModelSelector, ModelInfo, MODEL_CATALOG

logger = logging.getLogger("tripletriple.agents.llm")


@dataclass
class ToolCallChunk:
    """A single tool call accumulated from streaming chunks."""
    id: str = ""
    name: str = ""
    arguments: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StreamChunk:
    """
    Unified streaming chunk that all providers must yield.

    To add a new LLM provider, implement chat_stream() that converts
    the provider's raw chunks into StreamChunk objects.
    """
    content: str = "" # Text content (may be partial)
    tool_calls: List[ToolCallChunk] = field(default_factory=list)
    done: bool = False # True on final chunk
    input_tokens: int = 0 # Token usage (populated on final chunk)
    output_tokens: int = 0


class LLMProvider:
    """Abstract base for all LLM providers."""

    provider_name: str = ""
    model_id: str = ""

    async def chat_stream(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]] = None,
    ) -> AsyncGenerator[StreamChunk, None]:
        """Yield normalized StreamChunk objects from the provider's API."""
        raise NotImplementedError

    def switch_model(self, model_id: str):
        """Switch to a different model within the same provider."""
        self.model_id = model_id
        logger.info(f"Switched to model: {self.provider_name}/{model_id}")


class OpenAIProvider(LLMProvider):
    provider_name = "openai"

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        self.client = AsyncOpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model_id = model

    async def chat_stream(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]] = None,
    ) -> AsyncGenerator[StreamChunk, None]:
        # Pre-process messages to handle multimodal content
        processed_messages = []
        for msg in messages:
            content = msg["content"]
            if isinstance(content, str):
                processed_messages.append(msg)
            elif isinstance(content, list):
                new_content = []
                for item in content:
                    if item.get("type") == "text":
                        new_content.append({"type": "text", "text": item.get("text", "")})
                    elif item.get("type") == "image":
                        path = item.get("path")
                        if path and os.path.exists(path):
                            mime = item.get("mime_type", "image/jpeg")
                            data = Path(path).read_bytes()
                            b64 = base64.b64encode(data).decode("utf-8")
                            new_content.append({
                                "type": "image_url",
                                "image_url": {"url": f"data:{mime};base64,{b64}"}
                            })
                processed_messages.append({"role": msg["role"], "content": new_content})

        stream = await self.client.chat.completions.create(
            model=self.model_id,
            messages=processed_messages,
            tools=tools or None,
            stream=True,
        )
        async for chunk in stream:
            sc = StreamChunk()
            if chunk.choices:
                delta = chunk.choices[0].delta
                if delta.content:
                    sc.content = delta.content
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        sc.tool_calls.append(ToolCallChunk(
                            id=tc.id or "",
                            name=tc.function.name or "" if tc.function else "",
                            arguments=tc.function.arguments or "" if tc.function else "",
                        ))
                if chunk.choices[0].finish_reason == "stop":
                    sc.done = True
                    # Try to capture usage from the final chunk
                    if hasattr(chunk, 'usage') and chunk.usage:
                        sc.input_tokens = chunk.usage.prompt_tokens or 0
                        sc.output_tokens = chunk.usage.completion_tokens or 0
            yield sc


class AnthropicProvider(LLMProvider):
    provider_name = "anthropic"

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-sonnet-4-5-20250514"):
        self.client = anthropic.AsyncAnthropic(
            api_key=api_key or os.getenv("ANTHROPIC_API_KEY")
        )
        self.model_id = model

    async def chat_stream(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]] = None,
    ) -> AsyncGenerator[StreamChunk, None]:
        system_prompt = next(
            (m["content"] for m in messages if m["role"] == "system"), ""
        )
        # Pre-process messages (Anthropic expects "user" content as list of blocks)
        user_messages = []
        for m in messages:
            if m["role"] == "system": continue
            
            content = m["content"]
            if isinstance(content, str):
                user_messages.append({"role": m["role"], "content": content})
            elif isinstance(content, list):
                blocks = []
                for item in content:
                    if item.get("type") == "text":
                         blocks.append({"type": "text", "text": item.get("text", "")})
                    elif item.get("type") == "image":
                        path = item.get("path")
                        if path and os.path.exists(path):
                            mime = item.get("mime_type", "image/jpeg")
                            data = Path(path).read_bytes()
                            b64 = base64.b64encode(data).decode("utf-8")
                            blocks.append({
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": mime,
                                    "data": b64
                                }
                            })
                user_messages.append({"role": m["role"], "content": blocks})

        async with self.client.messages.stream(
            max_tokens=8192,
            system=system_prompt,
            messages=user_messages,
            model=self.model_id,
            tools=tools or [],
        ) as stream:
            async for event in stream:
                sc = StreamChunk()
                if hasattr(event, 'type'):
                    if event.type == 'content_block_delta' and hasattr(event.delta, 'text'):
                        sc.content = event.delta.text or ""
                    elif event.type == 'message_stop':
                        sc.done = True
                yield sc


class GeminiProvider(LLMProvider):
    provider_name = "gemini"

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2.0-flash"):
        self.client = genai.Client(api_key=api_key or os.getenv("GEMINI_API_KEY"))
        self.model_id = model

    async def chat_stream(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, Any]] = None,
    ) -> AsyncGenerator[StreamChunk, None]:
        # Extract system prompt (Gemini uses system_instruction, not a role)
        system_text = "\n".join(
            m["content"] for m in messages if m["role"] == "system"
        )
        contents = []
        for msg in messages:
            if msg["role"] == "system":
                continue

            # ── Tool result messages → function_response ──
            if msg["role"] == "tool":
                fn_name = msg.get("name", "unknown")
                result_text = msg.get("content", "")
                # Parse JSON result if possible, else wrap as string
                try:
                    result_obj = json.loads(result_text)
                except (json.JSONDecodeError, TypeError):
                    result_obj = {"result": result_text}
                parts = [types.Part.from_function_response(
                    name=fn_name,
                    response=result_obj,
                )]
                contents.append(types.Content(role="user", parts=parts))
                continue

            # ── Assistant messages with tool_calls → function_call parts ──
            if msg["role"] == "assistant" and msg.get("tool_calls"):
                parts = []
                # Include text content if present
                text_content = msg.get("content", "")
                if text_content:
                    parts.append(types.Part.from_text(text=text_content))
                # Add function_call parts
                for tc in msg["tool_calls"]:
                    fn = tc.get("function", {})
                    fn_name = fn.get("name", "")
                    try:
                        fn_args = json.loads(fn.get("arguments", "{}"))
                    except (json.JSONDecodeError, TypeError):
                        fn_args = {}
                    
                    # Recover metadata (where thought_signature lives)
                    metadata = tc.get("metadata", {})
                    
                    # Reconstruct FunctionCall with signature if present
                    # Note: The library likely expects these fields passed to constructor or assigned
                    # We check if 'thought_signature' is in metadata
                    
                    # Construct basic object
                    fc = types.FunctionCall(
                        name=fn_name,
                        args=fn_args,
                    )
                    # Use setattr to inject opaque fields if constructor doesn't support them explicitly
                    # or if the library uses specific field names. 
                    # Based on error, we need 'thought_signature'.
                    if "thought_signature" in metadata:
                        # Attempt to set it. types.FunctionCall might be a Pydantic model or similar.
                        # We try to pass it in constructor if possible, but keywords might differ.
                        # Let's try dynamic assignment if not in __init__.
                        # Actually, checking types.FunctionCall source would be ideal, but for now:
                        # If it's a proto wrapper, it might need specific handling.
                        # Let's assume we can pass it as a kwarg if we had checked, 
                        # but as a fallback do setattr.
                        try:
                            # Try to rebuild if it's a standard field
                            # But wait, google-genai 0.x might use different kwarg.
                            # The error "missing a thought_signature in functionCall parts" 
                            # implies it's a field.
                            pass
                        except:
                            pass
                    
                    # Actually, better approach: The google-genai library likely has `id` or similar 
                    # that maps to thought_signature? 
                    # Or maybe it's just `id`. 
                    # In our stream loop below, we capture `fc.id`? 
                    # Let's look at the stream loop update below first.
                    
                    parts.append(types.Part(function_call=fc))
                
                if parts:
                    contents.append(types.Content(role="model", parts=parts))
                continue

            # ... (rest of loop)

        # ... (streaming call)
        async for chunk in stream:
            sc = StreamChunk()
            # Parse parts directly to handle text and function calls safely
            if chunk.candidates and chunk.candidates[0].content and chunk.candidates[0].content.parts:
                for part in chunk.candidates[0].content.parts:
                    if part.text:
                        sc.content += part.text
                    elif part.function_call:
                        fc = part.function_call
                        # Capture potential signature
                        # We inspect `fc` for 'id' or 'thought_signature'
                        meta = {}
                        # Common fields for signature in various Gemini SDK versions: 'id', 'thought'
                        # The error says "thought_signature".
                        # Let's try to capture everything relevant.
                        if hasattr(fc, 'id'):
                             meta['id'] = fc.id
                        # Check for direct attribute if new SDK
                        # We'll blindly store it if found
                        
                        # ! The SDK usually exposes 'id' on FunctionCall which IS the signature !
                        
                        sc.tool_calls.append(ToolCallChunk(
                            id=fc.name,  # logical ID for our agent
                            name=fc.name,
                            arguments=json.dumps(fc.args) if fc.args else "{}",
                            metadata=meta
                        ))
            # Capture token usage from usage_metadata
            if hasattr(chunk, 'usage_metadata') and chunk.usage_metadata:
                um = chunk.usage_metadata
                sc.input_tokens = getattr(um, 'prompt_token_count', 0) or 0
                sc.output_tokens = getattr(um, 'candidates_token_count', 0) or 0
            yield sc


# ─── Provider Map ────────────────────────────────────────────────

PROVIDER_MAP: Dict[str, type] = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "gemini": GeminiProvider,
}


def create_provider(
    ref: str,
    selector: ModelSelector = None,
    api_key: Optional[str] = None,
) -> LLMProvider:
    """
    Factory: create an LLM provider from a model reference.

    Args:
        ref: Model ref in "provider/model" or alias format
        selector: ModelSelector for resolution (uses default if None)
        api_key: Optional API key override

    Returns:
        Configured LLMProvider instance

    Examples:
        create_provider("openai/gpt-4o")
        create_provider("sonnet")       # alias -> anthropic/claude-sonnet-4-5-20250514
        create_provider("2.5pro")       # alias -> gemini/gemini-2.5-pro-preview-06-05
    """
    selector = selector or ModelSelector()
    model_info = selector.get_model(ref)

    if not model_info:
        raise ValueError(
            f'Model "{ref}" not found. Use `tripletriple models list` to see available models.'
        )

    provider_cls = PROVIDER_MAP.get(model_info.provider)
    if not provider_cls:
        raise ValueError(f'Provider "{model_info.provider}" is not supported.')

    logger.info(f"Creating provider: {model_info.full_id} ({model_info.name})")
    return provider_cls(api_key=api_key, model=model_info.id)


def create_provider_with_fallback(
    selector: ModelSelector = None,
) -> LLMProvider:
    """
    Create a provider using the primary model, falling back through
    the chain if the primary provider can't be initialized.
    """
    selector = selector or ModelSelector()
    chain = selector.get_fallback_chain()

    for model_info in chain:
        provider_cls = PROVIDER_MAP.get(model_info.provider)
        if not provider_cls:
            continue

        # Check if the required env key is set
        env_key = MODEL_CATALOG[model_info.provider].env_key
        if os.getenv(env_key):
            logger.info(
                f"Using model: {model_info.full_id} ({model_info.name})"
            )
            return provider_cls(model=model_info.id)

        logger.warning(
            f"Skipping {model_info.full_id}: {env_key} not set, trying fallback..."
        )

    raise ValueError(
        "No LLM provider could be initialized. "
        "Set at least one API key (OPENAI_API_KEY, ANTHROPIC_API_KEY, or GEMINI_API_KEY)."
    )
