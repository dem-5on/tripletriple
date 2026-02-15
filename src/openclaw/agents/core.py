import json
import asyncio
import logging
from typing import AsyncGenerator, List, Dict, Any, Type, Optional, Union
from ..gateway.session import Session
from .base import Agent
from .tools import ToolRegistry, Tool
from .llm import LLMProvider, OpenAIProvider, StreamChunk
from .system_prompt import SystemPromptBuilder

logger = logging.getLogger("openclaw.agents.core")

MAX_TOOL_ITERATIONS = 10


class ReActAgent(Agent):
    def __init__(
        self,
        llm: LLMProvider = None,
        tools: List[Type[Tool]] = [],
        prompt_builder: Optional[SystemPromptBuilder] = None,
        tool_context: Optional[Dict[str, Any]] = None,
    ):
        self.llm = llm or OpenAIProvider()
        self.tool_registry = ToolRegistry()
        self.prompt_builder = prompt_builder
        self.tool_context = tool_context or {}
        for tool_cls in tools:
            self.tool_registry.register(tool_cls())

    async def process_message(
        self,
        session: Session,
        message: Union[str, List[Any]],
    ) -> AsyncGenerator[str, None]:
        """
        ReAct loop: LLM → tool execution → feed results back → repeat.

        Yields text chunks as the LLM generates them. When the LLM
        requests tool calls, executes them and feeds results back for
        the next iteration. Stops when the LLM produces a final text
        response with no tool calls, or after MAX_TOOL_ITERATIONS.
        """
        # Build messages list starting with system prompt
        messages: List[Dict[str, Any]] = []

        # Inject system prompt from workspace identity files
        if self.prompt_builder:
            is_main = session.entry.chat_type.value == "dm"
            system_prompt = self.prompt_builder.assemble(is_main_session=is_main)
            messages.append({"role": "system", "content": system_prompt})

        # Append conversation history
        messages.extend(
            {"role": m.role, "content": m.content}
            for m in session.messages
        )

        tools_schema = self.tool_registry.get_all_schemas()

        # ── ReAct Loop ──
        for iteration in range(MAX_TOOL_ITERATIONS):
            full_content = ""
            tool_calls: dict[int, dict] = {}
            total_input_tokens = 0
            total_output_tokens = 0

            # Stream LLM response
            async for chunk in self.llm.chat_stream(messages, tools=tools_schema):
                if chunk.content:
                    full_content += chunk.content
                    yield chunk.content

                if chunk.input_tokens:
                    total_input_tokens = chunk.input_tokens
                if chunk.output_tokens:
                    total_output_tokens = chunk.output_tokens

                for tc in chunk.tool_calls:
                    idx = len(tool_calls)
                    for k, v in tool_calls.items():
                        if v["id"] == tc.id and tc.id:
                            idx = k
                            break
                    if idx not in tool_calls:
                        tool_calls[idx] = {"id": "", "function": {"name": "", "arguments": ""}}
                    if tc.id:
                        tool_calls[idx]["id"] = tc.id
                    if tc.name:
                        tool_calls[idx]["function"]["name"] = tc.name
                    if tc.arguments:
                        tool_calls[idx]["function"]["arguments"] += tc.arguments

            # Update session token counts
            if total_input_tokens or total_output_tokens:
                session.add_tokens(total_input_tokens, total_output_tokens)

            # No tool calls → final response, exit loop
            if not tool_calls:
                break

            # ── Execute tools and build result messages ──

            # Append the assistant message with tool calls to conversation
            assistant_msg = {"role": "assistant", "content": full_content or ""}
            assistant_msg["tool_calls"] = [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["function"]["name"],
                        "arguments": tc["function"]["arguments"],
                    },
                }
                for tc in tool_calls.values()
            ]
            messages.append(assistant_msg)

            for tc in tool_calls.values():
                func_name = tc["function"]["name"]
                args_str = tc["function"]["arguments"]
                call_id = tc["id"]

                tool = self.tool_registry.get(func_name)
                if tool:
                    yield f"\n\n🔧 *{func_name}*\n"
                    try:
                        args = json.loads(args_str) if args_str else {}
                        # Pass tool context for context-aware tools
                        if hasattr(tool, 'needs_context') and tool.needs_context:
                            args["_context"] = {
                                **self.tool_context,
                                "current_session": session,
                            }
                        result = await tool.run(**args)
                        result_str = str(result) if result is not None else "Done."
                    except Exception as e:
                        logger.error(f"Tool {func_name} error: {e}", exc_info=True)
                        result_str = f"Error: {e}"
                else:
                    result_str = f"Error: Tool '{func_name}' not found."

                # Append tool result to conversation
                messages.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": func_name,
                    "content": result_str,
                })

            # Loop continues → LLM sees tool results and generates next response

        # If we hit the iteration limit, warn
        if tool_calls:
            yield "\n\n⚠️ Reached maximum tool iterations.\n"
