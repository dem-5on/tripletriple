import json
import asyncio
import logging
from .base import Agent
from typing import AsyncGenerator, List, Dict, Any, Type, Optional, Union
from ..gateway.session import Session
from .tools import ToolRegistry, Tool
from .system_prompt import SystemPromptBuilder
from .llm import LLMProvider, OpenAIProvider, StreamChunk

logger = logging.getLogger("tripletriple.agents.core")

MAX_TOOL_ITERATIONS = 20


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
        """
        # ── 1. Bootstrap Cleanup Check ──
        # Check if we started in bootstrap mode
        started_in_bootstrap = False
        if self.prompt_builder and self.prompt_builder.needs_bootstrap():
            started_in_bootstrap = True

        # ── 2. Memory Flush Check ──
        # If tokens are high, inject a "Flush Turn" system message
        FLUSH_THRESHOLD = 50000  # TODO: Make configurable
        flushing_memory = False
        
        if session.entry.tokens.total_tokens > FLUSH_THRESHOLD:
            logger.info(f"Memory flush triggered for {session.key} ({session.entry.tokens.total_tokens} tokens)")
            flushing_memory = True
            # We don't yield this message to the user, it is internal to the LLM
            # But here we just set a flag to inject instructions.
            # actually, we might want to do a separate turn? 
            # For now, let's just append a strong system instruction to the prompt.
            
        # Build messages list starting with system prompt
        messages: List[Dict[str, Any]] = []

        # Inject system prompt from workspace identity files
        if self.prompt_builder:
            is_main = session.entry.chat_type.value == "dm"
            system_prompt = self.prompt_builder.assemble(is_main_session=is_main)
            messages.append({"role": "system", "content": system_prompt})

        # Inject Memory Flush Instruction if needed
        if flushing_memory:
            messages.append({
                "role": "system",
                "content": (
                    "🚨 **CRITICAL: MEMORY FLUSH REQUIRED** 🚨\n"
                    "Your context window is getting full.\n"
                    "1. Review the conversation history.\n"
                    "2. Save any important facts, preferences, or tasks to memory files using your tools.\n"
                    "3. AFTER saving, you may continue with the user's request.\n"
                    "Do not complain about this. Just do it."
                )
            })

        # Append conversation history
        messages.extend(
            {"role": m.role, "content": m.content}
            for m in session.messages
        )

        # Pass actual Tool objects to the provider so it can generate the correct schema
        tools = list(self.tool_registry.get_all().values())

        # ── ReAct Loop ──
        for iteration in range(MAX_TOOL_ITERATIONS):
            full_content = ""
            tool_calls: dict[int, dict] = {}
            total_input_tokens = 0
            total_output_tokens = 0

            # Stream LLM response
            async for chunk in self.llm.chat_stream(messages, tools=tools):
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
                        tool_calls[idx] = {
                            "id": "", 
                            "function": {"name": "", "arguments": ""},
                            "metadata": {}
                        }
                    if tc.id:
                        tool_calls[idx]["id"] = tc.id
                    if tc.name:
                        tool_calls[idx]["function"]["name"] = tc.name
                    if tc.arguments:
                        tool_calls[idx]["function"]["arguments"] += tc.arguments
                    if tc.metadata:
                        tool_calls[idx]["metadata"].update(tc.metadata)

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
                    "metadata": tc.get("metadata", {})
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
                    # yield f"\n\n🔧 *{func_name}*\n"
                    try:
                        try:
                            args = json.loads(args_str) if args_str else {}
                        except json.JSONDecodeError as json_err:
                            # Fallback: sometimes LLM adds extra text or markdown
                            # Try to extract JSON from code block if present
                            import re
                            match = re.search(r'\{.*\}', args_str, re.DOTALL)
                            if match:
                                args = json.loads(match.group(0))
                            else:
                                raise json_err

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

        # ── Post-Processing ──

        # 1. Bootstrap Cleanup
        if started_in_bootstrap and self.prompt_builder:
            # Check if we are NO LONGER in bootstrap mode (meaning IDENTITY.md was created)
            if not self.prompt_builder.needs_bootstrap():
                # Cleanup BOOTSTRAP.md
                bootstrap_path = self.prompt_builder.config.root / "BOOTSTRAP.md"
                if bootstrap_path.exists():
                    try:
                        bootstrap_path.unlink()
                        logger.info("Bootstrap complete. Deleted BOOTSTRAP.md.")
                        # yield "\n\n✨ Identity established. Bootstrap complete."
                    except Exception as e:
                        logger.error(f"Failed to delete BOOTSTRAP.md: {e}")

        # 2. Pruning (if we just finished a flush turn)
        if flushing_memory:
            removed = session.prune_context(keep_last=20)
            if removed > 0:
                logger.info(f"Pruned {removed} messages from session {session.key}")
