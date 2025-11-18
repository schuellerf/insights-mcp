"""Enhanced MCP Agent implementation focused on extracting called tools and steps.

This implementation removes reliance on deprecated WorkflowCheckpointer and instead:
- Wraps tools to record invocations for validation in tests
- Streams workflow events to optionally log step progression
- Returns called tools for assertions and minimal reasoning steps for logs
"""

import asyncio
import json
import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union, cast

from deepeval.test_case import ToolCall
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.base.llms.types import (
    ChatResponse,
    LLMMetadata,
    MessageRole,
    TextBlock,
    ToolCallBlock,
)
from llama_index.core.llms import ChatMessage
from llama_index.core.tools import BaseTool
from llama_index.core.workflow import Context
from llama_index.llms.openai import OpenAI
from llama_index.llms.openai.utils import (
    from_openai_message,
    from_openai_token_logprobs,
    to_openai_message_dicts,
    update_tool_calls,
)
from llama_index.tools.mcp import BasicMCPClient, McpToolSpec
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_chunk import (
    ChatCompletionChunk,
    ChoiceDelta,
    ChoiceDeltaToolCall,
)


# pylint: disable=too-many-ancestors
class VllmRemote(OpenAI):
    """Extended OpenAI class that supports base_url for remote vLLM servers.

    This class extends OpenAI to work with remote vLLM servers that expose
    OpenAI-compatible APIs. It uses base_url parameter (mapped to api_base)
    and supports api_key for authentication. It also provides default metadata
    for custom model names that aren't recognized by OpenAI's model registry.
    """

    def __init__(
        self,
        model: str = "ibm-granite/granite-3.3-2b-instruct",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.1,
        **kwargs: Any,
    ) -> None:
        """Initialize VllmRemote with base_url and api_key support.

        Args:
            model: Model name/identifier
            base_url: Base URL of the remote vLLM server (e.g., "http://server:8000/v1")
            api_key: API key for authentication (optional)
            temperature: Sampling temperature
            **kwargs: Additional arguments passed to OpenAI
        """
        if not base_url:
            raise ValueError("base_url must be provided for VllmRemote")

        # Store model name for metadata override (set before super().__init__)
        object.__setattr__(self, "_custom_model_id", model)

        # Initialize parent OpenAI class with api_base parameter
        super().__init__(
            model=model,
            api_key=api_key or "",
            api_base=base_url,
            temperature=temperature,
            **kwargs,
        )

    @property
    def metadata(self) -> LLMMetadata:
        """Override metadata to provide context window for custom models."""
        # Check if this is a custom model by trying to get parent metadata
        # If it fails, we know it's a custom model
        try:
            # Try to get metadata from parent (works for known OpenAI models)
            parent_metadata = super().metadata
            # If we got here, it's a known model, return it
            return parent_metadata
        except (ValueError, AttributeError):
            # For custom models, provide default metadata
            model_name = getattr(self, "_custom_model_id", self.model)
            return LLMMetadata(
                context_window=8192,
                num_output=2048,
                is_chat_model=True,
                is_function_calling_model=True,
                model_name=model_name,
            )

    def _fix_tool_call_arguments(self, message_dict: Any) -> Any:
        """Fix tool call arguments to be JSON strings instead of dicts.

        The OpenAI API requires tool call arguments to be JSON strings, but
        LlamaIndex sometimes sends them as dicts. This fixes that issue.
        """
        message_dict = dict(message_dict)  # Create a copy to avoid mutating original

        # Handle tool_calls at top level
        if "tool_calls" in message_dict:
            tool_calls = message_dict["tool_calls"]
            if isinstance(tool_calls, list):
                fixed_tool_calls = []
                for tool_call in tool_calls:
                    fixed_tool_call = self._fix_single_tool_call(tool_call)
                    fixed_tool_calls.append(fixed_tool_call)
                message_dict["tool_calls"] = fixed_tool_calls

        # Handle tool_calls nested in content array (for multimodal messages)
        if "content" in message_dict and isinstance(message_dict["content"], list):
            content = message_dict["content"]
            fixed_content = []
            for item in content:
                if isinstance(item, dict) and "tool_calls" in item:
                    item = dict(item)
                    tool_calls = item["tool_calls"]
                    if isinstance(tool_calls, list):
                        fixed_tool_calls = [self._fix_single_tool_call(tc) for tc in tool_calls]
                        item["tool_calls"] = fixed_tool_calls
                fixed_content.append(item)
            message_dict["content"] = fixed_content

        return message_dict

    def _fix_single_tool_call(self, tool_call: Any) -> Any:
        """Fix a single tool call's arguments field."""
        if not isinstance(tool_call, dict):
            return tool_call

        tool_call = dict(tool_call)  # Create a copy

        if "function" in tool_call:
            function = tool_call["function"]
            if isinstance(function, dict) and "arguments" in function:
                function = dict(function)
                arguments = function["arguments"]
                # If arguments is a dict, convert it to JSON string
                if isinstance(arguments, dict):
                    function["arguments"] = json.dumps(arguments)
                    tool_call["function"] = function
                # If arguments is already a string, leave it as is

        return tool_call

    async def _achat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> Any:
        """Override _achat to fix tool call arguments before sending to API."""
        # Convert messages to OpenAI format
        message_dicts = to_openai_message_dicts(
            messages,
            model=self.model,
        )
        # Fix tool call arguments in each message
        fixed_messages: List[Any] = [self._fix_tool_call_arguments(msg) for msg in message_dicts]

        # Call parent's _achat with fixed messages by temporarily replacing the method
        # We need to call the OpenAI client directly with fixed messages
        aclient = self._get_aclient()
        model_kwargs = self._get_model_kwargs(**kwargs)

        if self.reuse_client:
            response: Any = await aclient.chat.completions.create(messages=fixed_messages, stream=False, **model_kwargs)
        else:
            async with aclient:
                response = await aclient.chat.completions.create(
                    messages=fixed_messages,
                    stream=False,
                    **model_kwargs,
                )

        # Type narrowing: when stream=False, response is ChatCompletion, not AsyncStream
        if not isinstance(response, ChatCompletion):
            raise ValueError("Unexpected response type from chat.completions.create")

        openai_message = response.choices[0].message
        message = from_openai_message(openai_message, modalities=self.modalities or ["text"])
        openai_token_logprobs = response.choices[0].logprobs
        logprobs = None
        if openai_token_logprobs and openai_token_logprobs.content:
            logprobs = from_openai_token_logprobs(openai_token_logprobs.content)

        return ChatResponse(
            message=message,
            raw=response,
            logprobs=logprobs,
            additional_kwargs=self._get_response_token_counts(response),
        )

    async def _astream_chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> Any:
        """Override _astream_chat to fix tool call arguments before sending to API."""
        # Convert messages to OpenAI format
        message_dicts = to_openai_message_dicts(
            messages,
            model=self.model,
        )
        # Fix tool call arguments in each message
        fixed_messages: List[Any] = [self._fix_tool_call_arguments(msg) for msg in message_dicts]

        # Call parent's streaming method with fixed messages
        aclient = self._get_aclient()

        async def gen():
            content = ""
            tool_calls: List[ChoiceDeltaToolCall] = []

            is_function = False
            first_chat_chunk = True
            async for response in await aclient.chat.completions.create(
                messages=fixed_messages,
                **self._get_model_kwargs(stream=True, **kwargs),
            ):
                blocks = []
                response = cast(ChatCompletionChunk, response)
                if len(response.choices) > 0:
                    if (
                        first_chat_chunk
                        and response.choices[0].delta.content is None
                        and response.choices[0].delta.tool_calls is None
                    ):
                        first_chat_chunk = False
                        continue
                    delta = response.choices[0].delta
                else:
                    delta = ChoiceDelta()
                first_chat_chunk = False

                if delta is None:
                    continue

                if delta.tool_calls:
                    is_function = True

                role = delta.role or MessageRole.ASSISTANT
                content_delta = delta.content or ""
                content += content_delta
                blocks.append(TextBlock(text=content))

                additional_kwargs = {}
                if is_function:
                    tool_calls = update_tool_calls(tool_calls, delta.tool_calls)
                    if tool_calls:
                        additional_kwargs["tool_calls"] = tool_calls
                        for tool_call in tool_calls:
                            if tool_call.function:
                                blocks.append(
                                    ToolCallBlock(
                                        tool_call_id=tool_call.id,
                                        tool_kwargs=tool_call.function.arguments or {},
                                        tool_name=tool_call.function.name or "",
                                    )
                                )

                yield ChatResponse(
                    message=ChatMessage(role=role, blocks=blocks),
                    raw=response,
                    additional_kwargs=additional_kwargs,
                )

        return gen()


class MCPAgentWrapper:  # pylint: disable=too-many-instance-attributes,too-few-public-methods
    """MCP agent wrapper that records tool calls and step progression.

    - Records tool calls for validation in tests
    - Optionally logs step progression if a logger is provided
    - Provides minimal reasoning steps useful for debugging output
    """

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(
        self,
        server_url: str,
        api_url: str,
        model_id: str,
        api_key: str,
        verbose_logger: Optional[logging.Logger] = None,
    ):  # pylint: disable=too-many-instance-attributes
        self.server_url = server_url
        self.api_url = api_url
        self.model_id = model_id
        self.api_key = api_key
        self.tools: Optional[List[Union[BaseTool, Callable]]] = []
        self.agent: Optional[FunctionAgent] = None
        self.context: Optional[Context] = None

        # Recorded data
        self._called_tools: List[ToolCall] = []
        self._step_names: List[str] = []

        # Set up logging for debugging
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Initialize LlamaIndex LLM with native vLLM support for remote servers
        self.llama_llm = VllmRemote(
            model=model_id,
            base_url=api_url,
            api_key=api_key,
            temperature=0.1,
        )
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        if verbose_logger:
            self.logger = verbose_logger

        # Run async initialization
        asyncio.run(self._initialize())

    async def _initialize(self):
        """Initialize MCP session and get available tools."""
        await self._init_mcp_tools()
        await self._setup_agent()

    async def _init_mcp_tools(self):
        """Initialize MCP tools using LlamaIndex MCP support."""
        try:
            # Support stdio transport by launching the server as a subprocess
            if self.server_url == "stdio":
                mcp_client = BasicMCPClient("python", args=["-m", "insights_mcp.server", "stdio"])
            else:
                mcp_client = BasicMCPClient(self.server_url)

            mcp_tool_spec = McpToolSpec(client=mcp_client)
            self.tools = await mcp_tool_spec.to_tool_list_async()

            logging.info("Initialized %d tools from MCP server", len(self.tools or []))
        except Exception as e:  # pylint: disable=broad-exception-caught
            self.logger.error("Failed to initialize MCP tools: %s", e)
            raise

    def _record_tool_call(self, tool_name: str, arguments: Optional[Dict[str, Any]] = None) -> None:
        """Record a tool call in a deepeval-compatible structure."""
        if len(self._called_tools) > 0 and self._called_tools[-1].name == tool_name:
            return
        args = arguments or {}
        self._called_tools.append(ToolCall(name=tool_name, input_parameters=args))

    def _wrap_one_tool(self, tool: Union[BaseTool, Callable]) -> Union[BaseTool, Callable]:
        """Monkey-patch a tool to record invocations while preserving behavior."""
        try:
            # Resolve tool name robustly and ensure it's str for typing
            tool_name: str
            if hasattr(tool, "metadata") and getattr(tool, "metadata") is not None:
                tool_name = str(getattr(tool.metadata, "name", "unknown"))
            else:
                name_attr = getattr(tool, "name", None)
                tool_name = (
                    str(name_attr) if name_attr is not None else (f"unknown class:{tool.__class__.__name__} {tool}")
                )

            # Prefer async path if available
            if hasattr(tool, "acall") and asyncio.iscoroutinefunction(getattr(tool, "acall")):
                original_acall = getattr(tool, "acall")

                async def wrapped_acall(*args: Any, **kwargs: Any) -> Any:  # type: ignore
                    self._record_tool_call(tool_name, kwargs)
                    return await original_acall(*args, **kwargs)

                setattr(tool, "acall", wrapped_acall)
                return tool

            # Some BaseTool implementations expose __call__ as async
            if hasattr(tool, "__call__") and asyncio.iscoroutinefunction(getattr(tool, "__call__")):
                original_call = getattr(tool, "__call__")

                async def wrapped_call(*args: Any, **kwargs: Any) -> Any:  # type: ignore
                    self._record_tool_call(tool_name, kwargs)
                    return await original_call(*args, **kwargs)

                setattr(tool, "__call__", wrapped_call)  # type: ignore
                return tool

            # Fallback: sync call path
            if hasattr(tool, "call") and callable(getattr(tool, "call")):
                original_sync_call = getattr(tool, "call")

                async def wrapped_sync(*args: Any, **kwargs: Any) -> Any:
                    self._record_tool_call(tool_name, kwargs)
                    return await asyncio.to_thread(original_sync_call, *args, **kwargs)

                # Prefer to expose async interface to agent
                setattr(tool, "acall", wrapped_sync)
                return tool

            if callable(tool):
                original_callable = tool

                async def wrapped_callable(*args: Any, **kwargs: Any) -> Any:
                    self._record_tool_call(tool_name, kwargs)
                    if asyncio.iscoroutinefunction(original_callable):
                        return await original_callable(*args, **kwargs)
                    return await asyncio.to_thread(original_callable, *args, **kwargs)

                # Expose as async entrypoint commonly used by tools
                setattr(tool, "acall", wrapped_callable)
                return tool

            return tool
        except Exception:  # pylint: disable=broad-exception-caught
            # If wrapping fails, return original tool unmodified
            return tool

    def _wrap_tools_for_recording(self) -> None:
        if not self.tools:
            return
        wrapped: List[Union[BaseTool, Callable]] = []
        for t in self.tools:
            wrapped.append(self._wrap_one_tool(t))
        self.tools = wrapped

    async def _setup_agent(self):
        """Setup LlamaIndex agent with MCP tools and optional verbose logging."""
        # Reset recordings for a new session
        self._called_tools = []
        self._step_names = []

        # Wrap tools first so the agent uses the wrapped versions
        self._wrap_tools_for_recording()

        self.agent = FunctionAgent(
            name="MCP Agent",
            description="Agent with MCP tools",
            llm=self.llama_llm,
            tools=self.tools,
        )
        self.context = Context(self.agent)

        self.logger.info("📝 Initialized workflow with event streaming for step logging")

    async def execute_with_reasoning(  # pylint: disable=too-many-locals
        self,
        user_msg: str,
        chat_history: Optional[List[ChatMessage]] = None,
        max_iterations: int = 10,
    ) -> Tuple[str, List[Dict[str, Any]], List[Any], List[ChatMessage]]:  # pylint: disable=too-many-locals,too-many-arguments
        """Execute agent, record tool calls and steps, return response and artifacts."""
        if chat_history is None:
            chat_history = []

        if not self.agent or not self.context:
            raise ValueError("Agent or context not initialized")

        # Stream events for optional step logging while the workflow runs
        self.logger.info("🎬 Starting workflow execution...")
        self.logger.info("📝 User message: %s", user_msg)

        handler = self.agent.run(
            user_msg=user_msg,
            ctx=self.context,
            chat_history=chat_history,
            max_iterations=max_iterations,
        )

        # Consume events to capture step progression
        async def _stream_events() -> None:
            async for ev in handler.stream_events():
                ev_name = ev.__class__.__name__
                self._step_names.append(ev_name)
                if self.logger and ev_name not in ["AgentStream"]:
                    data = f"{ev}"
                    if len(data) > 2000:
                        data = data[:1000] + "\n<… abbreviated log …>\n" + data[-1000:]
                    self.logger.debug("📡 Event %s: %s", ev_name, data)

        # Run streaming in background while awaiting result
        stream_task = asyncio.create_task(_stream_events())
        try:
            response = await handler
        finally:
            # Ensure streaming task cleaned up
            try:
                await asyncio.wait_for(stream_task, timeout=0.5)
            except asyncio.TimeoutError:
                stream_task.cancel()

        # Build minimal reasoning steps from recorded step names
        reasoning_steps: List[Dict[str, Any]] = [
            {"step_number": i + 1, "step_type": "event", "content": name} for i, name in enumerate(self._step_names)
        ]

        # Build updated chat history
        updated_history = chat_history + [ChatMessage(role="user", content=user_msg)]
        updated_history.append(ChatMessage(role="assistant", content=str(response)))

        # Return called tools as recorded
        tools_called: List[Any] = list(self._called_tools)

        self.logger.info("🔍 Agent response: %s", response)
        if tools_called:
            self.logger.info("🔧 Tools called: %s", [t.name for t in tools_called])
        else:
            self.logger.info("🔧 No tools called")

        return str(response), reasoning_steps, tools_called, updated_history
