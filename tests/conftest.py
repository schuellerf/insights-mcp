"""Pytest configuration and common fixtures."""

# Apply defensive patch for llama-index MCP schema violation bug
# This prevents TypeError when llama-index incorrectly generates additionalProperties: true
# (which violates MCP specification that expects explicit object properties)

import asyncio
import logging
import multiprocessing
import socket
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastmcp.tools.tool import ToolResult
from llama_index.core.tools import FunctionTool
from llama_index.tools.mcp import BasicMCPClient, McpToolSpec

# Add imports for mock client creation
from insights_mcp.client import InsightsClient
from insights_mcp.mcp import INSIGHTS_BASE_URL
from insights_mcp.server import create_mcp_server, reset_all_mcp_instances

from .llama_index_non_iterable_bool_patch import apply_llama_index_bool_patch
from .utils import (
    CustomVLLMModel,
    build_server_args,
    get_free_port,
    load_llm_configurations,
)
from .utils_agent import MCPAgentWrapper

TEST_CLIENT_ID = "test-client-id"
TEST_CLIENT_SECRET = "test-client-secret"
TEST_BLUEPRINT_UUID = "12345678-1234-1234-1234-123456789012"
TEST_BIND_HOST = "127.0.0.1"


@dataclass
# pylint: disable=too-many-instance-attributes
class MCPTestServerConfig:
    """Configuration for MCP server test instances."""

    client_id: str | None = TEST_CLIENT_ID
    client_secret: str | None = TEST_CLIENT_SECRET
    debug: bool = False
    oauth_enabled: bool = False
    proxy_url: str | None = None
    readonly: bool = False
    refresh_token: str | None = None
    stage: bool = False
    toolset: str | None = None
    transport: str = "stdio"

    def to_kwargs(self) -> dict:
        """Convert config to kwargs for create_mcp_server."""

        return vars(self).copy()


@dataclass
class ServerInstance:
    """Running server instance with URL and process."""

    url: str
    process: Any  # multiprocessing.Process, but avoid import for type hint


if apply_llama_index_bool_patch():
    print("✅ Patch applied successfully")
else:
    print("❌ Failed to apply patch")


# Load LLM configurations for fixtures
_, guardian_llm_config = load_llm_configurations()


@pytest.fixture
# pylint: disable=redefined-outer-name
def inmemory_test_mcp_server(request):
    """Create a unified Insights MCP server instance.

    Can be used for direct function calls to the MCP server.
    defaults to http transport. So mcp_test_client_network and mcp_tools_network work out of the box."""

    config = (
        getattr(request, "param", None)
        if hasattr(request, "param") and request.param
        else MCPTestServerConfig(transport="http")
    )

    verbosity = request.config.getoption("verbose", default=0)
    debug = verbosity >= 3

    server = create_mcp_server(
        toolset_list=config.toolset,
        readonly=config.readonly,
        stage=False,
        debug=debug,
        transport=config.transport,
        client_id=config.client_id,
        client_secret=config.client_secret,
        refresh_token=None,
        proxy_url=None,
        oauth_enabled=config.oauth_enabled,
    )
    yield server
    # Cleanup: reset all MCP instances after test
    reset_all_mcp_instances()


def is_port_open(host: str, port: int, timeout: float = 0.1) -> bool:
    """Check if a port is open using socket connection."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout)
            result = sock.connect_ex((host, port))
            return result == 0
    except Exception:  # pylint: disable=broad-exception-caught
        return False


def wait_for_port(host: str, port: int, timeout: float = 10.0) -> None:
    """Wait for a port to become available, raising an exception if timeout is reached."""
    start_time = time.time()

    while time.time() - start_time < timeout:
        if is_port_open(host, port):
            return
        time.sleep(0.1)

    raise RuntimeError(f"Port {port} on {host} not available within {timeout} seconds")


def _run_server_in_process(server_config_dict, transport, host, port):
    """Helper function to run server in a separate process with correct arguments."""
    # Recreate the server from the config dict
    server = create_mcp_server(
        toolset_list=server_config_dict["toolset"],
        readonly=server_config_dict["readonly"],
        stage=server_config_dict["stage"],
        debug=server_config_dict["debug"],
        transport=transport,
        client_id=server_config_dict["client_id"],
        client_secret=server_config_dict["client_secret"],
        refresh_token=server_config_dict["refresh_token"],
        proxy_url=server_config_dict["proxy_url"],
        oauth_enabled=server_config_dict["oauth_enabled"],
    )
    server.run(transport=transport, host=host, port=port)


@pytest.fixture
# pylint: disable=redefined-outer-name
async def inmemory_test_mcp_server_run_in_process(request, inmemory_test_mcp_server, verbose_logger):
    """Run the inmemory_test_mcp_server in a process.

    For real network transports, it runs the server in a background process."""

    config = MCPTestServerConfig()
    if (
        hasattr(request.node, "callspec")
        and request.node.callspec
        and "inmemory_test_mcp_server" in request.node.callspec.params
    ):
        config = request.node.callspec.params["inmemory_test_mcp_server"]

    test_transport = config.transport

    if test_transport == "stdio":
        raise ValueError("please use inmemory_test_mcp_server directly for stdio transport")

    port = get_free_port()

    # Run the server in a background process with correct keyword arguments
    # Use spawn method to avoid fork() deprecation warning in multi-threaded processes
    verbosity = request.config.getoption("verbose", default=0)
    debug = verbosity >= 3

    # Use the elegant dataclass method for serialization
    server_config_dict = config.to_kwargs()
    server_config_dict["debug"] = debug

    ctx = multiprocessing.get_context("spawn")
    process = ctx.Process(
        target=_run_server_in_process,
        args=(server_config_dict, test_transport, TEST_BIND_HOST, port),
        daemon=True,
    )
    process.start()

    verbose_logger.debug("🧪 Waiting for the server to be ready on port %s", port)
    # Wait for the server to be ready
    try:
        wait_for_port(TEST_BIND_HOST, port, timeout=10.0)
    except RuntimeError:
        process.terminate()
        raise
    verbose_logger.debug("🧪 Server is ready on port %s", port)
    url_path = ""
    if config.transport == "http":
        url_path = "mcp/"
    elif config.transport == "sse":
        url_path = "sse/"

    yield inmemory_test_mcp_server, process, f"http://{TEST_BIND_HOST}:{port}/{url_path}"

    process.terminate()
    await inmemory_test_mcp_server.aclose_clients()


@pytest.fixture
async def test_agent(  # pylint: disable=redefined-outer-name,too-many-arguments
    verbose_logger,
    request,
    inmemory_test_mcp_server,
):
    """Create and configure an inprocess MCP agent."""
    # Get llm_config from the test's parametrization
    llm_config = request.node.callspec.params["llm_config"]

    # Ensure tools see request headers even in in-process mode (no HTTP context)
    header_patcher = patch(
        "fastmcp.server.dependencies.get_http_headers",
        return_value={
            "insights-client-id": TEST_CLIENT_ID,
            "insights-client-secret": TEST_CLIENT_SECRET,
        },
    )
    header_patcher.start()

    # Build generic FunctionTool wrappers that call the server's ToolManager
    # TBD get more elegant?
    tools = []
    tools_dict = await inmemory_test_mcp_server._tool_manager.get_tools()  # pylint: disable=protected-access

    for tool_key, tool in tools_dict.items():
        # Extract description if available; fall back to key
        description = getattr(tool, "description", None)

        async def _make_tool_fn(key: str):  # type: ignore[no-untyped-def]
            async def _tool(**kwargs):  # type: ignore[no-untyped-def]
                # pylint: disable=protected-access
                result = await inmemory_test_mcp_server._tool_manager.call_tool(key, kwargs)
                if isinstance(result, ToolResult):
                    payload = result.to_mcp_result()
                    # Tuple: (content_blocks, structured)
                    if isinstance(payload, tuple):
                        content_blocks, structured = payload
                        if structured is not None:
                            return structured
                        # Concatenate text contents if present
                        return "".join(getattr(cb, "text", "") for cb in content_blocks)
                    # Only content blocks
                    content_blocks = payload
                    return "".join(getattr(cb, "text", "") for cb in content_blocks)
                return result

            return _tool

        tool_fn = await _make_tool_fn(tool_key)

        tools.append(
            FunctionTool.from_defaults(
                fn=tool_fn,
                name=tool_key,
                description=description or tool_key,
            )
        )

    agent = await MCPAgentWrapper.create(
        server_url="inprocess",
        api_url=llm_config["MODEL_API"],
        model_id=llm_config["MODEL_ID"],
        api_key=llm_config["USER_KEY"],
        verbose_logger=verbose_logger,
        tools_override=tools,
        system_prompt_override=getattr(inmemory_test_mcp_server, "instructions", ""),
    )

    verbose_logger.info("🧪 Testing the model (in-process tools): %s", agent.model_id)
    verbose_logger.info("🧪 with tools: %s", [tool.metadata.name for tool in tools])
    verbose_logger.info("🧪 orig tools_dict: %s", tools_dict.items())
    try:
        yield agent
    finally:
        # Ensure all Insights HTTP clients are closed before the loop shuts down
        # to avoid 'Event loop is closed' during AsyncClient.aclose()
        await inmemory_test_mcp_server.aclose_clients()
        header_patcher.stop()


@pytest.fixture
def guardian_agent(verbose_logger, request):  # pylint: disable=redefined-outer-name
    """Create and configure a guardian agent for evaluation."""
    # Get llm_config from the test's parametrization
    llm_config = request.node.callspec.params["llm_config"]

    # if there is a guardian LLM, use it for the guardian agent
    # otherwise, use the test LLM for the guardian agent
    if guardian_llm_config:
        agent = CustomVLLMModel(
            api_url=guardian_llm_config["MODEL_API"],
            model_id=guardian_llm_config["MODEL_ID"],
            api_key=guardian_llm_config["USER_KEY"],
        )
    else:
        agent = CustomVLLMModel(
            api_url=llm_config["MODEL_API"], model_id=llm_config["MODEL_ID"], api_key=llm_config["USER_KEY"]
        )

    verbose_logger.info("🧪 Verifying with the model: %s", agent.get_model_name())

    return agent


@pytest.fixture
def default_response_size():
    """Default response size for pagination tests."""
    return 7


@pytest.fixture
def test_client_credentials():
    """Test client credentials."""
    return {"client_id": "test-client-id", "client_secret": "test-client-secret"}


@pytest.fixture
# pylint: disable=redefined-outer-name
def mock_http_headers(test_client_credentials):
    """Mock HTTP headers with test credentials."""
    return {
        "image-builder-client-id": test_client_credentials["client_id"],
        "image-builder-client-secret": test_client_credentials["client_secret"],
    }


@pytest.fixture()
# pylint: disable=redefined-outer-name
async def mcp_test_client_network(inmemory_test_mcp_server_run_in_process):
    """Create a BasicMCPClient instance for the MCP server."""
    inmemory_test_mcp_server, process, url = inmemory_test_mcp_server_run_in_process

    client = BasicMCPClient(url)

    yield client

    process.terminate()
    process.join()
    await inmemory_test_mcp_server.aclose_clients()


@pytest.fixture()
# pylint: disable=redefined-outer-name
def mcp_test_client_stdio(request):
    """Create a BasicMCPClient instance for the MCP server."""

    config = getattr(request, "param", None) if hasattr(request, "param") and request.param else MCPTestServerConfig()

    args = build_server_args("stdio", config.toolset, config.readonly)
    client = BasicMCPClient("python", args=["-m", "insights_mcp.server"] + args)
    return client


@pytest.fixture()
def mcp_tools_stdio(mcp_test_client_stdio):  # pylint: disable=redefined-outer-name
    """Fetch tools from the MCP server via stdio transport.

    Assure the config is set on the mcp_test_client_stdio fixture.
    """

    tool_spec = McpToolSpec(client=mcp_test_client_stdio)

    async def _fetch():
        return await tool_spec.to_tool_list_async()

    return asyncio.run(_fetch())


@pytest.fixture()
async def mcp_tools_network(mcp_test_client_network):  # pylint: disable=redefined-outer-name
    """Fetch tools from the MCP server.
    via network transport (http or sse)."""
    tool_spec = McpToolSpec(client=mcp_test_client_network)
    return await tool_spec.to_tool_list_async()


@pytest.fixture
def verbose_logger(request):
    """Get a logger that respects pytest verbosity."""
    logger = logging.getLogger(__name__)

    verbosity = request.config.getoption("verbose", default=0)

    if verbosity >= 3:
        logger.setLevel(logging.DEBUG)
    elif verbosity == 2:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARNING)

    return logger


def create_mcp_server_low_level(server_class, mock_client):
    """Create a mock MCP server instance for any server class.

    Without the InsightsMCP server wrapper.
    """
    server = server_class()
    server.insights_client = mock_client
    server.register_tools()
    return server


def create_mock_client(client_id=TEST_CLIENT_ID, client_secret=TEST_CLIENT_SECRET, api_path=None):
    """Create a mock InsightsClient instance for any test."""

    client = Mock(spec=InsightsClient)
    client.client_id = client_id
    client.client_secret = client_secret
    client.insights_base_url = INSIGHTS_BASE_URL
    if api_path:
        client.api_path = api_path

    # Set up async mocks for the HTTP methods
    client.get = AsyncMock()
    client.post = AsyncMock()
    client.put = AsyncMock()

    return client


# No server-specific fixtures needed!
# Tests can import the server class directly and use create_mcp_server(ServerClass)


@contextmanager
# pylint: disable=too-many-arguments,too-many-positional-arguments,redefined-outer-name
def setup_mcp_mock(
    inmemory_test_mcp_server, mock_client, mock_response=None, side_effect=None, client_id=TEST_CLIENT_ID
):
    """Generic context manager for setting up MCP server mock patterns."""
    with patch("fastmcp.server.dependencies.get_http_headers") as mock_headers:
        mock_headers.return_value = {
            "insights-client-id": client_id,
            "insights-client-secret": TEST_CLIENT_SECRET,
        }

        if side_effect:
            mock_client.get.side_effect = side_effect
            mock_client.post.side_effect = side_effect
            mock_client.put.side_effect = side_effect
        elif mock_response is not None:
            mock_client.get.return_value = mock_response
            mock_client.post.return_value = mock_response
            mock_client.put.return_value = mock_response

        inmemory_test_mcp_server.clients[client_id] = mock_client
        yield mock_headers


def assert_api_error_result(result, error_message="API Error"):
    """Helper to assert API error results."""
    assert result.startswith(f"Error: {error_message}") or error_message.lower() in result.lower()


def assert_empty_response(result):
    """Helper to assert empty response results."""
    assert "[]" in result
