"""Pytest configuration and common fixtures."""

# Apply defensive patch for llama-index MCP schema violation bug
# This prevents TypeError when llama-index incorrectly generates additionalProperties: true
# (which violates MCP specification that expects explicit object properties)

import asyncio
import logging
import os
import threading
import time
from contextlib import contextmanager
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import requests
from llama_index.tools.mcp import BasicMCPClient, McpToolSpec

# Add imports for mock client creation
from insights_mcp.client import InsightsClient
from insights_mcp.mcp import INSIGHTS_BASE_URL_PROD
from tools.mock_http_server import MockRequestHandler, RoutesConfig

# pylint: disable=wrong-import-position
from .llama_index_non_iterable_bool_patch import apply_llama_index_bool_patch

if apply_llama_index_bool_patch():
    print("✅ Patch applied successfully")
else:
    print("❌ Failed to apply patch")

from .utils import CustomVLLMModel, cleanup_server_process, load_llm_configurations, start_insights_mcp_server
from .utils_agent import MCPAgentWrapper

# Load LLM configurations for fixtures
_, guardian_llm_config = load_llm_configurations()


@pytest.fixture
def test_agent(mcp_server_url, verbose_logger, request):  # pylint: disable=redefined-outer-name
    """Create and configure a simplified test agent for the current LLM configuration."""
    # Get llm_config from the test's parametrization
    llm_config = request.node.callspec.params["llm_config"]

    agent = MCPAgentWrapper(
        server_url=mcp_server_url,
        api_url=llm_config["MODEL_API"],
        model_id=llm_config["MODEL_ID"],
        api_key=llm_config["USER_KEY"],
        verbose_logger=verbose_logger,
    )
    verbose_logger.info("🧪 Testing the model: %s", agent.model_id)

    return agent


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


@pytest.fixture(scope="session")
def mock_insights_server(request):  # pylint: disable=too-many-branches,too-many-statements
    """Start a mock HTTP server for Insights API endpoints in-process.

    This fixture runs ThreadingHTTPServer in a background thread and sets
    INSIGHTS_BASE_URL, INSIGHTS_CLIENT_ID, INSIGHTS_CLIENT_SECRET, and
    INSIGHTS_TOKEN_ENDPOINT environment variables. The mock server will be used
    by the MCP server for API calls.

    Supports parametrization via indirect parametrization:
        @pytest.mark.parametrize("mock_insights_server",
            ["tests/fixtures/image_builder_mocks.yaml"],
            indirect=True)

        # For auth failure tests, add client_id/client_secret to test signature:
        @pytest.mark.parametrize("mock_insights_server",
            ["tests/fixtures/image_builder_mocks.yaml"],
            indirect=True)
        @pytest.mark.parametrize("client_id", ["invalid-id"])
        @pytest.mark.parametrize("client_secret", ["invalid-secret"])
        def test_something(mock_insights_server, client_id, client_secret):
            # client_id and client_secret will be passed to the fixture via callspec.params
            pass

    Args:
        request: Pytest request object for accessing parameters

    Yields:
        str: The base URL of the mock server (e.g., "http://127.0.0.1:12345")
    """
    logger = logging.getLogger(__name__)

    # Get mock config path from parameter or use default
    # Support both request.param (indirect parametrization) and callspec.params
    mock_config_path: str | Path | None = getattr(request, "param", None)
    if mock_config_path is None and hasattr(request.node, "callspec"):
        mock_config_path = request.node.callspec.params.get("mock_config_path")

    # Require mock config path to be specified
    if mock_config_path is None:
        pytest.fail(
            "No mock config path specified. "
            "Add @pytest.mark.parametrize('mock_insights_server', "
            "['src/image_builder_mcp/tests/fixtures/image_builder_mocks.yaml'], indirect=True) to your test."
        )

    # Get client credentials from parametrization or use defaults
    # Support both request.param (indirect parametrization) and callspec.params
    client_id: str | None = None
    client_secret: str | None = None
    if hasattr(request.node, "callspec"):
        client_id = request.node.callspec.params.get("client_id")
        client_secret = request.node.callspec.params.get("client_secret")

    # Default to test constants if not provided
    if client_id is None:
        client_id = TEST_CLIENT_ID
    if client_secret is None:
        client_secret = TEST_CLIENT_SECRET

    # Convert string to Path if needed
    mock_config_path = Path(mock_config_path)
    # If relative path, resolve relative to project root (if starts with src/) or tests directory
    if not mock_config_path.is_absolute():
        if str(mock_config_path).startswith("src/"):
            # Path relative to project root
            mock_config_path = Path(__file__).parent.parent / mock_config_path
        else:
            # Path relative to tests directory
            mock_config_path = Path(__file__).parent / mock_config_path

    if not mock_config_path.exists():
        pytest.skip(f"Mock config not found at {mock_config_path}")

    logger.info("Loading mock config from %s", mock_config_path)

    # Load routes configuration
    try:
        routes_config = RoutesConfig.from_yaml_file(str(mock_config_path))
    except Exception as e:  # pylint: disable=broad-exception-caught
        pytest.fail(f"Failed to load mock config from {mock_config_path}: {e}")

    # Create server on ephemeral port
    server_address = ("127.0.0.1", 0)
    httpd = ThreadingHTTPServer(server_address, MockRequestHandler)
    httpd.routes_config = routes_config  # type: ignore[attr-defined]

    # Get the actual port that was assigned
    port = httpd.server_address[1]
    mock_base_url = f"http://127.0.0.1:{port}"

    # Start server in background thread
    server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    server_thread.start()

    logger.info("Mock Insights server started at %s (in-process)", mock_base_url)

    # Wait a bit to ensure the server is ready
    time.sleep(0.1)

    # Verify the mock server is responding
    try:
        # Try a simple request to verify server is up
        response = requests.get(f"{mock_base_url}/", timeout=2)
        logger.debug("Mock server verified and responding (status: %s)", response.status_code)
    except Exception as e:  # pylint: disable=broad-exception-caught
        httpd.shutdown()
        server_thread.join(timeout=2)
        pytest.fail(f"Mock server not responding: {e}")

    # Set environment variables for MCP server to use
    os.environ["INSIGHTS_BASE_URL"] = mock_base_url
    os.environ["INSIGHTS_CLIENT_ID"] = client_id
    os.environ["INSIGHTS_CLIENT_SECRET"] = client_secret
    os.environ["INSIGHTS_TOKEN_ENDPOINT"] = f"{mock_base_url}/auth/realms/redhat-external/protocol/openid-connect/token"

    try:
        yield mock_base_url
    finally:
        # Cleanup: remove environment variables (no need to restore originals in tests)
        os.environ.pop("INSIGHTS_BASE_URL", None)
        os.environ.pop("INSIGHTS_CLIENT_ID", None)
        os.environ.pop("INSIGHTS_CLIENT_SECRET", None)
        os.environ.pop("INSIGHTS_TOKEN_ENDPOINT", None)

        # Shutdown the server
        try:
            httpd.shutdown()
            server_thread.join(timeout=2)
            logger.debug("Mock server stopped")
        except Exception as e:  # pylint: disable=broad-exception-caught
            logger.warning("Error stopping mock server: %s", e)


@pytest.fixture(scope="session")
def mcp_server_url(request):
    """Start MCP server and return the URL.

    Supports indirect parametrization with dict:
        @pytest.mark.parametrize("mcp_server_url", [
            {"transport": "http", "toolset": "image-builder"}
        ], indirect=True)
        def test_something(mcp_server_url):
            # Server will be started with specified transport and toolset
            pass

    Note: If mock_insights_server fixture is requested by a test, it will set
    INSIGHTS_BASE_URL environment variable, which will be used by the MCP server.
    """
    # Handle indirect parametrization with dict
    if hasattr(request, "param") and isinstance(request.param, dict):
        transport = request.param.get("transport", "http")
        toolset = request.param.get("toolset")
    else:
        # Default values if not parametrized
        transport = "http"
        toolset = None

    server_url, server_process = start_insights_mcp_server(transport, toolset=toolset)

    try:
        yield server_url
    finally:
        cleanup_server_process(server_process)


@pytest.fixture()
def mcp_tools(mcp_server_url):  # pylint: disable=redefined-outer-name
    """Fetch tools from the MCP server.

    For stdio transport, uses BasicMCPClient subprocess approach.
    For HTTP/SSE transports, connects to the running server.
    """
    if mcp_server_url == "stdio":
        # For stdio, use subprocess approach
        client = BasicMCPClient("python", args=["-m", "insights_mcp.server", "stdio"])
    else:
        # For HTTP/SSE, connect to running server
        client = BasicMCPClient(mcp_server_url)

    tool_spec = McpToolSpec(client=client)

    async def _fetch():
        return await tool_spec.to_tool_list_async()

    return asyncio.run(_fetch())


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


TEST_CLIENT_ID = "test-client-id"
TEST_CLIENT_SECRET = "test-client-secret"
TEST_BLUEPRINT_UUID = "12345678-1234-1234-1234-123456789012"


def create_mcp_server(server_class, client_id=TEST_CLIENT_ID, client_secret=TEST_CLIENT_SECRET):
    """Create a mock MCP server instance for any server class."""
    server = server_class()
    server.init_insights_client(
        client_id=client_id,
        client_secret=client_secret,
    )
    server.register_tools()
    return server


def create_mock_client(client_id=TEST_CLIENT_ID, client_secret=TEST_CLIENT_SECRET, api_path=None):
    """Create a mock InsightsClient instance for any test."""
    client = Mock(spec=InsightsClient)
    client.client_id = client_id
    client.client_secret = client_secret
    client.insights_base_url = INSIGHTS_BASE_URL_PROD
    if api_path:
        client.api_path = api_path
    return client


# No server-specific fixtures needed!
# Tests can import the server class directly and use create_mcp_server(ServerClass)


@contextmanager
# pylint: disable=too-many-arguments,too-many-positional-arguments
def setup_mcp_mock(
    server_module, mcp_server, mock_client, mock_response=None, side_effect=None, client_id=TEST_CLIENT_ID
):
    """Generic context manager for setting up MCP server mock patterns."""
    with patch.object(server_module, "get_http_headers") as mock_headers:
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

        mcp_server.clients[client_id] = mock_client
        yield mock_headers


def assert_api_error_result(result, error_message="API Error"):
    """Helper to assert API error results."""
    assert result.startswith(f"Error: {error_message}") or error_message.lower() in result.lower()


def assert_empty_response(result):
    """Helper to assert empty response results."""
    assert "[]" in result


def assert_instruction_in_result(result, instruction="[INSTRUCTION]"):
    """Helper to assert instruction text in result."""
    assert instruction in result
