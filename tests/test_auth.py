"""Test suite for authentication-related functionality."""

import pytest

# Clean import - no sys.path.insert needed with proper package structure!
from tests.conftest import MCPTestServerConfig


class TestAuthentication:
    """Test suite for authentication-related functionality."""

    # List of functions to test for authentication (excluding get_openapi)
    # TBD: change to dynamically getting from MCP server
    AUTH_FUNCTIONS = [
        ("image-builder__create_blueprint", {"data": {"name": "test", "description": "test"}}),
        ("image-builder__get_blueprints", {"limit": 7, "offset": 0, "search_string": ""}),
        ("image-builder__get_blueprint_details", {"blueprint_identifier": "12345678-1234-1234-1234-123456789012"}),
        ("image-builder__get_composes", {"limit": 7, "offset": 0, "search_string": ""}),
        ("image-builder__get_compose_details", {"compose_identifier": "12345678-1234-1234-1234-123456789012"}),
        ("image-builder__blueprint_compose", {"blueprint_uuid": "12345678-1234-1234-1234-123456789012"}),
    ]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("function_name,kwargs", AUTH_FUNCTIONS)
    @pytest.mark.parametrize(
        "inmemory_test_mcp_server",
        [MCPTestServerConfig(transport="http", client_id="wrong-client-id", client_secret="wrong-client-secret")],
        indirect=True,
    )
    async def test_function_wrong_auth(self, inmemory_test_mcp_server, mcp_test_client_network, function_name, kwargs):
        """Test that functions without authentication return error."""

        _ = inmemory_test_mcp_server  # pylint: disable=unused-variable

        result = await mcp_test_client_network.call_tool(function_name, kwargs)
        result = "\n".join([content.text for content in result.content])

        # Should return authentication error
        # The actual implementation makes API calls and gets 401 errors when no auth is provided
        assert "Invalid client or Invalid client credentials" in result

    @pytest.mark.asyncio
    @pytest.mark.parametrize("function_name,kwargs", AUTH_FUNCTIONS)
    @pytest.mark.parametrize(
        "inmemory_test_mcp_server",
        [MCPTestServerConfig(transport="http", client_id=None, client_secret=None)],
        indirect=True,
    )
    async def test_function_no_auth_error_message(
        self, inmemory_test_mcp_server, mcp_test_client_network, function_name, kwargs
    ):
        """Test that functions return the no_auth_error() message when authentication is missing."""
        # Create MCP server without default credentials

        _ = inmemory_test_mcp_server  # pylint: disable=unused-variable

        result = await mcp_test_client_network.call_tool(function_name, kwargs)
        result = "\n".join([content.text for content in result.content])

        # Should return authentication error message (not raise ValueError)
        assert "There seems to be a problem with the request" in result
        assert "tell the user that the MCP server setup is not valid" in result

    @pytest.mark.asyncio
    @pytest.mark.parametrize("function_name,kwargs", AUTH_FUNCTIONS)
    @pytest.mark.parametrize(
        "inmemory_test_mcp_server",
        [
            MCPTestServerConfig(transport="sse", client_id=None, client_secret=None),
            MCPTestServerConfig(transport="http", client_id=None, client_secret=None),
        ],
        indirect=True,
    )
    async def test_function_no_auth_error_message_network_transport(
        self,
        inmemory_test_mcp_server,
        mcp_test_client_network,
        function_name,
        kwargs,
    ):
        """Test that functions return the no_auth_error() message for SSE&HTTP transport.

        Tests the case when authentication is missing.
        """
        _ = inmemory_test_mcp_server  # pylint: disable=unused-variable
        basic_mcp_client = mcp_test_client_network

        tools = await basic_mcp_client.list_tools()
        tool_names = str(tools)
        assert function_name in tool_names, f"Function {function_name} not found in tools: {tool_names}"

        result_object = await basic_mcp_client.call_tool(function_name, kwargs)
        result = "\n".join([content.text for content in result_object.content])
        # Should return authentication error message (not raise ValueError)
        assert "There seems to be a problem with the request" in result
        assert "Client ID is required" in result or "insights-client-id" in result
