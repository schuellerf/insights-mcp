"""Test the MCP API.

Test includes:
- Generic MCP server functionality and transport validation
- Blueprint pattern tests are now in module-specific test files
"""


async def test_mcp_server_provides_tools(mcp_tools_stdio):
    """Test that the MCP server provides some tools."""
    assert len(mcp_tools_stdio) > 0, "MCP server should provide at least one tool"


async def test_all_tools_have_metadata(mcp_tools_stdio):
    """Test that all tools have proper metadata."""
    for tool in mcp_tools_stdio:
        # Check that the tool has the expected attributes
        assert tool.metadata.name is not None, f"Tool {tool.metadata.name} has None name"
        assert tool.metadata.description is not None, f"Tool {tool.metadata.name} has None description"
