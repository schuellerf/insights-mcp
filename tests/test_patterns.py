"""Test patterns - reusable test functions for MCP tools.

This module contains generic test functions that can be reused across different MCP modules.
The actual test parameters are defined in the specific module test files.
"""

from typing import Any, Dict


def assert_mcp_tool_descriptions_and_annotations(
    mcp_tools_stdio,
    subtests,
    tool_name: str,
    expected_desc: str,
    params: Dict[str, Dict[str, Any]],
):
    """Reusable test function to verify MCP tools include proper descriptions and annotations.

    Args:
        mcp_tools_stdio: List of MCP tools from the mcp_tools_stdio fixture
        subtests: pytest subtests fixture for granular test reporting
        tool_name: Name of the tool to test (e.g., "image-builder__get_blueprints")
        expected_desc: Expected start of the tool description
        params: Dictionary of parameter names to their expected schema properties
    """
    tools = mcp_tools_stdio

    # Build map for quick lookup
    name_to_tool = {getattr(t.metadata, "name", ""): t for t in tools}
    assert tool_name in name_to_tool, f"Tool not found: {tool_name}"
    tool = name_to_tool[tool_name]

    # Description check
    desc = getattr(tool.metadata, "description", "") or ""
    assert desc.startswith(expected_desc)

    fn_schema = getattr(tool.metadata, "fn_schema", None)
    assert fn_schema is not None, f"{tool_name}: fn_schema is None"
    assert hasattr(fn_schema, "model_json_schema"), f"{tool_name}: fn_schema.model_json_schema missing"
    schema_obj = fn_schema.model_json_schema()  # type: ignore[attr-defined]
    assert isinstance(schema_obj, dict), f"{tool_name}: invalid fn_schema (model_json_schema not dict)"

    props = schema_obj.get("properties", {}) or {}
    for param_name, expected_param_desc in params.items():
        with subtests.test(param=param_name):
            desc = props.get(param_name, {}).get("description", "")
            assert desc.startswith(expected_param_desc.get("description", ""))
            assert props.get(param_name, {}).get("default") == expected_param_desc.get("default")
            assert props.get(param_name, {}).get("type") == expected_param_desc.get("type")
            assert props.get(param_name, {}).get("anyOf") == expected_param_desc.get("anyOf")
    # Note: Testing defaults would be ideal but
    # default is null in FastMCP schema by design; actual defaulting occurs server-side


def assert_transport_types_expose_tool(mcp_tools_nework, request, tool_name: str):
    """Reusable test function to verify transport types can expose a specific tool.

    Args:
        mcp_tools_nework: List of MCP tools from the mcp_tools_nework fixture
        request: pytest request fixture to get transport information
        tool_name: Name of the tool to verify (e.g., "image-builder__get_blueprints")
    """
    # Get transport from the fixture parameter
    config = request.node.callspec.params["inmemory_test_mcp_server"]
    transport = config.transport

    # Build map for quick lookup
    tool_names = {getattr(t.metadata, "name", "") for t in mcp_tools_nework}

    # Verify tool is available
    assert tool_name in tool_names, (
        f"{tool_name} not found in tools for {transport} transport. Available tools: {tool_names}"
    )


def assert_stdio_transport_exposes_tool(mcp_tools_stdio, tool_name: str):
    """Reusable test function to verify stdio transport exposes a specific tool.

    Args:
        mcp_tools_stdio: List of MCP tools from the mcp_tools_stdio fixture
        tool_name: Name of the tool to verify (e.g., "image-builder__get_blueprints")
    """
    # Build map for quick lookup
    tool_names = {getattr(t.metadata, "name", "") for t in mcp_tools_stdio}

    # Verify tool is available
    assert tool_name in tool_names, f"{tool_name} not found in tools for stdio transport. Available tools: {tool_names}"
