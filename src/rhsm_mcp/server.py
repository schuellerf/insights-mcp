"""Red Hat Subscription Management (RHSM) MCP Server.

MCP server for Red Hat Subscription Management via Red Hat Insights API.
Provides tools to manage activation keys and subscription information in Red Hat services.
"""

from typing import Annotated, Any

from pydantic import Field

from insights_mcp.mcp import InsightsMCP

mcp = InsightsMCP(
    name="Insights RHSM MCP Server",
    toolset_name="rhsm",
    api_path="api/rhsm/v2",
    instructions="""
    This server provides tools to manage Red Hat Subscription Management (RHSM) for Red Hat services.
    You can get activation keys and subscription information.

    Insights RHSM requires correct RBAC permissions to be able to use the tools. Ensure that your
    Service Account has at least these roles:
    - Subscription Management Administrator (for full access)
    - Subscription Management Viewer (for read-only access)

    The RHSM REST API supports managing:
    - Activation keys for system registration
    - Subscription information
    - Organization details
    """,
)


@mcp.tool()
async def get_activation_keys(
    limit: Annotated[int, Field(default=20, description="Maximum number of activation keys to return (default: 20).")],
    offset: Annotated[
        int, Field(default=0, description="Number of activation keys to skip for pagination (default: 0).")
    ],
) -> dict[str, Any] | str:
    """Get the list of activation keys available to the authenticated user.

    🟢 CALL IMMEDIATELY - No information gathering required.

    This endpoint returns activation keys that can be used for RHEL system registration.
    Activation keys contain subscription and configuration information needed to register
    systems with Red Hat Subscription Management.

    If the user has more questions about the activation keys,
    ask the user to go to https://console.redhat.com/insights/connector/activation-keys

    Returns:
        List of activation keys with their details including names, descriptions,
        and associated subscriptions.
    """
    # Get all activation keys from the API (no pagination parameters)
    response = await mcp.insights_client.get("activation_keys")
    if isinstance(response, str):
        return response

    # Extract the body from the API response
    if isinstance(response, dict) and "body" in response:
        activation_keys = response["body"]
    else:
        activation_keys = response

    # Apply client-side pagination
    if isinstance(activation_keys, list):
        total_count = len(activation_keys)

        # Ensure offset and limit are non-negative
        offset = max(0, offset)
        limit = max(0, limit)

        # Ensure offset doesn't exceed total count
        offset = min(offset, total_count)

        # Calculate end index ensuring it doesn't exceed total count
        start_idx = offset
        end_idx = min(offset + limit, total_count)
        paginated_keys = activation_keys[start_idx:end_idx]

        return {
            "body": paginated_keys,
            "pagination": {"count": len(paginated_keys), "limit": limit, "offset": offset, "total": total_count},
        }

    return response


@mcp.tool()
async def get_organization_info() -> dict[str, Any] | str:
    """Get organization information for the authenticated user.

    🟢 CALL IMMEDIATELY - No information gathering required.

    This endpoint returns organization details including organization ID,
    account information, and subscription status.

    Returns:
        Organization information including org_id, account details, and metadata.
    """
    response = await mcp.insights_client.get("organization")
    if isinstance(response, str):
        return response
    return response
