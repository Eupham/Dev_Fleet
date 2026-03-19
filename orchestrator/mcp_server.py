import modal
from fleet_app import app
from orchestrator.core_app import orchestrator_image
from orchestrator.tool_sandbox import mcp

@app.function(image=orchestrator_image)
@modal.asgi_app()
def mcp_web():
    """ASGI web endpoint for the MCP server"""
    from fastapi import FastAPI

    mcp_app = mcp.http_app(transport="streamable-http", stateless_http=True)

    fastapi_app = FastAPI(lifespan=mcp_app.router.lifespan_context)
    fastapi_app.mount("/", mcp_app, "mcp")

    return fastapi_app

@app.function(image=orchestrator_image)
async def test_tool(tool_name: str | None = None):
    from fastmcp import Client
    from fastmcp.client.transports import StreamableHttpTransport

    if tool_name is None:
        tool_name = "execute_bash"

    transport = StreamableHttpTransport(url=f"{mcp_web.web_url}/mcp/")
    client = Client(transport)

    async with client:
        tools = await client.list_tools()

        for tool in tools:
            print(tool)
            if tool.name == tool_name:
                # We need to pass args to execute_bash. We can just test a simple echo.
                result = await client.call_tool(tool_name, {"code": "echo hello"})
                print(result.data)
                return

    raise Exception(f"could not find tool {tool_name}")
