import modal
from fleet_app import app
from orchestrator.core_app import orchestrator_image
from orchestrator.tool_sandbox import mcp

@app.function(
    image=orchestrator_image,
    min_containers=1,
    volumes={
        "/root/public": modal.Volume.from_name("dev-fleet-state-vol", create_if_missing=True),
        "/workspace": modal.Volume.from_name("dev-fleet-workspace", create_if_missing=True)
    },
    secrets=[modal.Secret.from_name("dev-fleet-mcp-secret")]
)
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def mcp_web():
    """ASGI web endpoint for the MCP server"""
    import os
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse

    # We must use a pure ASGI middleware because FastAPI's `@app.middleware("http")`
    # (which uses Starlette's BaseHTTPMiddleware) crashes when handling Server-Sent Events (SSE) streams
    # due to chunked transfer assertions upon disconnect.
    class MCPAuthMiddleware:
        def __init__(self, app):
            self.app = app

        async def __call__(self, scope, receive, send):
            if scope["type"] not in ["http", "websocket"]:
                return await self.app(scope, receive, send)

            expected_token = os.environ.get("MCP_AUTH_TOKEN")
            if not expected_token:
                async def send_500(message):
                    if message["type"] == "http.response.start":
                        message["status"] = 500
                    await send(message)
                return await self.app(scope, receive, send_500)

            # Extract headers (ASGI headers are lists of byte tuples)
            headers = dict(scope.get("headers", []))
            auth_header = headers.get(b"authorization", b"").decode("utf-8")

            if not auth_header or auth_header != f"Bearer {expected_token}":
                async def send_401(message):
                    if message["type"] == "http.response.start":
                        message["status"] = 401
                    await send(message)
                return await self.app(scope, receive, send_401)

            return await self.app(scope, receive, send)

    fastapi_app = FastAPI()

    mcp_app = mcp.sse_app()
    # Wrap the mounted SSE app in our pure ASGI auth middleware to avoid SSE assertion crashes
    protected_mcp_app = MCPAuthMiddleware(mcp_app)

    fastapi_app.router.lifespan_context = mcp_app.router.lifespan_context
    fastapi_app.mount("/", protected_mcp_app, "mcp")

    return fastapi_app

@app.function(image=orchestrator_image, timeout=1800, secrets=[modal.Secret.from_name("dev-fleet-mcp-secret")])
async def test_tool(tool_name: str | None = None):
    import os
    from fastmcp import Client
    from fastmcp.client.transports import SSETransport

    if tool_name is None:
        tool_name = "playwright_screenshot"

    expected_token = os.environ.get("MCP_AUTH_TOKEN")
    if not expected_token:
        raise ValueError("Missing MCP_AUTH_TOKEN in environment")

    mcp_url = await mcp_web.get_web_url.aio()

    transport = SSETransport(
        url=f"{mcp_url}/sse",
        headers={"Authorization": f"Bearer {expected_token}"}
    )
    client = Client(transport)

    async with client:
        tools = await client.list_tools()
        print("Available tools:", [t.name for t in tools])

        if any(t.name == tool_name for t in tools):
            # Target the deployed UI url, dropping any -dev suffix to test against the stable deployment
            ui_url = mcp_url.replace("mcp-web-dev", "ui").replace("mcp-web", "ui")
            print(f"Testing {tool_name} against UI url: {ui_url}...")

            result = await client.call_tool("playwright_screenshot", {"url": ui_url, "output_path": "/workspace/mcp_test_screenshot.png"})
            print(f"Screenshot Result: {result.data}")

            # Use `run_code` to execute a script that interacts with the UI as a user.
            # We save the screenshot directly to /app/public/ so it's statically served
            # by Chainlit and directly viewable via the browser by the user.
            script = f"""
import asyncio
from playwright.async_api import async_playwright
import os

async def interact():
    print(f"Connecting to UI at {ui_url} ...")
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        try:
            await page.goto("{ui_url}", timeout=30000)

            # Wait for Chainlit chat input to become available
            await page.wait_for_selector('textarea[id="chat-input"]', timeout=30000)

            prompt = "Research a random domain online. Build an app related to your research. Test it till it runs clean."
            print(f"Submitting prompt: {{prompt}}")
            # Type slowly to trigger React events properly
            await page.type('textarea[id="chat-input"]', prompt, delay=10)
            await page.press('textarea[id="chat-input"]', 'Enter')

            # Rather than waiting for a brittle selector like `#stop-button` which may not exist
            # or may time out, we just wait a fixed 3.5 minutes to let the agent compile tools
            # and run its LLM calls in the background. The user wants to see "till it runs clean."
            print("Waiting 200 seconds for agent to execute and print output to the screen...")
            await page.wait_for_timeout(200000)

            os.makedirs("/root/public", exist_ok=True)
            await page.screenshot(path="/root/public/chainlit_agent_finished.png", full_page=True)
            print("Success! UI automation completed and screenshot saved to /root/public/chainlit_agent_finished.png")

            # Ensure the URL is properly formed with a trailing slash
            base_url = "{ui_url}" if "{ui_url}".endswith("/") else f"{ui_url}/"
            print(f"View the test screenshot at: {{base_url}}public/chainlit_agent_finished.png")

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error driving UI: {{e}}")
        finally:
            await browser.close()

asyncio.run(interact())
"""
            print("Executing Chainlit UI driver script via MCP `run_code` tool...")
            automation_result = await client.call_tool("run_code", {"language": "python", "code": script})
            print(f"Automation Output:\\n{automation_result.data}")

            return

    raise Exception(f"could not find tool {tool_name}")
