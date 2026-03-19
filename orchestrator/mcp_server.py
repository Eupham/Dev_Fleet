import modal
from fleet_app import app
from orchestrator.core_app import orchestrator_image
from orchestrator.tool_sandbox import mcp

@app.function(
    image=orchestrator_image,
    min_containers=1,
    secrets=[modal.Secret.from_name("dev-fleet-mcp-secret")]
)
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def mcp_web():
    """ASGI web endpoint for the MCP server"""
    import os
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse

    fastapi_app = FastAPI()

    # Middleware to protect all routes (including mounted ones)
    @fastapi_app.middleware("http")
    async def auth_middleware(request: Request, call_next):
        # We use MCP_AUTH_TOKEN injected from the dev-fleet-mcp-secret Modal secret
        expected_token = os.environ.get("MCP_AUTH_TOKEN")
        if not expected_token:
            return JSONResponse(
                status_code=500,
                content={"detail": "Server misconfiguration: missing MCP_AUTH_TOKEN"}
            )

        auth_header = request.headers.get("Authorization")
        if not auth_header or auth_header != f"Bearer {expected_token}":
            return JSONResponse(
                status_code=401,
                content={"detail": "Unauthorized"},
                headers={"WWW-Authenticate": "Bearer"}
            )

        return await call_next(request)

    # Mount the MCP server's SSE application
    mcp_app = mcp.sse_app()

    # Append the lifespan handler
    fastapi_app.router.lifespan_context = mcp_app.router.lifespan_context
    fastapi_app.mount("/", mcp_app, "mcp")

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
            # Target the deployed UI url
            ui_url = mcp_url.replace("mcp-web", "ui")
            print(f"Testing {tool_name} against UI url: {ui_url}...")

            result = await client.call_tool("playwright_screenshot", {"url": ui_url, "output_path": "/workspace/mcp_test_screenshot.png"})
            print(f"Screenshot Result: {result.data}")

            # Use `run_code` to execute a script that interacts with the UI as a user
            script = f"""
import asyncio
from playwright.async_api import async_playwright

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
            await page.fill('textarea[id="chat-input"]', prompt)
            await page.press('textarea[id="chat-input"]', 'Enter')

            # Rather than waiting for a brittle selector like `#stop-button` which may not exist
            # or may time out, we just wait a fixed 3.5 minutes to let the agent compile tools
            # and run its LLM calls in the background. The user wants to see "till it runs clean."
            print("Waiting 200 seconds for agent to execute and print output to the screen...")
            await page.wait_for_timeout(200000)

            await page.screenshot(path="/workspace/chainlit_agent_finished.png", full_page=True)
            print("Success! UI automation completed and screenshot saved to /workspace/chainlit_agent_finished.png")

        except Exception as e:
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
