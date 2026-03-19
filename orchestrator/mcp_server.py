import modal
from fleet_app import app
from orchestrator.core_app import orchestrator_image
from orchestrator.tool_sandbox import mcp

@app.function(
    image=orchestrator_image,
    min_containers=1,
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
        # We use DF_MODAL_TOKEN_SECRET which should already be set in the modal environment
        expected_token = os.environ.get("DF_MODAL_TOKEN_SECRET")
        if not expected_token:
            # Fallback for dev testing if secret is missing
            expected_token = "dev_token_only"

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

@app.function(image=orchestrator_image, timeout=1800)
async def test_tool(tool_name: str | None = None):
    import os
    from fastmcp import Client
    from fastmcp.client.transports import SSETransport

    if tool_name is None:
        tool_name = "playwright_screenshot"

    expected_token = os.environ.get("DF_MODAL_TOKEN_SECRET", "dev_token_only")
    mcp_url = mcp_web.get_web_url()

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

            # Let the agent start processing
            print("Waiting 15 seconds for agent execution to kick off...")
            await page.wait_for_timeout(15000)

            await page.screenshot(path="/workspace/chainlit_agent_started.png", full_page=True)
            print("Success! Automation completed and screenshot saved to /workspace/chainlit_agent_started.png")

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
